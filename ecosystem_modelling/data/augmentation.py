"""
data/augmentation.py

Ecologically-principled graph augmentation for SSL pretraining.

Two views of the same ecosystem are created for contrastive learning.
Augmentation is NOT purely random — it respects ecological importance:
  - Strong predation edges are less likely to be dropped
  - Weak competition edges are dropped more freely
"""

import torch
import copy
import numpy as np
from typing import Tuple
from torch_geometric.data import Data

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import AUG_EDGE_DROP_RATE, AUG_WEIGHT_NOISE_STD


def augment_graph(data: Data,
                  edge_drop_rate: float = AUG_EDGE_DROP_RATE,
                  weight_noise_std: float = AUG_WEIGHT_NOISE_STD,
                  ) -> Tuple[Data, Data]:
    """
    Create two augmented views of the same ecosystem graph for contrastive SSL.

    View A — ecologically-weighted edge dropping:
        Edges with higher weight (stronger interactions) are LESS likely dropped.
        Mirrors ecological reality: you can remove weak competition links
        but not apex predation links without fundamentally changing the web.

    View B — edge weight perturbation + minor node feature noise:
        Weights shifted by Gaussian noise.
        Node features perturbed slightly (population variation).

    Returns (view_a, view_b).
    """
    view_a = _weighted_edge_drop(data, drop_rate=edge_drop_rate)
    view_b = _weight_perturbation(data, noise_std=weight_noise_std)
    return view_a, view_b


def mask_edges_for_reconstruction(data: Data,
                                   mask_rate: float = 0.2,
                                   ) -> Tuple[Data, torch.Tensor, torch.Tensor]:
    """
    Mask a fraction of edges for the masked-edge-reconstruction SSL objective.

    Masking probability is inversely proportional to edge weight:
        p_mask(e) ∝ 1 / (weight(e) + ε)
    so weak edges are masked more often (easier to reconstruct from context).

    Returns:
        masked_data   — Data object with masked edges removed
        masked_edges  — (2, K) tensor of masked edge indices
        masked_labels — (K, 2) tensor of [weight, edge_type] for masked edges
    """
    n_edges = data.edge_index.size(1)
    if n_edges == 0:
        return data, torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 2)

    # Masking probability inversely proportional to weight
    weights = data.edge_attr[:, 0].float()
    inv_w   = 1.0 / (weights + 1e-6)
    prob    = inv_w / inv_w.sum()
    n_mask  = max(1, int(mask_rate * n_edges))
    n_mask  = min(n_mask, n_edges - 1)

    mask_idx = torch.multinomial(prob, n_mask, replacement=False)
    keep_mask = torch.ones(n_edges, dtype=torch.bool)
    keep_mask[mask_idx] = False

    masked_data = Data(
        x          = data.x,
        edge_index = data.edge_index[:, keep_mask],
        edge_attr  = data.edge_attr[keep_mask],
        edge_type  = data.edge_type[keep_mask],
        trophic    = data.trophic,
        growth     = data.growth,
        num_nodes  = data.num_nodes,
    )
    masked_data.eco_name = getattr(data, "eco_name", "unknown")

    masked_edges  = data.edge_index[:, mask_idx]
    masked_labels = data.edge_attr[mask_idx]    # [K, 2]: weight, type

    return masked_data, masked_edges, masked_labels


# ── Internal helpers ──────────────────────────────────────────────────────────

def _weighted_edge_drop(data: Data, drop_rate: float) -> Data:
    """Drop edges with probability inversely proportional to weight."""
    n_edges = data.edge_index.size(1)
    if n_edges == 0:
        return data

    weights   = data.edge_attr[:, 0].float()
    norm_w    = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    # Strong edges have low drop prob, weak edges have high drop prob
    drop_prob = drop_rate * (1.0 - norm_w)
    keep      = torch.rand(n_edges) > drop_prob

    # Always keep at least 3 edges
    if keep.sum() < 3:
        keep[:3] = True

    out = Data(
        x          = data.x.clone(),
        edge_index = data.edge_index[:, keep],
        edge_attr  = data.edge_attr[keep],
        edge_type  = data.edge_type[keep],
        trophic    = data.trophic.clone(),
        growth     = data.growth.clone(),
        num_nodes  = data.num_nodes,
    )
    out.eco_name = getattr(data, "eco_name", "unknown")
    return out


def _weight_perturbation(data: Data, noise_std: float) -> Data:
    """Perturb edge weights and node features with Gaussian noise."""
    ea = data.edge_attr.clone()
    ea[:, 0] = (ea[:, 0] + torch.randn_like(ea[:, 0]) * noise_std).clamp(min=0.0)

    x_noisy = data.x.clone()
    x_noisy = x_noisy + torch.randn_like(x_noisy) * 0.02   # small feature noise

    out = Data(
        x          = x_noisy,
        edge_index = data.edge_index.clone(),
        edge_attr  = ea,
        edge_type  = data.edge_type.clone(),
        trophic    = data.trophic.clone(),
        growth     = data.growth.clone(),
        num_nodes  = data.num_nodes,
    )
    out.eco_name = getattr(data, "eco_name", "unknown")
    return out
