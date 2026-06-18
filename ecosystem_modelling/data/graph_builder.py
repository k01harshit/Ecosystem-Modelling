"""
data/graph_builder.py

Converts EcosystemData dicts (from fetch.py) into PyTorch Geometric Data objects.

Each species → node with a feature vector:
  [trophic_level, log_body_mass, diet_type (one-hot 4),
   habitat (one-hot 6), log_repro_rate, intrinsic_growth]
  → total dim = 16  (matches NODE_FEATURE_DIM in configs.py)

Each interaction → directed edge with:
  edge_attr  = [weight, edge_type_id]
  edge_type  = integer (0=predation, 1=competition, 2=mutualism, 3=parasitism)
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Data

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import EDGE_TYPES, NODE_FEATURE_DIM

logger = logging.getLogger(__name__)

EDGE_TYPE_MAP = {et: i for i, et in enumerate(EDGE_TYPES)}
DIET_TYPES  = 4
HABITAT_TYPES = 6


def build_pyg_graph(eco: Dict,
                    add_self_loops: bool = False) -> Optional[Data]:
    """
    Convert one EcosystemData dict to a PyTorch Geometric Data object.

    Returns None if the ecosystem has fewer than 3 species.
    """
    species = eco["species"]
    attrs   = eco["attributes"]
    interactions = eco["interactions"]

    if len(species) < 3:
        return None

    # ── Node features ─────────────────────────────────────────────────────
    sp2idx = {sp: i for i, sp in enumerate(species)}
    n = len(species)
    x = torch.zeros(n, NODE_FEATURE_DIM)

    for sp, idx in sp2idx.items():
        a = attrs.get(sp, {})
        x[idx] = _build_node_feature(a)

    # ── Edge index and attributes ──────────────────────────────────────────
    edge_list   = []
    edge_attrs  = []
    edge_types  = []

    for inter in interactions:
        src = inter.get("source", "")
        tgt = inter.get("target", "")
        if src not in sp2idx or tgt not in sp2idx:
            continue
        i, j = sp2idx[src], sp2idx[tgt]
        w  = float(inter.get("weight", 1.0))
        et = EDGE_TYPE_MAP.get(inter.get("type", "predation"), 0)
        edge_list.append([i, j])
        edge_attrs.append([w, float(et)])
        edge_types.append(et)

    if len(edge_list) == 0:
        return None

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_attrs, dtype=torch.float)
    edge_type  = torch.tensor(edge_types, dtype=torch.long)

    # ── Trophic levels as separate tensor (SSL target) ─────────────────────
    trophic = torch.tensor(
        [attrs.get(sp, {}).get("trophic_level", 2.0) for sp in species],
        dtype=torch.float,
    )

    # ── Intrinsic growth rates (for LV ODE) ───────────────────────────────
    growth = torch.tensor(
        [attrs.get(sp, {}).get("intrinsic_growth", 1.0) for sp in species],
        dtype=torch.float,
    )

    data = Data(
        x          = x,
        edge_index = edge_index,
        edge_attr  = edge_attr,
        edge_type  = edge_type,
        trophic    = trophic,
        growth     = growth,
        num_nodes  = n,
    )
    data.species_names = species
    data.eco_name      = eco.get("name", "unknown")
    data.source        = eco.get("source", "unknown")

    # Store LV matrix if available (synthetic ecosystems)
    if "lv_matrix" in eco:
        data.lv_matrix = torch.tensor(eco["lv_matrix"], dtype=torch.float)

    return data


def build_dataset(ecosystems: List[Dict]) -> List[Data]:
    """Convert a list of EcosystemData dicts to a list of PyG Data objects."""
    graphs = []
    for eco in ecosystems:
        g = build_pyg_graph(eco)
        if g is not None:
            graphs.append(g)
    logger.info(f"Built {len(graphs)} PyG graphs from {len(ecosystems)} ecosystems")
    return graphs


def insert_invasive_species(graph: Data,
                             invader_attrs: Dict,
                             predicted_edges: List[Dict]) -> Data:
    """
    Insert a new invasive species node into an existing ecosystem graph.

    Args:
        graph:           Original PyG Data object
        invader_attrs:   Attribute dict for the invader
        predicted_edges: List of {target_idx, type, weight, direction}
                         direction: "in" (invader eaten) or "out" (invader eats)

    Returns new Data object with invader node appended.
    """
    n     = graph.num_nodes
    x_inv = _build_node_feature(invader_attrs).unsqueeze(0)
    x_new = torch.cat([graph.x, x_inv], dim=0)

    new_edges = []
    new_attrs = []
    new_types = []
    inv_idx   = n  # new node index

    for pe in predicted_edges:
        tgt = pe["target_idx"]
        w   = float(pe.get("weight", 1.0))
        et  = EDGE_TYPE_MAP.get(pe.get("type", "predation"), 0)
        if pe.get("direction", "out") == "out":
            # invader → existing species (invader eats)
            new_edges.append([inv_idx, tgt])
        else:
            # existing species → invader
            new_edges.append([tgt, inv_idx])
        new_attrs.append([w, float(et)])
        new_types.append(et)

    if new_edges:
        new_edge_t = torch.tensor(new_edges, dtype=torch.long).t()
        new_edge_a = torch.tensor(new_attrs,  dtype=torch.float)
        new_edge_tp = torch.tensor(new_types, dtype=torch.long)
        edge_index = torch.cat([graph.edge_index, new_edge_t], dim=1)
        edge_attr  = torch.cat([graph.edge_attr,  new_edge_a], dim=0)
        edge_type  = torch.cat([graph.edge_type,  new_edge_tp], dim=0)
    else:
        edge_index = graph.edge_index
        edge_attr  = graph.edge_attr
        edge_type  = graph.edge_type

    # Trophic / growth for invader (from attrs)
    t_inv = torch.tensor([invader_attrs.get("trophic_level", 2.0)])
    g_inv = torch.tensor([invader_attrs.get("intrinsic_growth", 1.0)])

    augmented = Data(
        x          = x_new,
        edge_index = edge_index,
        edge_attr  = edge_attr,
        edge_type  = edge_type,
        trophic    = torch.cat([graph.trophic, t_inv]),
        growth     = torch.cat([graph.growth,  g_inv]),
        num_nodes  = n + 1,
    )
    augmented.eco_name      = graph.eco_name + "_invaded"
    augmented.invasive_idx  = inv_idx
    return augmented


# ── Feature builder ───────────────────────────────────────────────────────────

def _build_node_feature(a: Dict) -> torch.Tensor:
    """
    Build a fixed-size (NODE_FEATURE_DIM=16) feature vector for one species.

    Layout:
      [0]     trophic_level (normalised to [0,1] assuming max=6)
      [1]     log_body_mass (normalised)
      [2:6]   diet_type one-hot  (4 dims)
      [6:12]  habitat    one-hot  (6 dims)
      [12]    log_repro_rate (normalised)
      [13]    intrinsic_growth (normalised)
      [14-15] padding zeros
    """
    feat = torch.zeros(NODE_FEATURE_DIM)
    feat[0]  = float(a.get("trophic_level", 2.0)) / 6.0
    feat[1]  = float(a.get("log_body_mass",  2.0)) / 10.0
    diet = int(a.get("diet_type", 0)) % DIET_TYPES
    feat[2 + diet] = 1.0
    hab  = int(a.get("habitat", 0)) % HABITAT_TYPES
    feat[6 + hab]  = 1.0
    feat[12] = float(a.get("log_repro_rate",   0.0)) / 5.0
    feat[13] = float(a.get("intrinsic_growth", 1.0)) / 3.0
    return feat
