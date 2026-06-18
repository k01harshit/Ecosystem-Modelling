"""
models/lv_severity_model.py

LV-guided synthetic severity predictor — fully unsupervised.

The key insight: when we generate synthetic ecosystems, we know the true
structural disruption because we control the LV parameters. We can define
a ground-truth synthetic severity as a deterministic function of the ODE
outputs before and after a simulated invasion — no real event labels needed.

This module:
  1. Defines synthetic_severity from LV outputs (label-free by construction)
  2. Trains a small MLP to predict it from ODE features on synthetic data
  3. Uses the trained MLP to produce an LV-severity score for real events

The MLP never sees real event labels, severities, or outcome strings.
Its training signal comes entirely from synthetic ecosystem dynamics.

Synthetic severity definition (ecologically motivated):
    s = w1 * extinct_frac_change     (fraction of species lost post-invasion)
      + w2 * lambda_max_change        (how much the system becomes less stable)
      + w3 * biomass_change           (total biomass reduction)
      + w4 * invader_dominance        (invader final pop / mean native pop)

All four components are computable from LV simulation outputs alone.
"""

from __future__ import annotations

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


# ── Synthetic severity definition ─────────────────────────────────────────────

def compute_synthetic_severity(
    pre_ode:  Dict,
    post_ode: Dict,
) -> float:
    """
    Compute a label-free severity score from two ODE outputs:
    the pre-invasion system and the post-invasion system.

    This is the training signal for the MLP — derived purely from
    LV simulation, no ecological outcome labels involved.

    Components:
      extinct_change  : how many more species go extinct post-invasion
      stability_change: how much lambda_max increases (less stable)
      biomass_change  : fractional loss of total biomass
      invader_dom     : invader final pop relative to native mean
    """
    # Soft extinction fraction (differentiable proxy)
    def ext_frac(pop):
        return float(torch.sigmoid(-400.0 * (pop.detach().cpu() - 0.01)).mean())

    pre_ext  = ext_frac(pre_ode["final_pop"])
    post_ext = ext_frac(post_ode["final_pop"][:-1])  # exclude invader itself
    extinct_change = float(np.clip(post_ext - pre_ext, 0.0, 1.0))

    # Lambda_max change (positive = system became less stable)
    pre_lm  = float(np.clip(pre_ode["max_eigenvalue"],  -5.0, 5.0))
    post_lm = float(np.clip(post_ode["max_eigenvalue"], -5.0, 5.0))
    stability_change = float(np.clip((post_lm - pre_lm) / 5.0, 0.0, 1.0))

    # Biomass change (fraction lost)
    pre_bm  = float(pre_ode["final_pop"].detach().cpu().sum())
    post_bm = float(post_ode["final_pop"][:-1].detach().cpu().sum())
    biomass_change = float(np.clip(
        (pre_bm - post_bm) / (pre_bm + 1e-6), 0.0, 1.0))

    # Invader dominance
    inv_dom = float(np.clip(post_ode.get("invader_ratio", 0.0), 0.0, 1.0))

    severity = (
        0.40 * extinct_change    +
        0.25 * stability_change  +
        0.20 * biomass_change    +
        0.15 * inv_dom
    )
    return float(np.clip(severity, 0.0, 1.0))


# ── Feature extraction for the MLP (ODE-only, no trait features) ─────────────

def extract_lv_features(pre_ode: Dict, post_ode: Dict) -> np.ndarray:
    """
    Extract 8 ODE-derived features for the severity MLP.
    These are purely from LV simulation — no invader traits, no labels.

    Features:
      0  pre_extinct_frac       — pre-invasion extinction fraction
      1  post_extinct_frac      — post-invasion extinction fraction (natives)
      2  extinct_change         — post - pre extinction fraction
      3  pre_lambda_max         — pre-invasion Jacobian stability
      4  post_lambda_max        — post-invasion Jacobian stability
      5  lambda_change          — post - pre lambda_max
      6  biomass_change         — fractional biomass loss
      7  invader_dominance      — invader final pop / native mean
    """
    def ext_frac(pop):
        return float(torch.sigmoid(-400.0 * (pop.detach().cpu() - 0.01)).mean())

    pre_ext  = ext_frac(pre_ode["final_pop"])
    post_ext = ext_frac(post_ode["final_pop"][:-1])

    pre_lm  = float(np.clip(pre_ode["max_eigenvalue"],  -5.0, 5.0)) / 5.0
    post_lm = float(np.clip(post_ode["max_eigenvalue"], -5.0, 5.0)) / 5.0

    pre_bm  = float(pre_ode["final_pop"].detach().cpu().sum())
    post_bm = float(post_ode["final_pop"][:-1].detach().cpu().sum())
    bm_change = float(np.clip((pre_bm - post_bm) / (pre_bm + 1e-6), 0.0, 1.0))

    inv_dom = float(np.clip(post_ode.get("invader_ratio", 0.0), 0.0, 1.0))

    return np.array([
        pre_ext,
        post_ext,
        float(np.clip(post_ext - pre_ext, 0.0, 1.0)),
        pre_lm,
        post_lm,
        float(np.clip(post_lm - pre_lm, 0.0, 1.0)),
        bm_change,
        inv_dom,
    ], dtype=np.float32)


# ── Small MLP ─────────────────────────────────────────────────────────────────

class SeverityMLP(nn.Module):
    """
    Tiny MLP: 8 ODE features → scalar severity prediction in [0, 1].
    Trained on synthetic data only — no real event labels.
    """
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Training on synthetic data ─────────────────────────────────────────────────


def _fast_lv_simulate(A: torch.Tensor, r: torch.Tensor,
                       n_steps: int = 60, t_end: float = 15.0) -> torch.Tensor:
    """Lightweight Euler integrator. Longer integration for clearer severity signal."""
    x  = torch.full((r.size(0),), 0.5, device=A.device)
    dt = t_end / n_steps
    for _ in range(n_steps):
        x = (x + dt * x * (r + A @ x)).clamp(min=0.0, max=1e3)
    return x


def _fast_ode_out(A: torch.Tensor, r: torch.Tensor) -> dict:
    """Minimal ODE output dict for MLP training feature extraction."""
    final_pop = _fast_lv_simulate(A, r)
    max_re    = float(A.diagonal().max().item())
    inv_final = final_pop[-1].item()
    nat_mean  = final_pop[:-1].mean().item() if final_pop.size(0) > 1 else 1.0
    inv_ratio = float(np.clip(inv_final / (nat_mean + 1e-6), 0, 20))
    return {"final_pop": final_pop, "max_eigenvalue": max_re, "invader_ratio": inv_ratio}


def train_severity_mlp(
    dynamics,
    encoder,
    synthetic_graphs: List,
    n_invasions_per_graph: int = 3,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
) -> SeverityMLP:
    """
    Train the SeverityMLP entirely on synthetic ecosystem graphs.

    For each synthetic graph:
      1. Run the pre-invasion ODE to get baseline dynamics.
      2. Simulate n random 'invasions' by appending a random new node
         with random growth rate and interaction strengths.
      3. Run the post-invasion ODE.
      4. Compute synthetic_severity from the ODE outputs.
      5. Train the MLP to predict this severity from LV features.

    No real event labels, no outcome strings, no severity values from
    the real invasion dataset are used at any point.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    mlp = SeverityMLP().to(device)
    opt = optim.Adam(mlp.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    dynamics.eval()
    encoder.eval()
    for p in dynamics.parameters():
        p.requires_grad = False
    for p in encoder.parameters():
        p.requires_grad = False

    rng = np.random.default_rng(seed)
    logger.info(f"Training SeverityMLP: {len(synthetic_graphs)} graphs × "
                f"{n_invasions_per_graph} invasions (fast Euler, ~30s)...")

    for epoch in range(1, epochs + 1):
        mlp.train()
        total_loss = 0.0
        n_pairs = 0

        for g in synthetic_graphs:
            g = g.to(device)
            if g.num_nodes < 4:
                continue

            with torch.no_grad():
                node_emb, _ = encoder(g.x, g.edge_index, g.edge_type)
                A_pre   = dynamics.build_interaction_matrix(
                    node_emb, g.edge_index, g.edge_attr, g.num_nodes)
                pre_ode = _fast_ode_out(A_pre, g.growth)

            for _i in range(n_invasions_per_graph):
                # Cycle archetypes: 0=apex, 1=pathogen, 2=benign
                archetype = _i % 3
                inv_emb = _random_invader_embedding(
                    node_emb, rng, device, archetype=archetype)
                post_node_emb, post_ei, post_ea, post_growth = \
                    _append_invader(node_emb, g, inv_emb, rng, device,
                                   archetype=archetype)

                with torch.no_grad():
                    N2      = post_node_emb.size(0)
                    A_post  = dynamics.build_interaction_matrix(
                        post_node_emb, post_ei, post_ea, N2)
                    post_ode = _fast_ode_out(A_post, post_growth)

                # Synthetic severity (the training signal)
                syn_sev = compute_synthetic_severity(pre_ode, post_ode)
                target  = torch.tensor([syn_sev], dtype=torch.float32, device=device)

                # LV features (inputs to MLP)
                lv_feat = torch.tensor(
                    extract_lv_features(pre_ode, post_ode),
                    dtype=torch.float32, device=device).unsqueeze(0)

                # Forward + loss
                pred = mlp(lv_feat)
                loss = nn.functional.mse_loss(pred, target)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
                opt.step()

                total_loss += loss.item()
                n_pairs    += 1

        sched.step()
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"  SeverityMLP epoch {epoch:3d}/{epochs} | "
                        f"mse={total_loss/max(n_pairs,1):.5f}")

    # Unfreeze
    for p in dynamics.parameters():
        p.requires_grad = True
    for p in encoder.parameters():
        p.requires_grad = True

    logger.info("SeverityMLP training complete.")
    return mlp


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_lv_severity(
    mlp: SeverityMLP,
    dynamics,
    encoder,
    pre_graph,
    post_graph,
    device: str = "cpu",
) -> float:
    """
    Predict LV-guided severity for a real invasion event.
    The MLP was trained only on synthetic data — no real labels involved.
    Returns a float in [0, 1].
    """
    mlp.eval(); dynamics.eval(); encoder.eval()

    pre_g  = pre_graph.to(device)
    post_g = post_graph.to(device)

    pre_node_emb, _  = encoder(pre_g.x,  pre_g.edge_index,  pre_g.edge_type)
    post_node_emb, _ = encoder(post_g.x, post_g.edge_index, post_g.edge_type)

    pre_ode  = dynamics(pre_node_emb,  pre_g.edge_index,  pre_g.edge_attr,  pre_g.growth)
    post_ode = dynamics(post_node_emb, post_g.edge_index, post_g.edge_attr, post_g.growth)

    lv_feat = torch.tensor(
        extract_lv_features(pre_ode, post_ode),
        dtype=torch.float32, device=device
    ).unsqueeze(0)

    score = mlp(lv_feat).item()
    return float(np.clip(score, 0.0, 1.0))


# ── Synthetic invasion helpers ─────────────────────────────────────────────────

def _random_invader_embedding(
    node_emb: torch.Tensor,
    rng: np.random.Generator,
    device: str,
    archetype: int = 0,
) -> torch.Tensor:
    """Three ecologically distinct archetypes:
    0=apex predator, 1=fast pathogen/r-strategist, 2=benign visitor.
    Diversity in training data gives MLP meaningful severity variation.
    """
    N, D = node_emb.shape
    idx   = int(rng.integers(0, N))
    noise = torch.tensor(
        rng.normal(0, 0.2, D), dtype=torch.float32, device=device)
    return (node_emb[idx] + noise).unsqueeze(0)


def _append_invader(
    node_emb: torch.Tensor,
    g,
    inv_emb: torch.Tensor,
    rng: np.random.Generator,
    device: str,
    archetype: int = 0,
):
    """Archetype-specific invasion simulation:
    0=apex predator: few strong edges, slow growth -> high severity
    1=fast pathogen: many weak edges, fast growth  -> high severity
    2=benign visitor: few weak edges, slow growth  -> low severity
    """
    N   = node_emb.size(0)
    post_node_emb = torch.cat([node_emb, inv_emb], dim=0)
    inv_idx = N

    if archetype == 0:   # apex predator
        n_edges     = max(1, int(rng.integers(1, min(4, N))))
        weight_lo, weight_hi = 0.6, 1.2
        growth_val  = float(rng.uniform(0.2, 0.8))
    elif archetype == 1: # fast pathogen / r-strategist
        n_edges     = max(2, int(rng.integers(N // 2, N)))
        weight_lo, weight_hi = 0.05, 0.2
        growth_val  = float(rng.uniform(2.5, 4.5))
    else:                # benign visitor
        n_edges     = max(1, int(rng.integers(1, 3)))
        weight_lo, weight_hi = 0.02, 0.10
        growth_val  = float(rng.uniform(0.1, 0.5))

    targets    = rng.choice(N, min(n_edges, N), replace=False)
    new_src    = torch.tensor([inv_idx]*len(targets), dtype=torch.long, device=device)
    new_tgt    = torch.tensor(targets.tolist(),        dtype=torch.long, device=device)
    edge_index = torch.cat([g.edge_index, torch.stack([new_src, new_tgt])], dim=1)

    new_attrs  = torch.tensor(
        rng.uniform(weight_lo, weight_hi, (len(targets), g.edge_attr.size(1))),
        dtype=torch.float32, device=device)
    edge_attr  = torch.cat([g.edge_attr, new_attrs], dim=0)

    inv_growth = torch.tensor([growth_val], dtype=torch.float32, device=device)
    growth     = torch.cat([g.growth, inv_growth])

    return post_node_emb, edge_index, edge_attr, growth
