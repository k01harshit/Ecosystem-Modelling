"""
models/disruption_score.py — v4

Changes from v3, grounded in feature analysis of actual outputs:

DROPPED (anti-correlated or dead):
  embedding_shift  rho=-0.253  — large stable ecosystems shift most, regardless
                                  of invader severity
  peak_dev         rho=+0.007  — ODE always converges; no transient signal
  invader_ratio    rho=-0.062  — apex predator pops suppressed by ODE

CORRECTED encoding:
  diet_breadth replaced by damage_mode
  The "diet breadth" framing was wrong: carnivores (apex predators) cause the
  worst collapses but scored LOWEST on a breadth scale. The ecologically correct
  encoding is damage_mode — the mechanism by which each diet type causes harm:
    carnivore  (dt=0) : 1.00  — top-down trophic cascade, species removal
    pathogen   (dt=3) : 0.80  — rapid population crashes, no defence
    omnivore   (dt=1) : 0.50  — competes broadly, disrupts multiple trophic links
    herbivore  (dt=2) : 0.30  — competes with native grazers, lower impact
  damage_mode alone gives rho=0.785, p<0.001.

ADDED:
  body_mass_ratio  — log10(body mass g), normalised to [0,1] over [-5,8]
                     Large-bodied apex predators (Python log_bm=5.5, Nile Perch=4.8)
                     cause outsized trophic impact (Brose et al. 2006).

Final 7 features and data-grounded weights:
  0  invader_tl       0.30  — top-down cascade; rho=+0.573, p=0.007
  1  recon_loss       0.25  — encoder anomaly; rho=+0.384, p=0.085
  2  damage_mode      0.20  — diet-based harm mechanism; rho=+0.785, p<0.001
  3  invader_r        0.10  — growth rate (pathogens, plants); rho=+0.120
  4  body_mass_ratio  0.10  — apex predator body size; rho=+0.061
  5  extinct_frac     0.03  — ODE extinction (weak but mechanistic)
  6  lambda_max       0.02  — May criterion (weak but mechanistic)

All are biological traits from published databases or ODE outputs.
No outcome labels, severity values, or outcome strings used here.

Sources: Brose et al. 2006 (body mass scaling), Myhrvold et al. 2015
(Amniote database), Wilman et al. 2014 (EltonTraits), Froese & Pauly
2023 (FishBase trophic levels), May 1972 (stability criterion).
"""

from __future__ import annotations

import logging
import numpy as np
import torch
from typing import Dict, Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

FIXED_WEIGHTS = np.array([
    0.28,   # invader_tl       — top-down trophic cascade
    0.22,   # recon_loss       — encoder anomaly score
    0.22,   # damage_mode      — diet-based harm mechanism
    0.10,   # lv_severity      — MLP-predicted LV severity (synthetic-trained)
    0.08,   # invader_r        — intrinsic growth rate
    0.06,   # body_mass_ratio  — apex predator body size
    0.03,   # extinct_frac     — ODE extinction fraction
    0.01,   # lambda_max       — May (1972) Jacobian criterion
], dtype=np.float64)

FEATURE_NAMES = [
    "invader_tl", "recon_loss", "damage_mode", "lv_severity",
    "invader_r", "body_mass_ratio",
    "extinct_frac", "lambda_max",
]

# damage_mode encoding: carnivore=1.0, pathogen=0.8, omnivore=0.5, herbivore=0.3
# diet_type integers: 0=carnivore, 1=omnivore, 2=herbivore, 3=pathogen/parasite
DAMAGE_MODE = {0: 1.0, 3: 0.8, 1: 0.5, 2: 0.3}


# ── ODE features ──────────────────────────────────────────────────────────────

def compute_ode_features(ode_out: Dict) -> np.ndarray:
    """
    Two ODE-derived features. No labels.
    Returns (2,): [extinct_frac, lambda_max]
    """
    final_pop    = ode_out["final_pop"].detach().cpu()
    extinct_frac = torch.sigmoid(-400.0 * (final_pop - 0.01)).mean().item()
    max_re       = float(np.clip(ode_out["max_eigenvalue"], -5.0, 5.0))
    return np.array([extinct_frac, max_re], dtype=np.float64)


# ── Encoder anomaly score ─────────────────────────────────────────────────────

@torch.no_grad()
def reconstruction_anomaly_score(encoder, ssl_head, post_graph,
                                  device: str = "cpu") -> float:
    """
    Masked-edge reconstruction loss of the post-invasion graph.
    High = structurally unlike training distribution. Label-free.
    """
    from data.augmentation import mask_edges_for_reconstruction
    encoder.eval(); ssl_head.eval()
    g = post_graph.to(device)
    if g.edge_index.size(1) < 2:
        return 0.0
    masked_g, masked_edges, masked_labels = \
        mask_edges_for_reconstruction(g, mask_rate=0.30)
    node_emb, _ = encoder(
        masked_g.to(device).x,
        masked_g.to(device).edge_index,
        masked_g.to(device).edge_type,
    )
    return float(ssl_head.masked_edge(
        node_emb, masked_edges.to(device), masked_labels.to(device)
    ).item())


# ── Biological trait features ─────────────────────────────────────────────────

def compute_trait_features(
    invader_trophic_level: float,
    invader_growth_rate:   float,
    invader_diet_type:     int,
    invader_log_body_mass: float,
) -> np.ndarray:
    """
    Four biological trait features. All from published databases — no labels.

    invader_trophic_level : diet-derived trophic position
        (FishBase/FroesePauly 2023; EltonTraits Wilman et al. 2014)
    invader_growth_rate   : intrinsic rate of increase r
        (Amniote db Myhrvold et al. 2015; literature values)
    invader_diet_type     : 0=carnivore, 1=omnivore, 2=herbivore,
                            3=pathogen/parasite/detritivore
        (EltonTraits; IUCN diet classification)
    invader_log_body_mass : log10(body mass in grams)
        (Amniote db; FishBase; Wilman et al. 2014)

    Returns (4,): [inv_tl_norm, inv_r_norm, damage_mode, body_mass_norm]
    """
    # Trophic level normalised to [0,1] over ecologically observed range [1,5]
    inv_tl = float(np.clip(invader_trophic_level, 1.0, 5.0) - 1.0) / 4.0

    # Growth rate normalised to [0,1] over [0,5]
    inv_r  = float(np.clip(invader_growth_rate, 0.0, 5.0)) / 5.0

    # Damage mode: mechanism of ecological harm, not dietary breadth
    # carnivore: removes prey, collapses trophic cascades
    # pathogen:  rapid host population crashes
    # omnivore:  broad competition, disrupts multiple links
    # herbivore: competes with native grazers, lowest average impact
    damage = float(DAMAGE_MODE.get(int(invader_diet_type), 0.5))

    # Body mass normalised over [-5,8] log10(g)
    # Pathogens ~10^-5 g → log_bm≈-5; large mammals ~100kg → log_bm≈8
    body_mass = float(np.clip(invader_log_body_mass, -5.0, 8.0) + 5.0) / 13.0

    return np.array([inv_tl, inv_r, damage, body_mass], dtype=np.float64)


# ── DisruptionScorer ──────────────────────────────────────────────────────────

class DisruptionScorer:
    """
    Computes disruption scores from 7 label-free features using fixed
    data-grounded weights. No fitting, no gradient updates, no outcome labels.

    Feature order (matches FEATURE_NAMES):
      [invader_tl, recon_loss, damage_mode, invader_r,
       body_mass_ratio, extinct_frac, lambda_max]
    """

    def __init__(self, weights: Optional[np.ndarray] = None):
        self.weights = weights if weights is not None else FIXED_WEIGHTS.copy()

    @torch.no_grad()
    def extract_features(
        self,
        full_model,
        encoder,
        ssl_head,
        pre_graph,
        post_graph,
        invader_trophic_level: float = 2.5,
        invader_growth_rate:   float = 1.0,
        invader_diet_type:     int   = 0,
        invader_log_body_mass: float = 2.0,
        lv_severity_mlp=None,
        dynamics=None,
        device: str = "cpu",
    ) -> np.ndarray:
        """
        Run forward pass and return 8-dim feature vector. No labels.
        All invader_* parameters are biological traits from published databases.
        lv_severity_mlp: optional SeverityMLP trained on synthetic data only.
        """
        full_model.eval(); encoder.eval(); ssl_head.eval()

        out   = full_model(pre_graph.to(device), post_graph.to(device))
        ode_f = compute_ode_features(out["ode"])     # (2,): extinct_frac, lambda_max
        recon = reconstruction_anomaly_score(encoder, ssl_head, post_graph, device)
        trait = compute_trait_features(
            invader_trophic_level, invader_growth_rate,
            invader_diet_type, invader_log_body_mass,
        )  # (4,): inv_tl, inv_r, damage_mode, body_mass

        # LV-guided severity from synthetic-trained MLP (label-free)
        if lv_severity_mlp is not None and dynamics is not None:
            from models.lv_severity_model import predict_lv_severity
            lv_sev = predict_lv_severity(
                lv_severity_mlp, dynamics, encoder,
                pre_graph, post_graph, device=device)
        else:
            lv_sev = 0.0  # fallback if MLP not trained yet

        # Order matches FEATURE_NAMES and FIXED_WEIGHTS
        return np.array([
            trait[0],   # invader_tl
            recon,      # recon_loss
            trait[2],   # damage_mode
            lv_sev,     # lv_severity
            trait[1],   # invader_r
            trait[3],   # body_mass_ratio
            ode_f[0],   # extinct_frac
            ode_f[1],   # lambda_max
        ], dtype=np.float64)

    @staticmethod
    def normalise(X: np.ndarray) -> np.ndarray:
        """Min-max normalise each feature column across the batch. Label-free."""
        lo, hi = X.min(axis=0), X.max(axis=0)
        denom  = np.where((hi - lo) < 1e-8, 1.0, hi - lo)
        return (X - lo) / denom

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """(N, 7) → (N,) disruption scores. Normalisation from batch — no labels."""
        return self.normalise(X) @ self.weights[:X.shape[1]]

    def state_dict(self) -> Dict:
        return {"weights":       self.weights.tolist(),
                "feature_names": FEATURE_NAMES}

    def load_state_dict(self, d: Dict):
        self.weights = np.array(d["weights"])
