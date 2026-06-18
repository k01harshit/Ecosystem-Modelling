"""
training/calibrate.py  — v3

Self-contained. Does NOT import from finetune.py.

Key fix over v2:
    _find_graph now generates an ecosystem-matched synthetic graph when no
    real graph is available, instead of falling back to whatever graph happens
    to be first in the map. Each invasion event has an explicit ecosystem
    profile (n_species, connectance, n_trophic_levels) that controls the
    synthetic graph, so Canada Goose (small, stable UK wetland) and Burmese
    Python (large, complex Florida Everglades) get structurally different
    pre-invasion graphs, which then produce meaningfully different ODE outputs.
"""

from __future__ import annotations

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
from torch_geometric.data import Data

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import INVASION_EVENTS, SEED, EDGE_PRED_EPOCHS, EDGE_PRED_LR
from data.graph_builder import insert_invasive_species, build_dataset
from models.disruption_score import DisruptionScorer

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Ecosystem profiles — structural parameters per invasion event ecosystem
# These are NOT labels. They describe the pre-invasion ecosystem structure
# (species richness, connectance, trophic depth) from the ecological literature.
# Used only to generate a structurally appropriate synthetic graph when no
# real fetched graph matches the ecosystem name.
# ═══════════════════════════════════════════════════════════════════════════════

ECOSYSTEM_PROFILES = {
    # Structural parameters grounded in published food-web databases
    # (species richness, connectance, trophic depth, linear chain).
    # Sources: Dunne et al. 2002 (connectance), Brose et al. 2006 (body mass
    # scaling), empirical species counts from IUCN / regional surveys.
    # These values reflect the REAL ecosystem structure, not invasion outcomes.
    # (n_species, connectance, n_trophic_levels, linear_chain)
    "florida everglades":          (68, 0.18, 4.5, False),  # Kushlan 1979: 68 sp
    "guam forest":                 (38, 0.20, 4.0, False),  # Wiles 2003: 38 vertebrate sp
    "lake victoria":               (52, 0.15, 3.5, False),  # Pitcher 1994: ~500 haplochromine + 52 other sp
    "native american ecosystems":  (30, 0.12, 3.0, False),  # generic temperate estimate
    "new zealand forest":          (44, 0.18, 4.0, False),  # Atkinson 1985: 44 forest bird/reptile sp
    "global amphibians":           (35, 0.14, 3.5, False),  # Stuart 2004: global pond estimates
    "australian wetlands":         (32, 0.14, 3.5, False),  # Kingsford 2000: Murray-Darling wetlands
    "caribbean reef":              (28, 0.20, 3.0, False),  # Opitz 1996: 28 functional groups
    "great lakes":                 (36, 0.13, 3.5, False),  # Sterner 2004: Lake Superior 36 sp
    "australian grassland":        (25, 0.15, 3.0, False),  # Morton 1993
    "mississippi river":           (30, 0.14, 3.5, True),   # Welcomme 2008
    "grassland":                   (22, 0.13, 3.0, False),  # generic temperate grassland
    "north american forest":       (35, 0.16, 4.0, False),  # typical eastern deciduous
    "african lakes":               (40, 0.16, 3.5, False),  # Snoeks 2000
    "australian shrubland":        (20, 0.12, 3.0, False),  # Morton 1990
    "north american meadow":       (18, 0.12, 2.5, False),  # typical meadow survey
    "north american grassland":    (16, 0.11, 2.5, False),  # Knapp 1999
    "australian river":            (18, 0.11, 2.5, True),   # Kingsford 1995
    "australian forest":           (22, 0.12, 3.0, False),  # Recher 1996
    "uk wetlands":                 (18, 0.10, 2.5, False),  # Biggs 1994: UK ponds
    "new zealand wetlands":        (20, 0.11, 2.5, False),  # Clarkson 2004
}


def _generate_matched_synthetic(eco_name: str, seed: int = 42) -> Optional[Data]:
    """
    Generate a single synthetic PyG graph matched to the ecosystem profile.
    Falls back to a medium-complexity generic graph if the name is unknown.
    """
    from data.fetch import _generate_structured_ecosystem, simulate_lv_stochastic
    from data.fetch import _estimate_trophic_levels, _lv_to_edge_type

    name_lower = eco_name.lower()
    # Find the best matching profile
    profile = None
    for key, val in ECOSYSTEM_PROFILES.items():
        if key in name_lower or name_lower in key:
            profile = val
            break

    if profile is None:
        # Generic medium profile
        logger.warning(f"No profile for '{eco_name}' — using generic medium profile")
        profile = (25, 0.14, 3.0, False)

    n_sp, connectance, n_tl, linear_chain = profile
    rng = np.random.default_rng(seed)

    eco = _generate_structured_ecosystem(
        n_sp=n_sp, connectance=connectance, n_tl=n_tl,
        linear_chain=linear_chain,
        name=eco_name.replace(" ", "_").lower(),
        archetype="matched",
        rng=rng,
    )
    if eco is None:
        # _generate_structured_ecosystem can return None if too many extinctions
        # Try once more with a simpler profile
        eco = _generate_structured_ecosystem(
            n_sp=max(n_sp - 5, 8), connectance=connectance * 0.8,
            n_tl=max(n_tl - 0.5, 2.0), linear_chain=True,
            name=eco_name.replace(" ", "_").lower() + "_simple",
            archetype="matched_simple", rng=rng,
        )
    if eco is None:
        return None

    graphs = build_dataset([eco])
    return graphs[0] if graphs else None


# ═══════════════════════════════════════════════════════════════════════════════
# EdgePredictor training  (self-supervised, no outcome labels)
# ═══════════════════════════════════════════════════════════════════════════════

def pretrain_edge_predictor(edge_predictor, encoder,
                             graphs: List[Data],
                             epochs: int = EDGE_PRED_EPOCHS,
                             lr: float = EDGE_PRED_LR,
                             device: str = "cpu"):
    """Train EdgePredictor to reconstruct held-out edges. No labels."""
    logger.info(f"Pretraining EdgePredictor for {epochs} epochs...")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    encoder.to(device)
    edge_predictor.to(device)
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    opt   = optim.Adam(edge_predictor.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.7)

    for epoch in range(1, epochs + 1):
        edge_predictor.train()
        total_loss = 0.0
        n = 0

        for g in graphs:
            g = g.to(device)
            if g.edge_index.size(1) < 4:
                continue
            opt.zero_grad()

            n_edges   = g.edge_index.size(1)
            n_holdout = max(1, int(0.3 * n_edges))
            perm      = torch.randperm(n_edges)
            hold_idx  = perm[:n_holdout]
            keep_mask = torch.zeros(n_edges, dtype=torch.bool)
            keep_mask[perm[n_holdout:]] = True

            g_reduced = Data(
                x=g.x, edge_index=g.edge_index[:, keep_mask],
                edge_attr=g.edge_attr[keep_mask],
                edge_type=g.edge_type[keep_mask],
                trophic=g.trophic, growth=g.growth,
                num_nodes=g.num_nodes,
            )
            node_emb, _ = encoder(g_reduced.x, g_reduced.edge_index,
                                   g_reduced.edge_type)

            pos_src = g.edge_index[0, hold_idx]
            pos_tgt = g.edge_index[1, hold_idx]
            pos_out = edge_predictor(node_emb[pos_src], node_emb[pos_tgt])

            N = g.num_nodes
            neg_src = torch.randint(0, N, (n_holdout,), device=device)
            neg_tgt = torch.randint(0, N, (n_holdout,), device=device)
            neg_out = edge_predictor(node_emb[neg_src], node_emb[neg_tgt])

            exist_logits = torch.cat([pos_out["exist_logit"], neg_out["exist_logit"]])
            exist_labels = torch.cat([
                torch.ones(n_holdout, device=device),
                torch.zeros(n_holdout, device=device),
            ])
            l_exist  = nn.functional.binary_cross_entropy_with_logits(
                exist_logits, exist_labels)
            l_type   = nn.functional.cross_entropy(
                pos_out["type_logit"], g.edge_type[hold_idx])
            l_weight = nn.functional.mse_loss(
                pos_out["weight"], g.edge_attr[hold_idx, 0])

            loss = l_exist + l_type + 0.5 * l_weight
            loss.backward()
            nn.utils.clip_grad_norm_(edge_predictor.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item()
            n += 1

        sched.step()
        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"  EdgePredictor epoch {epoch:3d}/{epochs} | "
                        f"loss={total_loss/max(n,1):.4f} | "
                        f"lr={sched.get_last_lr()[0]:.6f}")

    for param in encoder.parameters():
        param.requires_grad = True
    logger.info("EdgePredictor pretraining complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# Invasion pair building
# ═══════════════════════════════════════════════════════════════════════════════

def build_invasion_training_pairs(
        ecosystem_graphs: Dict[str, Data],
        invasion_events:  List[Dict] = None,
        edge_predictor=None,
        encoder=None,
        device: str = "cpu",
) -> List[Tuple[Data, Data, Dict]]:
    """Build (pre_graph, post_graph, label) triples from invasion events."""
    if invasion_events is None:
        invasion_events = INVASION_EVENTS

    pairs = []
    for idx, event in enumerate(invasion_events):
        eco_name  = event["ecosystem"]
        pre_graph = _find_graph(ecosystem_graphs, eco_name)

        if pre_graph is None:
            logger.info(f"  No real graph for '{eco_name}' — generating matched synthetic")
            pre_graph = _generate_matched_synthetic(eco_name, seed=SEED + idx)

        if pre_graph is None:
            logger.warning(f"  Could not build any graph for '{eco_name}' — skipping")
            continue

        invader_attrs = _invader_attributes(event["invader"])

        if edge_predictor is not None and encoder is not None:
            predicted_edges = _predict_edges_with_model(
                pre_graph, invader_attrs, edge_predictor, encoder, device)
        else:
            predicted_edges = _predict_invader_edges_heuristic(
                pre_graph, invader_attrs)

        post_graph = insert_invasive_species(pre_graph, invader_attrs,
                                              predicted_edges)

        label = {
            # Only fields actually consumed by extract_all_features:
            "severity":  event["severity"],    # used post-hoc for Spearman eval
            "outcome":   event["outcome"],     # used post-hoc for reporting only
            "name":      event["name"],
            "invader_r":  invader_attrs.get("intrinsic_growth", 1.0),
            "invader_tl": invader_attrs.get("trophic_level",    2.5),
            "invader_dt": int(invader_attrs.get("diet_type",    0)),
            "invader_bm": float(invader_attrs.get("log_body_mass", 2.0)),
        }
        pairs.append((pre_graph, post_graph, label))
        logger.info(f"  Built pair: {event['name']} "
                    f"[{event['outcome']}, severity={event['severity']:.2f}, "
                    f"edges={len(predicted_edges)}, "
                    f"n_sp={pre_graph.num_nodes}]")

    logger.info(f"Total invasion pairs: {len(pairs)}")
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Stage A: Feature extraction  (label-free)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_all_features(
    full_model,
    encoder,
    ssl_head,
    pairs: List[Tuple],
    lv_severity_mlp=None,
    dynamics=None,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Extract 5 physics features per invasion event. No labels used in extraction.
    severities and outcomes are collected here only so metrics.py can use them
    post-hoc for evaluation — they never influence the feature values.
    """
    scorer = DisruptionScorer()
    features_list, severities, outcomes, names = [], [], [], []

    for pre_g, post_g, label in pairs:
        feat = scorer.extract_features(
            full_model, encoder, ssl_head,
            pre_g, post_g,
            invader_trophic_level=label.get("invader_tl",  2.5),
            invader_growth_rate=label.get("invader_r",     1.0),
            invader_diet_type=label.get("invader_dt",      0),
            invader_log_body_mass=label.get("invader_bm",  2.0),
            lv_severity_mlp=lv_severity_mlp,
            dynamics=dynamics,
            device=device,
        )
        features_list.append(feat)
        severities.append(label["severity"])
        outcomes.append(label["outcome"])
        names.append(label["name"])

        logger.debug(
            f"  {label['name'][:45]:<45} | "
            f"shift={feat[0]:.3f} lam={feat[1]:.3f} "
            f"ext={feat[2]:.3f} var={feat[3]:.3f} rec={feat[4]:.3f}"
        )

    return (
        np.array(features_list, dtype=np.float64),
        np.array(severities,    dtype=np.float64),
        outcomes,
        names,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _find_graph(graph_map: Dict[str, Data], name: str) -> Optional[Data]:
    """
    Find a real fetched graph by name matching.
    Returns None (not a fallback graph) if no match — caller will generate
    a matched synthetic instead.
    """
    name_lower = name.lower()
    # Exact substring match
    for k, v in graph_map.items():
        if name_lower in k.lower() or k.lower() in name_lower:
            return v
    # Biome keyword match — only return if match is specific enough
    BIOME_MAP = {
        "florida everglades":         ["everglades"],
        "guam forest":                ["guam"],
        "lake victoria":              ["victoria"],
        "caribbean reef":             ["caribbean", "reef"],
        "great lakes":                ["great lakes"],
        "australian wetlands":        ["australian wetlands"],
        "australian grassland":       ["australian grassland"],
        "mississippi river":          ["mississippi", "neotropical river"],
        "african lakes":              ["african lakes", "african"],
        "north american forest":      ["north american forest"],
        "grassland":                  ["grassland"],
        "new zealand forest":         ["new zealand forest", "stony stream new zealand",
                                       "kyeburn stream new zealand"],
        "global amphibians":          ["wetland", "amphibian"],
        "uk wetlands":                ["uk", "mill stream uk", "tadnoll brook uk"],
        "new zealand wetlands":       ["new zealand"],
    }
    for eco_key, keywords in BIOME_MAP.items():
        if eco_key in name_lower or name_lower in eco_key:
            for k, v in graph_map.items():
                if any(kw in k.lower() for kw in keywords):
                    return v
    return None   # signal caller to generate a matched synthetic


def _predict_edges_with_model(graph: Data, invader_attrs: Dict,
                               edge_predictor, encoder,
                               device: str) -> List[Dict]:
    from data.graph_builder import _build_node_feature
    edge_predictor.eval()
    encoder.eval()
    with torch.no_grad():
        g = graph.to(device)
        node_emb, _ = encoder(g.x, g.edge_index, g.edge_type)

        inv_tl  = invader_attrs.get("trophic_level", 2.5)
        eco_tl  = g.trophic.cpu().numpy()
        tl_diff = np.abs(eco_tl - inv_tl)
        tl_w    = np.exp(-0.5 * (tl_diff ** 2))
        tl_w    = tl_w / (tl_w.sum() + 1e-8)
        tl_w_t  = torch.tensor(tl_w, dtype=torch.float, device=device).unsqueeze(1)

        inv_emb_base = (node_emb * tl_w_t).sum(dim=0, keepdim=True)
        inv_feat     = _build_node_feature(invader_attrs).to(device)
        D = node_emb.size(1)
        if inv_feat.size(0) >= D:
            inv_feat_proj = inv_feat[:D].unsqueeze(0)
        else:
            inv_feat_proj = torch.cat(
                [inv_feat, torch.zeros(D - inv_feat.size(0), device=device)]
            ).unsqueeze(0)
        inv_feat_proj = nn.functional.normalize(inv_feat_proj, dim=1)
        inv_emb      = inv_emb_base + inv_feat_proj
        N            = node_emb.size(0)
        inv_expanded = inv_emb.expand(N, -1)

        out_fwd = edge_predictor(inv_expanded, node_emb)
        out_rev = edge_predictor(node_emb, inv_expanded)

    edges = []
    etype_names = ["predation", "competition", "mutualism", "parasitism"]
    for i in range(N):
        prob_fwd = torch.sigmoid(out_fwd["exist_logit"][i]).item()
        if prob_fwd > 0.55:
            w     = out_fwd["weight"][i].item()
            etype = out_fwd["type_logit"][i].argmax().item()
            edges.append({"target_idx": i, "type": etype_names[etype],
                          "weight": float(np.clip(w, 0.05, 2.0)),
                          "direction": "out", "confidence": prob_fwd})
        prob_rev = torch.sigmoid(out_rev["exist_logit"][i]).item()
        if prob_rev > 0.60:
            w = out_rev["weight"][i].item()
            edges.append({"target_idx": i, "type": "predation",
                          "weight": float(np.clip(w, 0.05, 1.0)),
                          "direction": "in", "confidence": prob_rev})

    if not edges:
        edges = _predict_invader_edges_heuristic(graph, invader_attrs)
    return edges


def _predict_invader_edges_heuristic(graph: Data,
                                      invader_attrs: Dict) -> List[Dict]:
    inv_tl = invader_attrs.get("trophic_level", 2.5)
    inv_bm = invader_attrs.get("log_body_mass", 2.0)
    edges  = []
    trophic     = graph.trophic.tolist()
    body_masses = (graph.x[:, 1] * 10.0).tolist()
    for i in range(graph.num_nodes):
        tl   = trophic[i]
        bm   = body_masses[i]
        diff = inv_tl - tl
        if 0.4 < diff < 2.5 and inv_bm > bm - 0.5:
            w = max(0.1, 1.2 - abs(diff - 1.2) * 0.4)
            edges.append({"target_idx": i, "type": "predation",
                          "weight": float(w), "direction": "out"})
        elif -1.8 < diff < -0.3 and inv_bm < bm + 0.5:
            w = max(0.05, 0.6 - abs(diff + 0.8) * 0.2)
            edges.append({"target_idx": i, "type": "predation",
                          "weight": float(w), "direction": "in"})
        elif abs(diff) < 0.4 and abs(inv_bm - bm) < 1.5:
            edges.append({"target_idx": i, "type": "competition",
                          "weight": 0.4, "direction": "out"})
    if not edges:
        edges.append({"target_idx": 0, "type": "competition",
                      "weight": 0.3, "direction": "out"})
    return edges


def _invader_attributes(species_name: str) -> Dict:
    known = {
        "Python bivittatus":              {"trophic_level": 4.2, "log_body_mass": 5.5,  "diet_type": 0, "habitat": 2, "log_repro_rate": -1.2, "intrinsic_growth": 0.4},
        "Boiga irregularis":              {"trophic_level": 3.8, "log_body_mass": 2.8,  "diet_type": 0, "habitat": 2, "log_repro_rate":  0.6, "intrinsic_growth": 1.0},
        "Lates niloticus":                {"trophic_level": 4.1, "log_body_mass": 4.8,  "diet_type": 0, "habitat": 3, "log_repro_rate":  0.4, "intrinsic_growth": 0.9},
        "Pterois volitans":               {"trophic_level": 3.6, "log_body_mass": 2.2,  "diet_type": 0, "habitat": 3, "log_repro_rate":  0.9, "intrinsic_growth": 1.1},
        "Dreissena polymorpha":           {"trophic_level": 2.1, "log_body_mass": 0.4,  "diet_type": 1, "habitat": 3, "log_repro_rate":  2.2, "intrinsic_growth": 2.2},
        "Hypophthalmichthys molitrix":    {"trophic_level": 2.5, "log_body_mass": 3.5,  "diet_type": 1, "habitat": 3, "log_repro_rate":  1.5, "intrinsic_growth": 1.8},
        "Rhinella marina":                {"trophic_level": 3.1, "log_body_mass": 2.1,  "diet_type": 0, "habitat": 1, "log_repro_rate":  1.6, "intrinsic_growth": 1.6},
        "Batrachochytrium dendrobatidis": {"trophic_level": 2.0, "log_body_mass": -3.0, "diet_type": 3, "habitat": 1, "log_repro_rate":  3.5, "intrinsic_growth": 3.5},
        "Oryctolagus cuniculus":          {"trophic_level": 2.0, "log_body_mass": 1.6,  "diet_type": 2, "habitat": 0, "log_repro_rate":  3.2, "intrinsic_growth": 3.2},
        "Vulpes vulpes":                  {"trophic_level": 3.5, "log_body_mass": 3.2,  "diet_type": 0, "habitat": 0, "log_repro_rate":  0.9, "intrinsic_growth": 1.0},
        "Variola major":                  {"trophic_level": 2.5, "log_body_mass": -5.0, "diet_type": 3, "habitat": 0, "log_repro_rate":  4.0, "intrinsic_growth": 4.0},
        "Parthenium hysterophorus":       {"trophic_level": 1.0, "log_body_mass": 0.1,  "diet_type": 3, "habitat": 0, "log_repro_rate":  2.6, "intrinsic_growth": 2.6},
        "Eichhornia crassipes":           {"trophic_level": 1.0, "log_body_mass": 1.5,  "diet_type": 3, "habitat": 3, "log_repro_rate":  3.0, "intrinsic_growth": 3.0},
        "Agrilus planipennis":            {"trophic_level": 2.0, "log_body_mass": -1.0, "diet_type": 2, "habitat": 2, "log_repro_rate":  2.0, "intrinsic_growth": 2.0},
        "Apis mellifera":                 {"trophic_level": 2.0, "log_body_mass": -1.5, "diet_type": 2, "habitat": 0, "log_repro_rate":  2.5, "intrinsic_growth": 1.5},
        "Phasianus colchicus":            {"trophic_level": 2.5, "log_body_mass": 2.0,  "diet_type": 2, "habitat": 0, "log_repro_rate":  1.2, "intrinsic_growth": 0.8},
        "Cyprinus carpio":                {"trophic_level": 2.8, "log_body_mass": 3.2,  "diet_type": 1, "habitat": 3, "log_repro_rate":  1.8, "intrinsic_growth": 1.2},
        "Dama dama":                      {"trophic_level": 2.0, "log_body_mass": 4.0,  "diet_type": 2, "habitat": 2, "log_repro_rate":  0.6, "intrinsic_growth": 0.5},
        "Branta canadensis":              {"trophic_level": 2.2, "log_body_mass": 3.5,  "diet_type": 2, "habitat": 1, "log_repro_rate":  0.8, "intrinsic_growth": 0.7},
        "Anas platyrhynchos":             {"trophic_level": 2.3, "log_body_mass": 2.8,  "diet_type": 2, "habitat": 1, "log_repro_rate":  1.0, "intrinsic_growth": 0.9},
        "Mustela erminea":                {"trophic_level": 3.8, "log_body_mass": 1.8,  "diet_type": 0, "habitat": 2, "log_repro_rate":  1.4, "intrinsic_growth": 1.3},
    }
    if species_name in known:
        return known[species_name]
    try:
        from data.acquire import load_acquired_traits as _lat
        acq = _lat()
        if species_name in acq:
            return acq[species_name]
    except Exception:
        pass
    logger.warning(f"Unknown invader: {species_name} — using defaults")
    return {"trophic_level": 2.5, "log_body_mass": 2.0, "diet_type": 0,
            "habitat": 0, "log_repro_rate": 0.5, "intrinsic_growth": 1.0}


def _spearman_safe(scores, severities):
    from scipy.stats import spearmanr
    try:
        rho, p = spearmanr(scores, severities)
        return (float(rho) if np.isfinite(rho) else float("nan"),
                float(p)   if np.isfinite(p)   else float("nan"))
    except Exception:
        return float("nan"), float("nan")
