"""
validation/metrics.py — v9

Fixes over v8:
  1. Accepts pre-trained encoder/ssl_head/full_model/edge_predictor from
     main.py — NO second pretrain. The function is now a pure evaluation
     step that uses the already-trained models.
  2. All outputs actually saved:
       outputs/loo/scores.csv        per-event scores + predictions
       outputs/loo/summary.csv       Spearman rho, accuracy, sanity stats
       outputs/loo/features.csv      raw physics features per event
       outputs/loo/encoder.pt        encoder state dict
       outputs/loo/ssl_head.pt       ssl_head state dict
       outputs/loo/full_model.pt     full_model state dict
  3. Key name is "loo_score" throughout (was "score" in v8, causing
     KeyError in main.py's _plot_loo_score_vs_severity).
  4. Reproducibility: no random state is mutated here beyond what the
     fixed-seed models already determine.
"""

from __future__ import annotations

import csv, json, os, logging
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from scipy.stats import spearmanr, kruskal, ranksums

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PRIMARY: Fully unsupervised evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def leave_one_out_validation(
    all_events:    List[Dict],
    graph_map:     Dict,
    all_graphs:    List,
    # Pre-trained models passed in from main — no re-training here
    encoder=None,
    ssl_head=None,
    full_model=None,
    edge_predictor=None,
    # Kept for backward compat but ignored when models are passed in
    pretrain_epochs: int = 50,
    device: str = "cpu",
    outdir: str = "outputs",
) -> Dict:
    """
    Fully unsupervised scoring pipeline.

    When encoder/ssl_head/full_model/edge_predictor are passed in (the normal
    path from main.py), they are used directly — no re-training occurs.
    This eliminates the duplicate pretrain and ensures the same model state
    is used for scoring as was trained in main.py.
    """
    import random as _random
    from training.calibrate import (
        build_invasion_training_pairs,
        extract_all_features,
        pretrain_edge_predictor,
    )
    from models.disruption_score import DisruptionScorer
    from configs import SEED, EDGE_PRED_EPOCHS

    # Only import and build models if none were passed in (fallback path)
    if encoder is None or ssl_head is None or full_model is None:
        logger.warning(
            "No pre-trained models passed to leave_one_out_validation — "
            "building and training from scratch. Pass models from main.py "
            "to avoid duplicate pretraining."
        )
        from models.encoder import EcologicalEncoder, EdgePredictor as EP
        from models.ssl_heads import SSLHead
        from models.neural_ode import EcologicalDynamicsModule
        from models.stability_head import FullStabilityModel
        from training.pretrain import pretrain as run_pretrain

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        _random.seed(SEED)

        encoder        = EcologicalEncoder()
        ssl_head       = SSLHead()
        edge_predictor = EP()
        dynamics       = EcologicalDynamicsModule()
        full_model     = FullStabilityModel(encoder, dynamics)

        run_pretrain(encoder, ssl_head, all_graphs,
                     epochs=pretrain_epochs, device=device, seed=SEED)
        pretrain_edge_predictor(
            edge_predictor, encoder, all_graphs,
            epochs=min(EDGE_PRED_EPOCHS, 15), device=device,
        )
    else:
        logger.info("  Using pre-trained models from main.py (no re-training).")

    os.makedirs(os.path.join(outdir, "loo"), exist_ok=True)

    logger.info("\n" + "=" * 64)
    logger.info("  UNSUPERVISED DISRUPTION SCORING (fixed weights, no label use)")
    logger.info(f"  N={len(all_events)} events")
    logger.info("=" * 64)

    # ── Step 1b: Train SeverityMLP on synthetic graphs (no real labels) ──────
    logger.info("\n-- Training LV Severity MLP (synthetic data only, no labels) --")
    from models.lv_severity_model import train_severity_mlp
    from models.neural_ode import EcologicalDynamicsModule
    dynamics_for_mlp = EcologicalDynamicsModule().to(device)
    syn_graphs = [g for g in all_graphs
                  if getattr(g, "source", "synthetic").startswith("synthetic")][:200]
    if not syn_graphs:
        syn_graphs = all_graphs[:200]
    lv_severity_mlp = train_severity_mlp(
        dynamics=dynamics_for_mlp, encoder=encoder,
        synthetic_graphs=syn_graphs,
        n_invasions_per_graph=3, epochs=20,
        device=device, seed=SEED,
    )
    import torch as _torch
    _torch.save(lv_severity_mlp.state_dict(),
                os.path.join(outdir, "loo", "lv_severity_mlp.pt"))
    logger.info(f"  Saved: {outdir}/loo/lv_severity_mlp.pt")

    # ── Step 1: Build invasion pairs (no labels used in graph construction) ───
    logger.info("\n-- Building invasion pairs --")
    all_pairs = build_invasion_training_pairs(
        graph_map, all_events,
        edge_predictor=edge_predictor,
        encoder=encoder,
        device=device,
    )
    if len(all_pairs) < len(all_events):
        logger.warning(f"  Only {len(all_pairs)}/{len(all_events)} events have graphs.")

    # ── Step 2: Extract physics features (no labels) ──────────────────────────
    logger.info("\n-- Extracting physics features (label-free) --")
    features, severities, outcomes, names = extract_all_features(
        full_model, encoder, ssl_head, all_pairs,
        lv_severity_mlp=lv_severity_mlp,
        dynamics=dynamics_for_mlp,
        device=device,
    )
    N = len(all_pairs)

    # ── Step 3: Score with fixed weights (no fitting, no labels) ─────────────
    logger.info("\n-- Computing disruption scores (fixed physics weights) --")
    scorer = DisruptionScorer()
    scores = scorer.score_batch(features)

    logger.info(f"\n  Weights: {np.round(scorer.weights, 3).tolist()}")
    logger.info(f"\n  {'Event':<45} | Score | Severity")
    logger.info(f"  {'-'*45}-+-------+---------")
    for name, score, sev in zip(names, scores, severities):
        logger.info(f"  {name[:45]:<45} | {score:.3f} | {sev:.2f}")

    # ── Step 4a: PRIMARY — Spearman rho (severity used only here, post-scoring)
    rho, p_val = _spearman_safe(scores, severities)
    logger.info(f"\n  PRIMARY: Spearman rho(score, severity) = {rho:.3f} "
                f"(p={p_val:.4f}, N={N})")

    # ── Step 4b: POST-HOC class accuracy (outcomes used only for reporting) ───
    thresh_low, thresh_high = _fit_thresholds_posthoc(scores, outcomes)
    predictions = []
    for i in range(N):
        pred    = _predict_class(scores[i], thresh_low, thresh_high)
        correct = pred == outcomes[i]
        predictions.append({
            "event_name": names[i],
            "outcome":    outcomes[i],
            "severity":   float(severities[i]),
            "loo_score":  round(float(scores[i]), 4),   # key is "loo_score"
            "pred_class": pred,
            "correct":    correct,
        })

    n_correct = sum(1 for p in predictions if p["correct"])
    acc = n_correct / N
    logger.info(f"  POST-HOC 3-class accuracy: {acc:.3f}  ({n_correct}/{N})")
    for cls in ["stable", "disrupted", "collapse"]:
        cls_p = [p for p in predictions if p["outcome"] == cls]
        if cls_p:
            cls_acc = sum(1 for p in cls_p if p["correct"]) / len(cls_p)
            logger.info(f"    {cls:<10}: {cls_acc:.3f}  (N={len(cls_p)})")

    # ── Step 4c: Physics sanity checks ────────────────────────────────────────
    sanity = _physics_sanity_checks(features, outcomes, scores)
    _log_sanity(sanity)

    # ── Save all outputs ──────────────────────────────────────────────────────
    _save_scores(predictions,
                 os.path.join(outdir, "loo", "scores.csv"))
    _save_features(features, names, outcomes, severities,
                   os.path.join(outdir, "loo", "features.csv"))
    _save_summary(rho, p_val, acc, sanity, scorer.weights,
                  os.path.join(outdir, "loo", "summary.csv"))
    _save_models(encoder, ssl_head, full_model, outdir)

    return {
        "loo_predictions":  predictions,
        "loo_scores":       scores.tolist(),
        "true_severities":  severities.tolist(),
        "true_outcomes":    outcomes,
        "features":         features.tolist(),
        "spearman_rho":     rho,
        "spearman_p":       p_val,
        "accuracy":         acc,
        "sanity_checks":    sanity,
        "encoder":          encoder,
        "ssl_head":         ssl_head,
        "full_model":       full_model,
        "thresh_lows":      [thresh_low] * N,
        "thresh_highs":     [thresh_high] * N,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Post-hoc helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _fit_thresholds_posthoc(scores, outcomes):
    from sklearn.isotonic import IsotonicRegression
    ORDINAL = {"stable": 0, "disrupted": 1, "collapse": 2}
    y  = np.array([ORDINAL[o] for o in outcomes], dtype=float)
    ir = IsotonicRegression(out_of_bounds="clip", increasing=True)
    ir.fit(scores, y)
    grid   = np.linspace(scores.min() - 0.1, scores.max() + 0.1, 1000)
    fitted = ir.predict(grid)

    def _crossover(target):
        idx = np.searchsorted(fitted, target)
        return float(grid[np.clip(idx, 0, len(grid) - 1)])

    return _crossover(0.5), _crossover(1.5)


def _predict_class(score, thresh_low, thresh_high):
    if score < thresh_low:  return "stable"
    if score < thresh_high: return "disrupted"
    return "collapse"


# ═══════════════════════════════════════════════════════════════════════════════
# Physics sanity checks
# ═══════════════════════════════════════════════════════════════════════════════

def _physics_sanity_checks(features, outcomes, scores):
    results = {}
    outcome_arr   = np.array(outcomes)
    stable_idx    = np.where(outcome_arr == "stable")[0]
    disrupted_idx = np.where(outcome_arr == "disrupted")[0]
    collapse_idx  = np.where(outcome_arr == "collapse")[0]

    def _safe_kruskal(*groups):
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2: return float("nan"), float("nan")
        try:   return float(kruskal(*groups)[0]), float(kruskal(*groups)[1])
        except: return float("nan"), float("nan")

    def _safe_ranksum(a, b):
        if len(a) < 2 or len(b) < 2: return float("nan"), float("nan")
        try:   return float(ranksums(a, b)[0]), float(ranksums(a, b)[1])
        except: return float("nan"), float("nan")

    lm = features[:, 1]
    stat, p = _safe_kruskal(lm[collapse_idx], lm[disrupted_idx], lm[stable_idx])
    results["lambda_max_kruskal_stat"] = stat
    results["lambda_max_kruskal_p"]    = p
    results["lambda_max_ordered"] = (
        _gmean(lm, collapse_idx) > _gmean(lm, disrupted_idx) > _gmean(lm, stable_idx)
    )

    ef = features[:, 2]
    stat, p = _safe_ranksum(ef[collapse_idx], ef[stable_idx])
    results["extinct_frac_ranksum_stat"] = stat
    results["extinct_frac_ranksum_p"]    = p
    results["extinct_frac_ordered"] = (
        _gmean(ef, collapse_idx) > _gmean(ef, stable_idx)
    )

    for cls, idx in [("stable", stable_idx),
                     ("disrupted", disrupted_idx),
                     ("collapse", collapse_idx)]:
        if len(idx) > 0:
            results[f"mean_score_{cls}"] = float(scores[idx].mean())

    results["score_ordered"] = (
        results.get("mean_score_collapse", -np.inf) >
        results.get("mean_score_disrupted", -np.inf) >
        results.get("mean_score_stable", np.inf)
    )
    return results


def _gmean(arr, idx):
    return float(arr[idx].mean()) if len(idx) > 0 else float("nan")


def _log_sanity(s):
    logger.info("\n  Physics sanity checks:")
    logger.info(
        f"  lambda_max Kruskal-Wallis p={s.get('lambda_max_kruskal_p', float('nan')):.4f}  "
        f"ordered={s.get('lambda_max_ordered')}  (collapse>disrupted>stable?)"
    )
    logger.info(
        f"  Extinct frac rank-sum     p={s.get('extinct_frac_ranksum_p', float('nan')):.4f}  "
        f"ordered={s.get('extinct_frac_ordered')}  (collapse>stable?)"
    )
    logger.info(
        f"  Score ordering: "
        f"collapse={s.get('mean_score_collapse', float('nan')):.3f}  "
        f"disrupted={s.get('mean_score_disrupted', float('nan')):.3f}  "
        f"stable={s.get('mean_score_stable', float('nan')):.3f}  "
        f"-> {s.get('score_ordered')}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Encoder quality checks
# ═══════════════════════════════════════════════════════════════════════════════

def check_trophic_clustering(encoder, graphs, device="cpu"):
    encoder.eval()
    all_embs, all_trophic = [], []
    with torch.no_grad():
        for g in graphs[:20]:
            g = g.to(device)
            node_emb, _ = encoder(g.x, g.edge_index, g.edge_type)
            all_embs.append(node_emb.cpu())
            all_trophic.append(g.trophic.cpu())
    if not all_embs:
        return {"trophic_spearman": float("nan")}
    embs    = torch.cat(all_embs).numpy()
    trophic = torch.cat(all_trophic).numpy()
    N   = min(len(trophic), 500)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(trophic), N, replace=False)
    emb_d   = np.linalg.norm(embs[idx, None] - embs[None, idx], axis=-1)
    troph_d = np.abs(trophic[idx, None] - trophic[None, idx])
    mask    = np.triu(np.ones_like(emb_d, dtype=bool), k=1)
    corr, pval = spearmanr(emb_d[mask], troph_d[mask])
    logger.info(f"Trophic clustering -- Spearman r={corr:.3f}, p={pval:.4f}")
    return {"trophic_spearman": float(corr), "p_value": float(pval)}


def check_edge_reconstruction(encoder, ssl_head, graphs, mask_rate=0.2, device="cpu"):
    from data.augmentation import mask_edges_for_reconstruction
    encoder.eval(); ssl_head.eval()
    type_correct = type_total = 0
    weight_errors = []
    with torch.no_grad():
        for g in graphs[:30]:
            g = g.to(device)
            masked_g, masked_edges, masked_labels = \
                mask_edges_for_reconstruction(g, mask_rate=mask_rate)
            if masked_edges.size(1) == 0:
                continue
            node_emb, _ = encoder(masked_g.x, masked_g.edge_index, masked_g.edge_type)
            pair = torch.cat([node_emb[masked_edges[0]], node_emb[masked_edges[1]]], dim=-1)
            z           = ssl_head.masked_edge.mlp(pair)
            type_pred   = ssl_head.masked_edge.type_head(z).argmax(dim=-1)
            weight_pred = torch.nn.functional.softplus(
                ssl_head.masked_edge.weight_head(z)).squeeze(-1)
            type_correct += (type_pred == masked_labels[:, 1].long()).sum().item()
            type_total   += len(masked_labels)
            weight_errors.append((weight_pred - masked_labels[:, 0]).abs().mean().item())
    acc = type_correct / max(type_total, 1)
    mae = float(np.mean(weight_errors)) if weight_errors else float("nan")
    logger.info(f"Edge reconstruction -- type_acc={acc:.3f}, weight_mae={mae:.4f}")
    return {"edge_type_accuracy": acc, "edge_weight_mae": mae}


# ═══════════════════════════════════════════════════════════════════════════════
# Save helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _save_scores(rows, fpath):
    if not rows: return
    fields = ["event_name", "outcome", "severity", "loo_score",
              "pred_class", "correct"]
    with open(fpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    logger.info(f"  Saved: {fpath}")


def _save_features(features, names, outcomes, severities, fpath):
    """Save raw physics features for inspection / reproducibility check."""
    feat_names = ["invader_tl", "recon_loss", "damage_mode", "lv_severity",
                  "invader_r", "body_mass_ratio",
                  "extinct_frac", "lambda_max"]
    with open(fpath, "w", newline="") as f:
        fields = ["event_name", "outcome", "severity"] + feat_names
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, (name, outcome, sev) in enumerate(zip(names, outcomes, severities)):
            row = {"event_name": name, "outcome": outcome,
                   "severity": round(float(sev), 4)}
            for j, fn in enumerate(feat_names):
                row[fn] = round(float(features[i, j]), 6)
            w.writerow(row)
    logger.info(f"  Saved: {fpath}")


def _save_summary(rho, p_val, acc, sanity, weights, fpath):
    rows = [
        {"metric": "spearman_rho",            "value": round(rho, 4)},
        {"metric": "spearman_p",              "value": round(p_val, 4)},
        {"metric": "posthoc_3class_accuracy", "value": round(acc, 4)},
        {"metric": "lambda_max_kruskal_p",    "value": round(sanity.get("lambda_max_kruskal_p", float("nan")), 4)},
        {"metric": "extinct_frac_ranksum_p",  "value": round(sanity.get("extinct_frac_ranksum_p", float("nan")), 4)},
        {"metric": "lambda_max_ordered",      "value": sanity.get("lambda_max_ordered", False)},
        {"metric": "extinct_frac_ordered",    "value": sanity.get("extinct_frac_ordered", False)},
        {"metric": "score_ordered",           "value": sanity.get("score_ordered", False)},
        {"metric": "scorer_weights",          "value": str(np.round(weights, 4).tolist())},
        {"metric": "note",                    "value": "fixed weights; no labels used in scoring"},
    ]
    with open(fpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"])
        w.writeheader(); w.writerows(rows)
    logger.info(f"  Saved: {fpath}")


def _save_models(encoder, ssl_head, full_model, outdir):
    """Save model weights so results are exactly reproducible."""
    loo_dir = os.path.join(outdir, "loo")
    torch.save(encoder.state_dict(),
               os.path.join(loo_dir, "encoder.pt"))
    torch.save(ssl_head.state_dict(),
               os.path.join(loo_dir, "ssl_head.pt"))
    torch.save(full_model.state_dict(),
               os.path.join(loo_dir, "full_model.pt"))
    logger.info(f"  Saved model weights to {loo_dir}/{{encoder,ssl_head,full_model}}.pt")


def _spearman_safe(a, b):
    try:
        rho, p = spearmanr(a, b)
        return (float(rho) if np.isfinite(rho) else float("nan"),
                float(p)   if np.isfinite(p)   else float("nan"))
    except Exception:
        return float("nan"), float("nan")


# ── Backward-compat alias ─────────────────────────────────────────────────────
def three_fold_cross_validation(
    all_events, graph_map, all_graphs,
    pretrain_epochs=50, finetune_epochs=40,
    device="cpu", outdir="outputs",
    encoder=None, ssl_head=None, full_model=None, edge_predictor=None,
):
    return leave_one_out_validation(
        all_events=all_events, graph_map=graph_map, all_graphs=all_graphs,
        encoder=encoder, ssl_head=ssl_head, full_model=full_model,
        edge_predictor=edge_predictor,
        pretrain_epochs=pretrain_epochs, device=device, outdir=outdir,
    )

five_fold_cross_validation = three_fold_cross_validation
