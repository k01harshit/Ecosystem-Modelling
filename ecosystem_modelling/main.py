"""
main.py — v10
Unsupervised disruption scoring. Fixes over v9:
  1. Full reproducibility: all random sources seeded before any operation.
  2. Pre-trained models passed into leave_one_out_validation — no duplicate
     pretrain run.
  3. All outputs saved: pretrain history, model checkpoints, scores, features,
     summary. Run once and results are fully recoverable.
  4. KMP_DUPLICATE_LIB_OK set programmatically so user doesn't need to.
"""

import os, sys, logging, argparse, random, json

# Fix OMP duplicate lib warning on Windows before any torch import
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import numpy as np

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import (SEED, SSL_PRETRAIN_EPOCHS, INVASION_EVENTS,
                     EDGE_PRED_EPOCHS, NUM_SYNTHETIC_GRAPHS)
from data.fetch import (fetch_web_of_life, fetch_all_real_networks,
                        generate_synthetic_ecosystems,
                        generate_diverse_synthetic_ecosystems)
from data.graph_builder import build_dataset
from models.encoder import EcologicalEncoder, EdgePredictor
from models.ssl_heads import SSLHead
from models.neural_ode import EcologicalDynamicsModule
from models.stability_head import FullStabilityModel
from training.pretrain import pretrain
from training.calibrate import pretrain_edge_predictor
from validation.metrics import (
    leave_one_out_validation,
    check_trophic_clustering,
    check_edge_reconstruction,
)
from validation.plots import save_all_loo_plots
from utils.viz import (plot_food_web, plot_lv_trajectory,
                       plot_training_curves, plot_trophic_embedding)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-fetch",  action="store_true")
    p.add_argument("--epochs",    type=int, default=None)
    p.add_argument("--device",    type=str, default="cpu")
    p.add_argument("--outdir",    type=str, default="outputs")
    p.add_argument("--local-networks", type=str, default="networks")
    return p.parse_args()


def _seed_everything(seed: int):
    """Seed every random source for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    # Make torch ops deterministic where possible
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def main():
    args   = parse_args()
    device = args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"

    # ── Seed everything before ANY other operation ────────────────────────────
    _seed_everything(SEED)

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "loo"),    exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "models"), exist_ok=True)

    pretrain_epochs = args.epochs or SSL_PRETRAIN_EPOCHS

    collapse_n  = sum(1 for e in INVASION_EVENTS if e["outcome"] == "collapse")
    disrupted_n = sum(1 for e in INVASION_EVENTS if e["outcome"] == "disrupted")
    stable_n    = sum(1 for e in INVASION_EVENTS if e["outcome"] == "stable")

    logger.info("=" * 64)
    logger.info("  Ecosystem GNN v10 — Unsupervised scoring")
    logger.info(f"  Events: {len(INVASION_EVENTS)} "
                f"(collapse={collapse_n}, disrupted={disrupted_n}, stable={stable_n})")
    logger.info(f"  SSL pretrain epochs: {pretrain_epochs} | seed: {SEED}")
    logger.info("=" * 64)

    # ── Stage 1: Data ─────────────────────────────────────────────────────────
    logger.info("\n-- Stage 1: Data --")
    all_ecosystems = []
    if not args.no_fetch:
        logger.info("Fetching Web of Life...")
        wol = fetch_web_of_life()
        logger.info(f"  Web of Life: {len(wol)} networks")
        all_ecosystems.extend(wol)
        logger.info("Fetching real networks...")
        real = fetch_all_real_networks(
            local_dir=args.local_networks,
            use_globalweb=True, use_globi=True)
        logger.info(f"  Real networks: {len(real)}")
        all_ecosystems.extend(real)
    else:
        logger.info("  Skipping API fetch")

    # Re-seed after network I/O (requests may have consumed random state)
    _seed_everything(SEED)

    logger.info(f"Generating {NUM_SYNTHETIC_GRAPHS} random LV ecosystems...")
    all_ecosystems.extend(
        generate_synthetic_ecosystems(n=NUM_SYNTHETIC_GRAPHS, seed=SEED))
    diverse = generate_diverse_synthetic_ecosystems(n_per_archetype=40, seed=SEED + 1)
    all_ecosystems.extend(diverse)
    logger.info(f"  Diverse structured: {len(diverse)}")

    all_graphs = build_dataset(all_ecosystems)
    graph_map  = {g.eco_name: g for g in all_graphs}
    logger.info(f"Total pretraining graphs: {len(all_graphs)}")

    # ── Stage 2: Models ───────────────────────────────────────────────────────
    logger.info("\n-- Stage 2: Models --")
    # Re-seed before model init so weights are deterministic
    _seed_everything(SEED)

    encoder        = EcologicalEncoder()
    ssl_head       = SSLHead()
    edge_predictor = EdgePredictor()
    dynamics       = EcologicalDynamicsModule()
    full_model     = FullStabilityModel(encoder, dynamics)

    logger.info(f"  Encoder:       {sum(p.numel() for p in encoder.parameters()):,}")
    logger.info(f"  EdgePredictor: {sum(p.numel() for p in edge_predictor.parameters()):,}")

    # ── Stage 3: SSL Pretrain (label-free) ───────────────────────────────────
    logger.info(f"\n-- Stage 3: SSL Pretrain ({pretrain_epochs} epochs, label-free) --")
    _seed_everything(SEED)   # seed before training for reproducible shuffles
    pretrain_history = pretrain(
        encoder, ssl_head, all_graphs,
        epochs=pretrain_epochs, device=device, seed=SEED)

    # Save pretrain history
    hist_path = os.path.join(args.outdir, "pretrain_history.json")
    with open(hist_path, "w") as f:
        json.dump(pretrain_history, f, indent=2)
    logger.info(f"  Saved: {hist_path}")

    # Save SSL model checkpoint
    torch.save(encoder.state_dict(),
               os.path.join(args.outdir, "models", "encoder_pretrained.pt"))
    torch.save(ssl_head.state_dict(),
               os.path.join(args.outdir, "models", "ssl_head_pretrained.pt"))
    logger.info(f"  Saved: {args.outdir}/models/encoder_pretrained.pt")

    # ── Stage 4: EdgePredictor (self-supervised) ──────────────────────────────
    logger.info("\n-- Stage 4: EdgePredictor (self-supervised) --")
    _seed_everything(SEED)
    pretrain_edge_predictor(
        edge_predictor, encoder, all_graphs,
        epochs=min(EDGE_PRED_EPOCHS, 15), device=device)

    torch.save(edge_predictor.state_dict(),
               os.path.join(args.outdir, "models", "edge_predictor.pt"))
    logger.info(f"  Saved: {args.outdir}/models/edge_predictor.pt")

    # ── Stage 5: Scoring + Evaluation ────────────────────────────────────────
    # Pass pre-trained models in — metrics.py will NOT re-train them
    logger.info("\n-- Stage 5: Unsupervised Scoring + Evaluation --")
    _seed_everything(SEED)
    loo_results = leave_one_out_validation(
        all_events=INVASION_EVENTS,
        graph_map=graph_map,
        all_graphs=all_graphs,
        encoder=encoder,
        ssl_head=ssl_head,
        full_model=full_model,
        edge_predictor=edge_predictor,
        pretrain_epochs=pretrain_epochs,   # ignored since models are passed
        device=device,
        outdir=args.outdir,
    )

    # ── Stage 6: Visualisations ───────────────────────────────────────────────
    logger.info("\n-- Stage 6: Visualisations --")
    sample_g = all_graphs[0]

    try:
        fig = plot_food_web(sample_g, title=f"Food Web: {sample_g.eco_name}")
        fig.savefig(os.path.join(args.outdir, "sample_food_web.png"),
                    dpi=150, bbox_inches="tight", facecolor="#0d1117")
        logger.info(f"  Saved: {args.outdir}/sample_food_web.png")
    except Exception as e:
        logger.warning(f"  Food web plot failed: {e}")

    try:
        full_model.eval()
        with torch.no_grad():
            g_d = sample_g.to(device)
            node_emb, _ = encoder(g_d.x, g_d.edge_index, g_d.edge_type)
            ode_out = dynamics(node_emb, g_d.edge_index, g_d.edge_attr, g_d.growth)
        fig = plot_lv_trajectory(
            ode_out["trajectory"], species_names=sample_g.species_names,
            extinct_mask=ode_out["extinct"],
            title=f"LV Dynamics: {sample_g.eco_name}")
        fig.savefig(os.path.join(args.outdir, "lv_trajectory.png"),
                    dpi=150, bbox_inches="tight", facecolor="#0d1117")
        logger.info(f"  Saved: {args.outdir}/lv_trajectory.png")
    except Exception as e:
        logger.warning(f"  LV trajectory plot failed: {e}")

    try:
        fig = plot_trophic_embedding(encoder, all_graphs, device=device)
        fig.savefig(os.path.join(args.outdir, "trophic_embedding.png"),
                    dpi=150, bbox_inches="tight", facecolor="#0d1117")
        logger.info(f"  Saved: {args.outdir}/trophic_embedding.png")
    except Exception as e:
        logger.warning(f"  Trophic embedding plot failed: {e}")

    try:
        save_all_loo_plots(loo_results, args.outdir)
        logger.info(f"  Saved: {args.outdir}/loo_score_vs_severity.png")
        logger.info(f"  Saved: {args.outdir}/loo_score_distribution_by_outcome.png")
        logger.info(f"  Saved: {args.outdir}/loo_mean_scores_by_outcome.png")
        logger.info(f"  Saved: {args.outdir}/loo_feature_correlations.png")
        logger.info(f"  Saved: {args.outdir}/loo_feature_vs_score_grid.png")
    except Exception as e:
        logger.warning(f"  LOO plot suite failed: {e}")

    # ── Stage 7: Encoder quality metrics ──────────────────────────────────────
    logger.info("\n-- Stage 7: Encoder Quality Metrics --")
    trophic_res = check_trophic_clustering(encoder, all_graphs, device=device)
    edge_res    = check_edge_reconstruction(encoder, ssl_head, all_graphs, device=device)

    # Save quality metrics
    quality = {
        "trophic_spearman":    trophic_res.get("trophic_spearman"),
        "edge_type_accuracy":  edge_res.get("edge_type_accuracy"),
        "edge_weight_mae":     edge_res.get("edge_weight_mae"),
    }
    with open(os.path.join(args.outdir, "encoder_quality.json"), "w") as f:
        json.dump(quality, f, indent=2)
    logger.info(f"  Saved: {args.outdir}/encoder_quality.json")

    # ── Final summary ─────────────────────────────────────────────────────────
    rho    = loo_results.get("spearman_rho", float("nan"))
    p      = loo_results.get("spearman_p",   float("nan"))
    acc    = loo_results.get("accuracy",     float("nan"))
    sanity = loo_results.get("sanity_checks", {})

    logger.info("\n" + "=" * 64)
    logger.info("  FINAL RESULTS")
    logger.info("=" * 64)
    logger.info(f"  Pretraining graphs    : {len(all_graphs)}")
    logger.info(f"  Trophic Spearman r    : {trophic_res.get('trophic_spearman', 0):.3f}")
    logger.info(f"  Edge recon accuracy   : {edge_res.get('edge_type_accuracy', 0):.3f}")
    logger.info(f"  Spearman rho          : {rho:.3f}  (p={p:.4f})")
    logger.info(f"  Post-hoc 3-class acc  : {acc:.3f}")
    logger.info(f"  lambda_max ordered    : {sanity.get('lambda_max_ordered')}")
    logger.info(f"  Extinct frac ordered  : {sanity.get('extinct_frac_ordered')}")
    logger.info(f"\n  Outputs written to: {os.path.abspath(args.outdir)}/")
    logger.info(f"    loo/scores.csv        per-event scores")
    logger.info(f"    loo/features.csv      raw physics features")
    logger.info(f"    loo/summary.csv       Spearman rho + sanity stats")
    logger.info(f"    loo/encoder.pt        encoder weights")
    logger.info(f"    loo/ssl_head.pt       ssl_head weights")
    logger.info(f"    loo/full_model.pt     full_model weights")
    logger.info(f"    models/               pretrained checkpoints")
    logger.info(f"    pretrain_history.json loss curves")
    logger.info("=" * 64)


def _plot_loo_score_vs_severity(loo_results: dict, outdir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preds      = loo_results["loo_predictions"]
    scores     = [p["loo_score"]  for p in preds]   # key is "loo_score"
    severities = [p["severity"]   for p in preds]
    outcomes   = [p["outcome"]    for p in preds]
    names      = [p["event_name"] for p in preds]

    colour_map = {"stable": "#4caf50", "disrupted": "#ff9800", "collapse": "#f44336"}
    colours    = [colour_map.get(o, "#aaaaaa") for o in outcomes]

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.scatter(scores, severities, c=colours, s=90, zorder=3,
               edgecolors="white", linewidths=0.5)

    for score, sev, name in zip(scores, severities, names):
        ax.annotate(name.split("—")[0].strip()[:22],
                    (score, sev), textcoords="offset points",
                    xytext=(5, 3), fontsize=6.5, color="#cccccc", alpha=0.85)

    rho = loo_results.get("spearman_rho", float("nan"))
    p   = loo_results.get("spearman_p",   float("nan"))
    ax.text(0.03, 0.96, f"Spearman rho = {rho:.3f}  (p = {p:.4f})",
            transform=ax.transAxes, fontsize=10, color="white", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e1e2e", alpha=0.7))

    for lbl, col in colour_map.items():
        ax.scatter([], [], c=col, label=lbl, s=70)
    ax.legend(framealpha=0.3, labelcolor="white", facecolor="#1e1e2e",
              edgecolor="gray", fontsize=9)

    ax.set_xlabel("Disruption score (unsupervised)", color="white", fontsize=11)
    ax.set_ylabel("True severity (documented)",      color="white", fontsize=11)
    ax.set_title("Score vs. Documented Severity",    color="white", fontsize=13)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    fpath = os.path.join(outdir, "loo_score_vs_severity.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    logger.info(f"  Saved: {fpath}")


if __name__ == "__main__":
    main()
