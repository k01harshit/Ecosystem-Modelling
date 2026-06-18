"""
demo.py — Self-contained demo using only synthetic data.
No internet connection required.

Demonstrates the full pipeline:
  1. Generate synthetic stable LV ecosystems
  2. Pretrain with SSL objectives
  3. Simulate Burmese-python-style invasion (apex predator, no natural enemies)
  4. Predict per-species risk
  5. Compare pre/post ODE trajectories
  6. Save all plots to outputs/demo/
"""

import os
import sys
import logging
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import SEED
from data.fetch import generate_synthetic_ecosystems
from data.graph_builder import (build_dataset, insert_invasive_species,
                                  build_pyg_graph)
from models.encoder import EcologicalEncoder
from models.ssl_heads import SSLHead
from models.neural_ode import EcologicalDynamicsModule
from models.stability_head import FullStabilityModel
from training.pretrain import pretrain
from training.finetune import finetune
from utils.viz import (plot_food_web, plot_lv_trajectory,
                        plot_training_curves, plot_species_risk)


def run_demo():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    outdir = "outputs/demo"
    os.makedirs(outdir, exist_ok=True)
    device = "cpu"

    # ── 1. Generate synthetic ecosystems ─────────────────────────────────────
    logger.info("Generating synthetic ecosystems...")
    ecosystems = generate_synthetic_ecosystems(n=100, seed=SEED)
    graphs     = build_dataset(ecosystems)
    logger.info(f"  {len(graphs)} graphs built")

    # Pick a medium-sized graph as our demo ecosystem
    demo_eco = sorted(ecosystems, key=lambda e: len(e["species"]))[len(ecosystems)//2]
    demo_g   = build_pyg_graph(demo_eco)
    logger.info(f"  Demo ecosystem: {demo_eco['name']}, "
                f"{len(demo_eco['species'])} species")

    # ── 2. Initialize models ──────────────────────────────────────────────────
    logger.info("\nInitializing models...")
    encoder  = EcologicalEncoder()
    ssl_head = SSLHead()
    dynamics = EcologicalDynamicsModule()
    full_model = FullStabilityModel(encoder, dynamics)

    total_params = sum(p.numel() for p in full_model.parameters())
    logger.info(f"  Total parameters: {total_params:,}")

    # ── 3. SSL Pretraining ────────────────────────────────────────────────────
    logger.info("\nSSL Pretraining (20 epochs)...")
    history = pretrain(encoder, ssl_head, graphs, epochs=20, device=device)

    fig = plot_training_curves(history, title="SSL Pretraining — Synthetic Ecosystems")
    fig.savefig(f"{outdir}/pretrain_loss.png", dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    plt.close(fig)
    logger.info(f"  Saved pretrain loss plot")

    # ── 4. Pre-invasion analysis ──────────────────────────────────────────────
    logger.info("\nAnalyzing pre-invasion ecosystem...")
    encoder.eval()
    dynamics.eval()

    with torch.no_grad():
        g_dev = demo_g.to(device)
        pre_node_emb, pre_graph_emb = encoder(
            g_dev.x, g_dev.edge_index, g_dev.edge_type)
        pre_ode = dynamics(pre_node_emb, g_dev.edge_index,
                           g_dev.edge_attr, g_dev.growth)

    # Plot pre-invasion food web
    fig = plot_food_web(demo_g, title="Demo Ecosystem — Before Invasion")
    fig.savefig(f"{outdir}/pre_invasion_foodweb.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)

    # Plot pre-invasion trajectory
    fig = plot_lv_trajectory(
        pre_ode["trajectory"],
        species_names=demo_eco["species"],
        title="Population Dynamics — Before Invasion",
    )
    fig.savefig(f"{outdir}/pre_invasion_trajectory.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)

    logger.info(f"  Pre-invasion resilience: {pre_ode['resilience']:.3f}")
    logger.info(f"  Pre-invasion stable: {pre_ode['stable']}")

    # ── 5. Insert Invasive Species ────────────────────────────────────────────
    # Simulate a "Burmese python" style invader:
    #   - Very high trophic level (4.5)
    #   - Apex predator with no natural predators in this ecosystem
    #   - Preys heavily on medium-trophic species
    logger.info("\nInserting invasive apex predator...")

    invader_attrs = {
        "trophic_level":   4.5,
        "log_body_mass":   5.2,
        "diet_type":       0,      # carnivore
        "habitat":         2,
        "log_repro_rate":  -0.5,
        "intrinsic_growth": 0.8,
    }

    # Predict edges: preys on all species below TL 3.0
    n_sp = demo_g.num_nodes
    predicted_edges = []
    for i, tl in enumerate(demo_g.trophic.tolist()):
        if tl < 3.0:
            w = max(0.3, 1.0 - abs(invader_attrs["trophic_level"] - tl - 1.5) * 0.3)
            predicted_edges.append({
                "target_idx": i,
                "type": "predation",
                "weight": w,
                "direction": "out",
            })

    post_g = insert_invasive_species(demo_g, invader_attrs, predicted_edges)
    logger.info(f"  Invader added with {len(predicted_edges)} predicted prey connections")

    # Post-invasion food web
    fig = plot_food_web(post_g, title="Demo Ecosystem — After Invasion",
                        highlight_species=n_sp)  # highlight invader
    fig.savefig(f"{outdir}/post_invasion_foodweb.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)

    # ── 6. Post-invasion dynamics ─────────────────────────────────────────────
    logger.info("\nAnalyzing post-invasion dynamics...")
    encoder.eval()
    dynamics.eval()

    with torch.no_grad():
        pg_dev = post_g.to(device)
        post_node_emb, post_graph_emb = encoder(
            pg_dev.x, pg_dev.edge_index, pg_dev.edge_type)
        post_ode = dynamics(post_node_emb, pg_dev.edge_index,
                            pg_dev.edge_attr, pg_dev.growth)

    # Plot post-invasion trajectory
    species_with_invader = demo_eco["species"] + ["INVADER"]
    fig = plot_lv_trajectory(
        post_ode["trajectory"],
        species_names=species_with_invader,
        extinct_mask=post_ode["extinct"],
        title="Population Dynamics — After Invasion (red=extinct)",
    )
    fig.savefig(f"{outdir}/post_invasion_trajectory.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)

    n_extinct = post_ode["extinct"][:n_sp].sum().item()
    logger.info(f"  Post-invasion resilience: {post_ode['resilience']:.3f}")
    logger.info(f"  Post-invasion stable: {post_ode['stable']}")
    logger.info(f"  Species predicted extinct: {n_extinct}/{n_sp}")

    # ── 7. Fine-tune and predict stability ────────────────────────────────────
    logger.info("\nBuilding invasion training pairs and fine-tuning (10 epochs)...")
    pairs = _build_synthetic_pairs_with_labels(graphs[:20], device)
    finetune_history = finetune(full_model, pairs, epochs=10, device=device)

    fig = plot_training_curves(finetune_history, title="Fine-tuning Loss")
    fig.savefig(f"{outdir}/finetune_loss.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)

    # Final stability prediction for demo invasion
    full_model.eval()
    with torch.no_grad():
        out = full_model(g_dev, pg_dev)

    stability    = out["stability"].item()
    biodiversity = out["biodiversity"].item()
    species_risk = out["species_risk"].detach().cpu().numpy()

    logger.info(f"\n  Stability score:    {stability:.3f} "
                f"({'STABLE' if stability > 0.5 else 'DISRUPTED'})")
    logger.info(f"  Biodiversity score: {biodiversity:.3f} "
                f"({biodiversity*100:.0f}% of species retained)")

    # Per-species risk
    fig = plot_species_risk(
        demo_eco["species"],
        species_risk,
        title="Per-Species Disruption Risk — After Invasion",
    )
    fig.savefig(f"{outdir}/species_risk.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)

    # ── 8. Summary comparison plot ────────────────────────────────────────────
    _plot_before_after_summary(pre_ode, post_ode, stability, biodiversity,
                                n_sp, outdir)

    # ── Final report ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("  DEMO COMPLETE")
    logger.info("=" * 55)
    logger.info(f"  Ecosystem: {demo_eco['name']}")
    logger.info(f"  Species:   {n_sp}")
    logger.info(f"  Pre-invasion resilience:  {pre_ode['resilience']:.3f}")
    logger.info(f"  Post-invasion resilience: {post_ode['resilience']:.3f}")
    logger.info(f"  Predicted stability:      {stability:.3f}")
    logger.info(f"  Predicted biodiversity:   {biodiversity:.3f}")
    logger.info(f"  Species extinct:          {n_extinct}/{n_sp}")
    logger.info(f"\n  Plots saved to: {outdir}/")
    logger.info("=" * 55)


def _build_synthetic_pairs_with_labels(graphs, device, n=10):
    """Build labeled (pre, post, label) triples from synthetic graphs."""
    from data.graph_builder import insert_invasive_species
    pairs = []
    for g in graphs[:n]:
        # High-impact invader
        inv_high = {"trophic_level": 4.5, "log_body_mass": 4.0, "diet_type": 0,
                    "habitat": 0, "log_repro_rate": -0.5, "intrinsic_growth": 1.5}
        N = g.num_nodes
        edges_high = [{"target_idx": i, "type": "predation",
                        "weight": 0.8, "direction": "out"}
                       for i in range(min(N, 4))]
        post_high = insert_invasive_species(g, inv_high, edges_high)
        pairs.append((g, post_high, {"stability": 0.1, "biodiversity": 0.3,
                                      "severity": 0.9, "outcome": "collapse",
                                      "name": f"high_impact_{g.eco_name}"}))

        # Low-impact invader
        inv_low = {"trophic_level": 2.0, "log_body_mass": 1.0, "diet_type": 2,
                   "habitat": 0, "log_repro_rate": 1.0, "intrinsic_growth": 0.8}
        edges_low = [{"target_idx": 0, "type": "competition",
                       "weight": 0.2, "direction": "out"}]
        post_low = insert_invasive_species(g, inv_low, edges_low)
        pairs.append((g, post_low, {"stability": 0.85, "biodiversity": 0.9,
                                     "severity": 0.1, "outcome": "stable",
                                     "name": f"low_impact_{g.eco_name}"}))
    return pairs


def _plot_before_after_summary(pre_ode, post_ode, stability,
                                 biodiversity, n_sp, outdir):
    """Side-by-side metrics comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    metrics = [
        ("Resilience",   pre_ode["resilience"],  post_ode["resilience"]),
        ("Stability",    1.0,                    stability),
        ("Biodiversity", 1.0,                    biodiversity),
    ]

    for ax, (name, pre_val, post_val) in zip(axes, metrics):
        ax.set_facecolor("#0d1117")
        bars = ax.bar(["Before", "After"], [pre_val, post_val],
                      color=["#2ecc71", "#e74c3c"], edgecolor="#ffffff22",
                      width=0.5)
        ax.set_ylim(0, 1.1)
        ax.set_title(name, color="white", fontsize=12, fontweight="bold")
        ax.tick_params(colors="#888888")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        for bar, val in zip(bars, [pre_val, post_val]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.03,
                    f"{val:.3f}", ha="center", color="white", fontsize=11,
                    fontweight="bold")

    fig.suptitle("Ecosystem Impact — Before vs After Invasion",
                 color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{outdir}/before_after_summary.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    logger.info(f"  Saved before/after summary plot")


if __name__ == "__main__":
    run_demo()
