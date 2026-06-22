"""
validation/plots.py — advanced validation visualisations for leave-one-out results.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
from typing import Dict, List, Optional

OUTCOME_COLORS = {"stable": "#009E73", "disrupted": "#E69F00", "collapse": "#D55E00"}
FEATURE_NAMES = [
    "invader_tl", "recon_loss", "damage_mode", "lv_severity",
    "invader_r", "body_mass_ratio", "extinct_frac", "lambda_max",
]


def _ensure_save_dir(save_path: Optional[str]):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)


def _extract_loo_arrays(loo_results: Dict):
    preds = loo_results.get("loo_predictions", [])
    if not preds:
        raise ValueError("loo_results must contain loo_predictions")

    scores = np.array([float(p["loo_score"]) for p in preds], dtype=float)
    severity = np.array([float(p["severity"]) for p in preds], dtype=float)
    outcomes = np.array([p["outcome"] for p in preds], dtype=str)
    names = [p["event_name"] for p in preds]
    feature_matrix = np.array(loo_results.get("features", []), dtype=float)
    return scores, severity, outcomes, names, feature_matrix


def plot_loo_score_vs_severity(loo_results: Dict, save_path: Optional[str] = None) -> plt.Figure:
    scores, severity, outcomes, names, _ = _extract_loo_arrays(loo_results)
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colours = [OUTCOME_COLORS.get(o, "#888888") for o in outcomes]
    ax.scatter(scores, severity, c=colours, s=80, alpha=0.85, edgecolors="black", linewidths=0.4)

    for score, sev, name in zip(scores, severity, names):
        ax.annotate(name.split("—")[0].strip()[:20], (score, sev),
                    textcoords="offset points", xytext=(4, 3), fontsize=7, color="#333333")

    rho = loo_results.get("spearman_rho", float("nan"))
    pval = loo_results.get("spearman_p", float("nan"))
    ax.text(0.02, 0.96, f"Spearman ρ = {rho:.3f} (p = {pval:.4f})",
            transform=ax.transAxes, fontsize=9, color="#222222",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f4f4f4", alpha=0.9))

    for label, color in OUTCOME_COLORS.items():
        ax.scatter([], [], c=color, label=label, s=50, alpha=0.8, edgecolors="black", linewidths=0.2)
    ax.legend(frameon=True, edgecolor="#cccccc", fontsize=8, facecolor="white")

    ax.set_xlabel("Disruption score", fontsize=10)
    ax.set_ylabel("Documented severity", fontsize=10)
    ax.set_title("LOO score vs documented severity", fontsize=11, pad=10)
    ax.grid(True, color="#eeeeee", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if save_path:
        _ensure_save_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_loo_score_distribution_by_outcome(loo_results: Dict, save_path: Optional[str] = None) -> plt.Figure:
    scores, _, outcomes, _, _ = _extract_loo_arrays(loo_results)
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    classes = ["collapse", "disrupted", "stable"]
    data = [scores[outcomes == cls] for cls in classes]
    positions = [1, 2, 3]

    bp = ax.boxplot(data, positions=positions, widths=0.6,
                    patch_artist=True, showfliers=False, medianprops={"color": "black"})
    for patch, cls in zip(bp["boxes"], classes):
        patch.set_facecolor(OUTCOME_COLORS[cls])
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)

    for pos, values, cls in zip(positions, data, classes):
        if values.size > 0:
            jitter = np.random.default_rng(42 + pos).uniform(-0.08, 0.08, size=values.size)
            ax.scatter(np.full(values.shape, pos) + jitter, values,
                       c=OUTCOME_COLORS[cls], edgecolors="black", linewidths=0.3,
                       s=30, alpha=0.75, zorder=3)
            ax.text(pos, -0.06, f"n={len(values)}", ha="center", fontsize=8, color="#444444",
                    transform=ax.get_xaxis_transform())

    ax.set_xticks(positions)
    ax.set_xticklabels([cls.capitalize() for cls in classes], fontsize=9)
    ax.set_ylabel("Disruption score", fontsize=10)
    ax.set_title("Score distribution by true outcome", fontsize=11, pad=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if save_path:
        _ensure_save_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_loo_mean_scores_by_outcome(loo_results: Dict, save_path: Optional[str] = None) -> plt.Figure:
    scores, _, outcomes, _, _ = _extract_loo_arrays(loo_results)
    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    classes = ["collapse", "disrupted", "stable"]
    means = [float(scores[outcomes == cls].mean()) if np.any(outcomes == cls) else float("nan")
             for cls in classes]
    counts = [int((outcomes == cls).sum()) for cls in classes]
    bars = ax.bar(classes, means, color=[OUTCOME_COLORS[c] for c in classes], alpha=0.8)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={count}", ha="center", va="bottom", fontsize=8, color="#333333")

    ax.set_ylabel("Mean disruption score", fontsize=10)
    ax.set_title("Mean score per outcome class", fontsize=11, pad=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if save_path:
        _ensure_save_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_loo_feature_correlation_matrix(loo_results: Dict, save_path: Optional[str] = None) -> plt.Figure:
    _, _, _, _, feature_matrix = _extract_loo_arrays(loo_results)
    if feature_matrix.size == 0:
        raise ValueError("loo_results must include feature matrix for correlation plot")

    labels = FEATURE_NAMES + ["loo_score", "severity"]
    score = np.array([float(p["loo_score"]) for p in loo_results["loo_predictions"]], dtype=float)
    severity = np.array([float(p["severity"]) for p in loo_results["loo_predictions"]], dtype=float)
    matrix = np.column_stack([feature_matrix, score, severity])
    corr = np.corrcoef(matrix, rowvar=False)

    fig, ax = plt.subplots(figsize=(8, 7), facecolor="white")
    fig.patch.set_facecolor("white")
    im = ax.imshow(corr, cmap="RdYlBu", vmin=-1, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Pearson correlation", fontsize=9)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr[i, j]) > 0.45 else "black", fontsize=7)

    ax.set_title("Feature and score correlation matrix", fontsize=11, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if save_path:
        _ensure_save_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_loo_feature_vs_score_grid(loo_results: Dict, save_path: Optional[str] = None) -> plt.Figure:
    scores, _, outcomes, _, feature_matrix = _extract_loo_arrays(loo_results)
    if feature_matrix.size == 0:
        raise ValueError("loo_results must include feature matrix for feature-vs-score plot")

    selected_indices = [2, 4, 6, 7]
    selected_labels = [FEATURE_NAMES[i] for i in selected_indices]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), facecolor="white")
    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    for ax, idx, label in zip(axes, selected_indices, selected_labels):
        values = feature_matrix[:, idx]
        for cls in np.unique(outcomes):
            mask = outcomes == cls
            ax.scatter(values[mask], scores[mask],
                       label=f"{cls} ({mask.sum()})",
                       c=OUTCOME_COLORS.get(cls, "#888888"), alpha=0.75, s=30,
                       edgecolors="black", linewidths=0.3)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Disruption score", fontsize=9)
        ax.grid(True, color="#eeeeee", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("Key feature relationships with disruption score", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        _ensure_save_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def save_all_loo_plots(loo_results: Dict, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    plot_loo_score_vs_severity(loo_results, os.path.join(outdir, "loo_score_vs_severity.png"))
    plot_loo_score_distribution_by_outcome(loo_results, os.path.join(outdir, "loo_score_distribution_by_outcome.png"))
    plot_loo_mean_scores_by_outcome(loo_results, os.path.join(outdir, "loo_mean_scores_by_outcome.png"))
    plot_loo_feature_correlation_matrix(loo_results, os.path.join(outdir, "loo_feature_correlations.png"))
    plot_loo_feature_vs_score_grid(loo_results, os.path.join(outdir, "loo_feature_vs_score_grid.png"))
