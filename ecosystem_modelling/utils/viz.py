"""
utils/viz.py — v9

Nature-journal style plots:
  - Pure white background everywhere (fig + axes)
  - Arial → DejaVu Sans font fallback (Windows compatible, no warnings)
  - Minimal axes, Tufte style (top+right spines removed)
  - Wong (2011) colourblind-safe palette
  - Smart label placement (only annotates notable points)
  - constrained_layout for no text clipping
  - 300 dpi output
"""

import torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import logging, warnings
from typing import List, Dict, Optional
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

# ── Suppress font-not-found warnings (Arial → DejaVu Sans fallback is fine) ──
warnings.filterwarnings("ignore", message="findfont")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ── Global rcParams — set ONCE at import, applies to every figure ─────────────
# Arial is standard on Windows/Mac; DejaVu Sans is the matplotlib default fallback.
# Listing both suppresses the "Helvetica not found" warnings entirely.
matplotlib.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Liberation Sans", "DejaVu Sans"],
    "figure.facecolor":     "white",      # outer figure background
    "axes.facecolor":       "white",      # axes background
    "axes.edgecolor":       "#333333",    # spine colour
    "axes.labelcolor":      "#333333",    # axis label colour
    "xtick.color":          "#333333",    # tick marks and labels
    "ytick.color":          "#333333",
    "text.color":           "#333333",    # all other text
    "grid.color":           "#DDDDDD",
    "grid.linewidth":       0.6,
    "figure.dpi":           150,          # screen preview quality
    "savefig.dpi":          300,          # output quality
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.05,
    "axes.spines.top":      False,        # remove top spine globally
    "axes.spines.right":    False,        # remove right spine globally
    "legend.frameon":       True,
    "legend.edgecolor":     "#CCCCCC",
    "legend.facecolor":     "white",
})

# ── Wong (2011) colourblind-safe palette ──────────────────────────────────────
NATURE_COLORS = {
    "collapse":  "#D55E00",   # vermilion
    "disrupted": "#E69F00",   # orange
    "stable":    "#009E73",   # bluish-green
    "blue":      "#0072B2",
    "sky":       "#56B4E9",
    "yellow":    "#F0E442",
    "pink":      "#CC79A7",
    "gray":      "#999999",
    "black":     "#000000",
}
OUTCOME_COLORS = {k: NATURE_COLORS[k] for k in ("collapse","disrupted","stable")}
EDGE_COLORS = {0:"#D55E00", 1:"#0072B2", 2:"#009E73", 3:"#CC79A7"}  # distinct per type
EDGE_LABELS = {0:"predation", 1:"competition", 2:"mutualism", 3:"parasitism"}

# ── Typography ────────────────────────────────────────────────────────────────
FONT_FAMILY     = ["Arial", "Liberation Sans", "DejaVu Sans"]
FONT_SIZE_TITLE = 10
FONT_SIZE_LABEL = 9
FONT_SIZE_TICK  = 8
FONT_SIZE_ANNOT = 7
FONT_SIZE_LEGEND= 8

def _nature_style(ax, title=None, xlabel=None, ylabel=None):
    """Apply Nature-journal axis style. rcParams handle most settings globally."""
    ax.tick_params(width=0.8, length=3, labelsize=FONT_SIZE_TICK,
                   colors="#333333", labelcolor="#333333")
    ax.set_facecolor("white")
    # Ensure spines are dark (rcParams may not propagate in all backends)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
        spine.set_linewidth(0.8)
    if title:
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight="bold",
                     pad=6, color="#111111")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABEL, labelpad=4, color="#333333")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL, labelpad=4, color="#333333")

def _nature_fig(w=7.0, h=4.5):
    """Create a Nature-style figure. constrained_layout prevents text clipping."""
    fig = plt.figure(figsize=(w, h), facecolor="white",
                     layout="constrained")
    fig.patch.set_facecolor("white")
    return fig

# ── Plot functions ────────────────────────────────────────────────────────────

def plot_food_web(graph: Data, title="Food Web", save_path=None, highlight_species=None):
    fig, ax = _nature_fig(8, 5.5), None
    ax = fig.add_subplot(111)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")
    N   = graph.num_nodes
    tl  = graph.trophic.numpy() if hasattr(graph, "trophic") else np.ones(N) * 2
    names = getattr(graph, "species_names", [f"sp_{i}" for i in range(N)])
    rng = np.random.default_rng(42)
    pos = {}
    for i in range(N):
        same = [j for j in range(N) if abs(tl[j] - tl[i]) < 0.3]
        pos[i] = ((same.index(i) - len(same)/2)*1.4 + rng.uniform(-0.15, 0.15), float(tl[i]))
    ei = graph.edge_index.numpy()
    etypes   = graph.edge_type.numpy() if hasattr(graph, "edge_type") else np.zeros(ei.shape[1])
    eweights = graph.edge_attr[:,0].numpy() if hasattr(graph, "edge_attr") else np.ones(ei.shape[1])
    for k in range(ei.shape[1]):
        s,t = ei[0,k], ei[1,k]; x0,y0=pos[s]; x1,y1=pos[t]
        c = EDGE_COLORS.get(int(etypes[k]), "#aaa")
        ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
            arrowprops=dict(arrowstyle="->", color=c,
                lw=min(1.5, 0.3+eweights[k]*0.6), alpha=min(0.8, 0.2+eweights[k]*0.5)))
    for i in range(N):
        x,y = pos[i]
        c = NATURE_COLORS["blue"] if i == highlight_species else "#4C8CBF"
        ax.scatter(x, y, s=120+tl[i]*60, c=c, zorder=5, edgecolors="white", lw=0.6, alpha=0.9)
        if tl[i] >= 3.8 or tl[i] <= 1.3:
            ax.text(x, y+0.14, str(names[i])[:13], ha="center", va="bottom",
                    fontsize=FONT_SIZE_ANNOT, color="#222",
                    bbox=dict(boxstyle="round,pad=0.1",fc="white",ec="none",alpha=0.8))
    for tl_val in np.arange(1, 6):
        ax.axhline(tl_val, color="#DDDDDD", lw=0.5, ls="--", zorder=0)
        ax.text(-9.5, tl_val, f"TL {tl_val:.0f}", color="#BBBBBB",
                fontsize=FONT_SIZE_ANNOT, va="center", ha="right", style="italic")
    patches = [mpatches.Patch(color=c, label=EDGE_LABELS[t]) for t,c in EDGE_COLORS.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=FONT_SIZE_LEGEND,
              framealpha=0.9, edgecolor="#ccc", facecolor="white")
    _nature_style(ax, title=title, xlabel="Species spread", ylabel="Trophic level")
    ax.spines["left"].set_visible(True); ax.spines["bottom"].set_visible(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    # constrained_layout handles spacing
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_lv_trajectory(trajectory, species_names=None, extinct_mask=None,
                        title="Population dynamics", save_path=None):
    T, N = trajectory.shape
    traj = trajectory.detach().cpu().numpy()
    times = np.linspace(0, 10, T)
    if species_names is None: species_names = [f"sp_{i}" for i in range(N)]
    fig = _nature_fig(8, 4)
    ax  = fig.add_subplot(111)
    cmap = plt.colormaps.get_cmap("tab20").resampled(N) if hasattr(plt, "colormaps") \
           else plt.cm.get_cmap("tab20", N)
    extinct = extinct_mask.cpu().numpy().astype(bool) if extinct_mask is not None \
              else np.zeros(N, dtype=bool)
    for i in range(N):
        ax.plot(times, traj[:,i], color=cmap(i),
                ls="--" if extinct[i] else "-",
                alpha=0.35 if extinct[i] else 0.80,
                lw=0.8 if extinct[i] else 1.2,
                label=str(species_names[i])[:18] if i < 10 else None)
    _nature_style(ax, title=title, xlabel="Time", ylabel="Population")
    if N <= 15:
        ax.legend(fontsize=FONT_SIZE_LEGEND, ncol=2, framealpha=0.8,
                  edgecolor="#ccc", facecolor="white")
    # constrained_layout handles spacing
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_training_curves(history: Dict, title="Training loss", save_path=None):
    fig = _nature_fig(4*len(history), 3.5)
    axes = fig.subplots(1, len(history))
    if len(history) == 1: axes = [axes]
    palette = [NATURE_COLORS["blue"], NATURE_COLORS["collapse"],
               NATURE_COLORS["stable"], NATURE_COLORS["pink"], NATURE_COLORS["gray"]]
    for ax, (key, vals), color in zip(axes, history.items(), palette):
        ax.plot(vals, color=color, lw=1.2)
        _nature_style(ax, title=key, xlabel="Epoch", ylabel="Loss")
    fig.suptitle(title, fontsize=FONT_SIZE_TITLE+1, fontweight="bold")
    # constrained_layout handles spacing
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_species_risk(species_names, risk_scores, title="Per-species disruption risk", save_path=None):
    idx = np.argsort(risk_scores)[::-1]
    names_s = [str(species_names[i])[:30] for i in idx]; risk_s = risk_scores[idx]
    fig = _nature_fig(7, max(3.5, len(names_s)*0.35))
    ax  = fig.add_subplot(111)
    colors = [NATURE_COLORS["collapse"] if r > 0.7 else
              NATURE_COLORS["disrupted"] if r > 0.4 else
              NATURE_COLORS["stable"] for r in risk_s]
    ax.barh(range(len(names_s)), risk_s, color=colors, edgecolor="none", height=0.6)
    ax.set_yticks(range(len(names_s))); ax.set_yticklabels(names_s, fontsize=FONT_SIZE_TICK)
    ax.set_xlim(0, 1); ax.axvline(0.5, color="#888", lw=0.8, ls="--")
    _nature_style(ax, title=title, xlabel="Disruption risk score")
    # constrained_layout handles spacing
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_cv_summary(cv_results: Dict, save_path=None):
    fold_results = cv_results.get("fold_results", [])
    summary      = cv_results.get("summary", {})
    n_folds      = len(fold_results)
    fig = _nature_fig(max(5, n_folds*2.2), 4)
    ax  = fig.add_subplot(111)
    metrics = ["test_auc", "test_severity_r", "test_binary_acc"]
    labels  = ["AUC", "Severity r", "Binary acc"]
    colors  = [NATURE_COLORS["blue"], NATURE_COLORS["collapse"], NATURE_COLORS["stable"]]
    hatches = ["", "//", ".."]
    x = np.arange(n_folds); bw = 0.25
    for i, (metric, label, color, hatch) in enumerate(zip(metrics, labels, colors, hatches)):
        vals = [fr.get(metric, 0) or 0 for fr in fold_results]
        bars = ax.bar(x+i*bw, vals, bw, label=label, color=color, alpha=0.85,
                      hatch=hatch, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if not np.isnan(val) and val > 0.05:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f"{val:.2f}", ha="center", va="bottom",
                        fontsize=FONT_SIZE_ANNOT)
    ax.set_xticks(x+bw); ax.set_xticklabels([f"Fold {fr['fold']}" for fr in fold_results], fontsize=FONT_SIZE_TICK)
    ax.set_ylim(0, 1.25); ax.axhline(0.5, color="#BBBBBB", lw=0.7, ls="--")
    for j, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        m = summary.get(f"{metric}_mean", 0) or 0
        s = summary.get(f"{metric}_std",  0) or 0
        ax.text(0.98, 0.97-j*0.09, f"{label}: {m:.3f} ± {s:.3f}",
                transform=ax.transAxes, ha="right", va="top", color=color,
                fontsize=FONT_SIZE_LEGEND,
                bbox=dict(facecolor="white", edgecolor="#ccc", alpha=0.85, boxstyle="round,pad=0.2"))
    _nature_style(ax, title=f"{n_folds}-fold cross-validation — test metrics",
                  ylabel="Score")
    ax.legend(fontsize=FONT_SIZE_LEGEND, framealpha=0.9, edgecolor="#ccc", facecolor="white")
    # constrained_layout handles spacing
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def _three_class_pred(pred_disruption: float) -> str:
    if pred_disruption > 0.66: return "collapse"
    if pred_disruption > 0.33: return "disrupted"
    return "stable"


def plot_confusion_matrix(all_predictions: List[Dict], save_path=None):
    classes = ["collapse", "disrupted", "stable"]
    n = len(classes); matrix = np.zeros((n, n), dtype=int)
    idx_map = {c: i for i, c in enumerate(classes)}
    test_preds = [p for p in all_predictions if p.get("split") == "test"]
    for p in test_preds:
        tc = p["outcome"]; pc = _three_class_pred(p["pred_disruption"])
        if tc in idx_map and pc in idx_map:
            matrix[idx_map[tc], idx_map[pc]] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_mat = np.zeros_like(matrix, dtype=float)
    np.divide(matrix, row_sums, out=norm_mat, where=row_sums > 0)
    fig = _nature_fig(5.5, 4.5)
    ax  = fig.add_subplot(111)
    im = ax.imshow(norm_mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(n):
        for j in range(n):
            tc = "white" if norm_mat[i,j] > 0.55 else "#222"
            ax.text(j, i, f"{matrix[i,j]}\n({norm_mat[i,j]*100:.0f}%)",
                    ha="center", va="center", color=tc,
                    fontsize=FONT_SIZE_TICK+1, fontweight="bold")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    col_labels = ["Pred: collapse", "Pred: disrupted", "Pred: stable"]
    row_labels = ["True: collapse", "True: disrupted", "True: stable"]
    ax.set_xticklabels(col_labels, fontsize=FONT_SIZE_TICK, rotation=15, ha="right")
    ax.set_yticklabels(row_labels, fontsize=FONT_SIZE_TICK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.04)
    cbar.set_label("Recall", fontsize=FONT_SIZE_LABEL)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)
    n_total = len(test_preds); n_correct = sum(1 for p in test_preds if p.get("correct"))
    _nature_style(ax, title="Confusion matrix — all CV test folds\n(row-normalised; diagonal = recall per class)")
    ax.text(0.5, -0.18, f"Overall accuracy: {n_correct}/{n_total} ({100*n_correct/max(n_total,1):.0f}%)",
            ha="center", transform=ax.transAxes, fontsize=FONT_SIZE_ANNOT+1, color="#555")
    # constrained_layout handles spacing
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_outcome_score_distributions(all_predictions: List[Dict], save_path=None):
    test_preds = [p for p in all_predictions if p.get("split") == "test"]
    classes = ["collapse", "disrupted", "stable"]
    scores_by_class = {c: [p["pred_disruption"] for p in test_preds if p["outcome"]==c] for c in classes}
    fig = _nature_fig(5.5, 4)
    ax  = fig.add_subplot(111)
    for pos, cls in zip([1, 2, 3], classes):
        scores = scores_by_class[cls]
        if not scores: continue
        color = OUTCOME_COLORS[cls]
        if len(scores) >= 3:
            parts = ax.violinplot(scores, positions=[pos], widths=0.5,
                                   showmedians=True, showextrema=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(color); pc.set_alpha(0.35); pc.set_edgecolor(color); pc.set_linewidth(0.8)
            for pn in ("cbars","cmins","cmaxes","cmedians"):
                if pn in parts:
                    parts[pn].set_edgecolor(color); parts[pn].set_linewidth(1.2)
        jitter = np.random.default_rng(42+pos).uniform(-0.10, 0.10, len(scores))
        ax.scatter(np.array([pos]*len(scores))+jitter, scores, c=color,
                   s=30, alpha=0.85, zorder=5, edgecolors="white", lw=0.5)
        ax.hlines(np.mean(scores), pos-0.18, pos+0.18, colors=color, lw=2.0, zorder=6)
        ax.text(pos, -0.07, f"n={len(scores)}", ha="center", color=color,
                fontsize=FONT_SIZE_ANNOT+1,
                transform=ax.get_xaxis_transform())
    ax.axhline(0.5, color="#BBBBBB", lw=0.8, ls="--", label="Decision boundary")
    ax.set_xticks([1,2,3]); ax.set_xticklabels(["Collapse","Disrupted","Stable"], fontsize=FONT_SIZE_TICK)
    ax.set_ylim(-0.05, 1.10)
    _nature_style(ax, title="Predicted score distributions by true outcome class",
                  ylabel="Predicted disruption score (1 − stability)")
    ax.legend(fontsize=FONT_SIZE_LEGEND, framealpha=0.9, edgecolor="#ccc", facecolor="white")
    # constrained_layout handles spacing
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_cv_event_heatmap(cv_results: Dict, save_path=None):
    fold_results = cv_results.get("fold_results", [])
    all_preds = cv_results.get("all_test_predictions", [])
    if not all_preds:
        for fr in cv_results.get("fold_results", []):
            all_preds.extend(fr.get("test_predictions", []))
    event_names = list(dict.fromkeys(p["event_name"] for p in all_preds))
    n_events = len(event_names); n_folds = len(fold_results)
    matrix = np.zeros((n_events, n_folds))
    outcomes = {}; sev_map = {}
    for p in all_preds:
        ei = event_names.index(p["event_name"]); fi = p["fold"]-1
        if p["split"] == "test":
            matrix[ei, fi] = 1.0 if p["correct"] else -1.0
        outcomes[p["event_name"]] = p["outcome"]
        sev_map[p["event_name"]]  = p["severity"]
    order_key = {"collapse":0, "disrupted":1, "stable":2}
    event_names_sorted = sorted(event_names,
        key=lambda n:(order_key.get(outcomes.get(n,"stable"),1), -sev_map.get(n,0)))
    sorted_idx = [event_names.index(n) for n in event_names_sorted]
    matrix_s = matrix[sorted_idx, :]
    matrix_s_display = (matrix_s + 1.0) / 2.0
    fig = _nature_fig(max(4.5, n_folds*1.6), max(5.5, n_events*0.38))
    ax  = fig.add_subplot(111)
    ax.set_facecolor("#F8F8F8")
    # Nature-style tricolour: red=wrong, light grey=training, green=correct
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "hm_nature", ["#D55E00", "#EEEEEE", "#009E73"], N=256)
    ax.imshow(matrix_s_display, cmap=cmap, vmin=0, vmax=1,
              aspect="auto", interpolation="nearest")
    for i in range(n_events):
        for j in range(n_folds):
            val = matrix_s[i,j]
            sym = "✓" if val > 0 else ("✗" if val < 0 else "·")
            color = "white" if abs(val) > 0.5 else "#999"
            ax.text(j, i, sym, ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold")
    ax.set_xticks(range(n_folds))
    ax.set_xticklabels([f"Fold {k+1}" for k in range(n_folds)], fontsize=FONT_SIZE_TICK)
    ax.set_yticks(range(n_events))
    yticklabels = [n.split("—")[0].strip()[:26] for n in event_names_sorted]
    ax.set_yticklabels(yticklabels, fontsize=FONT_SIZE_ANNOT+1)
    for ytick, en in zip(ax.get_yticklabels(), event_names_sorted):
        ytick.set_color(OUTCOME_COLORS.get(outcomes.get(en,"disrupted"),"#222"))
    _nature_style(ax, title="Per-event × per-fold results\n✓ correct (test)   ✗ wrong (test)   · training fold")
    patches = [mpatches.Patch(color=c, label=k) for k,c in OUTCOME_COLORS.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=FONT_SIZE_LEGEND,
              framealpha=0.9, edgecolor="#ccc", facecolor="white", title="True outcome",
              title_fontsize=FONT_SIZE_LEGEND)
    # constrained_layout handles spacing
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_trophic_risk_profile(species_names, risk_scores, trophic_levels,
                               title="Trophic risk profile", save_path=None):
    risk = np.array(risk_scores).flatten(); tl = np.array(trophic_levels).flatten()
    n = min(len(risk), len(tl), len(species_names))
    risk = risk[:n]; tl = tl[:n]; names = [str(species_names[i])[:20] for i in range(n)]
    order = np.argsort(tl); tl = tl[order]; risk = risk[order]; names = [names[i] for i in order]
    fig = _nature_fig(7.5, 4)
    ax  = fig.add_subplot(111)
    sc  = ax.scatter(tl, risk, c=tl, cmap="plasma", s=40+risk*120,
                     alpha=0.80, edgecolors="white", lw=0.5, zorder=5)
    threshold = np.percentile(risk, 70)
    labeled_x = []
    for x, y, name in zip(tl, risk, names):
        if y >= threshold:
            if not any(abs(x-lx) < 0.3 for lx in labeled_x):
                ax.annotate(name, (x,y), xytext=(4,4),
                            textcoords="offset points",
                            fontsize=FONT_SIZE_ANNOT, color="#333", alpha=0.95,
                            bbox=dict(boxstyle="round,pad=0.1",fc="white",ec="none",alpha=0.75))
                labeled_x.append(x)
    ax.axhline(0.5, color="#BBBBBB", lw=0.8, ls="--", label="Risk threshold")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Trophic level", fontsize=FONT_SIZE_LABEL)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.set_ylim(-0.05, 1.10)
    _nature_style(ax, title=title, xlabel="Trophic level", ylabel="Disruption risk score")
    ax.legend(fontsize=FONT_SIZE_LEGEND, framealpha=0.9, edgecolor="#ccc", facecolor="white")
    # constrained_layout handles spacing
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_severity_calibration(cv_results: Dict, save_path=None) -> plt.Figure:
    from scipy.stats import spearmanr
    all_preds = cv_results.get("all_test_predictions", [])
    if not all_preds:
        for fr in cv_results.get("fold_results", []):
            all_preds.extend(fr.get("test_predictions", []))
    if not all_preds:
        fig, ax = plt.subplots(); ax.text(0.5,0.5,"No data",ha="center",va="center"); return fig
    fig = _nature_fig(5.5, 5.0)
    ax  = fig.add_subplot(111)
    ax.plot([0,1],[0,1], color="#BBBBBB", ls="--", lw=0.8, label="Perfect calibration", zorder=1)
    markers = {"collapse":"^", "disrupted":"o", "stable":"s"}
    _to_label = {id(p) for p in all_preds if not p["correct"]}
    for outcome in ("collapse","disrupted","stable"):
        top2 = sorted([p for p in all_preds if p["outcome"]==outcome and p["correct"]],
                      key=lambda x: -x["severity"])[:2]
        _to_label.update(id(p) for p in top2)
    for p in all_preds:
        outcome = p["outcome"]; color = OUTCOME_COLORS.get(outcome,"#888")
        marker  = markers[outcome]
        edgecol = "white" if p["correct"] else "#333"
        lw      = 0.5 if p["correct"] else 1.5
        ax.scatter(p["severity"], p["pred_disruption"], c=color, marker=marker,
                   s=55, alpha=0.85, edgecolors=edgecol, lw=lw, zorder=5)
        if id(p) in _to_label:
            label = p["event_name"].split("—")[0].strip()[:18]
            dx = 6 if p["severity"] < 0.5 else -6
            dy = 4 if p["pred_disruption"] < 0.5 else -8
            ha = "left" if dx > 0 else "right"
            ax.annotate(label, (p["severity"], p["pred_disruption"]),
                        textcoords="offset points", xytext=(dx,dy),
                        fontsize=FONT_SIZE_ANNOT, color="#333", ha=ha,
                        bbox=dict(boxstyle="round,pad=0.1",fc="white",ec="none",alpha=0.75))
    xs = np.array([p["severity"]        for p in all_preds])
    ys = np.array([p["pred_disruption"] for p in all_preds])
    r_all, _ = spearmanr(xs, ys)
    _nature_style(ax, title=f"Severity calibration — all CV test events\nSpearman r = {r_all:.3f}",
                  xlabel="Documented severity (ground truth)",
                  ylabel="Predicted disruption score")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    legend_elems = [
        mpatches.Patch(color=OUTCOME_COLORS["collapse"],  label="Collapse"),
        mpatches.Patch(color=OUTCOME_COLORS["disrupted"], label="Disrupted"),
        mpatches.Patch(color=OUTCOME_COLORS["stable"],    label="Stable"),
        plt.scatter([], [], marker="o", c="gray", s=30, edgecolors="white", lw=0.5, label="Correct"),
        plt.scatter([], [], marker="o", c="gray", s=30, edgecolors="#333",  lw=1.5, label="Wrong"),
    ]
    ax.legend(handles=legend_elems, fontsize=FONT_SIZE_LEGEND,
              framealpha=0.9, edgecolor="#ccc", facecolor="white")
    # constrained_layout handles spacing
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_outcome_confidence(cv_results: Dict, save_path=None) -> plt.Figure:
    all_preds = cv_results.get("all_test_predictions", [])
    if not all_preds:
        for fr in cv_results.get("fold_results", []):
            all_preds.extend(fr.get("test_predictions", []))
    if not all_preds:
        fig, ax = plt.subplots(); ax.text(0.5,0.5,"No data",ha="center",va="center"); return fig
    OUTCOMES_ORD = ["stable","disrupted","collapse"]
    groups = {o:[p["pred_disruption"] for p in all_preds if p["outcome"]==o] for o in OUTCOMES_ORD}
    fig = _nature_fig(5.5, 4.0)
    ax  = fig.add_subplot(111)
    bp = ax.boxplot([groups[o] for o in OUTCOMES_ORD], positions=[1,2,3], widths=0.35,
                    patch_artist=True,
                    medianprops=dict(color="#222", lw=1.5),
                    whiskerprops=dict(color="#888", lw=0.8),
                    capprops=dict(color="#888", lw=0.8),
                    flierprops=dict(marker="o", markerfacecolor="#aaa", markersize=3, lw=0))
    for patch, outcome in zip(bp["boxes"], OUTCOMES_ORD):
        patch.set_facecolor(OUTCOME_COLORS[outcome]); patch.set_alpha(0.45); patch.set_linewidth(0.8)
    rng = np.random.default_rng(42)
    for pos, outcome in zip([1,2,3], OUTCOMES_ORD):
        vals = groups[outcome]
        if not vals: continue
        jitter = rng.uniform(-0.10, 0.10, len(vals))
        ax.scatter(np.full(len(vals),pos)+jitter, vals, c=OUTCOME_COLORS[outcome],
                   s=28, alpha=0.85, edgecolors="white", lw=0.5, zorder=5)
        ax.text(pos, -0.06, f"n={len(vals)}", ha="center",
                fontsize=FONT_SIZE_ANNOT+1, color="#666")
    ax.axhline(0.5, color="#BBBBBB", lw=0.8, ls="--", label="Decision boundary")
    ax.set_xticks([1,2,3]); ax.set_xticklabels(["Stable","Disrupted","Collapse"], fontsize=FONT_SIZE_TICK)
    ax.set_ylim(-0.10, 1.15)
    _nature_style(ax, title="Prediction confidence by true outcome class",
                  ylabel="Predicted disruption score")
    ax.legend(fontsize=FONT_SIZE_LEGEND, framealpha=0.9, edgecolor="#ccc", facecolor="white")
    # constrained_layout handles spacing
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_trophic_embedding(encoder, graphs, device="cpu", save_path=None) -> plt.Figure:
    from sklearn.decomposition import PCA
    encoder.eval()
    all_embs, all_trophic, all_names = [], [], []
    with torch.no_grad():
        for g in graphs[:15]:
            g = g.to(device)
            node_emb, _ = encoder(g.x, g.edge_index, g.edge_type)
            all_embs.append(node_emb.cpu().numpy())
            tl = g.trophic.cpu().numpy() if hasattr(g,"trophic") else np.ones(g.num_nodes)*2.0
            all_trophic.append(tl)
            names = getattr(g,"species_names",[f"sp_{i}" for i in range(g.num_nodes)])
            all_names.extend(names)
    embs    = np.vstack(all_embs)
    trophic = np.concatenate(all_trophic)
    pca     = PCA(n_components=2, random_state=42)
    coords  = pca.fit_transform(embs)
    var     = pca.explained_variance_ratio_
    fig = _nature_fig(6, 5)
    ax  = fig.add_subplot(111)
    sc  = ax.scatter(coords[:,0], coords[:,1], c=trophic,
                     cmap="viridis", s=30, alpha=0.75, edgecolors="none")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.040, pad=0.04)
    cbar.set_label("Trophic level", fontsize=FONT_SIZE_LABEL)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)
    named_idx = [i for i,n in enumerate(all_names) if not str(n).startswith("sp_")]
    rng = np.random.default_rng(42)
    if len(named_idx) > 15:
        named_idx = list(rng.choice(named_idx, 15, replace=False))
    done = []
    for i in named_idx:
        x,y = coords[i,0], coords[i,1]
        if any(abs(x-lx)<0.35 and abs(y-ly)<0.35 for lx,ly in done):
            continue
        ax.annotate(str(all_names[i])[:16], (x,y), textcoords="offset points",
                    xytext=(4,3), fontsize=FONT_SIZE_ANNOT, color="#333", alpha=0.9,
                    bbox=dict(boxstyle="round,pad=0.1",fc="white",ec="none",alpha=0.75))
        done.append((x,y))
    _nature_style(ax,
        title="Encoder embedding space — coloured by trophic level",
        xlabel=f"PC1 ({var[0]:.1%} variance)",
        ylabel=f"PC2 ({var[1]:.1%} variance)")
    # constrained_layout handles spacing
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_per_fold_confusion(cv_results: Dict, save_path=None) -> plt.Figure:
    OUTCOMES_ORD = ["stable","disrupted","collapse"]
    fold_results  = cv_results.get("fold_results",[])
    n = len(fold_results)
    if n == 0:
        fig, ax = plt.subplots(); ax.text(0.5,0.5,"No data",ha="center",va="center"); return fig
    fig, axes = plt.subplots(1, n, figsize=(3.5*n, 3.5), facecolor="white")
    if n == 1: axes = [axes]
    for ax, fr in zip(axes, fold_results):
        cm = fr.get("confusion_matrix",{})
        matrix = np.array([[cm.get(t,{}).get(p,0) for p in OUTCOMES_ORD]
                            for t in OUTCOMES_ORD], dtype=float)
        row_sums = matrix.sum(axis=1, keepdims=True)
        norm = np.zeros_like(matrix)
        np.divide(matrix, row_sums, out=norm, where=row_sums>0)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(["S","D","C"], fontsize=FONT_SIZE_TICK)
        ax.set_yticklabels(["S","D","C"], fontsize=FONT_SIZE_TICK)
        for i in range(3):
            for j in range(3):
                raw = int(matrix[i,j]); frac = norm[i,j]
                tc  = "white" if frac > 0.55 else "#222"
                ax.text(j, i, f"{raw}\n{frac:.0%}", ha="center", va="center",
                        color=tc, fontsize=FONT_SIZE_TICK, fontweight="bold")
        acc = fr.get("test_binary_acc", float("nan"))
        auc = fr.get("test_auc",        float("nan"))
        _nature_style(ax, title=f"Fold {fr['fold']}\nAcc={acc:.2f}  AUC={auc:.2f}")
    fig.suptitle("Per-fold confusion matrices  (S=Stable  D=Disrupted  C=Collapse)",
                 fontsize=FONT_SIZE_TITLE, fontweight="bold", y=1.02)
    # constrained_layout handles spacing
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
