"""
06_visualization.py
====================
MODULE 6 – Results Visualisation
----------------------------------
Produces all output figures described in the report (Section 4):

  Fig 1  – True-colour composite  (RGB from Haridwar bands)
  Fig 2  – Spectral indices panel (NDVI / NDWI / MNDWI / NDBI)
  Fig 6  – Change detection composite map (4 change types)
  Fig 7  – Model accuracy & F1-score bar chart
  Fig 12 – Per-module performance dashboard

All figures are saved to  outputs/figures/  as high-resolution PNGs.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")              # non-interactive backend (server safe)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import ConfusionMatrixDisplay

import config


log = logging.getLogger(__name__)

VIZ_DIR = config.OUT["viz_dir"]

# ──────────────────────────────────────────────────────────────────────────────
# Colour maps
# ──────────────────────────────────────────────────────────────────────────────
LC_COLORS = {
    10: "#a8d08d",  # Cropland      – light green
    20: "#1a7837",  # Forest        – dark green
    30: "#c6e58b",  # Grassland     – yellow-green
    40: "#f0c86b",  # Shrubland     – golden
    50: "#d73027",  # Urban         – red
    60: "#c9b299",  # Bare land     – tan
    80: "#4393c3",  # Water         – blue
    90: "#74add1",  # Wetland       – light blue
}

CHANGE_COLORS = {
    0: "#f7f7f7",   # No change       – white/grey
    1: "#d73027",   # Vegetation loss – red
    2: "#fc8d59",   # Urban expansion – orange
    3: "#4393c3",   # Water change    – blue
    4: "#762a83",   # Anomaly         – purple
}


def _save(fig: plt.Figure, filename: str, dpi: int = 150) -> None:
    path = os.path.join(VIZ_DIR, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved → %s", path)


def _percentile_stretch(arr: np.ndarray, lo=2, hi=98) -> np.ndarray:
    p_lo, p_hi = np.percentile(arr, lo), np.percentile(arr, hi)
    return np.clip((arr - p_lo) / (p_hi - p_lo + 1e-6), 0, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Individual figure functions
# ──────────────────────────────────────────────────────────────────────────────

def fig01_true_colour(ingest: IngestResult) -> None:
    """True-colour composite: Red = B3, Green = B2, Blue = synthetic."""
    log.info("  Fig 01 – True-colour composite …")
    R = _percentile_stretch(ingest.bands.get("B3_Red",   list(ingest.bands.values())[1]))
    G = _percentile_stretch(ingest.bands.get("B2_Green", list(ingest.bands.values())[0]))
    # Synthesise B (≈ 0.9 × G)
    B = np.clip(G * 0.90, 0, 1)
    rgb = np.stack([R, G, B], axis=-1)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(rgb)
    ax.set_title("True-Colour Composite\nHaridwar – ResourceSat-2 LISS-III (2019)",
                 fontsize=12, fontweight="bold")
    ax.axis("off")
    _save(fig, "fig01_true_colour.png")


def fig02_spectral_indices(preproc: PreprocResult) -> None:
    """2×2 panel of NDVI / NDWI / MNDWI / NDBI."""
    log.info("  Fig 02 – Spectral indices …")
    idx_names = ["NDVI", "NDWI", "MNDWI", "NDBI"]
    cmaps     = ["RdYlGn", "Blues", "Blues_r", "hot_r"]
    labels    = [
        "NDVI (Vegetation Index)",
        "NDWI (Water Index – Gao)",
        "MNDWI (Modified Water Index)",
        "NDBI (Built-up Index)",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, name, cmap, label in zip(axes.flat, idx_names, cmaps, labels):
        arr = preproc.indices.get(name)
        if arr is None:
            continue
        im = ax.imshow(arr, cmap=cmap, vmin=-1, vmax=1)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Spectral Indices – Haridwar Scene", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig02_spectral_indices.png")


def fig03_qa_confidence(preproc: PreprocResult) -> None:
    """QA confidence raster + histogram."""
    log.info("  Fig 03 – QA confidence raster …")
    qa = preproc.qa_confidence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im = ax1.imshow(qa, cmap="RdYlGn", vmin=0, vmax=1)
    ax1.set_title("QA Confidence Raster\n(0 = cloud  ·  1 = clear)", fontsize=10)
    ax1.axis("off")
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax2.hist(qa.ravel(), bins=50, color="#4393c3", edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Confidence Score", fontsize=10)
    ax2.set_ylabel("Pixel Count", fontsize=10)
    ax2.set_title(f"QA Confidence Distribution\n"
                  f"Mean = {qa.mean():.3f}", fontsize=10)
    ax2.axvline(qa.mean(), color="red", linestyle="--", label=f"Mean={qa.mean():.3f}")
    ax2.legend()

    plt.suptitle("Pre-processing Quality Assurance", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig03_qa_confidence.png")


def fig04_water_map(ingest: IngestResult,
                    model_result: ModelResult) -> None:
    """Predicted water map vs reference water mask."""
    log.info("  Fig 04 – Water body prediction vs reference …")
    ref  = ingest.water_mask_ref
    pred = model_result.water_map_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(ref,  cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title("Reference Water Mask (ISRO)", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    if pred is not None:
        axes[1].imshow(pred, cmap="Blues", vmin=0, vmax=1)
        axes[1].set_title("Predicted Water Map (RF Model)", fontsize=10, fontweight="bold")
    else:
        axes[1].text(0.5, 0.5, "Not available", ha="center", va="center")
    axes[1].axis("off")

    plt.suptitle("Water Body Classification – Haridwar", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig04_water_map.png")


def fig05_landcover_map(model_result: ModelResult) -> None:
    """Land-cover map with categorical colour legend."""
    log.info("  Fig 05 – Land-cover classification map …")
    lc_map = model_result.lc_map_pred
    if lc_map is None:
        log.warning("  LC map not available – skipping.")
        return

    classes = sorted(config.LANDCOVER_CLASSES.keys())
    cmap    = mcolors.ListedColormap([LC_COLORS[c] for c in classes])
    bounds  = [min(classes) - 5] + [c + 5 for c in classes]
    norm    = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(8, 7))
    im      = ax.imshow(lc_map, cmap=cmap, norm=norm)
    ax.set_title("Land-Cover Classification Map\n(Random Forest – 8 Classes)",
                 fontsize=11, fontweight="bold")
    ax.axis("off")

    patches = [
        mpatches.Patch(color=LC_COLORS[c],
                       label=f"{c} – {config.LANDCOVER_CLASSES[c]}")
        for c in classes
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8,
              framealpha=0.85, title="Land Cover Class")
    _save(fig, "fig05_landcover_map.png")


def fig06_change_detection(change_result: ChangeResult) -> None:
    """Composite change detection map."""
    log.info("  Fig 06 – Change detection map …")
    cmap   = mcolors.ListedColormap([CHANGE_COLORS[i] for i in range(5)])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im = axes[0].imshow(change_result.change_map, cmap=cmap, norm=norm)
    axes[0].set_title("Change Detection Map\n(T1→T2 Multi-Temporal Analysis)",
                      fontsize=10, fontweight="bold")
    axes[0].axis("off")
    labels = change_result.stats.get("label_map", {})
    patches = [mpatches.Patch(color=CHANGE_COLORS[i], label=labels.get(i, str(i)))
               for i in range(5)]
    axes[0].legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.85)

    # Change-type bar chart
    types  = ["Vegetation\nLoss", "Urban\nExpansion", "Water\nChange", "Anomaly"]
    counts = [
        change_result.stats.get("vegetation_loss_px", 0),
        change_result.stats.get("urban_expansion_px",  0),
        change_result.stats.get("water_change_px",     0),
        change_result.stats.get("anomaly_px",          0),
    ]
    colors = ["#d73027", "#fc8d59", "#4393c3", "#762a83"]
    bars   = axes[1].bar(types, counts, color=colors, edgecolor="white", linewidth=0.8)
    axes[1].set_ylabel("Number of Changed Pixels", fontsize=10)
    axes[1].set_title(f"Changed Pixels by Type\n"
                      f"Total change coverage: {change_result.stats.get('change_pct', 0):.2f}%",
                      fontsize=10, fontweight="bold")
    for bar, cnt in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(counts) * 0.01,
                     f"{cnt:,}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Change Detection & Analysis", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig06_change_detection.png")


def fig07_model_performance(model_result: ModelResult) -> None:
    """Bar chart comparing accuracy and F1-score across all models."""
    log.info("  Fig 07 – Model performance comparison …")
    names    = list(model_result.metrics.keys())
    acc      = [model_result.metrics[n]["accuracy"]   for n in names]
    f1_macro = [model_result.metrics[n]["f1_macro"]   for n in names]
    f1_wt    = [model_result.metrics[n]["f1_weighted"] for n in names]

    x    = np.arange(len(names))
    w    = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w,   acc,      w, label="Accuracy",      color="#4393c3", edgecolor="white")
    ax.bar(x,       f1_macro, w, label="F1-Macro",      color="#1a7837", edgecolor="white")
    ax.bar(x + w,   f1_wt,    w, label="F1-Weighted",   color="#d73027", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("AI Model Performance Summary\n(Accuracy · F1-Macro · F1-Weighted)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(0.9, linestyle="--", color="grey", linewidth=0.8, label="0.9 threshold")
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bar in ax.patches:
        ax.annotate(f"{bar.get_height():.3f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    _save(fig, "fig07_model_performance.png")


def fig08_confusion_matrices(model_result: ModelResult) -> None:
    """Confusion matrices for Water-RF and LC-RF."""
    log.info("  Fig 08 – Confusion matrices …")
    tasks = [
        ("Water_RF",  ["Non-Water", "Water"]),
        ("LC_RF",     [config.LANDCOVER_CLASSES[c]
                       for c in sorted(config.LANDCOVER_CLASSES)]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (name, labels) in zip(axes, tasks):
        if name not in model_result.metrics:
            ax.set_visible(False)
            continue
        m   = model_result.metrics[name]
        cm  = np.array(m["conf_matrix"])
        # Keep display names short
        short = [l[:10] for l in labels]
        disp  = ConfusionMatrixDisplay(confusion_matrix=cm,
                                       display_labels=short)
        disp.plot(ax=ax, colorbar=True, xticks_rotation=45,
                  cmap="Blues", values_format="d")
        ax.set_title(f"{name} – Confusion Matrix\n"
                     f"Accuracy={m['accuracy']:.3f}  F1={m['f1_macro']:.3f}",
                     fontsize=9, fontweight="bold")

    plt.suptitle("Confusion Matrices", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig08_confusion_matrices.png")


def fig09_feature_importance(model_result: ModelResult) -> None:
    """Feature importance for water and land-cover RF models."""
    log.info("  Fig 09 – Feature importance …")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pairs = [
        (model_result.feature_importance_water,
         ["Band2", "Band3", "Band4", "Band5"],
         "Water RF – Feature Importance", "#4393c3"),
        (model_result.feature_importance_lc,
         ["Blue", "Green", "Red", "NIR", "SWIR-1", "SWIR-2"],
         "Land-Cover RF – Feature Importance", "#1a7837"),
    ]
    for ax, (imp, names, title, color) in zip(axes, pairs):
        if imp is None:
            ax.set_visible(False)
            continue
        idx = np.argsort(imp)[::-1]
        ax.barh([names[i] for i in idx], [imp[i] for i in idx],
                color=color, edgecolor="white")
        ax.set_xlabel("Importance Score")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Feature Importance Analysis", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig09_feature_importance.png")


def fig10_delta_histograms(change_result: ChangeResult) -> None:
    """Histograms of per-index difference rasters."""
    log.info("  Fig 10 – Delta-index histograms …")
    delta = change_result.delta_indices
    n     = len(delta)
    if n == 0:
        return
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n == 1:
        axes = [axes]
    axes = np.array(axes).flat

    for ax, (name, arr) in zip(axes, delta.items()):
        ax.hist(arr.ravel(), bins=80, color="#4393c3", edgecolor="none",
                alpha=0.85, density=True)
        ax.axvline(0,            color="black",  linestyle="--", linewidth=1)
        ax.axvline(arr.mean(),   color="red",    linestyle="-",  linewidth=1.2,
                   label=f"mean={arr.mean():+.3f}")
        ax.set_title(f"Δ{name}", fontsize=10, fontweight="bold")
        ax.set_xlabel("T2 − T1 value")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide unused axes
    for ax in list(axes)[len(delta):]:
        ax.set_visible(False)

    plt.suptitle("Change Detection – Δ-Index Distributions",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig10_delta_histograms.png")


def fig11_bias_mitigation(ingest: IngestResult) -> None:
    """Class distribution before balancing (vizag land-cover)."""
    log.info("  Fig 11 – Class distribution (bias analysis) …")
    df  = ingest.df_vizag
    vc  = df["Landcover"].value_counts().sort_index()
    clabels = [f"{c}\n{config.LANDCOVER_CLASSES.get(c, '')}" for c in vc.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(clabels, vc.values,
                  color=[LC_COLORS.get(c, "#aaa") for c in vc.index],
                  edgecolor="white", linewidth=0.8)
    ax.axhline(vc.mean(), linestyle="--", color="red", linewidth=1.2,
               label=f"Mean = {vc.mean():.0f}")
    ax.set_ylabel("Sample Count")
    ax.set_title("Class Distribution – Vizag Land-Cover Dataset\n"
                 "(Before Balancing)", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 10, str(v),
                ha="center", fontsize=9)
    plt.tight_layout()
    _save(fig, "fig11_class_distribution.png")


def fig12_pipeline_dashboard(model_result: ModelResult,
                              change_result: ChangeResult,
                              preproc: PreprocResult) -> None:
    """One-page pipeline performance dashboard."""
    log.info("  Fig 12 – Pipeline dashboard …")
    fig = plt.figure(figsize=(16, 9))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel A: Model scores ────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    names = list(model_result.metrics.keys())
    accs  = [model_result.metrics[n]["accuracy"] for n in names]
    f1s   = [model_result.metrics[n]["f1_macro"] for n in names]
    x     = np.arange(len(names))
    ax_a.bar(x - 0.2, accs, 0.4, label="Accuracy", color="#4393c3")
    ax_a.bar(x + 0.2, f1s,  0.4, label="F1-Macro", color="#1a7837")
    ax_a.set_xticks(x); ax_a.set_xticklabels(names, fontsize=7, rotation=15)
    ax_a.set_ylim(0, 1.1); ax_a.legend(fontsize=7); ax_a.grid(axis="y", alpha=0.3)
    ax_a.set_title("Model Scores", fontsize=9, fontweight="bold")

    # ── Panel B: QA confidence histogram ────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    qa   = preproc.qa_confidence
    ax_b.hist(qa.ravel(), bins=40, color="#74add1", edgecolor="none")
    ax_b.axvline(qa.mean(), color="red", linestyle="--",
                 label=f"Mean={qa.mean():.2f}")
    ax_b.set_title("QA Confidence", fontsize=9, fontweight="bold")
    ax_b.legend(fontsize=7); ax_b.grid(alpha=0.3)

    # ── Panel C: Change types pie ────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    cs   = change_result.stats
    sizes = [cs.get("vegetation_loss_px", 1), cs.get("urban_expansion_px", 1),
             cs.get("water_change_px",    1), cs.get("anomaly_px",         1)]
    clbls = ["Veg Loss", "Urban Exp.", "Water Chg", "Anomaly"]
    colors= ["#d73027",  "#fc8d59",    "#4393c3",    "#762a83"]
    ax_c.pie(sizes, labels=clbls, colors=colors, autopct="%1.1f%%",
             textprops={"fontsize": 8})
    ax_c.set_title(f"Change Types\n(Total: {cs.get('change_pct',0):.1f}% scene)",
                   fontsize=9, fontweight="bold")

    # ── Panel D: NDVI map ────────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    ax_d.imshow(preproc.indices.get("NDVI", np.zeros((10,10))),
                cmap="RdYlGn", vmin=-1, vmax=1)
    ax_d.set_title("NDVI", fontsize=9, fontweight="bold"); ax_d.axis("off")

    # ── Panel E: Change map thumbnail ───────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    chmap = change_result.change_map
    cmap  = mcolors.ListedColormap([CHANGE_COLORS[i] for i in range(5)])
    norm  = mcolors.BoundaryNorm([-0.5,0.5,1.5,2.5,3.5,4.5], 5)
    ax_e.imshow(chmap, cmap=cmap, norm=norm)
    ax_e.set_title("Change Map", fontsize=9, fontweight="bold"); ax_e.axis("off")

    # ── Panel F: QA stats text ───────────────────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis("off")
    qa_s = preproc.qa_stats
    lines = [
        "QA SUMMARY",
        "─" * 28,
        f"Cloud pixels     : {qa_s.get('cloud_pct',0):.1f}%",
        f"Shadow pixels    : {qa_s.get('shadow_pct',0):.1f}%",
        f"Valid pixels     : {qa_s.get('valid_pct',0):.1f}%",
        f"Mean QA conf.    : {qa_s.get('mean_qa',0):.3f}",
        "",
        "CHANGE STATS",
        "─" * 28,
        f"Total changed    : {cs.get('change_pct',0):.2f}%",
        f"Veg loss pixels  : {cs.get('vegetation_loss_px',0):,}",
        f"Urban exp. pixels: {cs.get('urban_expansion_px',0):,}",
        f"Water chg pixels : {cs.get('water_change_px',0):,}",
        f"Anomaly pixels   : {cs.get('anomaly_px',0):,}",
    ]
    ax_f.text(0.05, 0.95, "\n".join(lines), transform=ax_f.transAxes,
              fontsize=8.5, va="top", fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f7f7",
                        edgecolor="#aaa"))

    fig.suptitle("Geodata Processing AI Pipeline – Performance Dashboard",
                 fontsize=13, fontweight="bold")
    _save(fig, "fig12_pipeline_dashboard.png", dpi=180)


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────
def run_visualization(ingest: IngestResult,
                      preproc: PreprocResult,
                      harmon: HarmonResult,
                      model_result: ModelResult,
                      change_result: ChangeResult) -> None:
    """Generate all 12 output figures."""
    log.info("═" * 60)
    log.info("MODULE 6 – RESULTS VISUALISATION")
    log.info("═" * 60)

    fig01_true_colour(ingest)
    fig02_spectral_indices(preproc)
    fig03_qa_confidence(preproc)
    fig04_water_map(ingest, model_result)
    fig05_landcover_map(model_result)
    fig06_change_detection(change_result)
    fig07_model_performance(model_result)
    fig08_confusion_matrices(model_result)
    fig09_feature_importance(model_result)
    fig10_delta_histograms(change_result)
    fig11_bias_mitigation(ingest)
    fig12_pipeline_dashboard(model_result, change_result, preproc)

    log.info("MODULE 6 complete │ %d figures saved to %s",
             12, VIZ_DIR)
