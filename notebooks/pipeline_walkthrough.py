"""
🛰️ Geodata Processing AI Pipeline — Walkthrough
-------------------------------------------------------------
Run this script from the repo root:
    notebooks/pipeline_walkthrough.py
"""

import os
import sys

# ── Setup: ensure src/ is on the path ────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # change to 'Agg' if running headless (no display)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

print("=" * 60)
print("  GEODATA PROCESSING AI PIPELINE — WALKTHROUGH")
print("  Amity University UP · Ayush Kumar Singh · 2025")
print("=" * 60)
print(f"  Working directory : {os.getcwd()}")
print("  Setup complete ✓")
print()


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — Data Ingestion
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 60)
print("MODULE 1 — DATA INGESTION")
print("─" * 60)

from module_runner import load

m1     = load('01_data_ingestion')
ingest = m1.run_ingestion()

print(f"  Bands loaded  : {list(ingest.bands.keys())}")
print(f"  Raster shape  : {list(ingest.bands.values())[0].shape}")
print(f"  CRS           : {ingest.crs}")
print(f"  water_train   : {ingest.df_water.shape}")
print(f"  vizag         : {ingest.df_vizag.shape}")
print(f"  Water coverage: {100 * ingest.water_mask_ref.mean():.1f}%")

print("\n  === water_train.csv ===")
print(ingest.df_water.describe().to_string())

print("\n  === vizag_sample_data.csv — class distribution ===")
print(ingest.df_vizag['Landcover'].value_counts().sort_index().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — Pre-processing & QA
# ══════════════════════════════════════════════════════════════════════════════
print()
print("─" * 60)
print("MODULE 2 — PRE-PROCESSING & QUALITY ASSURANCE")
print("─" * 60)

m2      = load('02_preprocessing')
preproc = m2.run_preprocessing(ingest)

print(f"  Indices computed : {list(preproc.indices.keys())}")
print(f"  QA stats         : {preproc.qa_stats}")

# ── Plot spectral indices ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, cmap) in zip(axes, [
    ('NDVI',  'RdYlGn'),
    ('NDWI',  'Blues'),
    ('NDBI',  'hot_r'),
]):
    im = ax.imshow(preproc.indices[name], cmap=cmap, vmin=-1, vmax=1)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle('Key Spectral Indices — Haridwar Scene',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/walkthrough_indices.png', dpi=120,
            bbox_inches='tight')
plt.show()
print("  Spectral indices plot saved → outputs/figures/walkthrough_indices.png")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — Bias Mitigation & Harmonisation
# ══════════════════════════════════════════════════════════════════════════════
print()
print("─" * 60)
print("MODULE 3 — BIAS MITIGATION & HARMONISATION")
print("─" * 60)

m3     = load('03_bias_harmonization')
harmon = m3.run_harmonisation(ingest, preproc)

print(f"  Water train shape : {harmon.X_water_train.shape}")
print(f"  LC train shape    : {harmon.X_lc_train.shape}")
print(f"  Raster features   : {harmon.raster_features.shape}")
print(f"  Bias report       : {harmon.bias_report['summary']}")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — AI Model Training & Inference  (Version 1: RF + MLP)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("─" * 60)
print("MODULE 4 — AI MODEL TRAINING & INFERENCE  (RF + MLP)")
print("─" * 60)

m4           = load('04_ai_models')
model_result = m4.run_ai_models(ingest, preproc, harmon)

print("\n  === Model Performance ===")
for name, m in model_result.metrics.items():
    print(f"    {name:<25}  Acc={m['accuracy']:.4f}  "
          f"F1(macro)={m['f1_macro']:.4f}  "
          f"F1(wt)={m['f1_weighted']:.4f}")

# ── Feature importance plot ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (imp, names, title, color) in zip(axes, [
    (model_result.feature_importance_water,
     ['Band2', 'Band3', 'Band4', 'Band5'],
     'Water RF — Feature Importance', '#4393c3'),
    (model_result.feature_importance_lc,
     ['Blue', 'Green', 'Red', 'NIR', 'SWIR-1', 'SWIR-2'],
     'Land-Cover RF — Feature Importance', '#1a7837'),
]):
    idx = np.argsort(imp)[::-1]
    ax.barh([names[i] for i in idx], [imp[i] for i in idx],
            color=color, edgecolor='white')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/walkthrough_feature_importance.png',
            dpi=120, bbox_inches='tight')
plt.show()
print("  Feature importance plot saved → outputs/figures/walkthrough_feature_importance.png")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — Change Detection
# ══════════════════════════════════════════════════════════════════════════════
print()
print("─" * 60)
print("MODULE 5 — CHANGE DETECTION")
print("─" * 60)

m5            = load('05_change_detection')
change_result = m5.run_change_detection(preproc, model_result)

cs = change_result.stats
print(f"  Total changed  : {cs['changed_pixels']:,} px  ({cs['change_pct']:.2f}%)")
print(f"  Vegetation loss: {cs['vegetation_loss_px']:,} px")
print(f"  Urban expansion: {cs['urban_expansion_px']:,} px")
print(f"  Water change   : {cs['water_change_px']:,} px")
print(f"  Anomaly        : {cs['anomaly_px']:,} px")

# ── Change map plot ──────────────────────────────────────────────────────────
COLORS = {
    0: '#f7f7f7',   # No change
    1: '#d73027',   # Vegetation loss
    2: '#fc8d59',   # Urban expansion
    3: '#4393c3',   # Water change
    4: '#762a83',   # Anomaly
}
cmap = mcolors.ListedColormap([COLORS[i] for i in range(5)])
norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], 5)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# Map
axes[0].imshow(change_result.change_map, cmap=cmap, norm=norm)
axes[0].set_title(
    f"Change Detection Map\n({cs['change_pct']:.2f}% scene changed)",
    fontsize=12, fontweight='bold'
)
axes[0].axis('off')
patches = [
    mpatches.Patch(color=COLORS[i], label=l)
    for i, l in enumerate(
        ['No Change', 'Veg Loss', 'Urban Exp.', 'Water Chg', 'Anomaly']
    )
]
axes[0].legend(handles=patches, loc='lower right', fontsize=9, framealpha=0.9)

# Bar chart
types  = ['Veg\nLoss', 'Urban\nExp.', 'Water\nChg', 'Anomaly']
counts = [
    cs['vegetation_loss_px'], cs['urban_expansion_px'],
    cs['water_change_px'],    cs['anomaly_px'],
]
bar_colors = ['#d73027', '#fc8d59', '#4393c3', '#762a83']
bars = axes[1].bar(types, counts, color=bar_colors, edgecolor='white')
axes[1].set_ylabel('Pixel Count')
axes[1].set_title('Changed Pixels by Type', fontsize=12, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)
for bar, cnt in zip(bars, counts):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(counts) * 0.01,
                 f'{cnt:,}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/figures/walkthrough_change_detection.png',
            dpi=120, bbox_inches='tight')
plt.show()
print("  Change map plot saved → outputs/figures/walkthrough_change_detection.png")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6 — Full Visualisation (all 12 figures)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("─" * 60)
print("MODULE 6 — FULL VISUALISATION  (12 figures)")
print("─" * 60)

m6 = load('06_visualization')
m6.run_visualization(ingest, preproc, harmon, model_result, change_result)
print("  All 12 figures saved → outputs/figures/")

# ── Display the dashboard ────────────────────────────────────────────────────
dashboard_path = 'outputs/figures/fig12_pipeline_dashboard.png'
if os.path.exists(dashboard_path):
    from PIL import Image
    img = Image.open(dashboard_path)
    plt.figure(figsize=(16, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Pipeline Performance Dashboard',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("  WALKTHROUGH COMPLETE")
print(f"  Outputs saved to: {os.path.abspath('outputs/')}")
print("=" * 60)
