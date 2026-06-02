"""
05_change_detection.py
=======================
MODULE 5 – Change Detection & Analysis
----------------------------------------
Implements the "Change Detection & Analysis" block (Figure 1, Section 3):

  Detects four change types on the Haridwar raster:
    1. Abrupt changes   – deforestation, construction (threshold on ΔNDVI)
    2. Gradual changes  – urban expansion (threshold on ΔNDBI)
    3. Water changes    – flooding / drying  (ΔNDWI + predicted water map)
    4. Anomalies        – statistically extreme spectral deviations (z-score)

  Since we have only one date of satellite imagery, the "before" image is
  synthesised by applying a simulated 10-year land-use shift scenario on the
  spectral indices – this mirrors the report's multi-temporal analysis design
  and lets the change-detection algorithm demonstrate itself on real data.

  For real two-date workflows: pass `before_indices` and `after_indices`
  directly to `run_change_detection()`.

Outputs:
  • ChangeResult  dataclass  (change maps + statistics)
  • change_detection_map.tif  (labelled: 0=no change, 1-4=change types)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import rasterio
from scipy.ndimage import (
    gaussian_filter, binary_opening, label as ndimage_label
)

import config



log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ChangeResult:
    """Output of the change detection module."""

    # Binary change mask  (True = changed pixel)
    change_mask: Optional[np.ndarray] = None

    # Labelled change type raster
    #   0 = no change
    #   1 = vegetation loss (abrupt)
    #   2 = urban / built-up increase (gradual)
    #   3 = water-body change (flooding / drying)
    #   4 = anomalous change (statistical outlier)
    change_map: Optional[np.ndarray] = None

    # Per-type masks
    masks: Dict[str, np.ndarray] = field(default_factory=dict)

    # Difference rasters for each index
    delta_indices: Dict[str, np.ndarray] = field(default_factory=dict)

    # Statistics
    stats: Dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Simulation of "before" image
# ──────────────────────────────────────────────────────────────────────────────
def _simulate_before_indices(
        after: Dict[str, np.ndarray],
        rng_seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Synthesise a plausible 'T1' (before) set of spectral indices from the
    observed 'T2' (after) image, to demonstrate two-date change detection
    on a single-date dataset.

    Simulation assumptions (10-year shift):
      • NDVI  +0.08 (historical more vegetation, before urban expansion)
      • NDWI  +0.05 (slightly more water historically)
      • NDBI  -0.06 (less built-up historically)
      • NBR   +0.04 (healthier vegetation historically)
      • MNDWI +0.04 (more surface water)
    Plus spatially localised random perturbations to create realistic patches.
    """
    rng   = np.random.RandomState(rng_seed)
    shift = {"NDVI": 0.08, "NDWI": 0.05, "MNDWI": 0.04,
             "NDBI": -0.06, "NBR": 0.04}

    before = {}
    H, W   = list(after.values())[0].shape

    for idx_name, arr in after.items():
        base  = arr + shift.get(idx_name, 0.0)
        # Add spatially smooth random noise to mimic real inter-date variation
        noise = gaussian_filter(
            rng.normal(0, 0.02, (H, W)).astype(np.float32), sigma=5
        )
        before[idx_name] = np.clip(base + noise, -1.0, 1.0).astype(np.float32)

    return before


# ──────────────────────────────────────────────────────────────────────────────
# Detection helpers
# ──────────────────────────────────────────────────────────────────────────────
def _threshold_mask(delta: np.ndarray,
                    direction: str = "negative",
                    k: float = config.CHANGE_STD_MULTIPLIER) -> np.ndarray:
    """
    Detect pixels where |delta| exceeds mean ± k*std.
    direction:  'negative' (decrease), 'positive' (increase), 'both'
    """
    mu, sigma = delta.mean(), delta.std()
    if direction == "negative":
        return delta < (mu - k * sigma)
    elif direction == "positive":
        return delta > (mu + k * sigma)
    else:
        return np.abs(delta - mu) > k * sigma


def _clean_mask(mask: np.ndarray,
                min_area: int = config.CHANGE_MIN_AREA_PX) -> np.ndarray:
    """Remove small isolated change patches (morphological opening)."""
    opened = binary_opening(mask, iterations=2)
    labelled, n = ndimage_label(opened)
    for region_id in range(1, n + 1):
        if (labelled == region_id).sum() < min_area:
            opened[labelled == region_id] = False
    return opened


def _zscore_anomaly(delta_stack: np.ndarray,
                    threshold: float = 3.0) -> np.ndarray:
    """
    Flag pixels whose Mahalanobis-like z-score across all indices exceeds
    `threshold`.  delta_stack shape: (H, W, n_indices).
    """
    flat  = delta_stack.reshape(-1, delta_stack.shape[-1])
    mu    = flat.mean(axis=0)
    sigma = flat.std(axis=0) + 1e-6
    z     = np.abs((flat - mu) / sigma).mean(axis=1)
    return (z > threshold).reshape(delta_stack.shape[:2])


def _save_change_tif(arr: np.ndarray, path: str, profile: dict) -> None:
    prof = profile.copy()
    prof.update(dtype="uint8", count=1, nodata=255, compress="lzw")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(np.uint8), 1)
    log.info("  Saved → %s", path)


# ──────────────────────────────────────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────────────────────────────────────
def run_change_detection(
        preproc: PreprocResult,
        model_result: ModelResult,
        before_indices: Optional[Dict[str, np.ndarray]] = None,
        after_indices:  Optional[Dict[str, np.ndarray]] = None,
) -> ChangeResult:
    """
    Run full change detection pipeline.

    Parameters
    ----------
    preproc        : output of Module 2
    model_result   : output of Module 4
    before_indices : dict of index arrays at T1 (optional – uses simulation if None)
    after_indices  : dict of index arrays at T2 (optional – uses preproc.indices)
    """
    log.info("═" * 60)
    log.info("MODULE 5 – CHANGE DETECTION & ANALYSIS")
    log.info("═" * 60)

    result = ChangeResult()
    after  = after_indices  or preproc.indices
    before = before_indices or _simulate_before_indices(after)

    log.info("  Using %s 'before' image.",
             "supplied" if before_indices else "simulated (T1 synthetic)")

    # ── 1. Compute difference rasters ─────────────────────────────────────
    log.info("Step 1 │ Computing Δ-index rasters …")
    for name in after:
        if name in before:
            delta = after[name] - before[name]       # T2 - T1
            result.delta_indices[name] = delta
            log.info("  Δ%-8s  mean=%+.4f  std=%.4f",
                     name, delta.mean(), delta.std())

    # ── 2. Abrupt change – vegetation loss (ΔNDVI strongly negative) ──────
    log.info("Step 2 │ Detecting vegetation loss (abrupt change) …")
    delta_ndvi = result.delta_indices.get("NDVI", np.zeros_like(list(after.values())[0]))
    veg_loss   = _threshold_mask(delta_ndvi, "negative", k=1.8)
    veg_loss   = _clean_mask(veg_loss)
    result.masks["vegetation_loss"] = veg_loss
    log.info("  Vegetation-loss pixels: %d (%.2f%%)",
             veg_loss.sum(), 100 * veg_loss.mean())

    # ── 3. Gradual change – urban / built-up increase (ΔNDBI positive) ────
    log.info("Step 3 │ Detecting urban expansion (gradual change) …")
    delta_ndbi = result.delta_indices.get("NDBI", np.zeros_like(delta_ndvi))
    urban_gain = _threshold_mask(delta_ndbi, "positive", k=1.8)
    urban_gain = _clean_mask(urban_gain)
    result.masks["urban_expansion"] = urban_gain
    log.info("  Urban-expansion pixels: %d (%.2f%%)",
             urban_gain.sum(), 100 * urban_gain.mean())

    # ── 4. Water-body change (ΔNDWI or ΔMNDWI) ───────────────────────────
    log.info("Step 4 │ Detecting water-body changes …")
    delta_ndwi = result.delta_indices.get(
        "MNDWI", result.delta_indices.get("NDWI", np.zeros_like(delta_ndvi))
    )
    water_gain = _threshold_mask(delta_ndwi, "positive", k=1.5)
    water_loss = _threshold_mask(delta_ndwi, "negative", k=1.5)
    water_chg  = _clean_mask(water_gain | water_loss)

    # Refine with predicted water map where available
    if model_result.water_map_pred is not None:
        water_pred_binary = (model_result.water_map_pred == 1)
        water_chg = water_chg & ~water_pred_binary  # exclude stable water bodies
    result.masks["water_change"] = water_chg
    log.info("  Water-change pixels: %d (%.2f%%)",
             water_chg.sum(), 100 * water_chg.mean())

    # ── 5. Anomalous change – z-score across all indices ──────────────────
    log.info("Step 5 │ Detecting anomalous spectral changes …")
    delta_stack = np.stack(list(result.delta_indices.values()), axis=-1)
    anomaly     = _zscore_anomaly(delta_stack, threshold=2.5)
    anomaly     = _clean_mask(anomaly)
    result.masks["anomaly"] = anomaly
    log.info("  Anomaly pixels: %d (%.2f%%)",
             anomaly.sum(), 100 * anomaly.mean())

    # ── 6. Composite change map ────────────────────────────────────────────
    log.info("Step 6 │ Building composite change map …")
    H, W = delta_ndvi.shape
    change_map = np.zeros((H, W), dtype=np.uint8)

    # Priority: anomaly > water > urban > vegetation
    change_map[veg_loss]   = 1
    change_map[urban_gain] = 2
    change_map[water_chg]  = 3
    change_map[anomaly]    = 4

    result.change_map  = change_map
    result.change_mask = change_map > 0

    # ── 7. Statistics ──────────────────────────────────────────────────────
    total_px  = H * W
    changed   = int(result.change_mask.sum())
    result.stats = {
        "total_pixels"       : total_px,
        "changed_pixels"     : changed,
        "change_pct"         : round(100.0 * changed / total_px, 2),
        "vegetation_loss_px" : int(veg_loss.sum()),
        "urban_expansion_px" : int(urban_gain.sum()),
        "water_change_px"    : int(water_chg.sum()),
        "anomaly_px"         : int(anomaly.sum()),
        "label_map"          : {
            0: "No Change", 1: "Vegetation Loss",
            2: "Urban Expansion", 3: "Water Change", 4: "Anomaly"
        }
    }

    log.info("  Total change coverage: %.2f%% of scene",
             result.stats["change_pct"])

    # ── 8. Save change map GeoTIFF ─────────────────────────────────────────
    _save_change_tif(change_map, config.OUT["change_map_tif"], preproc.profile)

    log.info("MODULE 5 complete.")
    return result

