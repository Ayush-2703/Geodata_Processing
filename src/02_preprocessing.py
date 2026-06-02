"""
02_preprocessing.py
====================
MODULE 2 – Pre-processing & Quality Assurance
-----------------------------------------------
Implements the Pre-processing & Quality Assurance sub-system described in
Section 3 (Methods) of the training report:

  1. Radiometric normalisation  – raw DN → [0, 1] reflectance
  2. Noise reduction            – Gaussian spatial filtering
  3. Cloud / shadow masking     – brightness + NDVI heuristic (>92 % accuracy)
  4. Spectral index computation – NDVI, NDWI, MNDWI, NBR
  5. Missing-data interpolation – nearest-neighbour in-painting
  6. Quality Assurance raster   – per-pixel confidence [0, 1]
  7. Save preprocessed stack    – multi-band GeoTIFF for downstream modules

Inputs  : IngestResult  (from Module 1)
Outputs : PreprocResult (passed to Modules 3-5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.ndimage import gaussian_filter, label as ndimage_label
import rasterio
from rasterio.transform import Affine

import config


log = logging.getLogger(__name__)

NODATA = -9999.0


# ──────────────────────────────────────────────────────────────────────────────
# Output container
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PreprocResult:
    """Holds all pre-processed raster arrays and derived indices."""

    # Normalised reflectance bands  {name: 2D array [0,1]}
    reflectance: Dict[str, np.ndarray] = field(default_factory=dict)

    # Smoothed reflectance (after Gaussian filtering)
    smoothed: Dict[str, np.ndarray] = field(default_factory=dict)

    # Derived spectral indices  {index_name: 2D array [-1,1]}
    indices: Dict[str, np.ndarray] = field(default_factory=dict)

    # Boolean masks
    cloud_mask:  Optional[np.ndarray] = None   # True = cloud pixel
    shadow_mask: Optional[np.ndarray] = None   # True = shadow pixel
    valid_mask:  Optional[np.ndarray] = None   # True = usable pixel

    # QA confidence  [0 = unusable … 1 = high confidence]
    qa_confidence: Optional[np.ndarray] = None

    # Raster metadata (pass-through from IngestResult)
    profile:    dict = field(default_factory=dict)
    transform:  Optional[Affine] = None
    crs:        Optional[str] = None

    # QA statistics for reporting
    qa_stats: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Step helpers
# ──────────────────────────────────────────────────────────────────────────────
def _normalise_dn(band: np.ndarray, scale: float) -> np.ndarray:
    """Scale raw DN to [0, 1] reflectance; clamp to [0, 1]."""
    ref = band.astype(np.float32) * scale
    return np.clip(ref, 0.0, 1.0)


def _safe_ratio(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute (a - b) / (a + b) safely."""
    denom = a + b
    denom = np.where(np.abs(denom) < eps, eps, denom)
    return np.clip((a - b) / denom, -1.0, 1.0)


def _detect_clouds(
    green: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    bright_thresh: float = config.CLOUD_THRESHOLD,
) -> np.ndarray:
    """
    Heuristic cloud detection (>92 % accuracy as per report):
      • High brightness in all visible bands
      • Negative or near-zero NDVI (clouds suppress vegetation signal)
    Returns boolean mask (True = cloud).
    """
    brightness   = (green + red) / 2.0
    ndvi         = _safe_ratio(nir, red)
    cloud_mask   = (brightness > bright_thresh) & (ndvi < 0.1)
    return cloud_mask


def _detect_shadows(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """
    Simple shadow detection: very low NIR + low SWIR reflectance.
    Returns boolean mask (True = shadow).
    """
    return (nir < 0.05) & (swir < 0.05)


def _inpaint_missing(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Fill masked pixels using nearest-neighbour interpolation.
    `mask` is True where data is invalid.
    """
    from scipy.ndimage import distance_transform_edt
    if not mask.any():
        return arr.copy()
    filled = arr.copy()
    _, idx = distance_transform_edt(mask, return_indices=True)
    filled[mask] = arr[idx[0][mask], idx[1][mask]]
    return filled


def _compute_qa_confidence(
    cloud_mask: np.ndarray,
    shadow_mask: np.ndarray,
    reflectance_bands: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Compute per-pixel QA confidence score in [0, 1]:
      1.0 = clear pixel, normal reflectance
      0.5 = shadow-affected
      0.0 = cloud-contaminated
    Penalises anomalously high or saturated reflectance values.
    """
    confidence = np.ones(cloud_mask.shape, dtype=np.float32)
    confidence[shadow_mask] -= 0.5
    confidence[cloud_mask]   = 0.0

    # Penalise saturation (reflectance very close to 1)
    for arr in reflectance_bands.values():
        confidence -= 0.05 * (arr > 0.95).astype(np.float32)

    return np.clip(confidence, 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Main preprocessing function
# ──────────────────────────────────────────────────────────────────────────────
def run_preprocessing(ingest: "IngestResult") -> PreprocResult:
    """
    Run the full preprocessing & QA pipeline on an IngestResult.

    Parameters
    ----------
    ingest : IngestResult
        Output of Module 1 (data_ingestion).

    Returns
    -------
    PreprocResult
    """
    log.info("═" * 60)
    log.info("MODULE 2 – PRE-PROCESSING & QUALITY ASSURANCE")
    log.info("═" * 60)

    result = PreprocResult(
        profile   = ingest.profile.copy(),
        transform = ingest.transform,
        crs       = ingest.crs,
    )

    # ── 1. Radiometric normalisation ─────────────────────────────────────────
    log.info("Step 1 │ Radiometric normalisation (DN → reflectance) …")
    for name, arr in ingest.bands.items():
        result.reflectance[name] = _normalise_dn(arr, config.DN_SCALE_FACTOR)
        log.info("  %-18s  range [%.3f, %.3f]",
                 name,
                 result.reflectance[name].min(),
                 result.reflectance[name].max())

    # ── 2. Gaussian noise reduction ──────────────────────────────────────────
    log.info("Step 2 │ Gaussian noise reduction (σ=%.1f) …",
             config.GAUSSIAN_SIGMA)
    for name, ref in result.reflectance.items():
        result.smoothed[name] = gaussian_filter(
            ref, sigma=config.GAUSSIAN_SIGMA
        ).astype(np.float32)

    # Extract named bands (use smoothed for indices)
    green = result.smoothed.get("B2_Green", list(result.smoothed.values())[0])
    red   = result.smoothed.get("B3_Red",   list(result.smoothed.values())[1])
    nir   = result.smoothed.get("B4_NIR",   list(result.smoothed.values())[2])
    swir  = result.smoothed.get("B5_SWIR",  list(result.smoothed.values())[3]) \
            if len(result.smoothed) > 3 else nir

    # ── 3. Cloud / shadow detection ──────────────────────────────────────────
    log.info("Step 3 │ Cloud & shadow detection …")
    result.cloud_mask  = _detect_clouds(green, red, nir)
    result.shadow_mask = _detect_shadows(nir, swir)
    result.valid_mask  = ~(result.cloud_mask | result.shadow_mask)
    cloud_pct  = 100.0 * result.cloud_mask.mean()
    shadow_pct = 100.0 * result.shadow_mask.mean()
    valid_pct  = 100.0 * result.valid_mask.mean()
    log.info("  Cloud: %.1f%%   Shadow: %.1f%%   Valid: %.1f%%",
             cloud_pct, shadow_pct, valid_pct)

    # ── 4. Missing-data interpolation ────────────────────────────────────────
    log.info("Step 4 │ In-painting masked pixels …")
    invalid_mask = ~result.valid_mask
    if invalid_mask.any():
        for name in result.smoothed:
            result.smoothed[name] = _inpaint_missing(
                result.smoothed[name], invalid_mask
            )
        log.info("  In-painted %.0f pixels.", invalid_mask.sum())
    else:
        log.info("  No pixels needed in-painting.")

    # Re-extract after inpainting
    green = result.smoothed.get("B2_Green", list(result.smoothed.values())[0])
    red   = result.smoothed.get("B3_Red",   list(result.smoothed.values())[1])
    nir   = result.smoothed.get("B4_NIR",   list(result.smoothed.values())[2])
    swir  = result.smoothed.get("B5_SWIR",  list(result.smoothed.values())[3]) \
            if len(result.smoothed) > 3 else nir

    # ── 5. Spectral index computation ────────────────────────────────────────
    log.info("Step 5 │ Computing spectral indices …")

    result.indices["NDVI"]  = _safe_ratio(nir,   red)    # Vegetation
    result.indices["NDWI"]  = _safe_ratio(green, nir)    # Water (Gao)
    result.indices["MNDWI"] = _safe_ratio(green, swir)   # Modified water
    result.indices["NDBI"]  = _safe_ratio(swir,  nir)    # Built-up index
    result.indices["NBR"]   = _safe_ratio(nir,   swir)   # Burn ratio / soil

    for idx_name, idx_arr in result.indices.items():
        log.info("  %-8s  mean=%+.3f  std=%.3f",
                 idx_name, idx_arr.mean(), idx_arr.std())

    # ── 6. QA confidence raster ──────────────────────────────────────────────
    log.info("Step 6 │ Computing QA confidence raster …")
    result.qa_confidence = _compute_qa_confidence(
        result.cloud_mask, result.shadow_mask, result.reflectance
    )
    result.qa_stats = {
        "cloud_pct" : round(cloud_pct,  2),
        "shadow_pct": round(shadow_pct, 2),
        "valid_pct" : round(valid_pct,  2),
        "mean_qa"   : round(float(result.qa_confidence.mean()), 3),
    }
    log.info("  Mean QA confidence: %.3f", result.qa_stats["mean_qa"])

    # ── 7. Save preprocessed GeoTIFF stack ───────────────────────────────────
    log.info("Step 7 │ Saving preprocessed stack → %s …",
             config.OUT["preprocessed_tif"])
    _save_multiband_tif(result)

    log.info("MODULE 2 complete │ indices=%s", list(result.indices.keys()))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# GeoTIFF writer
# ──────────────────────────────────────────────────────────────────────────────
def _save_multiband_tif(preproc: PreprocResult) -> None:
    """Write smoothed reflectance + indices to a stacked GeoTIFF."""
    layers = list(preproc.smoothed.values()) + list(preproc.indices.values())
    names  = list(preproc.smoothed.keys())  + list(preproc.indices.keys())

    profile = preproc.profile.copy()
    profile.update(
        driver  = "GTiff",
        dtype   = "float32",
        count   = len(layers),
        nodata  = NODATA,
        compress= "lzw",
    )

    with rasterio.open(config.OUT["preprocessed_tif"], "w", **profile) as dst:
        for i, (arr, name) in enumerate(zip(layers, names), start=1):
            dst.write(arr.astype(np.float32), i)
            dst.update_tags(i, name=name)

    # Save QA confidence separately
    qa_profile = preproc.profile.copy()
    qa_profile.update(dtype="float32", count=1, nodata=NODATA)
    with rasterio.open(config.OUT["qa_confidence_tif"], "w", **qa_profile) as dst:
        dst.write(preproc.qa_confidence.astype(np.float32), 1)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────
