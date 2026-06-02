"""
01_data_ingestion.py
=====================
MODULE 1 – Data Ingestion
--------------------------
Loads all geospatial data sources into a unified in-memory structure:
  • Multi-band satellite TIF files  (Haridwar – IIRS/ISRO ResourceSat-2)
  • Reference Water Map TIF          (ISRO binary water mask)
  • water_train.csv                  (Band 2-5 + Water label)
  • vizag_sample_data.csv            (Blue/Green/Red/NIR/SWIR + Landcover label)

Returns a typed dictionary  ``IngestResult``  consumed by every downstream module.

Corresponds to the **Data Ingestion Module** in the architecture block diagram
(Figure 1) of the training report.
"""

from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

import config

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(module)s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data container
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class IngestResult:
    """Holds all raw data loaded during ingestion."""

    # Raster arrays  {band_name: 2D np.ndarray}
    bands: Dict[str, np.ndarray] = field(default_factory=dict)

    # Shared raster metadata (from first band)
    profile: dict = field(default_factory=dict)
    transform: Optional[Affine] = None
    crs: Optional[str] = None

    # Reference water mask (binary 0/1)
    water_mask_ref: Optional[np.ndarray] = None

    # Tabular datasets
    df_water: Optional[pd.DataFrame] = None     # water_train.csv
    df_vizag: Optional[pd.DataFrame] = None     # vizag_sample_data.csv

    # Summary stats collected during ingestion
    stats: Dict[str, dict] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _load_single_band_tif(path: str, name: str) -> tuple[np.ndarray, dict]:
    """Read a single-band GeoTIFF and return (array, rasterio_profile)."""
    if not os.path.exists(path):
        log.error("File not found: %s", path)
        sys.exit(1)
    with rasterio.open(path) as src:
        arr     = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    log.info("  Loaded %-18s │ shape=%s  min=%.0f  max=%.0f",
             name, arr.shape, arr.min(), arr.max())
    return arr, profile


def _band_stats(arr: np.ndarray) -> dict:
    """Return descriptive statistics for a 2D array."""
    valid = arr[np.isfinite(arr) & (arr > 0)]
    return {
        "min"   : float(np.nanmin(arr)),
        "max"   : float(np.nanmax(arr)),
        "mean"  : float(np.nanmean(valid)) if valid.size else 0.0,
        "std"   : float(np.nanstd(valid))  if valid.size else 0.0,
        "nodata_pct": round(100.0 * np.sum(arr <= 0) / arr.size, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main ingestion function
# ──────────────────────────────────────────────────────────────────────────────
def run_ingestion(
    user_tif_paths: Optional[Dict[str, str]] = None,
    user_csv_paths: Optional[Dict[str, str]] = None,
) -> IngestResult:
    """
    Execute the full ingestion pipeline.

    Parameters
    ----------
    user_tif_paths : dict, optional
        Additional TIF files to ingest.  Key = band name, value = file path.
        E.g. {"B6_SWIR2": "/my/data/band6.tif"}
    user_csv_paths : dict, optional
        Additional CSV files.  Key = "water" | "vizag", value = file path.
        Pass to override the default paths from config.py.

    Returns
    -------
    IngestResult
    """
    log.info("═" * 60)
    log.info("MODULE 1 – DATA INGESTION")
    log.info("═" * 60)

    result = IngestResult()

    # ── 1. Satellite bands ───────────────────────────────────────────────────
    log.info("Loading Haridwar satellite bands …")
    band_paths = dict(config.HARIDWAR_BANDS)          # copy defaults
    if user_tif_paths:
        band_paths.update(user_tif_paths)
        log.info("  +%d user-supplied TIF(s) added.", len(user_tif_paths))

    profile_set = False
    for name, path in band_paths.items():
        arr, profile = _load_single_band_tif(path, name)
        result.bands[name] = arr
        result.stats[name] = _band_stats(arr)
        if not profile_set:
            result.profile   = profile
            result.transform = profile.get("transform")
            result.crs       = str(profile.get("crs", "Unknown"))
            profile_set      = True

    log.info("  CRS: %s", result.crs)
    log.info("  Grid size: %d × %d pixels", *list(result.bands.values())[0].shape)

    # ── 2. Reference water mask ──────────────────────────────────────────────
    log.info("Loading reference Water Map …")
    wm_arr, _ = _load_single_band_tif(config.WATER_MAP_TIF, "Water_Map")
    result.water_mask_ref = (wm_arr > 0).astype(np.uint8)
    water_pct = 100.0 * result.water_mask_ref.mean()
    log.info("  Water coverage in reference mask: %.1f %%", water_pct)

    # ── 3. Tabular CSV datasets ──────────────────────────────────────────────
    log.info("Loading tabular datasets …")

    # water_train.csv
    water_csv = (user_csv_paths or {}).get("water", config.WATER_TRAIN_CSV)
    df_w = pd.read_csv(water_csv, index_col=0)
    df_w.columns = [c.strip() for c in df_w.columns]
    result.df_water = df_w
    log.info("  water_train.csv  │ rows=%d  water=%.1f%%",
             len(df_w), 100 * df_w["Water"].mean())

    # vizag_sample_data.csv
    vizag_csv = (user_csv_paths or {}).get("vizag", config.VIZAG_CSV)
    df_v = pd.read_csv(vizag_csv)
    df_v.columns = [c.strip() for c in df_v.columns]
    result.df_vizag = df_v
    class_counts = df_v["Landcover"].value_counts().to_dict()
    log.info("  vizag_sample_data.csv │ rows=%d  classes=%s",
             len(df_v), class_counts)

    log.info("MODULE 1 complete │ bands=%d  tabular_rows=%d",
             len(result.bands), len(df_w) + len(df_v))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = run_ingestion()
    print("\nIngested bands:", list(result.bands.keys()))
    print("df_water shape:", result.df_water.shape)
    print("df_vizag shape:", result.df_vizag.shape)
