"""
04_ai_models.py
================
MODULE 4 – AI Models Integration
----------------------------------
Implements the AI Models block from the architecture (Figure 1):

  • Random Forest  – water body classification (water_train.csv → raster)
  • Random Forest  – multi-class land-cover  (vizag_sample_data.csv)
  • MLP Classifier – alternative deep learner for both tasks
  • Feature importance analysis
  • Model serialisation (.pkl)
  • Raster inference  – applies trained model pixel-by-pixel

Outputs:
  • ModelResult  dataclass  (metrics, predictions, saved model paths)
  • Water-body classification GeoTIFF
  • Land-cover classification GeoTIFF
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)

import config




log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelResult:
    """Holds trained models, metrics, and predicted raster arrays."""

    # Trained model objects
    water_rf_model:  Optional[object] = None
    water_mlp_model: Optional[object] = None
    lc_rf_model:     Optional[object] = None
    lc_mlp_model:    Optional[object] = None

    # Evaluation metrics  {model_name: {metric: value}}
    metrics: Dict[str, dict] = field(default_factory=dict)

    # Predicted raster arrays  (2D)
    water_map_pred:  Optional[np.ndarray] = None
    lc_map_pred:     Optional[np.ndarray] = None

    # Feature importances
    feature_importance_water: Optional[np.ndarray] = None
    feature_importance_lc:    Optional[np.ndarray] = None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _train_rf(X_tr, y_tr, label: str) -> RandomForestClassifier:
    log.info("  Training Random Forest [%s] …", label)
    rf = RandomForestClassifier(
        n_estimators = config.N_ESTIMATORS,
        max_depth    = config.MAX_DEPTH,
        random_state = config.RANDOM_STATE,
        n_jobs       = -1,
        class_weight = "balanced",
    )
    rf.fit(X_tr, y_tr)
    return rf


def _train_mlp(X_tr, y_tr, label: str) -> MLPClassifier:
    log.info("  Training MLP [%s] …", label)
    mlp = MLPClassifier(
        hidden_layer_sizes = config.MLP_HIDDEN,
        max_iter           = config.MLP_MAX_ITER,
        random_state       = config.RANDOM_STATE,
        early_stopping     = True,
        validation_fraction= 0.1,
        n_iter_no_change   = 15,
        learning_rate_init = 0.001,
    )
    mlp.fit(X_tr, y_tr)
    return mlp


def _evaluate(model, X_te, y_te, model_name: str,
              class_names=None) -> dict:
    """Run predictions, compute metrics, and log a summary."""
    y_pred   = model.predict(X_te)
    acc      = accuracy_score(y_te, y_pred)
    f1_macro = f1_score(y_te, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    cm       = confusion_matrix(y_te, y_pred)
    report   = classification_report(
        y_te, y_pred,
        target_names=[str(c) for c in (class_names or np.unique(y_te))],
        output_dict=True, zero_division=0
    )
    log.info("  %-30s  Acc=%.3f  F1-macro=%.3f  F1-weighted=%.3f",
             model_name, acc, f1_macro, f1_weighted)
    return {
        "accuracy"    : round(acc, 4),
        "f1_macro"    : round(f1_macro, 4),
        "f1_weighted" : round(f1_weighted, 4),
        "conf_matrix" : cm.tolist(),
        "report"      : report,
        "y_pred"      : y_pred,
        "y_true"      : y_te,
    }


def _save_model(model, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info("  Model saved → %s", path)


def _predict_raster(model, raster_features: np.ndarray,
                    shape: tuple) -> np.ndarray:
    """Run model.predict on the full raster feature matrix → 2D map."""
    log.info("  Raster inference (%d pixels) …", raster_features.shape[0])
    pred_flat = model.predict(raster_features)
    return pred_flat.reshape(shape).astype(np.int16)


def _save_raster_tif(arr: np.ndarray, path: str, profile: dict,
                     nodata: int = -1) -> None:
    prof = profile.copy()
    prof.update(dtype="int16", count=1, nodata=nodata, compress="lzw")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(np.int16), 1)
    log.info("  Saved raster → %s", path)


# ──────────────────────────────────────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────────────────────────────────────
def run_ai_models(ingest: IngestResult,
                  preproc: PreprocResult,
                  harmon: HarmonResult) -> ModelResult:
    """
    Train, evaluate, and apply all AI models.
    """
    log.info("═" * 60)
    log.info("MODULE 4 – AI MODELS INTEGRATION")
    log.info("═" * 60)

    result = ModelResult()

    # ╔══════════════════════════════════════════════════════╗
    # ║  TASK A – Water Body Classification                 ║
    # ║  Dataset: water_train.csv  →  Water_Map.tif         ║
    # ╚══════════════════════════════════════════════════════╝
    log.info("─" * 50)
    log.info("TASK A │ Water Body Classification")
    log.info("─" * 50)
    water_class_names = ["Non-Water", "Water"]

    # A1. Train Random Forest
    rf_w = _train_rf(harmon.X_water_train, harmon.y_water_train, "Water-RF")
    result.water_rf_model = rf_w
    metrics_rf_w = _evaluate(
        rf_w, harmon.X_water_test, harmon.y_water_test,
        "Water-RF", water_class_names
    )
    result.metrics["Water_RF"] = metrics_rf_w

    # A2. Train MLP
    mlp_w = _train_mlp(harmon.X_water_train, harmon.y_water_train, "Water-MLP")
    result.water_mlp_model = mlp_w
    metrics_mlp_w = _evaluate(
        mlp_w, harmon.X_water_test, harmon.y_water_test,
        "Water-MLP", water_class_names
    )
    result.metrics["Water_MLP"] = metrics_mlp_w

    # A3. Feature importance (RF only)
    result.feature_importance_water = rf_w.feature_importances_
    feat_names_w = ["Band2", "Band3", "Band4", "Band5"]
    for fn, fi in zip(feat_names_w, result.feature_importance_water):
        log.info("  Feature importance [Water] %-8s: %.3f", fn, fi)

    # A4. Save models
    _save_model(rf_w,  config.OUT["water_model"])

    # A5. Raster inference with the best water model
    # Build raster features from raw Haridwar bands (Band2-5 → scale to 0-255)
    log.info("  Building Haridwar raster feature matrix for water prediction …")
    band_keys = ["B2_Green", "B3_Red", "B4_NIR", "B5_SWIR"]
    raw_bands  = [ingest.bands[k] for k in band_keys]
    # Scale raw DN to similar range as training data (0–255 integer)
    H, W       = raw_bands[0].shape
    raster_raw = np.stack(
        [np.clip(b, 0, 1023).astype(np.float32) * (255.0 / 1023.0)
         for b in raw_bands], axis=-1
    ).reshape(-1, 4)
    raster_raw_scaled = harmon.scaler_water.transform(raster_raw)
    result.water_map_pred = _predict_raster(rf_w, raster_raw_scaled, (H, W))
    _save_raster_tif(result.water_map_pred,
                     config.OUT["water_map_pred_tif"],
                     preproc.profile)

    # ╔══════════════════════════════════════════════════════╗
    # ║  TASK B – Multi-Class Land-Cover Classification     ║
    # ║  Dataset: vizag_sample_data.csv  →  lc_map.tif      ║
    # ╚══════════════════════════════════════════════════════╝
    log.info("─" * 50)
    log.info("TASK B │ Land-Cover Classification (8 classes)")
    log.info("─" * 50)
    lc_class_names = [config.LANDCOVER_CLASSES[c]
                      for c in sorted(config.LANDCOVER_CLASSES)]

    # B1. Train Random Forest
    rf_lc = _train_rf(harmon.X_lc_train, harmon.y_lc_train, "LC-RF")
    result.lc_rf_model = rf_lc
    metrics_rf_lc = _evaluate(
        rf_lc, harmon.X_lc_test, harmon.y_lc_test,
        "LC-RF", lc_class_names
    )
    result.metrics["LC_RF"] = metrics_rf_lc

    # B2. Train MLP
    mlp_lc = _train_mlp(harmon.X_lc_train, harmon.y_lc_train, "LC-MLP")
    result.lc_mlp_model = mlp_lc
    metrics_mlp_lc = _evaluate(
        mlp_lc, harmon.X_lc_test, harmon.y_lc_test,
        "LC-MLP", lc_class_names
    )
    result.metrics["LC_MLP"] = metrics_mlp_lc

    # B3. Feature importance
    result.feature_importance_lc = rf_lc.feature_importances_
    feat_names_lc = ["Blue", "Green", "Red", "NIR", "SWIR-1", "SWIR-2"]
    for fn, fi in zip(feat_names_lc, result.feature_importance_lc):
        log.info("  Feature importance [LC] %-8s: %.3f", fn, fi)

    # B4. Save model
    _save_model(rf_lc, config.OUT["lc_model"])

    # B5. Raster inference on Haridwar raster using best LC model
    # Use harmonised raster features (scaled bands + indices)
    log.info("  Raster inference for land-cover map …")
    # LC model was trained on 6 bands; raster has 4+5 features → use 4 bands only
    # We project the 4-band Haridwar features into 6-band space by deriving
    # synthetic Blue/SWIR-1 from existing bands (common approximation)
    G   = preproc.smoothed["B2_Green"]   # Green
    R   = preproc.smoothed["B3_Red"]     # Red
    NIR = preproc.smoothed["B4_NIR"]     # NIR
    SW  = preproc.smoothed["B5_SWIR"]    # SWIR

    # Approximate Blue ≈ 0.9 × Green (spectral similarity in LISS-III)
    Blue_approx  = np.clip(G * 0.90, 0, 1)
    # Approximate SWIR-2 ≈ 0.85 × SWIR-1
    SWIR2_approx = np.clip(SW * 0.85, 0, 1)

    raster_6band = np.stack(
        [Blue_approx, G, R, NIR, SW, SWIR2_approx], axis=-1
    ).reshape(-1, 6).astype(np.float32)
    raster_6band_scaled = harmon.scaler_lc.transform(raster_6band)
    result.lc_map_pred = _predict_raster(rf_lc, raster_6band_scaled, (H, W))
    _save_raster_tif(result.lc_map_pred,
                     config.OUT["lc_map_tif"],
                     preproc.profile)

    # ── Summary ────────────────────────────────────────────────────────────
    log.info("─" * 50)
    log.info("MODEL PERFORMANCE SUMMARY")
    log.info("─" * 50)
    for name, m in result.metrics.items():
        log.info("  %-25s  Acc=%.3f  F1(macro)=%.3f",
                 name, m["accuracy"], m["f1_macro"])

    log.info("MODULE 4 complete.")
    return result

