"""
config.py
=========
Central configuration for the Geodata Processing AI Pipeline.
Edit paths and parameters here to adapt to your own dataset.
"""

import os

# ─────────────────────────────────────────────
# ROOT PATHS
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# HARIDWAR SATELLITE BANDS  (IIRS / ISRO data)
# Band mapping: B2=Green, B3=Red, B4=NIR, B5=SWIR
# Source: ResourceSat-2 LISS-III, 04-May-2019
# ─────────────────────────────────────────────
HARIDWAR_BANDS = {
    "B2_Green": os.path.join(DATA_DIR, "haridwarBand2.tif"),
    "B3_Red"  : os.path.join(DATA_DIR, "haridwarBand3.tif"),
    "B4_NIR"  : os.path.join(DATA_DIR, "haridwarBand4.tif"),
    "B5_SWIR" : os.path.join(DATA_DIR, "haridwarBand5.tif"),
}

# Reference water mask produced by ISRO (binary: 1 = water, 0 = non-water)
WATER_MAP_TIF  = os.path.join(DATA_DIR, "Water_Map.tif")

# ─────────────────────────────────────────────
# CSV DATASETS
# ─────────────────────────────────────────────
# water_train.csv  – Band 2-5 reflectance + binary Water label (1 / 0)
WATER_TRAIN_CSV = os.path.join(DATA_DIR, "water_train.csv")

# vizag_sample_data.csv – Blue/Green/Red/NIR/SWIR-1/SWIR-2 + Landcover label
#   Classes: 10=Cropland, 20=Forest, 30=Grassland, 40=Shrubland,
#            50=Urban, 60=Bare land, 80=Water, 90=Wetland
VIZAG_CSV = os.path.join(DATA_DIR, "vizag_sample_data.csv")

# Landcover class map (for labels in plots and reports)
LANDCOVER_CLASSES = {
    10: "Cropland",
    20: "Forest",
    30: "Grassland",
    40: "Shrubland",
    50: "Urban",
    60: "Bare Land",
    80: "Water",
    90: "Wetland",
}

# ─────────────────────────────────────────────
# PREPROCESSING PARAMETERS
# ─────────────────────────────────────────────
# DN → reflectance scale factor (ResourceSat-2 LISS-III)
DN_SCALE_FACTOR  = 1.0 / 1023.0   # 10-bit sensor
GAUSSIAN_SIGMA   = 1.0             # spatial noise reduction kernel width
CLOUD_THRESHOLD  = 0.80            # reflectance above this → probable cloud
NDVI_VEG_THRESH  = 0.3             # NDVI ≥ threshold → vegetation
NDWI_WATER_THRESH= 0.0             # NDWI ≥ 0 → probable water

# ─────────────────────────────────────────────
# AI MODEL PARAMETERS
# ─────────────────────────────────────────────
RANDOM_STATE     = 42
TEST_SIZE        = 0.20            # 80/20 train-test split
N_ESTIMATORS     = 200             # Random Forest trees
MAX_DEPTH        = 15              # RF max depth
MLP_HIDDEN       = (128, 64, 32)   # MLP hidden layer sizes
MLP_MAX_ITER     = 500

# ─────────────────────────────────────────────
# CHANGE DETECTION PARAMETERS
# ─────────────────────────────────────────────
CHANGE_STD_MULTIPLIER = 2.0        # |ΔNDxI| > mean + k*std → change
CHANGE_MIN_AREA_PX    = 50         # ignore change patches smaller than this

# ─────────────────────────────────────────────
# OUTPUT FILE NAMES
# ─────────────────────────────────────────────
OUT = {
    # preprocessed raster (reflectance + indices stacked)
    "preprocessed_tif"  : os.path.join(OUTPUT_DIR, "preprocessed_stack.tif"),
    # QA confidence raster
    "qa_confidence_tif" : os.path.join(OUTPUT_DIR, "qa_confidence.tif"),
    # water classification map (from RF on raster)
    "water_map_pred_tif": os.path.join(OUTPUT_DIR, "water_map_predicted.tif"),
    # land-cover classification map
    "lc_map_tif"        : os.path.join(OUTPUT_DIR, "landcover_map.tif"),
    # change detection binary raster
    "change_map_tif"    : os.path.join(OUTPUT_DIR, "change_detection_map.tif"),
    # saved model files
    "water_model"       : os.path.join(OUTPUT_DIR, "model_water_rf.pkl"),
    "lc_model"          : os.path.join(OUTPUT_DIR, "model_landcover_rf.pkl"),
    # visualisation output folder
    "viz_dir"           : os.path.join(OUTPUT_DIR, "figures"),
    # pipeline summary report
    "report_txt"        : os.path.join(OUTPUT_DIR, "pipeline_report.txt"),
}
os.makedirs(OUT["viz_dir"], exist_ok=True)
