# Module API Reference

## Module 1 — Data Ingestion (`01_data_ingestion.py`)

### `run_ingestion(user_tif_paths=None, user_csv_paths=None) → IngestResult`

Loads all data sources into memory.

**Parameters:**
- `user_tif_paths` *(dict, optional)*: `{"band_name": "/path/to.tif"}` — extra TIF bands to add
- `user_csv_paths` *(dict, optional)*: `{"water": "/path.csv", "vizag": "/path.csv"}` — override CSV paths

**Returns:** `IngestResult` dataclass with fields:
| Field | Type | Description |
|-------|------|-------------|
| `bands` | `Dict[str, np.ndarray]` | Raw DN arrays per band |
| `profile` | `dict` | rasterio profile of first band |
| `crs` | `str` | Coordinate reference system |
| `water_mask_ref` | `np.ndarray` | Binary reference water mask |
| `df_water` | `pd.DataFrame` | water_train CSV data |
| `df_vizag` | `pd.DataFrame` | vizag CSV data |
| `stats` | `Dict` | Per-band descriptive statistics |

---

## Module 2 — Pre-processing (`02_preprocessing.py`)

### `run_preprocessing(ingest: IngestResult) → PreprocResult`

**Returns:** `PreprocResult` with fields:
| Field | Type | Description |
|-------|------|-------------|
| `reflectance` | `Dict[str, np.ndarray]` | Normalised [0,1] bands |
| `smoothed` | `Dict[str, np.ndarray]` | Gaussian-filtered bands |
| `indices` | `Dict[str, np.ndarray]` | NDVI, NDWI, MNDWI, NDBI, NBR |
| `cloud_mask` | `np.ndarray` | Boolean cloud mask |
| `valid_mask` | `np.ndarray` | Boolean usable-pixel mask |
| `qa_confidence` | `np.ndarray` | Per-pixel confidence [0–1] |
| `qa_stats` | `dict` | Cloud/shadow/valid percentages |

---

## Module 3 — Bias Mitigation (`03_bias_harmonization.py`)

### `run_harmonisation(ingest, preproc) → HarmonResult`

**Returns:** `HarmonResult` with fields:
| Field | Type | Description |
|-------|------|-------------|
| `X_water_train/test` | `np.ndarray` | Scaled water features |
| `y_water_train/test` | `np.ndarray` | Water labels |
| `scaler_water` | `StandardScaler` | Fitted scaler |
| `X_lc_train/test` | `np.ndarray` | Scaled LC features |
| `y_lc_train/test` | `np.ndarray` | LC labels |
| `raster_features` | `np.ndarray` | Flattened [H×W, C] feature matrix |
| `bias_report` | `dict` | CV, class counts, imbalance flags |

---

## Module 4 — AI Models (`04_ai_models.py`)

### `run_ai_models(ingest, preproc, harmon) → ModelResult`

**Returns:** `ModelResult` with fields:
| Field | Type | Description |
|-------|------|-------------|
| `water_rf_model` | `RandomForestClassifier` | Trained water RF |
| `water_mlp_model` | `MLPClassifier` | Trained water MLP |
| `lc_rf_model` | `RandomForestClassifier` | Trained LC RF |
| `lc_mlp_model` | `MLPClassifier` | Trained LC MLP |
| `metrics` | `Dict[str, dict]` | Accuracy, F1, confusion matrix per model |
| `water_map_pred` | `np.ndarray` | Predicted water raster (2D) |
| `lc_map_pred` | `np.ndarray` | Predicted land-cover raster (2D) |
| `feature_importance_*` | `np.ndarray` | RF feature importances |

---

## Module 5 — Change Detection (`05_change_detection.py`)

### `run_change_detection(preproc, model_result, before_indices=None, after_indices=None) → ChangeResult`

**Parameters:**
- `before_indices` *(dict, optional)*: T1 index arrays. If `None`, synthetic T1 is generated.
- `after_indices` *(dict, optional)*: T2 index arrays. If `None`, uses `preproc.indices`.

**Returns:** `ChangeResult` with fields:
| Field | Type | Description |
|-------|------|-------------|
| `change_map` | `np.ndarray uint8` | 0=none, 1=veg loss, 2=urban, 3=water, 4=anomaly |
| `change_mask` | `np.ndarray bool` | Any-change binary mask |
| `masks` | `Dict[str, np.ndarray]` | Per-type boolean masks |
| `delta_indices` | `Dict[str, np.ndarray]` | T2−T1 per index |
| `stats` | `dict` | Pixel counts per change type, total % |

---

## Module 6 — Visualisation (`06_visualization.py`)

### `run_visualization(ingest, preproc, harmon, model_result, change_result) → None`

Generates all 12 figures to `outputs/figures/`. No return value.

**Figures produced:** `fig01` through `fig12` (see README for descriptions).
