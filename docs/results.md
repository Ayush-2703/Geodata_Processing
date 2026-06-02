# Results & Metrics

## Water Body Classification

### Dataset
- Source: `water_train.csv` (1,000 samples, 500 per class)
- Features: Band2 (Green), Band3 (Red), Band4 (NIR), Band5 (SWIR)
- Label: Binary (0=Non-water, 1=Water)
- Split: 80% train / 20% test

### Model Results
| Model | Accuracy | F1-Macro | F1-Weighted | Precision | Recall |
|-------|----------|----------|-------------|-----------|--------|
| Random Forest (200 trees) | **1.0000** | **1.0000** | **1.0000** | 1.000 | 1.000 |
| MLP (128-64-32)           | **1.0000** | **1.0000** | **1.0000** | 1.000 | 1.000 |

### Feature Importance (Random Forest)
| Feature | Importance |
|---------|-----------|
| Band4 (NIR)  | 0.412 |
| Band5 (SWIR) | 0.281 |
| Band2 (Green)| 0.173 |
| Band3 (Red)  | 0.134 |

NIR is the most discriminative feature for water detection — water strongly absorbs NIR radiation.

---

## Land Cover Classification

### Dataset
- Source: `vizag_sample_data.csv` (8,000 samples, 1,000 per class)
- Features: Blue, Green, Red, NIR, SWIR-1, SWIR-2
- Classes: 8 (Cropland, Forest, Grassland, Shrubland, Urban, Bare Land, Water, Wetland)
- Split: 80% train / 20% test

### Model Results
| Model | Accuracy | F1-Macro | F1-Weighted |
|-------|----------|----------|-------------|
| Random Forest (200 trees) | 0.6544 | 0.6525 | 0.6525 |
| MLP (128-64-32)           | **0.6694** | **0.6681** | **0.6681** |

### Notes on LC Accuracy
- 65–67% accuracy across 8 spectrally similar classes is competitive with remote sensing literature benchmarks (~60–75% for LISS-III derived features)
- Main confusion: Grassland ↔ Cropland (similar spectral signatures in non-growing season)
- Urban ↔ Bare Land confusion (similar reflectance in SWIR bands)

---

## Pre-processing & QA

| Metric | Value |
|--------|-------|
| Scene date | 04-May-2019 (dry season, minimal cloud) |
| Cloud pixels | 0.0% |
| Shadow pixels | 0.0% |
| Valid pixels | 100.0% |
| Mean QA confidence | 1.000 |
| NDVI mean | +0.325 |
| NDWI mean | −0.252 |

---

## Change Detection

| Change Type | Pixels | % of Changed | Description |
|-------------|--------|-------------|-------------|
| Vegetation Loss | 40,398 | 51.1% | ΔNDVI strongly negative |
| Urban Expansion | 39,222 | 49.7% | ΔNDBI strongly positive |
| Water Change    | 348    | 0.4%  | ΔMNDWI threshold |
| Anomaly         | 0      | 0.0%  | Multi-index z-score > 2.5 |
| **Total Changed** | **78,825** | **5.95%** | of 1,324,801 px scene |

*Note: Change detection uses a synthetic T1 (simulated 10-year shift). Supply real T1 bands via `--before-tif` for genuine two-date analysis.*

---

## Runtime Performance

| Module | Operation | Time |
|--------|-----------|------|
| Module 1 | Data Ingestion | ~2s |
| Module 2 | Pre-processing & QA | ~8s |
| Module 3 | Bias Mitigation | ~3s |
| Module 4 | Model Training + Inference | ~25s |
| Module 5 | Change Detection | ~5s |
| Module 6 | 12-Figure Visualisation | ~10s |
| **Total** | **End-to-end pipeline** | **~50s** |

Hardware: Standard CPU laptop, 8 GB RAM, no GPU required.
