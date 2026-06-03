# Changelog

All notable changes to this project will be documented in this file.


### Added
- **Module 1** — Data Ingestion: multi-band GeoTIFF + CSV loader with user-extensible paths
- **Module 2** — Pre-processing & QA: radiometric normalisation, Gaussian filtering, cloud/shadow detection, NDVI/NDWI/MNDWI/NDBI/NBR indices, QA confidence raster
- **Module 3** — Bias Mitigation & Harmonisation: CV-based bias analysis, oversampling, StandardScaler, raster feature matrix
- **Module 4** — AI Models: Random Forest (200 trees) + MLP (128-64-32) for water and land-cover classification, raster pixel-wise inference, feature importance
- **Module 5** — Change Detection: spectral index differencing, four change type detection (vegetation loss, urban expansion, water change, anomaly), morphological cleaning
- **Module 6** — Visualisation: 12 publication-quality PNG figures + pipeline dashboard
- `run_pipeline.py` — CLI runner with `--tif`, `--csv`, `--before-tif`, `--no-viz` flags
- GitHub Actions CI workflow
- Full test suite (4 modules)
- Python notebook walkthrough
- Comprehensive documentation

### Performance
- Water body classification: Accuracy = 1.0000, F1-macro = 1.0000
- Land-cover classification: Accuracy = 0.6694 (MLP), F1-macro = 0.6681
- Change detection: 78,825 changed pixels (5.95% of 1151×1151 scene)
- Pipeline runtime: ~50 seconds on standard hardware
