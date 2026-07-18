<div align="center">
 
![🛰️ Geodata Processing Using Artificial Intelligence](https://capsule-render.vercel.app/api?type=waving&color=0:022C22,100:059669&height=250&section=header&text=🛰️%20Geodata%20Processing%20Using%20AI&fontSize=55&fontColor=FDE68A&fontAlignY=36&animation=fadeIn&desc=End-to-end%20geospatial%20AI%20pipeline—from%20raw%20satellite%20bands%20to%20land-cover%20maps,%20water%20body%20detection%20and%20change%20analysis&descSize=15&descAlignY=58)

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-32%20passing-22C55E?style=flat-square&logo=pytest)](tests/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?style=flat-square&logo=github-actions&logoColor=white)](.github/workflows/ci.yml)
[![ISRO](https://img.shields.io/badge/Data-ISRO%20ResourceSat--2-E53935?style=flat-square)](data/README_data.md)

<br/>

**Ayush Kumar Singh** 

</div>

---

## 📌 What This Project Does

This pipeline processes **multi-spectral satellite imagery** from ISRO's ResourceSat-2 LISS-III sensor and classifies every pixel into land-cover types, detects water bodies, and flags areas of change between two dates — all using AI.

It ships in **two versions** that use different model architectures, explained in full detail below.

<br/>

<div align="center">

| | Version 1 · `run_pipeline.py` | Version 2 · `run_pipeline_dl.py` |
|:---|:---:|:---:|
| **Nickname** | Baseline | Paper-Compliant |
| **Models** | Random Forest · MLP | CNN · LSTM · Transformer · Ensemble |
| **Framework** | scikit-learn | PyTorch 2.x |
| **Water Accuracy** | **1.0000** | **1.0000** |
| **Land-Cover Accuracy** | **0.6694** | **0.6681** |
| **Training Time** | ~30 sec | ~5–10 min |
| **GPU Required** | ❌ | ❌ (CPU mode) |
| **Matches paper architecture** | Partial | ✅ Full |

</div>

---

## 🗂️ Repository Structure

```
geodata-processing-ai/
│
├── 🚀 run_pipeline.py            ← Version 1 entry point  (RF + MLP)
├── 🧠 run_pipeline_dl.py         ← Version 2 entry point  (CNN + LSTM + Transformer)
│
├── src/
│   ├── config.py                 ← All paths, thresholds, hyperparameters
│   ├── module_runner.py          ← Module loader utility
│   ├── 01_data_ingestion.py      ← Load satellite TIFs + CSV datasets
│   ├── 02_preprocessing.py       ← Radiometric correction · indices · QA
│   ├── 03_bias_harmonization.py  ← Bias analysis · oversampling · scaling
│   ├── 04_ai_models.py           ← Version 1: Random Forest + MLP
│   ├── 04_ai_models_dl.py        ← Version 2: CNN + LSTM + Transformer
│   ├── 05_change_detection.py    ← Multi-temporal change detection
│   ├── 06_visualization.py       ← 12 publication-quality figures
│   └── dl_models.py              ← PyTorch model definitions
│
├── data/
│   ├── sample/
│   │   ├── water_train.csv       ← 1,000 rows · Band2–5 + Water label
│   │   └── vizag_sample_data.csv ← 8,000 rows · 6 bands + 8-class Landcover
│   └── README_data.md            ← How to download the TIF files
│
├── tests/                        ← 32 unit tests (pytest)
├── notebooks/
│   └── pipeline_walkthrough.ipynb
├── docs/
│   ├── architecture.md
│   ├── modules.md
│   └── results.md
├── assets/                       ← Output figure previews
├── MODEL_JUSTIFICATION.txt       ← Full explanation of why two versions exist
├── requirements.txt
├── setup.py
├── CHANGELOG.md
└── LICENSE
```

---

## ⚡ Quick Start

### 1 · Clone & Install

```bash
git clone https://github.com/Ayush-2703/Geodata_Processing.git
cd Geodata_Processing
pip install -r requirements.txt
```

> Version 2 needs PyTorch:
> ```bash
> pip install torch
> ```

### 2 · Download Satellite Data

```bash
# Haridwar TIF bands
git clone https://github.com/iirs-isro/IIRS_ISRO_Geoprocessing-using-Python.git
cp IIRS_ISRO_Geoprocessing-using-Python/haridwar/haridwarBand*.tif data/

# Water Map reference
git clone https://github.com/iirs-isro/ISRO-Geodata-Processing-using-Python-and-Machine-Learning.git
cp ISRO-Geodata-Processing-using-Python-and-Machine-Learning/Water_Map.tif data/
```

### 3 · Run

```bash
# Version 1 — RF + MLP  (~50 seconds, any laptop)
python run_pipeline.py

# Version 2 — CNN + LSTM + Transformer  (~10 min, CPU)
python run_pipeline_dl.py
```

---

## 🏗️ Pipeline Architecture

Both versions share the same 6-module structure. Only **Module 4** differs.

```
  [ Satellite TIFs ]  [ Water Map ]  [ water_train.csv ]  [ vizag.csv ]
          └──────────────────┴────────────────┴──────────────────┘
                                      │
                      ┌───────────────▼────────────────┐
                      │   MODULE 1 · Data Ingestion    │
                      │   Raster loader · CSV loader   │
                      └───────────────┬────────────────┘
                                      │
                      ┌───────────────▼───────────────────────┐
                      │   MODULE 2 · Pre-processing & QA      │
                      │   DN→Reflectance · Gaussian filter    │
                      │   Cloud/shadow detection (>92% acc)   │
                      │   NDVI · NDWI · MNDWI · NDBI · NBR    │
                      │   QA confidence raster [0–1]          │
                      └───────────────┬───────────────────────┘
                                      │
                      ┌───────────────▼───────────────────────┐
                      │   MODULE 3 · Bias Mitigation          │
                      │   CV bias check · Oversampling        │
                      │   StandardScaler · Feature matrix     │
                      └───────────────┬───────────────────────┘
                                      │
               ┌──────────────────────┴──────────────────────┐
               │                                             │
  ┌────────────▼──────────────┐          ┌───────────────────▼──────────────────┐
  │   MODULE 4 · Version 1    │          │      MODULE 4 · Version 2            │
  │                           │          │                                      │
  │  RandomForestClassifier   │          │  SpectralCNN   (U-Net encoder-dec)   │
  │    n_estimators = 200     │          │  TemporalLSTM  (BiLSTM + GRU)        │
  │    max_depth    = 15      │          │  SpectralTransformer (MHA, Pre-LN)   │
  │                           │          │  EnsembleFusion (learnable weights)  │
  │  MLPClassifier            │          │  GradientSaliency (XAI)              │
  │    layers: 128→64→32      │          │                                      │
  │    early_stopping = True  │          │  PyTorch 2.x · ~10 min · CPU only    │
  │                           │          └───────────────────┬──────────────────┘
  │  ~30 sec · no GPU         │                              │
  └────────────┬──────────────┘                              │
               └──────────────────────┬──────────────────────┘
                                      │
                      ┌───────────────▼───────────────────────┐
                      │   MODULE 5 · Change Detection         │
                      │   Δ-Index differencing (T2 − T1)      │
                      │   Vegetation loss · Urban expansion   │
                      │   Water change · Anomaly detection    │
                      │   Morphological cleaning              │
                      └───────────────┬───────────────────────┘
                                      │
                      ┌───────────────▼───────────────────────┐
                      │   MODULE 6 · Visualisation            │
                      │   12 publication-quality figures      │
                      │   GeoTIFF prediction outputs          │
                      └───────────────────────────────────────┘
```

---

## 🧠 The Two Versions — Full Explanation

### Version 1 · Random Forest + MLP

Fast, reliable, and highly accurate on the available ISRO tabular datasets. The right tool for the available data format.

```python
# Water body classification
RandomForestClassifier(n_estimators=200, max_depth=15, class_weight="balanced")

# Land-cover (8-class)
MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, early_stopping=True)
```

**Why this version exists:**
The ISRO training CSVs are **tabular** — one row = one pixel's band values. CNNs require 2D spatial patches; LSTMs require time-series with multiple dates. RF and MLP are the scientifically correct choice for this data format and run in ~50 seconds on any laptop.

> Full technical reasoning → [`MODEL_JUSTIFICATION.txt`](MODEL_JUSTIFICATION.txt)

---

### Version 2 · CNN + LSTM + Transformer

Implements the **exact architecture** stated in the research paper using PyTorch 2.x.

---

#### 🔷 SpectralCNN
> *Paper: "U-Net and ResNet architectures to extract spatial features at various scales"*

```
Input  (B, 4, H, W)
   Encoder:  [Conv→BN→ReLU] × 3  +  MaxPool
                    ↕  skip connections
   Decoder:  [ConvTranspose] × 3  +  skip concat
Output (B, num_classes, H, W)
```

---

#### 🔷 TemporalLSTM
> *Paper: "LSTM and GRU variants to capture temporal dynamics in time-series data"*

```
Input  (B, T, features)
   BiLSTM × 2 layers   hidden=128   dropout=0.3
   GRU refinement
   LayerNorm → Linear(128→64→classes)
Output (B, num_classes)
```

---

#### 🔷 SpectralTransformer
> *Paper: "Self-attention mechanisms facilitate integration of contextual information across spatial and temporal dimensions"*

```
Input  (B, seq_len, 1)   ← each band = one token
   Linear projection   → d_model=64
   Positional encoding (sinusoidal)
   TransformerEncoder × 2   nhead=4   FFN=256   GELU   Pre-LN
   Global average pool
   Linear → num_classes
Output (B, num_classes)
```

---

#### 🔷 EnsembleFusion
> *Paper: "multi-modal fusion framework that achieves both computational efficiency and analytical strength"*

```
CNN probs  ──┐
LSTM probs ──┼──►  Learnable weighted softmax  ──►  Final prediction
TRF probs  ──┘     (weights optimised via AdamW on training set)
```

---

#### 🔷 GradientSaliency (XAI)
> *Paper: "explainable AI components mitigate the black box issues linked with deep learning"*

```python
saliency = GradientSaliency(model).compute(x, target_class=1)
# Returns same shape as input — high values = most important features
```

---

## 📊 Results

### Water Body Classification

| Model | Version | Accuracy | F1-Macro | F1-Weighted |
|-------|---------|:--------:|:--------:|:-----------:|
| Random Forest | v1 | **1.0000** | **1.0000** | **1.0000** |
| MLP | v1 | **1.0000** | **1.0000** | **1.0000** |
| CNN | v2 | **1.0000** | **1.0000** | **1.0000** |
| LSTM | v2 | **1.0000** | **1.0000** | **1.0000** |
| Transformer | v2 | **1.0000** | **1.0000** | **1.0000** |
| **Ensemble** | v2 | **1.0000** | **1.0000** | **1.0000** |

### Land-Cover Classification (8 classes)

| Model | Version | Accuracy | F1-Macro |
|-------|---------|:--------:|:--------:|
| Random Forest | v1 | 0.6544 | 0.6525 |
| MLP | v1 | 0.6694 | 0.6681 |
| CNN | v2 | **0.6681** | **0.6670** |
| LSTM | v2 | *in training* | — |
| Transformer | v2 | *in training* | — |

> Classes: `Cropland · Forest · Grassland · Shrubland · Urban · Bare Land · Water · Wetland`

### Change Detection

| Metric | Value |
|--------|-------|
| Scene | 1,151 × 1,151 px · EPSG:4326 |
| **Total changed** | **78,825 px (5.95%)** |
| Vegetation loss | 40,398 px · 51.1% |
| Urban expansion | 39,222 px · 49.7% |
| Water change | 348 px · 0.4% |
| QA confidence | 1.000 (no cloud) |

### Runtime

| | Training | Full pipeline |
|--|:--:|:--:|
| **v1** (RF + MLP) | ~30 sec | **~50 sec** |
| **v2** (DL models) | ~5–10 min | **~12 min** |

---

## 🗃️ Datasets

| File | Type | Rows / Size | Description |
|------|------|-------------|-------------|
| `haridwarBand2–5.tif` | GeoTIFF | 1151×1151 px | ResourceSat-2 LISS-III · 04-May-2019 |
| `Water_Map.tif` | GeoTIFF | 1151×1151 px | Binary reference water mask |
| `water_train.csv` | CSV | 1,000 rows | Band2–5 + Water label (50/50) |
| `vizag_sample_data.csv` | CSV | 8,000 rows | 6 bands + 8-class Landcover |

---

## 🔬 Spectral Indices

| Index | Formula | Application |
|-------|---------|-------------|
| **NDVI** | (NIR−Red)/(NIR+Red) | Vegetation density |
| **NDWI** | (Green−NIR)/(Green+NIR) | Surface water |
| **MNDWI** | (Green−SWIR)/(Green+SWIR) | Modified water |
| **NDBI** | (SWIR−NIR)/(SWIR+NIR) | Built-up / urban |
| **NBR** | (NIR−SWIR)/(NIR+SWIR) | Burn / bare soil |

---

## 🖼️ Output Figures

<p align="center">
  <img src="https://github.com/user-attachments/assets/fb59d494-47b6-40c4-92a9-b6622b15f794" width="270"/>
  <img src="https://github.com/user-attachments/assets/bdb3bde1-9e38-4e5c-97f9-5f7750134e97" width="270"/>
  <img src="https://github.com/user-attachments/assets/d7c4255b-379f-4510-85d6-d4317198c8b1" width="270"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/8897222f-6c0e-4018-bcae-35077e5ac9cc" width="440"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/cf772a5b-434b-49b6-b5c1-557de32ecb33" width="860"/>
</p>

---

## 🛠️ CLI Reference

```bash
# Both runners accept the same flags:

--tif NAME=PATH          Add extra satellite band
--csv water=PATH         Override water training CSV
--csv vizag=PATH         Override land-cover CSV
--before-tif NAME=PATH   Real T1 bands for two-date change detection
--no-viz                 Skip figure generation
```

**Examples:**

```bash
# Fastest possible run
python run_pipeline.py --no-viz

# Real two-date change detection
python run_pipeline_dl.py \
  --before-tif B2_Green=/T1/b2.tif B3_Red=/T1/b3.tif \
               B4_NIR=/T1/b4.tif  B5_SWIR=/T1/b5.tif

# Use your own data
python run_pipeline.py \
  --csv water=/my/water.csv \
  --tif B2_Green=/my/b2.tif B3_Red=/my/b3.tif
```

---

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

| File | Tests | Covers |
|------|:-----:|--------|
| `test_ingestion.py` | 6 | CSV loading, TIF reading, label validation |
| `test_preprocessing.py` | 9 | Index ranges, cloud detection, normalisation |
| `test_models.py` | 10 | RF accuracy, MLP convergence, feature importance |
| `test_change_detection.py` | 7 | Delta computation, thresholds, anomaly detection |

**32 / 32 passing ✅**

---

## 📦 Requirements

```
# Both versions
rasterio>=1.3.0   scikit-learn>=1.2.0   pandas>=1.5.0
numpy>=1.23.0     matplotlib>=3.6.0     scipy>=1.9.0

# Version 2 only
torch>=2.0.0
```

---

## 🙏 Acknowledgements

- **Ms. Garima Srivastava** — guidance and supervision throughout
- **IIRS / ISRO** — Haridwar satellite imagery and online course datasets
- Open-source community — scikit-learn, PyTorch, rasterio, NumPy, Matplotlib

---
## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.  
You're free to use, fork, and build on this for personal and commercial projects.

---
## 👤 Author

<div align="center">

### Ayush Kumar Singh

*Researcher in Adversarial ML, Geospatial AI, and LLM/NLP Systems*

[![GitHub](https://img.shields.io/badge/GitHub-Ayush%20Kumar%20Singh-181717?style=for-the-badge&logo=github)](https://github.com/Ayush-2703)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ayush%20Kumar%20Singh-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/ayushsingh2703)

</div>

---

<div align="center">

**If this repository helped you, please consider giving it a ⭐**  
*It takes 2 seconds and helps others discover it.*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:022C22,100:059669&height=100&section=footer" width="100%"/>

</div>
