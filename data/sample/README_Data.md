# Data Download Instructions

The satellite TIF files are **not included** in this repository due to their size.
Download them from the sources below and place them in this `data/` folder.

---

## Required Files

```
data/
├── haridwarBand2.tif     ← Green band  (ResourceSat-2 LISS-III)
├── haridwarBand3.tif     ← Red band
├── haridwarBand4.tif     ← NIR band
├── haridwarBand5.tif     ← SWIR band
└── Water_Map.tif         ← Binary reference water mask
```

---

## Source 1 — IIRS/ISRO GitHub (Haridwar bands)

```bash
git clone https://github.com/iirs-isro/IIRS_ISRO_Geoprocessing-using-Python.git
cp IIRS_ISRO_Geoprocessing-using-Python/haridwar/haridwarBand*.tif data/
```

## Source 2 — ISRO ML Course (Water Map + CSVs)

```bash
git clone https://github.com/iirs-isro/ISRO-Geodata-Processing-using-Python-and-Machine-Learning.git
cp ISRO-Geodata-Processing-using-Python-and-Machine-Learning/Water_Map.tif data/
```

---

## Included Sample Data (`data/sample/`)

These CSV files **are included** in the repo and ready to use:

| File | Rows | Description |
|------|------|-------------|
| `water_train.csv`       | 1,000 | Band2–5 reflectance + binary Water label |
| `vizag_sample_data.csv` | 8,000 | 6-band reflectance + 8-class Landcover label |

---

## Using Your Own Data

Point the pipeline to any custom TIF or CSV using CLI flags:

```bash
# Custom satellite bands
python run_pipeline.py --tif B2_Green=/path/b2.tif B3_Red=/path/b3.tif \
                            B4_NIR=/path/b4.tif  B5_SWIR=/path/b5.tif

# Custom CSV datasets
python run_pipeline.py --csv water=/path/my_water.csv \
                            vizag=/path/my_landcover.csv

# Real two-date change detection (supply T1 bands)
python run_pipeline.py --before-tif B2_Green=/T1/b2.tif \
                                    B3_Red=/T1/b3.tif   \
                                    B4_NIR=/T1/b4.tif   \
                                    B5_SWIR=/T1/b5.tif
```

### CSV Format Requirements

**water_train.csv** — must contain columns:
```
Band2, Band3, Band4, Band5, Water
```

**vizag_sample_data.csv** — must contain columns:
```
Blue, Green, Red, NIR, SWIR-1, SWIR-2, Landcover
```
