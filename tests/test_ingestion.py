"""
test_ingestion.py
========================
Unit tests for Module 1 – Data Ingestion.
Run with: pytest tests/test_ingestion.py -v
"""
import os, sys, pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ── fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def sample_water_csv(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    df  = pd.DataFrame({
        "Band2": np.random.randint(60, 200, 100),
        "Band3": np.random.randint(50, 180, 100),
        "Band4": np.random.randint(40, 300, 100),
        "Band5": np.random.randint(10, 150, 100),
        "Water": np.random.randint(0, 2, 100),
    })
    p = tmp / "water_train.csv"
    df.to_csv(p, index=True)
    return str(p)


@pytest.fixture(scope="module")
def sample_vizag_csv(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    df  = pd.DataFrame({
        "Blue":     np.random.uniform(0.02, 0.2, 160),
        "Green":    np.random.uniform(0.03, 0.25, 160),
        "Red":      np.random.uniform(0.02, 0.3, 160),
        "NIR":      np.random.uniform(0.1, 0.6, 160),
        "SWIR-1":   np.random.uniform(0.05, 0.4, 160),
        "SWIR-2":   np.random.uniform(0.03, 0.35, 160),
        "Landcover": np.tile([10,20,30,40,50,60,80,90], 20),
    })
    p = tmp / "vizag_sample_data.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture(scope="module")
def sample_tif(tmp_path_factory):
    """Create a tiny synthetic single-band GeoTIFF."""
    import rasterio
    from rasterio.transform import from_bounds
    tmp  = tmp_path_factory.mktemp("tif")
    path = str(tmp / "test_band.tif")
    data = (np.random.randint(100, 500, (32, 32))).astype(np.uint16)
    transform = from_bounds(77.0, 29.0, 78.0, 30.0, 32, 32)
    with rasterio.open(path, "w", driver="GTiff", height=32, width=32,
                       count=1, dtype="uint16", crs="EPSG:4326",
                       transform=transform) as dst:
        dst.write(data, 1)
    return path


# ── tests ─────────────────────────────────────────────────────────────────────
class TestDataIngestion:

    def test_water_csv_loads(self, sample_water_csv):
        df = pd.read_csv(sample_water_csv, index_col=0)
        assert len(df) == 100
        assert "Water" in df.columns
        assert set(["Band2","Band3","Band4","Band5"]).issubset(df.columns)

    def test_vizag_csv_loads(self, sample_vizag_csv):
        df = pd.read_csv(sample_vizag_csv)
        assert len(df) == 160
        assert "Landcover" in df.columns
        assert df["Landcover"].nunique() == 8

    def test_tif_reads_correctly(self, sample_tif):
        import rasterio
        with rasterio.open(sample_tif) as src:
            arr = src.read(1)
        assert arr.shape == (32, 32)
        assert arr.dtype == np.uint16
        assert arr.min() >= 100
        assert arr.max() < 500

    def test_band_stats_non_negative(self, sample_tif):
        import rasterio
        with rasterio.open(sample_tif) as src:
            arr = src.read(1).astype(np.float32)
        assert arr.mean() > 0
        assert arr.std() >= 0

    def test_water_labels_binary(self, sample_water_csv):
        df = pd.read_csv(sample_water_csv, index_col=0)
        assert set(df["Water"].unique()).issubset({0, 1})

    def test_vizag_landcover_classes(self, sample_vizag_csv):
        df  = pd.read_csv(sample_vizag_csv)
        expected = {10, 20, 30, 40, 50, 60, 80, 90}
        assert set(df["Landcover"].unique()) == expected
