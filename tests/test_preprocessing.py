"""
test_preprocessing.py
============================
Unit tests for Module 2 – Pre-processing & QA.
"""
import numpy as np
import pytest


def safe_ratio(a, b, eps=1e-6):
    denom = a + b
    denom = np.where(np.abs(denom) < eps, eps, denom)
    return np.clip((a - b) / denom, -1.0, 1.0)


def normalise_dn(band, scale):
    return np.clip(band.astype(np.float32) * scale, 0.0, 1.0)


def detect_clouds(green, red, nir, threshold=0.80):
    brightness = (green + red) / 2.0
    ndvi = safe_ratio(nir, red)
    return (brightness > threshold) & (ndvi < 0.1)


class TestPreprocessing:

    def test_normalise_range(self):
        arr = np.array([0, 256, 512, 1023], dtype=np.float32)
        ref = normalise_dn(arr, 1.0/1023.0)
        assert ref.min() >= 0.0
        assert ref.max() <= 1.0

    def test_normalise_clamps(self):
        arr = np.array([-10, 2000], dtype=np.float32)
        ref = normalise_dn(arr, 1.0/1023.0)
        assert ref[0] == 0.0
        assert ref[1] == 1.0

    def test_ndvi_range(self):
        nir = np.random.uniform(0.1, 0.8, (50, 50)).astype(np.float32)
        red = np.random.uniform(0.05, 0.4, (50, 50)).astype(np.float32)
        ndvi = safe_ratio(nir, red)
        assert ndvi.min() >= -1.0
        assert ndvi.max() <= 1.0

    def test_ndvi_vegetation_positive(self):
        nir = np.full((10, 10), 0.5, dtype=np.float32)
        red = np.full((10, 10), 0.1, dtype=np.float32)
        ndvi = safe_ratio(nir, red)
        assert (ndvi > 0).all(), "Dense vegetation should have positive NDVI"

    def test_ndwi_water_positive(self):
        green = np.full((10, 10), 0.3, dtype=np.float32)
        nir   = np.full((10, 10), 0.05, dtype=np.float32)
        ndwi  = safe_ratio(green, nir)
        assert (ndwi > 0).all(), "Water should have positive NDWI"

    def test_cloud_mask_bright_pixels(self):
        green = np.full((10, 10), 0.95, dtype=np.float32)
        red   = np.full((10, 10), 0.90, dtype=np.float32)
        nir   = np.full((10, 10), 0.05, dtype=np.float32)
        mask  = detect_clouds(green, red, nir)
        assert mask.all(), "Bright low-NDVI pixels should be flagged as cloud"

    def test_cloud_mask_clear_vegetation(self):
        green = np.full((10, 10), 0.1, dtype=np.float32)
        red   = np.full((10, 10), 0.08, dtype=np.float32)
        nir   = np.full((10, 10), 0.5, dtype=np.float32)
        mask  = detect_clouds(green, red, nir)
        assert not mask.any(), "Dark high-NDVI pixels should not be flagged"

    def test_safe_ratio_no_division_by_zero(self):
        a = np.zeros((10, 10), dtype=np.float32)
        b = np.zeros((10, 10), dtype=np.float32)
        result = safe_ratio(a, b)
        assert np.isfinite(result).all(), "safe_ratio must not produce NaN or Inf"

    def test_output_shape_preserved(self):
        arr  = np.random.rand(64, 64).astype(np.float32)
        ref  = normalise_dn(arr * 1023, 1.0/1023.0)
        assert ref.shape == arr.shape
