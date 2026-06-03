"""
test_change_detection.py
================================
Unit tests for Module 5 – Change Detection.
"""
import numpy as np
import pytest


def safe_ratio(a, b, eps=1e-6):
    denom = a + b
    denom = np.where(np.abs(denom) < eps, eps, denom)
    return np.clip((a - b) / denom, -1.0, 1.0)


def threshold_mask(delta, direction="negative", k=2.0):
    mu, sigma = delta.mean(), delta.std()
    if direction == "negative":
        return delta < (mu - k * sigma)
    elif direction == "positive":
        return delta > (mu + k * sigma)
    else:
        return np.abs(delta - mu) > k * sigma


def zscore_anomaly(delta_stack, threshold=3.0):
    flat  = delta_stack.reshape(-1, delta_stack.shape[-1])
    mu    = flat.mean(axis=0)
    sigma = flat.std(axis=0) + 1e-6
    z     = np.abs((flat - mu) / sigma).mean(axis=1)
    return (z > threshold).reshape(delta_stack.shape[:2])


class TestChangeDetection:

    def test_delta_computation(self):
        after  = np.ones((10, 10), dtype=np.float32) * 0.5
        before = np.ones((10, 10), dtype=np.float32) * 0.3
        delta  = after - before
        assert np.allclose(delta, 0.2)

    def test_threshold_mask_negative(self):
        delta = np.zeros((20, 20), dtype=np.float32)
        delta[0, 0] = -5.0      # extreme outlier
        mask  = threshold_mask(delta, "negative", k=2.0)
        assert mask[0, 0], "Extreme negative outlier should be flagged"
        assert not mask[1:, 1:].any(), "Background should not be flagged"

    def test_threshold_mask_positive(self):
        delta = np.zeros((20, 20), dtype=np.float32)
        delta[5, 5] = 8.0
        mask  = threshold_mask(delta, "positive", k=2.0)
        assert mask[5, 5]

    def test_no_change_if_uniform(self):
        delta = np.ones((30, 30), dtype=np.float32) * 0.1
        mask  = threshold_mask(delta, "both", k=2.0)
        assert not mask.any(), "Uniform delta should produce no change"

    def test_change_map_labels(self):
        H, W = 20, 20
        change_map = np.zeros((H, W), dtype=np.uint8)
        change_map[2, 2] = 1  # vegetation loss
        change_map[3, 3] = 2  # urban
        change_map[4, 4] = 3  # water
        change_map[5, 5] = 4  # anomaly
        assert set(np.unique(change_map)) == {0, 1, 2, 3, 4}

    def test_zscore_anomaly_detects_outlier(self):
        base  = np.random.normal(0, 0.05, (20, 20, 3)).astype(np.float32)
        base[10, 10, :] = 10.0   # extreme outlier pixel
        mask = zscore_anomaly(base, threshold=3.0)
        assert mask[10, 10], "Extreme pixel should be flagged as anomaly"

    def test_change_percentage_reasonable(self):
        H, W = 100, 100
        delta = np.random.normal(0, 0.02, (H, W)).astype(np.float32)
        delta[10:15, 10:15] = -0.5     # deliberate change patch
        mask  = threshold_mask(delta, "negative", k=2.0)
        pct   = 100.0 * mask.mean()
        assert 0 < pct < 50, f"Change pct out of reasonable range: {pct:.1f}%"

    def test_delta_dtype_preserved(self):
        a = np.random.rand(10, 10).astype(np.float32)
        b = np.random.rand(10, 10).astype(np.float32)
        d = a - b
        assert d.dtype == np.float32
