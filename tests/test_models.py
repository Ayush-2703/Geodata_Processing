"""
test_models.py
=====================
Unit tests for Module 4 – AI Models.
"""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def water_dataset():
    X, y = make_classification(
        n_samples=400, n_features=4, n_informative=4,
        n_redundant=0, n_classes=2, random_state=42
    )
    scaler = StandardScaler()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    return X_tr, X_te, y_tr, y_te


@pytest.fixture
def lc_dataset():
    X, y = make_classification(
        n_samples=800, n_features=6, n_informative=6,
        n_redundant=0, n_classes=8, n_clusters_per_class=1,
        random_state=42
    )
    scaler = StandardScaler()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    return X_tr, X_te, y_tr, y_te


class TestRandomForest:

    def test_rf_trains_without_error(self, water_dataset):
        X_tr, _, y_tr, _ = water_dataset
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_tr, y_tr)
        assert hasattr(rf, "estimators_")

    def test_rf_water_accuracy(self, water_dataset):
        X_tr, X_te, y_tr, y_te = water_dataset
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        rf.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, rf.predict(X_te))
        assert acc >= 0.70, f"RF accuracy too low: {acc:.3f}"

    def test_rf_feature_importance_sums_to_one(self, water_dataset):
        X_tr, _, y_tr, _ = water_dataset
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_tr, y_tr)
        assert abs(rf.feature_importances_.sum() - 1.0) < 1e-5

    def test_rf_lc_multiclass(self, lc_dataset):
        X_tr, X_te, y_tr, y_te = lc_dataset
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        rf.fit(X_tr, y_tr)
        f1 = f1_score(y_te, rf.predict(X_te), average="macro", zero_division=0)
        assert f1 >= 0.40, f"LC F1-macro too low: {f1:.3f}"

    def test_rf_predict_shape(self, water_dataset):
        X_tr, X_te, y_tr, _ = water_dataset
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X_tr, y_tr)
        preds = rf.predict(X_te)
        assert preds.shape == (len(X_te),)

    def test_rf_proba_shape(self, water_dataset):
        X_tr, X_te, y_tr, _ = water_dataset
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X_tr, y_tr)
        proba = rf.predict_proba(X_te)
        assert proba.shape[0] == len(X_te)
        assert abs(proba.sum(axis=1).mean() - 1.0) < 1e-5


class TestMLP:

    def test_mlp_trains_without_error(self, water_dataset):
        X_tr, _, y_tr, _ = water_dataset
        mlp = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=200, random_state=42)
        mlp.fit(X_tr, y_tr)
        assert hasattr(mlp, "coefs_")

    def test_mlp_water_accuracy(self, water_dataset):
        X_tr, X_te, y_tr, y_te = water_dataset
        mlp = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300, random_state=42,
                            early_stopping=True)
        mlp.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, mlp.predict(X_te))
        assert acc >= 0.65, f"MLP accuracy too low: {acc:.3f}"

    def test_mlp_loss_decreases(self, water_dataset):
        X_tr, _, y_tr, _ = water_dataset
        mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, random_state=42)
        mlp.fit(X_tr, y_tr)
        assert mlp.loss_ < 1.0, "Training loss should be < 1.0 after convergence"
