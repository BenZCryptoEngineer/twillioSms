"""Tests for evaluation metrics."""

import numpy as np

from tennis_miner.evaluation.metrics import (
    compute_auc, compute_logloss, compute_calibration,
    paired_bootstrap_test, evaluate_kill_point_1,
)


def test_compute_auc_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_auc(y_true, y_pred) == 1.0


def test_compute_logloss():
    y_true = np.array([0, 1])
    y_pred = np.array([0.1, 0.9])
    loss = compute_logloss(y_true, y_pred)
    assert loss > 0
    assert loss < 0.5  # predictions are good


def test_calibration():
    y_true = np.random.binomial(1, 0.5, 1000).astype(float)
    y_pred = np.clip(y_true + np.random.randn(1000) * 0.1, 0.01, 0.99)
    cal = compute_calibration(y_true, y_pred, n_bins=5)
    assert len(cal["bin_pred_mean"]) == 5
    assert len(cal["bin_actual_mean"]) == 5


def test_paired_bootstrap():
    np.random.seed(42)
    n = 500
    y_true = np.random.binomial(1, 0.5, n).astype(float)
    # Model B is strictly better
    y_pred_a = np.clip(0.5 + (y_true - 0.5) * 0.2 + np.random.randn(n) * 0.3, 0.01, 0.99)
    y_pred_b = np.clip(0.5 + (y_true - 0.5) * 0.6 + np.random.randn(n) * 0.1, 0.01, 0.99)
    result = paired_bootstrap_test(y_true, y_pred_a, y_pred_b, n_bootstrap=1000)
    assert result["observed_diff"] > 0
    assert result["p_value"] < 0.05


def test_evaluate_kill_point_1():
    np.random.seed(42)
    n = 1000
    y_true = np.random.binomial(1, 0.5, n).astype(float)
    baseline = np.clip(0.5 + np.random.randn(n) * 0.3, 0.01, 0.99)
    better = np.clip(0.5 + (y_true - 0.5) * 0.8 + np.random.randn(n) * 0.05, 0.01, 0.99)
    result = evaluate_kill_point_1(y_true, baseline, better)
    assert "kill_point_1" in result
    assert "baseline" in result
    assert "sequence_model" in result
