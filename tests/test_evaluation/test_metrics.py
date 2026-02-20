"""Comprehensive tests for evaluation metrics."""

import numpy as np
import pytest

from tennis_miner.evaluation.metrics import (
    compute_auc, compute_logloss, compute_brier, compute_calibration,
)
from tennis_miner.evaluation.significance import paired_bootstrap_test
from tennis_miner.evaluation.kill_point import evaluate_kill_point_1


class TestAUC:
    def test_perfect(self):
        assert compute_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0

    def test_random(self):
        auc = compute_auc(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5]))
        assert abs(auc - 0.5) < 0.01

    def test_single_class_returns_0_5(self):
        """All same label → AUC should return 0.5 (degenerate)."""
        assert compute_auc(np.ones(5), np.random.rand(5)) == 0.5

    def test_inverted(self):
        auc = compute_auc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1]))
        assert auc == 0.0


class TestLogLoss:
    def test_good_predictions(self):
        loss = compute_logloss(np.array([0, 1]), np.array([0.1, 0.9]))
        assert loss < 0.5

    def test_bad_predictions(self):
        loss = compute_logloss(np.array([0, 1]), np.array([0.9, 0.1]))
        assert loss > 1.0

    def test_clips_extremes(self):
        """Should not crash on predictions of exactly 0 or 1."""
        loss = compute_logloss(np.array([0, 1]), np.array([0.0, 1.0]))
        assert np.isfinite(loss)


class TestBrier:
    def test_perfect(self):
        b = compute_brier(np.array([0, 1]), np.array([0.0, 1.0]))
        assert b == 0.0

    def test_worst(self):
        b = compute_brier(np.array([0, 1]), np.array([1.0, 0.0]))
        assert b == 1.0


class TestCalibration:
    def test_num_bins(self):
        y = np.random.binomial(1, 0.5, 100).astype(float)
        p = np.clip(y + np.random.randn(100) * 0.1, 0.01, 0.99)
        cal = compute_calibration(y, p, n_bins=5)
        assert len(cal["bin_pred_mean"]) == 5
        assert len(cal["bin_actual_mean"]) == 5
        assert len(cal["bin_counts"]) == 5

    def test_empty_bins(self):
        """Extreme predictions → some bins empty → should handle NaN."""
        y = np.array([0, 0, 1, 1], dtype=float)
        p = np.array([0.01, 0.02, 0.98, 0.99])
        cal = compute_calibration(y, p, n_bins=10)
        assert np.isnan(cal["bin_pred_mean"]).sum() > 0  # some bins empty

    def test_perfectly_calibrated(self):
        np.random.seed(42)
        n = 10000
        p = np.random.rand(n)
        y = (np.random.rand(n) < p).astype(float)
        cal = compute_calibration(y, p, n_bins=10)
        assert cal["max_calibration_error"] < 0.05

    def test_all_same_prediction(self):
        y = np.array([0, 1, 0, 1], dtype=float)
        p = np.array([0.5, 0.5, 0.5, 0.5])
        cal = compute_calibration(y, p, n_bins=10)
        assert np.isfinite(cal["mean_calibration_error"])


class TestBootstrap:
    def test_better_model_significant(self):
        np.random.seed(42)
        n = 500
        y = np.random.binomial(1, 0.5, n).astype(float)
        a = np.clip(0.5 + np.random.randn(n) * 0.3, 0.01, 0.99)
        b = np.clip(0.5 + (y - 0.5) * 0.6 + np.random.randn(n) * 0.1, 0.01, 0.99)
        r = paired_bootstrap_test(y, a, b, n_bootstrap=1000)
        assert r["observed_diff"] > 0
        assert r["p_value"] < 0.05

    def test_equal_models_not_significant(self):
        np.random.seed(42)
        n = 100
        y = np.random.binomial(1, 0.5, n).astype(float)
        p = np.clip(0.5 + np.random.randn(n) * 0.2, 0.01, 0.99)
        r = paired_bootstrap_test(y, p, p, n_bootstrap=500)
        assert abs(r["observed_diff"]) < 0.01
        assert r["p_value"] > 0.1

    def test_degenerate_labels(self):
        """All same label — should handle gracefully."""
        y = np.ones(50)
        a = np.random.rand(50)
        b = np.random.rand(50)
        r = paired_bootstrap_test(y, a, b, n_bootstrap=100)
        assert "p_value" in r


class TestKillPoint1:
    def test_clear_pass(self):
        np.random.seed(42)
        n = 1000
        y = np.random.binomial(1, 0.5, n).astype(float)
        baseline = np.clip(0.5 + np.random.randn(n) * 0.3, 0.01, 0.99)
        better = np.clip(0.5 + (y - 0.5) * 0.8 + np.random.randn(n) * 0.05, 0.01, 0.99)
        r = evaluate_kill_point_1(y, baseline, better)
        assert r["kill_point_1"]["OVERALL_PASS"] is True

    def test_clear_fail(self):
        np.random.seed(42)
        n = 200
        y = np.random.binomial(1, 0.5, n).astype(float)
        preds = np.clip(0.5 + np.random.randn(n) * 0.2, 0.01, 0.99)
        r = evaluate_kill_point_1(y, preds, preds)
        assert r["kill_point_1"]["OVERALL_PASS"] is False

    def test_result_structure(self):
        y = np.array([0, 0, 1, 1], dtype=float)
        a = np.array([0.3, 0.4, 0.6, 0.7])
        b = np.array([0.1, 0.2, 0.8, 0.9])
        r = evaluate_kill_point_1(y, a, b)
        assert "baseline" in r
        assert "sequence_model" in r
        assert "comparison" in r
        assert "kill_point_1" in r
        assert "thresholds" in r["kill_point_1"]
