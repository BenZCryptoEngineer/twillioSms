"""
Evaluation metrics for Kill Point #1 validation.

Computes AUC, log-loss, calibration, and statistical significance tests
to determine if shot-sequence V outperforms score-only baseline.
"""

import logging

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

log = logging.getLogger(__name__)


def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute ROC AUC score."""
    return roc_auc_score(y_true, y_pred)


def compute_logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute binary cross-entropy (log loss)."""
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return log_loss(y_true, y_pred_clipped)


def compute_brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Brier score (MSE of probabilities)."""
    return brier_score_loss(y_true, y_pred)


def compute_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute calibration metrics across probability deciles.

    Returns:
        dict with bin_edges, bin_pred_mean, bin_actual_mean,
        max_calibration_error, mean_calibration_error
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_pred_mean = []
    bin_actual_mean = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_pred >= lo) & (y_pred <= hi)
        else:
            mask = (y_pred >= lo) & (y_pred < hi)

        if mask.sum() == 0:
            bin_pred_mean.append(np.nan)
            bin_actual_mean.append(np.nan)
            bin_counts.append(0)
        else:
            bin_pred_mean.append(y_pred[mask].mean())
            bin_actual_mean.append(y_true[mask].mean())
            bin_counts.append(mask.sum())

    bin_pred_mean = np.array(bin_pred_mean)
    bin_actual_mean = np.array(bin_actual_mean)
    bin_counts = np.array(bin_counts)

    # Calibration errors (only for non-empty bins)
    valid = ~np.isnan(bin_pred_mean)
    errors = np.abs(bin_pred_mean[valid] - bin_actual_mean[valid])

    return {
        "bin_edges": bin_edges,
        "bin_pred_mean": bin_pred_mean,
        "bin_actual_mean": bin_actual_mean,
        "bin_counts": bin_counts,
        "max_calibration_error": errors.max() if len(errors) > 0 else float("nan"),
        "mean_calibration_error": errors.mean() if len(errors) > 0 else float("nan"),
    }


def paired_bootstrap_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric_fn=roc_auc_score,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap test for comparing two models.

    Tests if model B's metric is significantly better than model A's.

    Args:
        y_true: ground truth labels
        y_pred_a: predictions from model A (baseline)
        y_pred_b: predictions from model B (shot-sequence)
        metric_fn: metric function (higher = better)
        n_bootstrap: number of bootstrap samples

    Returns:
        dict with observed_diff, p_value, ci_lower, ci_upper, significant
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    observed_a = metric_fn(y_true, y_pred_a)
    observed_b = metric_fn(y_true, y_pred_b)
    observed_diff = observed_b - observed_a

    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        # Skip degenerate samples
        if len(np.unique(yt)) < 2:
            continue
        diff = metric_fn(yt, y_pred_b[idx]) - metric_fn(yt, y_pred_a[idx])
        diffs.append(diff)

    diffs = np.array(diffs)

    p_value = (diffs <= 0).mean()
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)

    return {
        "metric": metric_fn.__name__,
        "observed_a": observed_a,
        "observed_b": observed_b,
        "observed_diff": observed_diff,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant_at_01": p_value < 0.01,
        "n_bootstrap": len(diffs),
    }


def evaluate_kill_point_1(
    y_true: np.ndarray,
    baseline_preds: np.ndarray,
    sequence_preds: np.ndarray,
    thresholds: dict = None,
) -> dict:
    """Full Kill Point #1 evaluation.

    Compares baseline vs shot-sequence model on all Phase 1 criteria.

    Args:
        thresholds: dict with min_auc_improvement, min_logloss_improvement,
                    calibration_tolerance, significance_level

    Returns:
        dict with all metrics and PASS/FAIL decision
    """
    thresholds = thresholds or {
        "min_auc_improvement": 0.03,
        "min_logloss_improvement": 0.02,
        "calibration_tolerance": 0.05,
        "significance_level": 0.01,
    }

    # Individual metrics
    baseline_auc = compute_auc(y_true, baseline_preds)
    sequence_auc = compute_auc(y_true, sequence_preds)
    auc_improvement = sequence_auc - baseline_auc

    baseline_logloss = compute_logloss(y_true, baseline_preds)
    sequence_logloss = compute_logloss(y_true, sequence_preds)
    logloss_improvement = baseline_logloss - sequence_logloss  # lower is better

    baseline_brier = compute_brier(y_true, baseline_preds)
    sequence_brier = compute_brier(y_true, sequence_preds)

    # Calibration
    baseline_cal = compute_calibration(y_true, baseline_preds)
    sequence_cal = compute_calibration(y_true, sequence_preds)

    # Statistical significance
    auc_test = paired_bootstrap_test(
        y_true, baseline_preds, sequence_preds,
        metric_fn=roc_auc_score,
    )

    # Kill point decisions
    auc_pass = auc_improvement > thresholds["min_auc_improvement"]
    logloss_pass = logloss_improvement > thresholds["min_logloss_improvement"]
    cal_pass = sequence_cal["max_calibration_error"] < thresholds["calibration_tolerance"]
    sig_pass = auc_test["significant_at_01"]

    overall_pass = auc_pass and logloss_pass and sig_pass

    result = {
        "baseline": {
            "auc": baseline_auc,
            "logloss": baseline_logloss,
            "brier": baseline_brier,
            "calibration": baseline_cal,
        },
        "sequence_model": {
            "auc": sequence_auc,
            "logloss": sequence_logloss,
            "brier": sequence_brier,
            "calibration": sequence_cal,
        },
        "comparison": {
            "auc_improvement": auc_improvement,
            "logloss_improvement": logloss_improvement,
            "brier_improvement": baseline_brier - sequence_brier,
            "bootstrap_test": auc_test,
        },
        "kill_point_1": {
            "auc_pass": auc_pass,
            "logloss_pass": logloss_pass,
            "calibration_pass": cal_pass,
            "significance_pass": sig_pass,
            "OVERALL_PASS": overall_pass,
            "thresholds": thresholds,
        },
    }

    verdict = "PASS" if overall_pass else "FAIL"
    log.info(f"Kill Point #1: {verdict}")
    log.info(f"  AUC: {baseline_auc:.4f} -> {sequence_auc:.4f} (delta={auc_improvement:.4f}, need>{thresholds['min_auc_improvement']}): {'PASS' if auc_pass else 'FAIL'}")
    log.info(f"  Log-loss: {baseline_logloss:.4f} -> {sequence_logloss:.4f} (delta={logloss_improvement:.4f}, need>{thresholds['min_logloss_improvement']}): {'PASS' if logloss_pass else 'FAIL'}")
    log.info(f"  Calibration max error: {sequence_cal['max_calibration_error']:.4f} (need<{thresholds['calibration_tolerance']}): {'PASS' if cal_pass else 'FAIL'}")
    log.info(f"  Significance p={auc_test['p_value']:.6f} (need<{thresholds['significance_level']}): {'PASS' if sig_pass else 'FAIL'}")

    return result
