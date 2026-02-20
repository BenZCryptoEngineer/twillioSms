"""
Core evaluation metrics: AUC, log-loss, calibration, Brier score.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss


def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_pred))


def compute_logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(log_loss(y_true, np.clip(y_pred, 1e-7, 1 - 1e-7)))


def compute_brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_pred))


def compute_calibration(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10,
) -> dict:
    """Calibration metrics across probability bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    pred_means, actual_means, counts = [], [], []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_pred >= lo) & (y_pred <= hi if i == n_bins - 1 else y_pred < hi)
        n = int(mask.sum())
        counts.append(n)
        if n == 0:
            pred_means.append(np.nan)
            actual_means.append(np.nan)
        else:
            pred_means.append(float(y_pred[mask].mean()))
            actual_means.append(float(y_true[mask].mean()))

    pred_arr = np.array(pred_means)
    actual_arr = np.array(actual_means)
    valid = ~np.isnan(pred_arr)
    errors = np.abs(pred_arr[valid] - actual_arr[valid]) if valid.any() else np.array([])

    return {
        "bin_edges": edges,
        "bin_pred_mean": pred_arr,
        "bin_actual_mean": actual_arr,
        "bin_counts": np.array(counts),
        "max_calibration_error": float(errors.max()) if len(errors) > 0 else float("nan"),
        "mean_calibration_error": float(errors.mean()) if len(errors) > 0 else float("nan"),
    }
