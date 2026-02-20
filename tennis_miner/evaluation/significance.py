"""
Statistical significance tests for model comparison.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def paired_bootstrap_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric_fn=roc_auc_score,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """Test if model B significantly outperforms model A.

    Returns dict with observed_diff, p_value, 95% CI, significance flag.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    obs_a = metric_fn(y_true, y_pred_a)
    obs_b = metric_fn(y_true, y_pred_b)

    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        diffs.append(metric_fn(yt, y_pred_b[idx]) - metric_fn(yt, y_pred_a[idx]))

    diffs = np.array(diffs)

    return {
        "metric": metric_fn.__name__,
        "observed_a": float(obs_a),
        "observed_b": float(obs_b),
        "observed_diff": float(obs_b - obs_a),
        "p_value": float((diffs <= 0).mean()) if len(diffs) > 0 else 1.0,
        "ci_lower": float(np.percentile(diffs, 2.5)) if len(diffs) > 0 else 0.0,
        "ci_upper": float(np.percentile(diffs, 97.5)) if len(diffs) > 0 else 0.0,
        "significant_at_01": bool((diffs <= 0).mean() < 0.01) if len(diffs) > 0 else False,
        "n_bootstrap": len(diffs),
    }
