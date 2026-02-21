# evaluation/ — Metrics & Kill-Point Gates

## Purpose

Computes metrics, runs significance tests, and makes go/no-go decisions at kill-point gates.

## Files

| File | Responsibility |
|------|---------------|
| `metrics.py` | AUC, log-loss, Brier score, calibration |
| `significance.py` | Paired bootstrap test for model comparison |
| `kill_point.py` | KP-1 gate evaluation with thresholds |
| `report.py` | Report generation (JSON + text + plots) |

## Metrics (metrics.py)

| Function | Returns | Edge Cases |
|----------|---------|------------|
| `compute_auc(y_true, y_pred)` | float | Single class → 0.5 |
| `compute_logloss(y_true, y_pred)` | float | Clips predictions to [1e-7, 1-1e-7] |
| `compute_brier(y_true, y_pred)` | float | Perfect=0.0, worst=1.0 |
| `compute_calibration(y_true, y_pred, n_bins)` | dict | Empty bins → NaN |

### Calibration Output

```python
{
    "bin_edges": np.ndarray,          # (n_bins+1,)
    "bin_pred_mean": np.ndarray,      # (n_bins,) — may contain NaN
    "bin_actual_mean": np.ndarray,    # (n_bins,)
    "bin_counts": np.ndarray,         # (n_bins,) — int
    "max_calibration_error": float,
    "mean_calibration_error": float,
}
```

## Significance Testing (significance.py)

`paired_bootstrap_test(y_true, pred_a, pred_b, n_bootstrap=10000, metric="auc")`

Paired bootstrap comparison:
1. Sample N indices with replacement
2. Compute metric(pred_b) - metric(pred_a) for each sample
3. P-value = fraction of samples where A >= B

Returns:
```python
{
    "mean_diff": float,          # mean improvement of B over A
    "ci_lower": float,           # 2.5th percentile
    "ci_upper": float,           # 97.5th percentile
    "p_value": float,
    "significant_at_05": bool,
    "significant_at_01": bool,
}
```

Handles degenerate cases (single class labels → p=0.5).

## Kill Point #1 (kill_point.py)

`evaluate_kill_point_1(y_true, baseline_preds, sequence_preds, thresholds=None)`

### Default Thresholds

| Criterion | Threshold | Meaning |
|-----------|-----------|---------|
| AUC improvement | > 0.03 | Sequence model AUC - Baseline AUC |
| Log-loss improvement | > 0.02 | Baseline LL - Sequence LL (lower is better) |
| Calibration | < 0.05 | Max error across decile bins |
| Significance | p < 0.01 | Paired bootstrap test |

### Decision Logic

```
OVERALL_PASS = auc_pass AND logloss_pass AND significance_pass
```

Note: Calibration is tracked but not a hard gate for PASS (it's informational for Phase 1).

### Return Structure

```python
{
    "baseline": {"auc", "logloss", "brier", "calibration"},
    "sequence_model": {"auc", "logloss", "brier", "calibration"},
    "comparison": {"auc_improvement", "logloss_improvement", "brier_improvement", "bootstrap_test"},
    "kill_point_1": {
        "auc_pass": bool,
        "logloss_pass": bool,
        "calibration_pass": bool,
        "significance_pass": bool,
        "OVERALL_PASS": bool,
        "thresholds": dict,
    },
}
```

## Report Generation (report.py)

`generate_report(kp_result, output_dir="reports/")` → path to report

Outputs:
1. `kp1_report.json` — Full result dict
2. `kp1_report.txt` — Human-readable summary
3. `kp1_calibration.png` — Calibration plot (if matplotlib available)

## Dependencies

- Imports from: `core` only (interfaces)
- Imported by: `orchestration`
- External: `sklearn.metrics`, `numpy`, `matplotlib` (optional)
