"""
Kill-point gate evaluation.

Centralizes all go/no-go decisions with quantitative thresholds.
"""

import logging

import numpy as np

from tennis_miner.evaluation.metrics import (
    compute_auc, compute_logloss, compute_brier, compute_calibration,
)
from tennis_miner.evaluation.significance import paired_bootstrap_test

log = logging.getLogger(__name__)

DEFAULT_KP1_THRESHOLDS = {
    "min_auc_improvement": 0.03,
    "min_logloss_improvement": 0.02,
    "calibration_tolerance": 0.05,
    "significance_level": 0.01,
}


def evaluate_kill_point_1(
    y_true: np.ndarray,
    baseline_preds: np.ndarray,
    sequence_preds: np.ndarray,
    thresholds: dict | None = None,
) -> dict:
    """Full KP-1 evaluation: does shot-sequence V beat score-only baseline?"""
    t = {**DEFAULT_KP1_THRESHOLDS, **(thresholds or {})}

    b_auc = compute_auc(y_true, baseline_preds)
    s_auc = compute_auc(y_true, sequence_preds)
    auc_delta = s_auc - b_auc

    b_ll = compute_logloss(y_true, baseline_preds)
    s_ll = compute_logloss(y_true, sequence_preds)
    ll_delta = b_ll - s_ll

    b_brier = compute_brier(y_true, baseline_preds)
    s_brier = compute_brier(y_true, sequence_preds)

    s_cal = compute_calibration(y_true, sequence_preds)
    b_cal = compute_calibration(y_true, baseline_preds)

    boot = paired_bootstrap_test(y_true, baseline_preds, sequence_preds)

    auc_pass = auc_delta > t["min_auc_improvement"]
    ll_pass = ll_delta > t["min_logloss_improvement"]
    cal_pass = s_cal["max_calibration_error"] < t["calibration_tolerance"]
    sig_pass = boot["significant_at_01"]
    overall = auc_pass and ll_pass and sig_pass

    verdict = "PASS" if overall else "FAIL"
    log.info(f"Kill Point #1: {verdict}")
    log.info(f"  AUC:     {b_auc:.4f} → {s_auc:.4f} (Δ={auc_delta:+.4f}, need>{t['min_auc_improvement']}): {'PASS' if auc_pass else 'FAIL'}")
    log.info(f"  LogLoss: {b_ll:.4f} → {s_ll:.4f} (Δ={ll_delta:+.4f}, need>{t['min_logloss_improvement']}): {'PASS' if ll_pass else 'FAIL'}")
    log.info(f"  Calib:   max_err={s_cal['max_calibration_error']:.4f} (need<{t['calibration_tolerance']}): {'PASS' if cal_pass else 'FAIL'}")
    log.info(f"  Signif:  p={boot['p_value']:.6f} (need<{t['significance_level']}): {'PASS' if sig_pass else 'FAIL'}")

    return {
        "baseline": {"auc": b_auc, "logloss": b_ll, "brier": b_brier, "calibration": b_cal},
        "sequence_model": {"auc": s_auc, "logloss": s_ll, "brier": s_brier, "calibration": s_cal},
        "comparison": {
            "auc_improvement": auc_delta,
            "logloss_improvement": ll_delta,
            "brier_improvement": b_brier - s_brier,
            "bootstrap_test": boot,
        },
        "kill_point_1": {
            "auc_pass": auc_pass,
            "logloss_pass": ll_pass,
            "calibration_pass": cal_pass,
            "significance_pass": sig_pass,
            "OVERALL_PASS": overall,
            "thresholds": t,
        },
    }
