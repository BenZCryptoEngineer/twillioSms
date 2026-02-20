"""
Report generation for Phase 1 Kill Point evaluation.

Produces evaluation report with metrics, calibration curves,
feature importance, and go/no-go decision.
"""

import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np

log = logging.getLogger(__name__)


def generate_kp1_report(
    eval_result: dict,
    output_dir: str = "reports",
    include_plots: bool = True,
) -> str:
    """Generate Kill Point #1 evaluation report.

    Returns path to the generated report file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON report with full metrics
    json_path = out / f"kp1_report_{timestamp}.json"

    # Convert numpy types for JSON serialization
    report_data = _make_serializable(eval_result)
    report_data["generated_at"] = datetime.now().isoformat()

    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)

    # Text summary
    txt_path = out / f"kp1_report_{timestamp}.txt"
    summary = _format_text_report(eval_result)
    with open(txt_path, "w") as f:
        f.write(summary)

    # Calibration plot
    if include_plots:
        try:
            _generate_plots(eval_result, out, timestamp)
        except ImportError:
            log.warning("matplotlib not available, skipping plots")

    log.info(f"Report saved to {json_path}")
    return str(json_path)


def _format_text_report(result: dict) -> str:
    """Format evaluation results as readable text report."""
    kp = result["kill_point_1"]
    baseline = result["baseline"]
    seq = result["sequence_model"]
    comp = result["comparison"]

    verdict = "PASS — Shot sequence data adds significant predictive value." if kp["OVERALL_PASS"] else \
              "FAIL — Shot sequence data does not add sufficient predictive value. Project terminates."

    lines = [
        "=" * 70,
        "TENNIS MINER — Kill Point #1 Evaluation Report",
        "=" * 70,
        "",
        f"Decision: {verdict}",
        "",
        "-" * 40,
        "Metric Comparison",
        "-" * 40,
        f"{'Metric':<25} {'Baseline':>10} {'Sequence':>10} {'Delta':>10} {'Pass?':>8}",
        f"{'AUC':<25} {baseline['auc']:>10.4f} {seq['auc']:>10.4f} {comp['auc_improvement']:>+10.4f} {'YES' if kp['auc_pass'] else 'NO':>8}",
        f"{'Log-Loss':<25} {baseline['logloss']:>10.4f} {seq['logloss']:>10.4f} {comp['logloss_improvement']:>+10.4f} {'YES' if kp['logloss_pass'] else 'NO':>8}",
        f"{'Brier Score':<25} {baseline['brier']:>10.4f} {seq['brier']:>10.4f} {comp['brier_improvement']:>+10.4f} {'---':>8}",
        "",
        "-" * 40,
        "Statistical Significance (Paired Bootstrap)",
        "-" * 40,
        f"  p-value: {comp['bootstrap_test']['p_value']:.6f}",
        f"  95% CI: [{comp['bootstrap_test']['ci_lower']:.4f}, {comp['bootstrap_test']['ci_upper']:.4f}]",
        f"  Significant at p<0.01: {'YES' if kp['significance_pass'] else 'NO'}",
        "",
        "-" * 40,
        "Calibration",
        "-" * 40,
        f"  Sequence model max calibration error: {seq['calibration']['max_calibration_error']:.4f}",
        f"  Within tolerance ({kp['thresholds']['calibration_tolerance']}): {'YES' if kp['calibration_pass'] else 'NO'}",
        "",
        "-" * 40,
        "Kill Point Criteria",
        "-" * 40,
        f"  AUC improvement > {kp['thresholds']['min_auc_improvement']}: {'PASS' if kp['auc_pass'] else 'FAIL'}",
        f"  Log-loss improvement > {kp['thresholds']['min_logloss_improvement']}: {'PASS' if kp['logloss_pass'] else 'FAIL'}",
        f"  Calibration < {kp['thresholds']['calibration_tolerance']}: {'PASS' if kp['calibration_pass'] else 'FAIL'}",
        f"  Statistical significance p < {kp['thresholds']['significance_level']}: {'PASS' if kp['significance_pass'] else 'FAIL'}",
        "",
        f"  OVERALL: {verdict}",
        "",
        "=" * 70,
    ]
    return "\n".join(lines)


def _generate_plots(result: dict, output_dir: Path, timestamp: str):
    """Generate calibration and comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Calibration plot
    ax = axes[0]
    for label, model_key in [("Baseline", "baseline"), ("Sequence", "sequence_model")]:
        cal = result[model_key]["calibration"]
        valid = ~np.isnan(cal["bin_pred_mean"])
        if valid.any():
            ax.plot(
                cal["bin_pred_mean"][valid],
                cal["bin_actual_mean"][valid],
                "o-", label=label,
            )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Actual Win Rate")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Metric comparison bar chart
    ax = axes[1]
    metrics = ["AUC", "1 - Log-Loss", "1 - Brier"]
    baseline_vals = [
        result["baseline"]["auc"],
        1 - result["baseline"]["logloss"],
        1 - result["baseline"]["brier"],
    ]
    seq_vals = [
        result["sequence_model"]["auc"],
        1 - result["sequence_model"]["logloss"],
        1 - result["sequence_model"]["brier"],
    ]
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width / 2, baseline_vals, width, label="Baseline")
    ax.bar(x + width / 2, seq_vals, width, label="Sequence")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = output_dir / f"kp1_plots_{timestamp}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log.info(f"Plots saved to {plot_path}")


def _make_serializable(obj):
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
