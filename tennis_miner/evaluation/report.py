"""Report generation: JSON + text + plots."""

import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np

log = logging.getLogger(__name__)


def generate_report(eval_result: dict, output_dir: str = "reports") -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = out / f"kp1_report_{ts}.json"
    data = _serialize(eval_result)
    data["generated_at"] = datetime.now().isoformat()
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    txt_path = out / f"kp1_report_{ts}.txt"
    with open(txt_path, "w") as f:
        f.write(_format_text(eval_result))

    try:
        _save_plots(eval_result, out, ts)
    except ImportError:
        log.warning("matplotlib not available, skipping plots")

    log.info(f"Report: {json_path}")
    return str(json_path)


def _format_text(r: dict) -> str:
    kp = r["kill_point_1"]
    b, s, c = r["baseline"], r["sequence_model"], r["comparison"]
    v = "PASS" if kp["OVERALL_PASS"] else "FAIL"

    return "\n".join([
        "=" * 70,
        "TENNIS MINER — Kill Point #1 Report",
        "=" * 70, "",
        f"Decision: {v}", "",
        f"{'Metric':<20} {'Baseline':>10} {'Sequence':>10} {'Delta':>10} {'Pass':>6}",
        "-" * 60,
        f"{'AUC':<20} {b['auc']:>10.4f} {s['auc']:>10.4f} {c['auc_improvement']:>+10.4f} {'Y' if kp['auc_pass'] else 'N':>6}",
        f"{'Log-Loss':<20} {b['logloss']:>10.4f} {s['logloss']:>10.4f} {c['logloss_improvement']:>+10.4f} {'Y' if kp['logloss_pass'] else 'N':>6}",
        f"{'Brier':<20} {b['brier']:>10.4f} {s['brier']:>10.4f} {c['brier_improvement']:>+10.4f} {'---':>6}",
        "",
        f"Bootstrap p-value: {c['bootstrap_test']['p_value']:.6f}",
        f"95% CI: [{c['bootstrap_test']['ci_lower']:.4f}, {c['bootstrap_test']['ci_upper']:.4f}]",
        f"Calibration max error: {s['calibration']['max_calibration_error']:.4f}",
        "", "=" * 70,
    ])


def _save_plots(r: dict, out: Path, ts: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for label, key in [("Baseline", "baseline"), ("Sequence", "sequence_model")]:
        cal = r[key]["calibration"]
        valid = ~np.isnan(cal["bin_pred_mean"])
        if valid.any():
            ax1.plot(cal["bin_pred_mean"][valid], cal["bin_actual_mean"][valid], "o-", label=label)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Calibration")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    names = ["AUC", "1-LogLoss", "1-Brier"]
    bv = [r["baseline"]["auc"], 1 - r["baseline"]["logloss"], 1 - r["baseline"]["brier"]]
    sv = [r["sequence_model"]["auc"], 1 - r["sequence_model"]["logloss"], 1 - r["sequence_model"]["brier"]]
    x = np.arange(len(names))
    ax2.bar(x - 0.18, bv, 0.35, label="Baseline")
    ax2.bar(x + 0.18, sv, 0.35, label="Sequence")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_title("Model Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out / f"kp1_plots_{ts}.png", dpi=150)
    plt.close()


def _serialize(obj):
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
