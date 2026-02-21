"""
Phase-aware pipeline runner.

This is the ONLY module that imports from all other modules.
It wires ingestion → features → models → evaluation.

Key fix vs old code: proper train/val/test split (no data leakage).
"""

import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from tennis_miner.orchestration.config import load_config
from tennis_miner.ingestion.mcp import MCPLoader
from tennis_miner.ingestion.court_vision import CourtVisionLoader
from tennis_miner.ingestion.validator import DataValidator
from tennis_miner.features.sequence import build_dataset
from tennis_miner.models.baseline import LogisticBaseline
from tennis_miner.models.lstm import LSTMModel
from tennis_miner.models.transformer import TransformerModel
from tennis_miner.evaluation.kill_point import evaluate_kill_point_1
from tennis_miner.evaluation.report import generate_report

log = logging.getLogger(__name__)


def run_phase1(config_path: str = "configs/default.yaml") -> dict:
    """Phase 1 end-to-end: data → features → train → evaluate → report."""
    cfg = load_config(config_path)
    p1 = cfg["phase1"]

    # ── 1. Load data ──
    log.info("Phase 1 Step 1: Loading data")
    all_matches = []

    mcp_dir = cfg["data_acquisition"]["mcp"]["clone_dir"]
    if Path(mcp_dir).exists():
        loader = MCPLoader(mcp_dir)
        matches = loader.load()
        all_matches.extend(matches)

    cv_dir = Path(cfg["paths"]["raw_data"]) / "court_vision"
    if cv_dir.exists():
        loader = CourtVisionLoader(str(cv_dir))
        matches = loader.load()
        all_matches.extend(matches)

    if not all_matches:
        log.error("No data. Run data acquisition first.")
        return {"error": "no_data"}

    # ── 2. Validate ──
    log.info("Phase 1 Step 2: Validating data")
    validator = DataValidator()
    validation = validator.validate(all_matches)
    log.info(f"Validation: {validation['stats']}")

    # ── 3. Feature engineering ──
    log.info("Phase 1 Step 3: Feature engineering")
    max_rally = p1["sequence_model"]["max_rally_length"]
    full_bundle = build_dataset(all_matches, max_rally_length=max_rally)

    if full_bundle.n_samples == 0:
        log.error("Dataset has 0 samples after feature engineering")
        return {"error": "empty_dataset"}

    # ── 4. Split: train / val / test (by match, no leakage) ──
    log.info("Phase 1 Step 4: Splitting data")
    unique_ids = list(set(full_bundle.match_ids))
    train_ids, temp_ids = train_test_split(
        unique_ids, test_size=0.3, random_state=p1["random_seed"]
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.5, random_state=p1["random_seed"]
    )

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    train_mask = np.array([m in train_set for m in full_bundle.match_ids])
    val_mask = np.array([m in val_set for m in full_bundle.match_ids])
    test_mask = np.array([m in test_set for m in full_bundle.match_ids])

    train_data = full_bundle.subset(train_mask)
    val_data = full_bundle.subset(val_mask)
    test_data = full_bundle.subset(test_mask)

    log.info(
        f"Split: train={train_data.n_samples}, "
        f"val={val_data.n_samples}, test={test_data.n_samples}"
    )

    # ── 5. Train baseline ──
    log.info("Phase 1 Step 5: Training baseline (score-only)")
    baseline = LogisticBaseline()
    baseline.fit(train_data, val_data)
    baseline_preds = baseline.predict_proba(test_data)

    models_dir = Path(cfg["paths"].get("models", "tennis_miner/models/saved"))
    models_dir.mkdir(parents=True, exist_ok=True)
    baseline.save(str(models_dir / "baseline.joblib"))

    # ── 6. Train sequence model ──
    log.info("Phase 1 Step 6: Training sequence model")
    seq_cfg = p1["sequence_model"]
    model_type = seq_cfg.get("model_type", "lstm")

    if model_type == "transformer":
        model = TransformerModel(
            d_model=seq_cfg.get("embedding_dim", 64),
            num_layers=seq_cfg.get("num_layers", 2),
            dropout=seq_cfg.get("dropout", 0.2),
            lr=seq_cfg.get("learning_rate", 0.001),
            epochs=seq_cfg.get("max_epochs", 100),
            batch_size=seq_cfg.get("batch_size", 512),
            max_seq_len=max_rally,
        )
    else:
        model = LSTMModel(
            embed_dim=seq_cfg.get("embedding_dim", 32),
            hidden_dim=seq_cfg.get("hidden_dim", 128),
            num_layers=seq_cfg.get("num_layers", 2),
            dropout=seq_cfg.get("dropout", 0.2),
            lr=seq_cfg.get("learning_rate", 0.001),
            epochs=seq_cfg.get("max_epochs", 100),
            batch_size=seq_cfg.get("batch_size", 512),
        )

    model.fit(train_data, val_data)
    sequence_preds = model.predict_proba(test_data)
    model.save(str(models_dir / f"sequence_{model_type}.pt"))

    # ── 7. Kill Point #1 ──
    log.info("Phase 1 Step 7: Kill Point #1 evaluation")
    kp_cfg = p1.get("kill_point", {})
    result = evaluate_kill_point_1(
        y_true=test_data.labels,
        baseline_preds=baseline_preds,
        sequence_preds=sequence_preds,
        thresholds=kp_cfg if kp_cfg else None,
    )

    report_path = generate_report(result)
    result["report_path"] = report_path
    result["data_stats"] = validation["stats"]

    return result
