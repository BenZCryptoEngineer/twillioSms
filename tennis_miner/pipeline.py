"""
Phase 1 training and evaluation pipeline.

Orchestrates the full flow: data loading -> feature engineering ->
model training -> Kill Point #1 evaluation.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from tennis_miner.data.loader_mcp import load_mcp_matches
from tennis_miner.data.loader_court_vision import load_court_vision_matches
from tennis_miner.data.loader_slam_pbp import load_slam_pbp_matches
from tennis_miner.data.feature_engineering import build_phase1_dataset
from tennis_miner.models.baseline import LogisticBaseline, MLPBaseline, train_mlp_baseline
from tennis_miner.models.sequence_model import (
    ShotSequenceLSTM, ShotSequenceTransformer, train_sequence_model,
)
from tennis_miner.evaluation.metrics import evaluate_kill_point_1
from tennis_miner.evaluation.report import generate_kp1_report
from tennis_miner.utils.config import load_config

log = logging.getLogger(__name__)


def run_phase1(config_path: str = "configs/default.yaml") -> dict:
    """Run the complete Phase 1 pipeline.

    1. Load data from all sources
    2. Build feature matrices
    3. Train baseline model (score-only)
    4. Train shot-sequence model (score + shots)
    5. Evaluate Kill Point #1

    Returns the evaluation result dict.
    """
    config = load_config(config_path)
    p1 = config["phase1"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Running Phase 1 pipeline on device: {device}")

    # ---- Step 1: Load data ----
    log.info("Step 1: Loading data...")
    all_matches = []

    # MCP (primary source for Phase 1)
    mcp_dir = config["data_acquisition"]["mcp"]["clone_dir"]
    if Path(mcp_dir).exists():
        mcp_matches = load_mcp_matches(mcp_dir)
        all_matches.extend(mcp_matches)
    else:
        log.warning(f"MCP data not found at {mcp_dir}. Run acquire_mcp.py first.")

    # Court Vision (spatial data — used in Phase 1 for additional points)
    cv_dir = Path(config["paths"]["raw_data"]) / "court_vision"
    if cv_dir.exists():
        cv_matches = load_court_vision_matches(str(cv_dir))
        all_matches.extend(cv_matches)

    if not all_matches:
        log.error("No data loaded. Run Phase 0 acquisition scripts first.")
        return {"error": "No data available"}

    log.info(f"Total matches loaded: {len(all_matches)}")

    # ---- Step 2: Feature engineering ----
    log.info("Step 2: Building feature matrices...")
    max_rally = p1["sequence_model"]["max_rally_length"]
    dataset = build_phase1_dataset(all_matches, max_rally_length=max_rally)

    n_points = len(dataset["labels"])
    log.info(f"Dataset: {n_points} points, "
             f"server win rate: {dataset['labels'].mean():.3f}")

    # Train/test split (stratified by match to prevent data leakage)
    unique_matches = list(set(dataset["match_ids"]))
    train_matches, test_matches = train_test_split(
        unique_matches, test_size=p1["test_size"], random_state=p1["random_seed"]
    )
    train_match_set = set(train_matches)

    train_mask = np.array([m in train_match_set for m in dataset["match_ids"]])
    test_mask = ~train_mask

    def split_data(data, mask):
        return {
            "score_features": data["score_features"][mask],
            "shot_sequences": data["shot_sequences"][mask],
            "sequence_lengths": data["sequence_lengths"][mask],
            "labels": data["labels"][mask],
        }

    train_data = split_data(dataset, train_mask)
    test_data = split_data(dataset, test_mask)

    log.info(f"Train: {len(train_data['labels'])} points, "
             f"Test: {len(test_data['labels'])} points")

    # ---- Step 3: Train baseline ----
    log.info("Step 3: Training baseline model (score-only)...")
    baseline = LogisticBaseline()
    baseline.fit(train_data["score_features"], train_data["labels"])
    baseline_preds = baseline.predict_proba(test_data["score_features"])

    # Save baseline
    models_dir = Path(config["paths"].get("models", "tennis_miner/models/saved"))
    models_dir.mkdir(parents=True, exist_ok=True)
    baseline.save(str(models_dir / "baseline_logistic.joblib"))

    # ---- Step 4: Train shot-sequence model ----
    log.info("Step 4: Training shot-sequence model...")
    seq_config = p1["sequence_model"]
    model_type = seq_config.get("model_type", "lstm")

    if model_type == "transformer":
        model = ShotSequenceTransformer(
            embedding_dim=seq_config.get("embedding_dim", 32),
            d_model=seq_config.get("embedding_dim", 64),
            num_layers=seq_config.get("num_layers", 2),
            dropout=seq_config.get("dropout", 0.2),
            max_seq_len=max_rally,
        )
    else:
        model = ShotSequenceLSTM(
            embedding_dim=seq_config.get("embedding_dim", 32),
            hidden_dim=seq_config.get("hidden_dim", 128),
            num_layers=seq_config.get("num_layers", 2),
            dropout=seq_config.get("dropout", 0.2),
        )

    history = train_sequence_model(
        model=model,
        train_data=train_data,
        val_data=test_data,
        lr=seq_config.get("learning_rate", 0.001),
        epochs=seq_config.get("max_epochs", 100),
        batch_size=seq_config.get("batch_size", 512),
        device=device,
    )

    # Save sequence model
    torch.save(model.state_dict(), str(models_dir / f"sequence_{model_type}.pt"))

    # Get predictions
    model.eval()
    model = model.to(device)
    score_norm = (test_data["score_features"] - history["score_mean"]) / (history["score_std"] + 1e-8)
    with torch.no_grad():
        seq_preds = model.predict_proba(
            torch.tensor(test_data["shot_sequences"], dtype=torch.float32, device=device),
            torch.tensor(test_data["sequence_lengths"], dtype=torch.long, device=device),
            torch.tensor(score_norm, dtype=torch.float32, device=device),
        ).cpu().numpy()

    # ---- Step 5: Kill Point #1 evaluation ----
    log.info("Step 5: Evaluating Kill Point #1...")
    kp_thresholds = p1.get("kill_point", {})
    result = evaluate_kill_point_1(
        y_true=test_data["labels"],
        baseline_preds=baseline_preds,
        sequence_preds=seq_preds,
        thresholds={
            "min_auc_improvement": kp_thresholds.get("min_auc_improvement", 0.03),
            "min_logloss_improvement": kp_thresholds.get("min_logloss_improvement", 0.02),
            "calibration_tolerance": kp_thresholds.get("calibration_tolerance", 0.05),
            "significance_level": kp_thresholds.get("significance_level", 0.01),
        },
    )

    # Generate report
    report_path = generate_kp1_report(result)
    log.info(f"Full report: {report_path}")

    return result
