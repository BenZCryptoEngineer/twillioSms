"""Shot-sequence feature extraction for LSTM/Transformer models."""

import logging

import numpy as np

from tennis_miner.core.schema import Match, Point, Shot, DataBundle
from tennis_miner.features.encoder import (
    SHOT_TYPE_ENC, DIRECTION_ENC, DEPTH_ENC, OUTCOME_ENC,
)
from tennis_miner.features.score import extract_score_features

log = logging.getLogger(__name__)

SHOT_FEATURE_DIM = 6


def extract_shot_features(shot: Shot) -> np.ndarray:
    """6-dim vector per shot: [is_server, type, dir, depth, outcome, shot_num_norm]."""
    return np.array([
        1.0 if shot.player == "server" else 0.0,
        SHOT_TYPE_ENC.get(shot.shot_type, 0),
        DIRECTION_ENC.get(shot.direction, 0),
        DEPTH_ENC.get(shot.depth, 0),
        OUTCOME_ENC.get(shot.outcome, 0),
        min(shot.shot_num / 50.0, 1.0),
    ], dtype=np.float32)


def extract_shot_sequence(point: Point, max_length: int = 50) -> np.ndarray:
    """(max_length, 6) padded shot sequence matrix."""
    seq = np.zeros((max_length, SHOT_FEATURE_DIM), dtype=np.float32)
    for i, shot in enumerate(point.shots[:max_length]):
        seq[i] = extract_shot_features(shot)
    return seq


def build_dataset(
    matches: list[Match],
    max_rally_length: int = 50,
    require_shots: bool = True,
) -> DataBundle:
    """Convert list[Match] → DataBundle ready for model training.

    Args:
        require_shots: If True (default), skip points without shot data.
                       If False, include all points (shot sequences will be zeros).
    """
    score_feats, shot_seqs, seq_lens, labels, match_ids = [], [], [], [], []

    for match in matches:
        for point in match.points:
            if require_shots and not point.has_shots:
                continue

            score_feats.append(extract_score_features(point, match))
            shot_seqs.append(extract_shot_sequence(point, max_rally_length))
            seq_lens.append(min(len(point.shots), max_rally_length))
            labels.append(1.0 if point.server_won else 0.0)
            match_ids.append(match.match_id)

    n = len(labels)
    if n == 0:
        log.warning("build_dataset produced 0 samples")
        return DataBundle(
            score_features=np.zeros((0, 8), dtype=np.float32),
            shot_sequences=np.zeros((0, max_rally_length, SHOT_FEATURE_DIM), dtype=np.float32),
            sequence_lengths=np.zeros(0, dtype=np.int32),
            labels=np.zeros(0, dtype=np.float32),
            match_ids=[],
        )

    bundle = DataBundle(
        score_features=np.array(score_feats, dtype=np.float32),
        shot_sequences=np.array(shot_seqs, dtype=np.float32),
        sequence_lengths=np.array(seq_lens, dtype=np.int32),
        labels=np.array(labels, dtype=np.float32),
        match_ids=match_ids,
    )

    log.info(
        f"Built dataset: {bundle.n_samples} samples from {len(matches)} matches, "
        f"server win rate: {bundle.server_win_rate:.3f}"
    )
    return bundle
