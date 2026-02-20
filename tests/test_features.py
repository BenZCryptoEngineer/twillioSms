"""Tests for feature engineering."""

import numpy as np

from tennis_miner.data.schema import (
    Match, Point, Shot, Surface,
    ShotType, ShotDirection, ShotOutcome, ShotDepth,
)
from tennis_miner.data.feature_engineering import (
    extract_score_features, extract_shot_features,
    extract_shot_sequence, build_phase1_dataset,
)


def _make_test_match():
    shots = [
        Shot(shot_num=1, player="server", shot_type=ShotType.FIRST_SERVE,
             direction=ShotDirection.WIDE_AD),
        Shot(shot_num=2, player="returner", shot_type=ShotType.FOREHAND,
             direction=ShotDirection.CROSSCOURT),
        Shot(shot_num=3, player="server", shot_type=ShotType.BACKHAND,
             direction=ShotDirection.DOWN_THE_LINE, outcome=ShotOutcome.WINNER),
    ]
    point = Point(
        point_id="test_001", match_id="match_001",
        set_num=1, game_num=3, point_num=2,
        server="Player1", returner="Player2",
        server_won=True, shots=shots, rally_length=3,
    )
    match = Match(
        match_id="match_001", tournament="test", year=2023,
        surface=Surface.HARD, player1="Player1", player2="Player2",
        winner="Player1", score="6-4",
    )
    match.points = [point]
    return match


def test_score_features_shape():
    match = _make_test_match()
    feats = extract_score_features(match.points[0], match)
    assert feats.shape == (8,)
    assert feats.dtype == np.float32


def test_shot_features_shape():
    shot = Shot(shot_num=1, player="server", shot_type=ShotType.FOREHAND)
    feats = extract_shot_features(shot)
    assert feats.shape == (6,)


def test_shot_sequence_padding():
    match = _make_test_match()
    seq = extract_shot_sequence(match.points[0], max_length=10)
    assert seq.shape == (10, 6)
    # First 3 rows should be non-zero, rest padded
    assert seq[0].sum() > 0
    assert seq[2].sum() > 0
    assert seq[3].sum() == 0  # padded


def test_build_phase1_dataset():
    match = _make_test_match()
    dataset = build_phase1_dataset([match], max_rally_length=10)
    assert dataset["score_features"].shape == (1, 8)
    assert dataset["shot_sequences"].shape == (1, 10, 6)
    assert dataset["sequence_lengths"].shape == (1,)
    assert dataset["labels"].shape == (1,)
    assert dataset["labels"][0] == 1.0  # server won
