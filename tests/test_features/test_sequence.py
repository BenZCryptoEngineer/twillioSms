"""Comprehensive tests for feature engineering."""

import numpy as np
import pytest

from tennis_miner.core.schema import (
    Match, Point, Shot, Surface, DataBundle,
    ShotType, ShotDirection, ShotOutcome,
)
from tennis_miner.features.score import extract_score_features, SCORE_FEATURE_DIM
from tennis_miner.features.sequence import (
    extract_shot_features, extract_shot_sequence, build_dataset, SHOT_FEATURE_DIM,
)
from tennis_miner.features.spatial import extract_spatial_features, spatial_mask


def _make_match(n_points=3, with_shots=True):
    m = Match(
        match_id="m1", tournament="AO", year=2023, surface=Surface.HARD,
        player1="A", player2="B", winner="A", score="6-4",
    )
    for i in range(n_points):
        shots = []
        if with_shots:
            shots = [
                Shot(shot_num=1, player="server", shot_type=ShotType.FIRST_SERVE),
                Shot(shot_num=2, player="returner", shot_type=ShotType.FOREHAND),
                Shot(shot_num=3, player="server", outcome=ShotOutcome.WINNER),
            ]
        m.points.append(Point(
            point_id=f"p{i}", match_id="m1",
            set_num=1, game_num=1, point_num=i + 1,
            server="A", returner="B",
            server_won=(i % 2 == 0),
            shots=shots,
        ))
    return m


class TestScoreFeatures:
    def test_shape(self):
        m = _make_match(1)
        f = extract_score_features(m.points[0], m)
        assert f.shape == (SCORE_FEATURE_DIM,)
        assert f.dtype == np.float32

    def test_server_encoding(self):
        m = _make_match(1)
        f = extract_score_features(m.points[0], m)
        assert f[3] == 1.0  # server == player1

    def test_tiebreak_flag(self):
        m = _make_match(1)
        m.points[0].is_tiebreak = True
        f = extract_score_features(m.points[0], m)
        assert f[7] == 1.0


class TestShotFeatures:
    def test_shape(self):
        s = Shot(shot_num=1, player="server")
        f = extract_shot_features(s)
        assert f.shape == (SHOT_FEATURE_DIM,)

    def test_server_flag(self):
        s = Shot(shot_num=1, player="server")
        assert extract_shot_features(s)[0] == 1.0

    def test_returner_flag(self):
        s = Shot(shot_num=1, player="returner")
        assert extract_shot_features(s)[0] == 0.0

    def test_shot_num_normalization(self):
        s = Shot(shot_num=100, player="server")
        f = extract_shot_features(s)
        assert f[5] == 1.0  # capped at 1.0


class TestShotSequence:
    def test_padding(self):
        m = _make_match(1)
        seq = extract_shot_sequence(m.points[0], max_length=10)
        assert seq.shape == (10, SHOT_FEATURE_DIM)
        assert seq[0].sum() > 0  # first shot
        assert seq[2].sum() > 0  # third shot
        assert seq[3].sum() == 0  # padded

    def test_truncation(self):
        m = _make_match(1)
        seq = extract_shot_sequence(m.points[0], max_length=2)
        assert seq.shape == (2, SHOT_FEATURE_DIM)
        assert seq[1].sum() > 0  # second shot present

    def test_empty_shots(self):
        m = _make_match(1, with_shots=False)
        seq = extract_shot_sequence(m.points[0], max_length=5)
        assert seq.sum() == 0.0


class TestBuildDataset:
    def test_basic(self):
        m = _make_match(3)
        b = build_dataset([m], max_rally_length=10)
        assert b.n_samples == 3
        assert b.score_features.shape == (3, SCORE_FEATURE_DIM)
        assert b.shot_sequences.shape == (3, 10, SHOT_FEATURE_DIM)

    def test_labels(self):
        m = _make_match(3)
        b = build_dataset([m])
        # Points 0, 2 won by server; point 1 not
        assert b.labels[0] == 1.0
        assert b.labels[1] == 0.0
        assert b.labels[2] == 1.0

    def test_skip_no_shots(self):
        m = _make_match(3, with_shots=False)
        b = build_dataset([m], require_shots=True)
        assert b.n_samples == 0

    def test_include_no_shots(self):
        m = _make_match(3, with_shots=False)
        b = build_dataset([m], require_shots=False)
        assert b.n_samples == 3

    def test_multiple_matches(self):
        m1 = _make_match(2)
        m2 = Match(
            match_id="m2", tournament="RG", year=2023, surface=Surface.CLAY,
            player1="C", player2="D", winner="C", score="6-3",
        )
        m2.points.append(Point(
            point_id="px", match_id="m2", set_num=1, game_num=1, point_num=1,
            server="C", returner="D", server_won=True,
            shots=[Shot(shot_num=1, player="server", outcome=ShotOutcome.ACE)],
        ))
        b = build_dataset([m1, m2])
        assert b.n_samples == 3
        assert "m2" in b.match_ids

    def test_empty_input(self):
        b = build_dataset([])
        assert b.n_samples == 0


class TestSpatial:
    def test_extract_with_data(self):
        s = Shot(
            shot_num=1, player="server",
            ball_landing_x=1.5, ball_landing_y=-3.0, serve_speed_kmh=200.0,
        )
        f = extract_spatial_features(s)
        assert f.shape == (7,)
        assert f[0] == 1.5
        assert f[1] == -3.0
        assert abs(f[2] - 200.0 / 250.0) < 0.01

    def test_extract_missing(self):
        s = Shot(shot_num=1, player="server")
        f = extract_spatial_features(s)
        assert f.sum() == 0.0

    def test_mask(self):
        s = Shot(shot_num=1, player="server", ball_landing_x=1.0)
        m = spatial_mask(s)
        assert m[0] == 1.0  # x present
        assert m[1] == 0.0  # y missing
