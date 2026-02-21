"""Comprehensive tests for core domain objects."""

import pytest
import numpy as np

from tennis_miner.core.schema import (
    Shot, Point, Match, DataBundle, Surface,
    ShotType, ShotDirection, ShotOutcome, ShotDepth,
)


# ── Shot ─────────────────────────────────────────────────────────────

class TestShot:
    def test_basic_creation(self):
        s = Shot(shot_num=1, player="server", shot_type=ShotType.FOREHAND)
        assert s.shot_num == 1
        assert s.player == "server"

    def test_spatial_fields_default_none(self):
        s = Shot(shot_num=1, player="server")
        assert s.ball_landing_x is None
        assert not s.has_spatial

    def test_has_spatial_true(self):
        s = Shot(shot_num=1, player="server", ball_landing_x=1.5, ball_landing_y=3.0)
        assert s.has_spatial

    def test_has_spatial_partial_false(self):
        s = Shot(shot_num=1, player="server", ball_landing_x=1.5)
        assert not s.has_spatial  # needs both x and y

    def test_is_point_ending_winner(self):
        s = Shot(shot_num=1, player="server", outcome=ShotOutcome.WINNER)
        assert s.is_point_ending

    def test_is_point_ending_in_play(self):
        s = Shot(shot_num=1, player="server", outcome=ShotOutcome.IN_PLAY)
        assert not s.is_point_ending

    def test_is_point_ending_ace(self):
        s = Shot(shot_num=1, player="server", outcome=ShotOutcome.ACE)
        assert s.is_point_ending

    def test_invalid_shot_num_zero(self):
        with pytest.raises(ValueError, match="shot_num"):
            Shot(shot_num=0, player="server")

    def test_invalid_shot_num_negative(self):
        with pytest.raises(ValueError, match="shot_num"):
            Shot(shot_num=-1, player="server")

    def test_invalid_player(self):
        with pytest.raises(ValueError, match="player"):
            Shot(shot_num=1, player="spectator")

    def test_invalid_negative_speed(self):
        with pytest.raises(ValueError, match="serve_speed_kmh"):
            Shot(shot_num=1, player="server", serve_speed_kmh=-10.0)

    def test_zero_speed_valid(self):
        s = Shot(shot_num=1, player="server", serve_speed_kmh=0.0)
        assert s.serve_speed_kmh == 0.0


# ── Point ────────────────────────────────────────────────────────────

class TestPoint:
    def test_basic_creation(self):
        p = Point(
            point_id="p1", match_id="m1", set_num=1, game_num=1, point_num=1,
            server="A", returner="B", server_won=True,
        )
        assert p.server_score == "0"
        assert not p.has_shots

    def test_auto_rally_length(self):
        """rally_length=0 with shots present should auto-calculate."""
        shots = [Shot(shot_num=1, player="server"), Shot(shot_num=2, player="returner")]
        p = Point(
            point_id="p1", match_id="m1", set_num=1, game_num=1, point_num=1,
            server="A", returner="B", server_won=True, shots=shots,
        )
        assert p.rally_length == 2

    def test_explicit_rally_length_preserved(self):
        shots = [Shot(shot_num=1, player="server")]
        p = Point(
            point_id="p1", match_id="m1", set_num=1, game_num=1, point_num=1,
            server="A", returner="B", server_won=True, shots=shots, rally_length=5,
        )
        assert p.rally_length == 5

    def test_invalid_set_num(self):
        with pytest.raises(ValueError, match="set_num"):
            Point(
                point_id="p1", match_id="m1", set_num=0, game_num=1, point_num=1,
                server="A", returner="B", server_won=True,
            )

    def test_empty_server_name(self):
        with pytest.raises(ValueError, match="server"):
            Point(
                point_id="p1", match_id="m1", set_num=1, game_num=1, point_num=1,
                server="", returner="B", server_won=True,
            )

    def test_has_spatial_false(self):
        p = Point(
            point_id="p1", match_id="m1", set_num=1, game_num=1, point_num=1,
            server="A", returner="B", server_won=True,
            shots=[Shot(shot_num=1, player="server")],
        )
        assert not p.has_spatial

    def test_has_spatial_true(self):
        p = Point(
            point_id="p1", match_id="m1", set_num=1, game_num=1, point_num=1,
            server="A", returner="B", server_won=True,
            shots=[Shot(shot_num=1, player="server", ball_landing_x=1.0, ball_landing_y=2.0)],
        )
        assert p.has_spatial


# ── Match ────────────────────────────────────────────────────────────

class TestMatch:
    def test_basic_creation(self):
        m = Match(
            match_id="m1", tournament="AO", year=2023, surface=Surface.HARD,
            player1="A", player2="B", winner="A", score="6-4 6-3",
        )
        assert m.n_points == 0
        assert not m.has_spatial

    def test_invalid_empty_id(self):
        with pytest.raises(ValueError, match="match_id"):
            Match(
                match_id="", tournament="AO", year=2023, surface=Surface.HARD,
                player1="A", player2="B", winner="A", score="",
            )

    def test_invalid_year(self):
        with pytest.raises(ValueError, match="year"):
            Match(
                match_id="m1", tournament="AO", year=1800, surface=Surface.HARD,
                player1="A", player2="B", winner="A", score="",
            )

    def test_n_points_with_shots(self):
        m = Match(
            match_id="m1", tournament="AO", year=2023, surface=Surface.HARD,
            player1="A", player2="B", winner="A", score="",
        )
        m.points = [
            Point(point_id="p1", match_id="m1", set_num=1, game_num=1, point_num=1,
                  server="A", returner="B", server_won=True,
                  shots=[Shot(shot_num=1, player="server")]),
            Point(point_id="p2", match_id="m1", set_num=1, game_num=1, point_num=2,
                  server="A", returner="B", server_won=False),
        ]
        assert m.n_points == 2
        assert m.n_points_with_shots == 1


# ── DataBundle ───────────────────────────────────────────────────────

class TestDataBundle:
    def test_basic_creation(self):
        b = DataBundle(
            score_features=np.zeros((5, 8)),
            shot_sequences=np.zeros((5, 10, 6)),
            sequence_lengths=np.zeros(5, dtype=np.int32),
            labels=np.ones(5),
            match_ids=["m1"] * 5,
        )
        assert b.n_samples == 5
        assert b.server_win_rate == 1.0

    def test_mismatched_shapes(self):
        with pytest.raises(ValueError, match="score_features"):
            DataBundle(
                score_features=np.zeros((3, 8)),
                shot_sequences=np.zeros((5, 10, 6)),
                sequence_lengths=np.zeros(5, dtype=np.int32),
                labels=np.ones(5),
                match_ids=["m1"] * 5,
            )

    def test_mismatched_match_ids(self):
        with pytest.raises(ValueError, match="match_ids"):
            DataBundle(
                score_features=np.zeros((5, 8)),
                shot_sequences=np.zeros((5, 10, 6)),
                sequence_lengths=np.zeros(5, dtype=np.int32),
                labels=np.ones(5),
                match_ids=["m1"] * 3,
            )

    def test_subset(self):
        b = DataBundle(
            score_features=np.arange(20).reshape(5, 4).astype(float),
            shot_sequences=np.zeros((5, 10, 6)),
            sequence_lengths=np.array([1, 2, 3, 4, 5], dtype=np.int32),
            labels=np.array([0, 1, 0, 1, 1], dtype=float),
            match_ids=["a", "b", "c", "d", "e"],
        )
        mask = np.array([True, False, True, False, True])
        sub = b.subset(mask)
        assert sub.n_samples == 3
        assert sub.match_ids == ["a", "c", "e"]

    def test_empty_bundle(self):
        b = DataBundle(
            score_features=np.zeros((0, 8)),
            shot_sequences=np.zeros((0, 10, 6)),
            sequence_lengths=np.zeros(0, dtype=np.int32),
            labels=np.zeros(0),
            match_ids=[],
        )
        assert b.n_samples == 0
        assert b.server_win_rate == 0.0
