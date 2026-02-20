"""Comprehensive tests for MCP shot parsing."""

import pytest
from tennis_miner.ingestion.mcp import (
    parse_shot_string, parse_rally, tokenize_rally, _infer_server_won,
)
from tennis_miner.core.schema import (
    Shot, ShotType, ShotDirection, ShotOutcome, ShotDepth,
)


class TestTokenizeRally:
    def test_basic(self):
        assert tokenize_rally("f1b3f2*") == ["f1", "b3", "f2*"]

    def test_single_shot(self):
        assert tokenize_rally("f1") == ["f1"]

    def test_empty(self):
        assert tokenize_rally("") == []

    def test_none(self):
        assert tokenize_rally(None) == []

    def test_whitespace(self):
        assert tokenize_rally("  f1b2  ") == ["f1", "b2"]

    def test_complex_rally(self):
        tokens = tokenize_rally("f1b3f27b4*")
        assert len(tokens) == 4
        assert tokens[2] == "f27"

    def test_all_outcomes(self):
        tokens = tokenize_rally("f1*b2#f3@")
        assert tokens == ["f1*", "b2#", "f3@"]

    def test_only_letters(self):
        """Each letter starts a new token."""
        tokens = tokenize_rally("fb")
        assert tokens == ["f", "b"]


class TestParseShotString:
    def test_forehand_zone1(self):
        s = parse_shot_string("f1", 1, "server")
        assert s.shot_type == ShotType.FOREHAND
        assert s.direction == ShotDirection.WIDE_AD

    def test_backhand_winner(self):
        s = parse_shot_string("b3*", 2, "returner")
        assert s.shot_type == ShotType.BACKHAND
        assert s.direction == ShotDirection.CENTER_AD
        assert s.outcome == ShotOutcome.WINNER

    def test_forced_error(self):
        s = parse_shot_string("f4#", 1, "server")
        assert s.outcome == ShotOutcome.FORCED_ERROR

    def test_unforced_error(self):
        s = parse_shot_string("b5@", 1, "returner")
        assert s.outcome == ShotOutcome.UNFORCED_ERROR
        assert s.direction == ShotDirection.BODY_DEUCE

    def test_deep_shot(self):
        s = parse_shot_string("f27", 1, "server")
        assert s.direction == ShotDirection.BODY_AD
        assert s.depth == ShotDepth.DEEP

    def test_mid_shot(self):
        s = parse_shot_string("f28", 1, "server")
        assert s.depth == ShotDepth.MIDDLE

    def test_short_shot(self):
        s = parse_shot_string("f29", 1, "server")
        assert s.depth == ShotDepth.SHORT

    def test_volley(self):
        s = parse_shot_string("v3", 1, "server")
        assert s.shot_type == ShotType.FOREHAND_VOLLEY

    def test_backhand_volley(self):
        s = parse_shot_string("z1", 1, "returner")
        assert s.shot_type == ShotType.BACKHAND_VOLLEY

    def test_overhead(self):
        s = parse_shot_string("o3*", 1, "server")
        assert s.shot_type == ShotType.OVERHEAD
        assert s.outcome == ShotOutcome.WINNER

    def test_lob(self):
        s = parse_shot_string("l2", 1, "returner")
        assert s.shot_type == ShotType.LOB

    def test_drop_shot(self):
        s = parse_shot_string("u1*", 1, "server")
        assert s.shot_type == ShotType.DROP_SHOT

    def test_empty_string(self):
        s = parse_shot_string("", 1, "server")
        assert s.shot_type == ShotType.UNKNOWN

    def test_none_input(self):
        s = parse_shot_string(None, 1, "server")
        assert s.shot_type == ShotType.UNKNOWN

    def test_unknown_type_char(self):
        s = parse_shot_string("x1", 1, "server")
        assert s.shot_type == ShotType.UNKNOWN
        assert s.direction == ShotDirection.WIDE_AD

    def test_outcome_only(self):
        """Shot string that is just an outcome marker after stripping."""
        s = parse_shot_string("f*", 1, "server")
        assert s.shot_type == ShotType.FOREHAND
        assert s.outcome == ShotOutcome.WINNER

    def test_direction_zone6(self):
        s = parse_shot_string("f6", 1, "server")
        assert s.direction == ShotDirection.WIDE_DEUCE

    def test_no_direction(self):
        s = parse_shot_string("f", 1, "server")
        assert s.direction == ShotDirection.UNKNOWN


class TestParseRally:
    def test_basic_3_shot_rally(self):
        shots = parse_rally("f1b3f2*")
        assert len(shots) == 3
        assert shots[0].player == "server"
        assert shots[1].player == "returner"
        assert shots[2].player == "server"
        assert shots[2].outcome == ShotOutcome.WINNER

    def test_single_ace(self):
        shots = parse_rally("f1*")
        assert len(shots) == 1
        assert shots[0].player == "server"
        assert shots[0].outcome == ShotOutcome.WINNER

    def test_long_rally(self):
        """10-shot rally."""
        rally = "f1b2f3b4f5b6f1b2f3b4@"
        shots = parse_rally(rally)
        assert len(shots) == 10
        assert shots[-1].outcome == ShotOutcome.UNFORCED_ERROR
        # Check alternation
        for i, s in enumerate(shots):
            assert s.player == ("server" if i % 2 == 0 else "returner")
            assert s.shot_num == i + 1

    def test_empty(self):
        assert parse_rally("") == []
        assert parse_rally(None) == []


class TestInferServerWon:
    def test_server_winner(self):
        shots = [
            Shot(shot_num=1, player="server", outcome=ShotOutcome.IN_PLAY),
            Shot(shot_num=2, player="returner", outcome=ShotOutcome.IN_PLAY),
            Shot(shot_num=3, player="server", outcome=ShotOutcome.WINNER),
        ]
        assert _infer_server_won(shots) is True

    def test_returner_winner(self):
        shots = [
            Shot(shot_num=1, player="server"),
            Shot(shot_num=2, player="returner", outcome=ShotOutcome.WINNER),
        ]
        assert _infer_server_won(shots) is False

    def test_server_error(self):
        shots = [
            Shot(shot_num=1, player="server", outcome=ShotOutcome.UNFORCED_ERROR),
        ]
        assert _infer_server_won(shots) is False

    def test_returner_error(self):
        shots = [
            Shot(shot_num=1, player="server"),
            Shot(shot_num=2, player="returner", outcome=ShotOutcome.UNFORCED_ERROR),
        ]
        assert _infer_server_won(shots) is True

    def test_ace(self):
        shots = [Shot(shot_num=1, player="server", outcome=ShotOutcome.ACE)]
        assert _infer_server_won(shots) is True

    def test_double_fault(self):
        shots = [Shot(shot_num=1, player="server", outcome=ShotOutcome.DOUBLE_FAULT)]
        assert _infer_server_won(shots) is False

    def test_empty_defaults_true(self):
        assert _infer_server_won([]) is True

    def test_in_play_defaults_true(self):
        shots = [Shot(shot_num=1, player="server", outcome=ShotOutcome.IN_PLAY)]
        assert _infer_server_won(shots) is True
