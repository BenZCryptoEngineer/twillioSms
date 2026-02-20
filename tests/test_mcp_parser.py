"""Tests for MCP shot string parsing."""

from tennis_miner.data.loader_mcp import parse_shot_string, parse_rally, _tokenize_rally
from tennis_miner.data.schema import ShotType, ShotDirection, ShotOutcome, ShotDepth


def test_parse_basic_forehand():
    shot = parse_shot_string("f1", shot_num=1, player="server")
    assert shot.shot_type == ShotType.FOREHAND
    assert shot.direction == ShotDirection.WIDE_AD
    assert shot.outcome == ShotOutcome.IN_PLAY


def test_parse_winner():
    shot = parse_shot_string("b3*", shot_num=2, player="returner")
    assert shot.shot_type == ShotType.BACKHAND
    assert shot.direction == ShotDirection.CENTER_AD
    assert shot.outcome == ShotOutcome.WINNER


def test_parse_error():
    shot = parse_shot_string("f2@", shot_num=3, player="server")
    assert shot.shot_type == ShotType.FOREHAND
    assert shot.outcome == ShotOutcome.UNFORCED_ERROR


def test_parse_deep_shot():
    shot = parse_shot_string("f27", shot_num=1, player="server")
    assert shot.shot_type == ShotType.FOREHAND
    assert shot.direction == ShotDirection.BODY_AD
    assert shot.depth == ShotDepth.DEEP


def test_tokenize_rally():
    tokens = _tokenize_rally("f1b3f2*")
    assert len(tokens) == 3
    assert tokens[0] == "f1"
    assert tokens[1] == "b3"
    assert tokens[2] == "f2*"


def test_parse_rally():
    shots = parse_rally("f1b3f2*")
    assert len(shots) == 3
    assert shots[0].player == "server"
    assert shots[1].player == "returner"
    assert shots[2].player == "server"
    assert shots[2].outcome == ShotOutcome.WINNER


def test_parse_empty():
    shots = parse_rally("")
    assert shots == []

    shots = parse_rally(None)
    assert shots == []
