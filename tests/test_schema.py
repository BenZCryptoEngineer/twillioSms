"""Tests for data schema."""

from tennis_miner.data.schema import (
    Shot, Point, Match, Surface,
    ShotType, ShotDirection, ShotOutcome, ShotDepth,
)


def test_shot_creation():
    shot = Shot(shot_num=1, player="server", shot_type=ShotType.FOREHAND)
    assert shot.shot_num == 1
    assert shot.player == "server"
    assert shot.shot_type == ShotType.FOREHAND
    assert shot.ball_landing_x is None


def test_point_creation():
    shots = [
        Shot(shot_num=1, player="server", shot_type=ShotType.FIRST_SERVE),
        Shot(shot_num=2, player="returner", shot_type=ShotType.FOREHAND),
        Shot(shot_num=3, player="server", shot_type=ShotType.BACKHAND, outcome=ShotOutcome.WINNER),
    ]
    point = Point(
        point_id="test_001",
        match_id="match_001",
        set_num=1,
        game_num=1,
        point_num=1,
        server="Djokovic",
        returner="Nadal",
        server_won=True,
        shots=shots,
        rally_length=3,
    )
    assert len(point.shots) == 3
    assert point.server_won is True
    assert point.rally_length == 3


def test_match_creation():
    match = Match(
        match_id="test_match",
        tournament="australian_open",
        year=2023,
        surface=Surface.HARD,
        player1="Djokovic",
        player2="Nadal",
        winner="Djokovic",
        score="6-4 6-3",
        source="mcp",
    )
    assert match.surface == Surface.HARD
    assert match.source == "mcp"
