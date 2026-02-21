"""
Tennis Miner domain objects.

All data sources normalize into these types. Every module in the project
depends on this file; this file depends on nothing else.

Validation rules are enforced at construction time via __post_init__.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ── Enums ──────────────────────────────────────────────────────────────

class Surface(Enum):
    HARD = "hard"
    CLAY = "clay"
    GRASS = "grass"
    CARPET = "carpet"


class ShotType(Enum):
    FIRST_SERVE = "1st_serve"
    SECOND_SERVE = "2nd_serve"
    FOREHAND = "forehand"
    BACKHAND = "backhand"
    FOREHAND_VOLLEY = "fh_volley"
    BACKHAND_VOLLEY = "bh_volley"
    OVERHEAD = "overhead"
    LOB = "lob"
    DROP_SHOT = "drop_shot"
    HALF_VOLLEY = "half_volley"
    SWINGING_VOLLEY = "swing_volley"
    TRICK_SHOT = "trick"
    UNKNOWN = "unknown"


class ShotDirection(Enum):
    WIDE_AD = 1
    BODY_AD = 2
    CENTER_AD = 3
    CENTER_DEUCE = 4
    BODY_DEUCE = 5
    WIDE_DEUCE = 6
    DOWN_THE_LINE = 7
    CROSSCOURT = 8
    MIDDLE = 9
    UNKNOWN = 0


class ShotOutcome(Enum):
    IN_PLAY = "in_play"
    WINNER = "winner"
    FORCED_ERROR = "forced_error"
    UNFORCED_ERROR = "unforced_error"
    ACE = "ace"
    DOUBLE_FAULT = "double_fault"
    LET = "let"
    UNKNOWN = "unknown"


class ShotDepth(Enum):
    DEEP = "deep"
    MIDDLE = "middle"
    SHORT = "short"
    UNKNOWN = "unknown"


# ── Domain Objects ─────────────────────────────────────────────────────

@dataclass
class Shot:
    """A single shot in a rally."""
    shot_num: int
    player: str  # "server" or "returner"
    shot_type: ShotType = ShotType.UNKNOWN
    direction: ShotDirection = ShotDirection.UNKNOWN
    depth: ShotDepth = ShotDepth.UNKNOWN
    outcome: ShotOutcome = ShotOutcome.IN_PLAY

    # Phase 2+ spatial fields
    ball_landing_x: Optional[float] = None
    ball_landing_y: Optional[float] = None
    serve_speed_kmh: Optional[float] = None
    player_x: Optional[float] = None
    player_y: Optional[float] = None
    opponent_x: Optional[float] = None
    opponent_y: Optional[float] = None

    def __post_init__(self):
        if self.shot_num < 1:
            raise ValueError(f"shot_num must be >= 1, got {self.shot_num}")
        if self.player not in ("server", "returner"):
            raise ValueError(f"player must be 'server' or 'returner', got '{self.player}'")
        if self.serve_speed_kmh is not None and self.serve_speed_kmh < 0:
            raise ValueError(f"serve_speed_kmh must be >= 0, got {self.serve_speed_kmh}")

    @property
    def has_spatial(self) -> bool:
        return self.ball_landing_x is not None and self.ball_landing_y is not None

    @property
    def is_point_ending(self) -> bool:
        return self.outcome in (
            ShotOutcome.WINNER, ShotOutcome.ACE,
            ShotOutcome.FORCED_ERROR, ShotOutcome.UNFORCED_ERROR,
            ShotOutcome.DOUBLE_FAULT,
        )


@dataclass
class Point:
    """A single point containing a rally of shots."""
    point_id: str
    match_id: str
    set_num: int
    game_num: int
    point_num: int
    server: str
    returner: str
    server_won: bool
    shots: list[Shot] = field(default_factory=list)
    rally_length: int = 0

    # Score context
    server_score: str = "0"
    returner_score: str = "0"
    server_sets_won: int = 0
    returner_sets_won: int = 0
    server_games_won: int = 0
    returner_games_won: int = 0
    is_tiebreak: bool = False
    is_break_point: bool = False

    def __post_init__(self):
        if self.set_num < 1:
            raise ValueError(f"set_num must be >= 1, got {self.set_num}")
        if self.game_num < 1:
            raise ValueError(f"game_num must be >= 1, got {self.game_num}")
        if not self.server:
            raise ValueError("server name cannot be empty")
        if not self.returner:
            raise ValueError("returner name cannot be empty")
        if self.rally_length == 0 and self.shots:
            self.rally_length = len(self.shots)

    @property
    def has_shots(self) -> bool:
        return len(self.shots) > 0

    @property
    def has_spatial(self) -> bool:
        return self.has_shots and any(s.has_spatial for s in self.shots)


@dataclass
class Match:
    """A complete tennis match."""
    match_id: str
    tournament: str
    year: int
    surface: Surface
    player1: str
    player2: str
    winner: str
    score: str
    points: list[Point] = field(default_factory=list)
    source: str = ""
    match_duration_mins: Optional[float] = None

    def __post_init__(self):
        if not self.match_id:
            raise ValueError("match_id cannot be empty")
        if self.year < 1900 or self.year > 2100:
            raise ValueError(f"year out of range: {self.year}")
        if not self.player1 or not self.player2:
            raise ValueError("player names cannot be empty")

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def n_points_with_shots(self) -> int:
        return sum(1 for p in self.points if p.has_shots)

    @property
    def has_spatial(self) -> bool:
        return any(p.has_spatial for p in self.points)


# ── Data Transfer Objects ──────────────────────────────────────────────

@dataclass
class DataBundle:
    """Standardized container for model-ready data.

    This is the contract between features/ and models/.
    Every field is a numpy array with matching first dimension (N samples).
    """
    score_features: np.ndarray      # (N, score_dim)
    shot_sequences: np.ndarray      # (N, max_rally_len, shot_dim)
    sequence_lengths: np.ndarray    # (N,)
    labels: np.ndarray              # (N,) — 1.0 = server won
    match_ids: list[str]            # (N,) — for stratified splitting

    def __post_init__(self):
        n = len(self.labels)
        if self.score_features.shape[0] != n:
            raise ValueError(
                f"score_features has {self.score_features.shape[0]} rows, "
                f"expected {n}"
            )
        if self.shot_sequences.shape[0] != n:
            raise ValueError(
                f"shot_sequences has {self.shot_sequences.shape[0]} rows, "
                f"expected {n}"
            )
        if len(self.match_ids) != n:
            raise ValueError(
                f"match_ids has {len(self.match_ids)} entries, expected {n}"
            )

    @property
    def n_samples(self) -> int:
        return len(self.labels)

    @property
    def server_win_rate(self) -> float:
        return float(self.labels.mean()) if self.n_samples > 0 else 0.0

    def subset(self, mask: np.ndarray) -> "DataBundle":
        """Return a new DataBundle filtered by boolean mask."""
        ids = [m for m, keep in zip(self.match_ids, mask) if keep]
        return DataBundle(
            score_features=self.score_features[mask],
            shot_sequences=self.shot_sequences[mask],
            sequence_lengths=self.sequence_lengths[mask],
            labels=self.labels[mask],
            match_ids=ids,
        )
