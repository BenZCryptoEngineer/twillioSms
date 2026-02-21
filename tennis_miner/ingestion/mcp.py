"""
Match Charting Project (MCP) data loader.

Parses MCP's shot-by-shot encoding into validated domain objects.

MCP shot encoding:
  Type:      f=FH, b=BH, r=FH-slice, s=BH-slice, v=FH-volley, z=BH-volley,
             o=overhead, u=drop, l=lob, p=half-volley, j=FH-swing, k=BH-swing
  Direction: 1-6 zone mapping, 7=deep, 8=mid, 9=short
  Outcome:   *=winner, #=forced error, @=unforced error
"""

import logging
from pathlib import Path

import pandas as pd

from tennis_miner.core.schema import (
    Match, Point, Shot, Surface,
    ShotType, ShotDirection, ShotOutcome, ShotDepth,
)
from tennis_miner.core.interfaces import BaseLoader
from tennis_miner.ingestion.base import find_column, get_value, safe_int

log = logging.getLogger(__name__)

SHOT_TYPE_MAP = {
    "f": ShotType.FOREHAND,
    "b": ShotType.BACKHAND,
    "r": ShotType.FOREHAND,
    "s": ShotType.BACKHAND,
    "v": ShotType.FOREHAND_VOLLEY,
    "z": ShotType.BACKHAND_VOLLEY,
    "o": ShotType.OVERHEAD,
    "u": ShotType.DROP_SHOT,
    "l": ShotType.LOB,
    "p": ShotType.HALF_VOLLEY,
    "j": ShotType.SWINGING_VOLLEY,
    "k": ShotType.SWINGING_VOLLEY,
    "t": ShotType.TRICK_SHOT,
}

SURFACE_MAP = {
    "Hard": Surface.HARD, "hard": Surface.HARD,
    "Clay": Surface.CLAY, "clay": Surface.CLAY,
    "Grass": Surface.GRASS, "grass": Surface.GRASS,
    "Carpet": Surface.CARPET, "carpet": Surface.CARPET,
}


# ── Shot Parsing ───────────────────────────────────────────────────────

def tokenize_rally(rally_str: str) -> list[str]:
    """Split continuous MCP rally string into shot tokens.

    Each shot starts with a letter (type) followed by digits/modifiers.
    Example: "f1b3f2*" → ["f1", "b3", "f2*"]
    """
    if not rally_str or not isinstance(rally_str, str):
        return []

    tokens = []
    current = ""
    for ch in rally_str.strip():
        if ch.isalpha() and current:
            tokens.append(current)
            current = ch
        else:
            current += ch
    if current:
        tokens.append(current)
    return tokens


def parse_shot_string(shot_str: str, shot_num: int, player: str) -> Shot:
    """Parse one MCP shot token (e.g. "f27*") into a Shot object."""
    if not shot_str or not isinstance(shot_str, str):
        return Shot(shot_num=shot_num, player=player)

    shot_str = shot_str.strip()
    if not shot_str:
        return Shot(shot_num=shot_num, player=player)

    # Outcome (trailing marker)
    outcome = ShotOutcome.IN_PLAY
    if shot_str.endswith("*"):
        outcome = ShotOutcome.WINNER
        shot_str = shot_str[:-1]
    elif shot_str.endswith("#"):
        outcome = ShotOutcome.FORCED_ERROR
        shot_str = shot_str[:-1]
    elif shot_str.endswith("@"):
        outcome = ShotOutcome.UNFORCED_ERROR
        shot_str = shot_str[:-1]

    if not shot_str:
        return Shot(shot_num=shot_num, player=player, outcome=outcome)

    # Type (first char)
    shot_type = SHOT_TYPE_MAP.get(shot_str[0].lower(), ShotType.UNKNOWN)

    # Direction + depth (remaining digits)
    direction = ShotDirection.UNKNOWN
    depth = ShotDepth.UNKNOWN
    for ch in shot_str[1:]:
        if not ch.isdigit():
            continue
        d = int(ch)
        if 1 <= d <= 6 and direction == ShotDirection.UNKNOWN:
            try:
                direction = ShotDirection(d)
            except ValueError:
                pass
        elif d == 7:
            depth = ShotDepth.DEEP
        elif d == 8:
            depth = ShotDepth.MIDDLE
        elif d == 9:
            depth = ShotDepth.SHORT

    return Shot(
        shot_num=shot_num,
        player=player,
        shot_type=shot_type,
        direction=direction,
        depth=depth,
        outcome=outcome,
    )


def parse_rally(rally_str: str) -> list[Shot]:
    """Parse full MCP rally string into alternating server/returner shots."""
    tokens = tokenize_rally(rally_str)
    shots = []
    for i, token in enumerate(tokens):
        player = "server" if i % 2 == 0 else "returner"
        shots.append(parse_shot_string(token, shot_num=i + 1, player=player))
    return shots


# ── Match Loading ──────────────────────────────────────────────────────

class MCPLoader(BaseLoader):
    """Loads Match Charting Project data."""

    def __init__(self, mcp_dir: str):
        self.mcp_dir = Path(mcp_dir)

    def load(self) -> list[Match]:
        matches = []
        for gender in ["m", "w"]:
            matches_file = self.mcp_dir / f"charting-{gender}-matches.csv"
            points_file = self.mcp_dir / f"charting-{gender}-points.csv"

            if not matches_file.exists() or not points_file.exists():
                log.warning(f"MCP {gender} files not found in {self.mcp_dir}")
                continue

            matches_df = pd.read_csv(matches_file, low_memory=False)
            points_df = pd.read_csv(points_file, low_memory=False)
            loaded = self._build_matches(matches_df, points_df, gender)
            matches.extend(loaded)
            log.info(f"MCP: loaded {len(loaded)} {gender} matches")

        log.info(f"MCP total: {len(matches)} matches")
        return matches

    def validate(self, matches: list[Match]) -> list[str]:
        warnings = []
        for m in matches:
            if m.n_points == 0:
                warnings.append(f"{m.match_id}: no points")
            if m.n_points_with_shots == 0:
                warnings.append(f"{m.match_id}: no shots in any point")
            shot_ratio = m.n_points_with_shots / max(m.n_points, 1)
            if shot_ratio < 0.5:
                warnings.append(
                    f"{m.match_id}: only {shot_ratio:.0%} of points have shot data"
                )
        return warnings

    def _build_matches(
        self, matches_df: pd.DataFrame, points_df: pd.DataFrame, gender: str,
    ) -> list[Match]:
        results = []
        match_id_col = find_column(points_df, ["match_id", "matchid", "Match Id"])
        if match_id_col is None:
            log.error("Cannot find match_id column in MCP points file")
            return results

        mid_col = find_column(matches_df, ["match_id", "matchid", "Match Id"])
        if mid_col is None:
            return results

        grouped = points_df.groupby(match_id_col)

        for _, row in matches_df.iterrows():
            match_id = str(row[mid_col])
            surface_str = str(row.get("Surface", row.get("surface", "Hard")))

            player1 = str(row.get("Player 1", row.get("player1", "Unknown")))
            player2 = str(row.get("Player 2", row.get("player2", "Unknown")))

            if not player1 or player1 == "nan":
                player1 = "Unknown1"
            if not player2 or player2 == "nan":
                player2 = "Unknown2"

            year_val = row.get("Year", row.get("year"))
            year = safe_int(year_val, 2000)
            if year < 1900:
                year = 2000

            match = Match(
                match_id=f"mcp_{gender}_{match_id}",
                tournament=str(row.get("Tournament", row.get("tournament", ""))),
                year=year,
                surface=SURFACE_MAP.get(surface_str, Surface.HARD),
                player1=player1,
                player2=player2,
                winner=player1,
                score="",
                source="mcp",
            )

            if match_id in grouped.groups:
                match.points = self._parse_points(
                    grouped.get_group(match_id), match
                )

            if match.points:
                results.append(match)

        return results

    def _parse_points(self, points_df: pd.DataFrame, match: Match) -> list[Point]:
        points = []
        for _, row in points_df.iterrows():
            set_num = safe_int(get_value(row, ["Set", "set", "SetNo"]), 1)
            game_num = safe_int(get_value(row, ["Game", "game", "Gm"]), 1)
            point_num = safe_int(
                get_value(row, ["Point", "point", "Pt"]), len(points) + 1
            )
            if set_num < 1:
                set_num = 1
            if game_num < 1:
                game_num = 1

            server_flag = str(get_value(row, ["Svr", "svr", "Server"]) or "1")
            server = match.player1 if server_flag in ("1", "1.0") else match.player2
            returner = match.player2 if server == match.player1 else match.player1

            rally_val = get_value(row, ["Rally", "rally", "1st", "2nd"])
            rally_str = str(rally_val) if rally_val is not None else ""
            shots = parse_rally(rally_str)

            pts_won = get_value(row, ["PtWinner", "isSvrWinner", "ServerWon"])
            if pts_won is not None:
                server_won = str(pts_won) in ("1", "1.0", "True", "true")
            else:
                server_won = _infer_server_won(shots)

            point = Point(
                point_id=f"{match.match_id}_S{set_num}G{game_num}P{point_num}",
                match_id=match.match_id,
                set_num=set_num,
                game_num=game_num,
                point_num=point_num,
                server=server,
                returner=returner,
                server_won=server_won,
                shots=shots,
                rally_length=len(shots),
            )

            score_val = get_value(row, ["Score", "score", "Pts"])
            if score_val is not None:
                parts = str(score_val).split("-")
                if len(parts) == 2:
                    point.server_score = parts[0].strip()
                    point.returner_score = parts[1].strip()

            points.append(point)

        return points


def _infer_server_won(shots: list[Shot]) -> bool:
    """Infer point winner from last shot outcome."""
    if not shots:
        return True
    last = shots[-1]
    if last.outcome in (ShotOutcome.WINNER, ShotOutcome.ACE):
        return last.player == "server"
    if last.outcome in (ShotOutcome.UNFORCED_ERROR, ShotOutcome.FORCED_ERROR, ShotOutcome.DOUBLE_FAULT):
        return last.player != "server"
    return True
