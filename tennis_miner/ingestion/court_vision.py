"""
Court Vision (Hawk-Eye) data loader.

Parses JSON files downloaded from the Infosys Court Vision API
into validated domain objects with spatial coordinates.
"""

import json
import logging
from pathlib import Path

from tennis_miner.core.schema import (
    Match, Point, Shot, Surface,
    ShotType, ShotOutcome, ShotDirection, ShotDepth,
)
from tennis_miner.core.interfaces import BaseLoader
from tennis_miner.ingestion.base import safe_int, safe_float

log = logging.getLogger(__name__)

SURFACE_BY_TOURNAMENT = {
    "australian_open": Surface.HARD,
    "roland_garros": Surface.CLAY,
}

SHOT_TYPE_KEYWORDS = {
    "forehand volley": ShotType.FOREHAND_VOLLEY,
    "backhand volley": ShotType.BACKHAND_VOLLEY,
    "forehand": ShotType.FOREHAND,
    "backhand": ShotType.BACKHAND,
    "overhead": ShotType.OVERHEAD,
    "smash": ShotType.OVERHEAD,
    "lob": ShotType.LOB,
    "drop shot": ShotType.DROP_SHOT,
    "drop": ShotType.DROP_SHOT,
    "second serve": ShotType.SECOND_SERVE,
    "serve": ShotType.FIRST_SERVE,
}

DIRECTION_KEYWORDS = {
    "down the line": ShotDirection.DOWN_THE_LINE,
    "dtl": ShotDirection.DOWN_THE_LINE,
    "crosscourt": ShotDirection.CROSSCOURT,
    "cross": ShotDirection.CROSSCOURT,
    "middle": ShotDirection.MIDDLE,
    "wide": ShotDirection.WIDE_AD,
    "body": ShotDirection.BODY_AD,
    "center": ShotDirection.CENTER_AD,
}


class CourtVisionLoader(BaseLoader):
    """Loads Court Vision JSON data."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load(self) -> list[Match]:
        matches = []
        json_files = sorted(self.data_dir.rglob("*.json"))
        json_files = [f for f in json_files if f.name != "download_summary.json"]

        log.info(f"Court Vision: found {len(json_files)} files")

        for fpath in json_files:
            try:
                with open(fpath) as f:
                    data = json.load(f)
                match = self._parse_match(data, fpath)
                if match and match.points:
                    matches.append(match)
            except Exception as e:
                log.warning(f"Failed to parse {fpath.name}: {e}")

        log.info(f"Court Vision: loaded {len(matches)} matches")
        return matches

    def validate(self, matches: list[Match]) -> list[str]:
        warnings = []
        for m in matches:
            spatial_points = sum(1 for p in m.points if p.has_spatial)
            if spatial_points == 0:
                warnings.append(f"{m.match_id}: no spatial data in any point")
            if m.n_points < 30:
                warnings.append(f"{m.match_id}: only {m.n_points} points (suspiciously few)")
        return warnings

    def _parse_match(self, data: dict, fpath: Path) -> Match | None:
        details = data.get("match_details", {})
        rally_data = data.get("rally_analysis")
        match_info = data.get("match_info", {})
        tournament = data.get("tournament", "")

        if not details and not rally_data:
            return None

        player1 = (
            _extract_player(details, "player1")
            or _extract_player(match_info, "player1")
            or "Player1"
        )
        player2 = (
            _extract_player(details, "player2")
            or _extract_player(match_info, "player2")
            or "Player2"
        )

        year = safe_int(data.get("year"), 2000)
        if year < 1900:
            year = 2000

        match = Match(
            match_id=f"cv_{data.get('match_id', fpath.stem)}",
            tournament=tournament,
            year=year,
            surface=SURFACE_BY_TOURNAMENT.get(tournament, Surface.HARD),
            player1=player1,
            player2=player2,
            winner=_extract_winner(details) or player1,
            score=str(details.get("score", "")),
            source="court_vision",
        )

        if rally_data:
            match.points = _parse_rally_analysis(rally_data, match)
        elif details:
            match.points = _parse_from_details(details, match)

        return match


# ── Helper functions ───────────────────────────────────────────────────

def _extract_player(data: dict, key: str) -> str | None:
    if key in data:
        val = data[key]
        if isinstance(val, dict):
            return val.get("name") or val.get("fullName") or val.get("shortName")
        return str(val) if val else None
    for nested in ("players", "matchInfo"):
        if nested in data and isinstance(data[nested], dict) and key in data[nested]:
            return str(data[nested][key])
    return None


def _extract_winner(details: dict) -> str | None:
    if "winner" in details:
        w = details["winner"]
        return w.get("name") if isinstance(w, dict) else str(w)
    return None


def _parse_rally_analysis(rally_data, match: Match) -> list[Point]:
    rallies = rally_data if isinstance(rally_data, list) else rally_data.get("rallies", [])
    points = []

    for i, rally in enumerate(rallies):
        set_num = max(safe_int(rally.get("setNo", rally.get("set")), 1), 1)
        game_num = max(safe_int(rally.get("gameNo", rally.get("game")), 1), 1)
        point_num = safe_int(rally.get("pointNo", rally.get("point")), i + 1)
        if point_num < 1:
            point_num = i + 1

        server_name = rally.get("server", match.player1)
        is_p1 = _is_player1(server_name, match.player1)
        server = match.player1 if is_p1 else match.player2
        returner = match.player2 if is_p1 else match.player1

        shots = _parse_shots(rally.get("shots", rally.get("strokes", [])))

        server_won = rally.get("serverWon", rally.get("isServerWinner"))
        if server_won is None:
            server_won = _infer_winner(shots)

        points.append(Point(
            point_id=f"{match.match_id}_S{set_num}G{game_num}P{point_num}",
            match_id=match.match_id,
            set_num=set_num,
            game_num=game_num,
            point_num=point_num,
            server=server,
            returner=returner,
            server_won=bool(server_won),
            shots=shots,
            rally_length=len(shots),
        ))

    return points


def _parse_shots(shots_data: list) -> list[Shot]:
    shots = []
    for i, s in enumerate(shots_data):
        player = "server" if i % 2 == 0 else "returner"
        type_str = str(s.get("shotType", s.get("type", ""))).lower()
        dir_str = str(s.get("direction", s.get("zone", ""))).lower()

        shots.append(Shot(
            shot_num=i + 1,
            player=player,
            shot_type=_map_shot_type(type_str, i),
            direction=_map_direction(dir_str),
            depth=ShotDepth.UNKNOWN,
            outcome=_map_outcome(s, i == len(shots_data) - 1),
            ball_landing_x=safe_float(s.get("landingX") or s.get("bounceX") or s.get("x")),
            ball_landing_y=safe_float(s.get("landingY") or s.get("bounceY") or s.get("y")),
            serve_speed_kmh=safe_float(s.get("speed") or s.get("serveSpeed")),
        ))
    return shots


def _parse_from_details(details: dict, match: Match) -> list[Point]:
    points = []
    sets_data = details.get("sets", details.get("setData", []))
    if not isinstance(sets_data, list):
        return points

    counter = 0
    for s_idx, sd in enumerate(sets_data):
        games = sd.get("games", []) if isinstance(sd, dict) else []
        for g_idx, gd in enumerate(games):
            for p_idx, pt in enumerate(gd.get("points", []) if isinstance(gd, dict) else []):
                counter += 1
                points.append(Point(
                    point_id=f"{match.match_id}_p{counter}",
                    match_id=match.match_id,
                    set_num=s_idx + 1,
                    game_num=g_idx + 1,
                    point_num=p_idx + 1,
                    server=match.player1,
                    returner=match.player2,
                    server_won=bool(pt.get("serverWon", True)),
                    rally_length=safe_int(pt.get("rallyLength"), 0),
                ))
    return points


def _map_shot_type(type_str: str, shot_idx: int) -> ShotType:
    if shot_idx == 0:
        return ShotType.FIRST_SERVE
    for kw, st in SHOT_TYPE_KEYWORDS.items():
        if kw in type_str:
            return st
    return ShotType.UNKNOWN


def _map_direction(dir_str: str) -> ShotDirection:
    for kw, d in DIRECTION_KEYWORDS.items():
        if kw in dir_str:
            return d
    return ShotDirection.UNKNOWN


def _map_outcome(shot_data: dict, is_last: bool) -> ShotOutcome:
    o = str(shot_data.get("outcome", shot_data.get("result", ""))).lower()
    if "ace" in o:
        return ShotOutcome.ACE
    if "winner" in o:
        return ShotOutcome.WINNER
    if "unforced" in o or "ue" in o:
        return ShotOutcome.UNFORCED_ERROR
    if "forced" in o or "fe" in o:
        return ShotOutcome.FORCED_ERROR
    if "double" in o or "df" in o:
        return ShotOutcome.DOUBLE_FAULT
    if is_last:
        return ShotOutcome.WINNER
    return ShotOutcome.IN_PLAY


def _is_player1(name: str, player1: str) -> bool:
    if not name or not player1:
        return True
    n, p = name.lower().strip(), player1.lower().strip()
    return n in p or p in n


def _infer_winner(shots: list[Shot]) -> bool:
    if not shots:
        return True
    last = shots[-1]
    if last.outcome in (ShotOutcome.WINNER, ShotOutcome.ACE):
        return last.player == "server"
    if last.outcome in (ShotOutcome.UNFORCED_ERROR, ShotOutcome.FORCED_ERROR):
        return last.player != "server"
    return True
