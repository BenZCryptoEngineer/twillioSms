"""
Grand Slam Point-by-Point data loader (Jeff Sackmann).

No shot sequences — supplements MCP with point-level features.
"""

import logging
from pathlib import Path

import pandas as pd

from tennis_miner.core.schema import Match, Point, Surface
from tennis_miner.core.interfaces import BaseLoader
from tennis_miner.ingestion.base import find_column, get_value, safe_int

log = logging.getLogger(__name__)

SLAM_SURFACE = {
    "ausopen": Surface.HARD,
    "frenchopen": Surface.CLAY,
    "wimbledon": Surface.GRASS,
    "usopen": Surface.HARD,
}


class SlamPBPLoader(BaseLoader):
    """Loads Grand Slam point-by-point data."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load(self) -> list[Match]:
        matches = []
        csv_files = sorted(self.data_dir.rglob("*.csv"))
        point_files = [f for f in csv_files if "points" in f.name.lower()]

        if not point_files:
            log.warning(f"No PBP CSV files in {self.data_dir}")
            return matches

        for fpath in point_files:
            try:
                matches.extend(self._load_file(fpath))
            except Exception as e:
                log.warning(f"Failed to load {fpath.name}: {e}")

        log.info(f"Slam PBP: loaded {len(matches)} matches")
        return matches

    def validate(self, matches: list[Match]) -> list[str]:
        warnings = []
        for m in matches:
            if m.n_points == 0:
                warnings.append(f"{m.match_id}: no points")
            if m.n_points > 500:
                warnings.append(f"{m.match_id}: {m.n_points} points (unusually many)")
        return warnings

    def _load_file(self, fpath: Path) -> list[Match]:
        df = pd.read_csv(fpath, low_memory=False)
        matches = []

        slam = "unknown"
        for key in SLAM_SURFACE:
            if key in fpath.name.lower() or key in str(fpath.parent).lower():
                slam = key
                break

        surface = SLAM_SURFACE.get(slam, Surface.HARD)
        match_col = find_column(df, ["match_id", "Match_Id", "matchId"])
        if match_col is None:
            return matches

        for match_id, group in df.groupby(match_col):
            first = group.iloc[0]
            p1 = str(get_value(first, ["player1", "Player1", "server1"]) or "Player1")
            p2 = str(get_value(first, ["player2", "Player2", "server2"]) or "Player2")

            year = safe_int(get_value(first, ["year", "Year"]), 2000)
            if year < 1900:
                year = 2000

            match = Match(
                match_id=f"pbp_{match_id}",
                tournament=slam,
                year=year,
                surface=surface,
                player1=p1,
                player2=p2,
                winner=p1,
                score="",
                source="slam_pbp",
            )

            points = []
            for _, row in group.iterrows():
                set_num = max(safe_int(get_value(row, ["SetNo", "set_no", "Set"]), 1), 1)
                game_num = max(safe_int(get_value(row, ["GameNo", "game_no", "Game"]), 1), 1)
                point_num = safe_int(
                    get_value(row, ["PointNo", "point_no", "Point"]),
                    len(points) + 1,
                )
                if point_num < 1:
                    point_num = len(points) + 1

                server_id = get_value(row, ["PointServer", "Svr", "server"])
                is_p1_serving = str(server_id) in ("1", "1.0")
                server = p1 if is_p1_serving else p2
                returner = p2 if is_p1_serving else p1

                winner_val = get_value(row, ["PointWinner", "PtWinner", "isSvrWinner"])
                if winner_val is not None:
                    pw = str(winner_val)
                    if pw in ("1", "1.0"):
                        server_won = is_p1_serving
                    elif pw in ("2", "2.0"):
                        server_won = not is_p1_serving
                    else:
                        server_won = True
                else:
                    server_won = True

                rally_len = safe_int(
                    get_value(row, ["RallyCount", "rally_count", "Rally"]), 0
                )

                point = Point(
                    point_id=f"{match.match_id}_S{set_num}G{game_num}P{point_num}",
                    match_id=match.match_id,
                    set_num=set_num,
                    game_num=game_num,
                    point_num=point_num,
                    server=server,
                    returner=returner,
                    server_won=server_won,
                    rally_length=rally_len,
                )

                # Score context
                p1s = get_value(row, ["P1Score", "p1_score"])
                p2s = get_value(row, ["P2Score", "p2_score"])
                if p1s is not None and p2s is not None:
                    point.server_score = str(p1s if is_p1_serving else p2s)
                    point.returner_score = str(p2s if is_p1_serving else p1s)

                p1sets = get_value(row, ["P1SetsWon", "p1_sets"])
                p2sets = get_value(row, ["P2SetsWon", "p2_sets"])
                if p1sets is not None and p2sets is not None:
                    point.server_sets_won = safe_int(p1sets if is_p1_serving else p2sets)
                    point.returner_sets_won = safe_int(p2sets if is_p1_serving else p1sets)

                points.append(point)

            match.points = points
            if points:
                matches.append(match)

        return matches
