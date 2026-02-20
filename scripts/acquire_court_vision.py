#!/usr/bin/env python3
"""
Phase 0: Download all Court Vision (Hawk-Eye) tracking data.

Downloads match data from Australian Open (2020+) and Roland Garros (2019+)
via the courtvisionpython library. This data may become unavailable at any time.

Usage:
    python scripts/acquire_court_vision.py
    python scripts/acquire_court_vision.py --tournament australian_open --year 2023
"""

import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

try:
    from courtvisionpython import CourtVision
except ImportError:
    CourtVision = None

from tennis_miner.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

TOURNAMENT_IDS = {
    "australian_open": "ausopen",
    "roland_garros": "rolandgarros",
}


def download_tournament_year(
    cv: "CourtVision",
    tournament: str,
    year: int,
    output_dir: Path,
    rate_limit: float = 1.0,
    max_retries: int = 3,
):
    """Download all match data for a tournament/year combination."""
    tourney_key = TOURNAMENT_IDS.get(tournament, tournament)
    year_dir = output_dir / tournament / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Fetching match list: {tournament} {year}")

    try:
        matches = cv.get_matches(tournament=tourney_key, year=year)
    except Exception as e:
        log.error(f"Failed to get match list for {tournament} {year}: {e}")
        return {"tournament": tournament, "year": year, "matches": 0, "errors": 1}

    if not matches:
        log.warning(f"No matches found for {tournament} {year}")
        return {"tournament": tournament, "year": year, "matches": 0, "errors": 0}

    stats = {"tournament": tournament, "year": year, "matches": 0, "errors": 0}
    match_list = matches if isinstance(matches, list) else [matches]

    for match in match_list:
        match_id = match.get("matchId") or match.get("id") or str(match)
        match_file = year_dir / f"{match_id}.json"

        if match_file.exists():
            log.info(f"  Skipping {match_id} (already downloaded)")
            stats["matches"] += 1
            continue

        for attempt in range(max_retries):
            try:
                detail = cv.get_match_details(
                    tournament=tourney_key, year=year, match_id=match_id
                )
                # Also try to get rally/point-level data
                rally_data = None
                try:
                    rally_data = cv.get_rally_analysis(
                        tournament=tourney_key, year=year, match_id=match_id
                    )
                except Exception:
                    pass

                record = {
                    "match_id": match_id,
                    "tournament": tournament,
                    "year": year,
                    "match_info": match,
                    "match_details": detail,
                    "rally_analysis": rally_data,
                    "downloaded_at": datetime.utcnow().isoformat(),
                }

                with open(match_file, "w") as f:
                    json.dump(record, f, indent=2, default=str)

                stats["matches"] += 1
                log.info(f"  Downloaded {match_id}")
                break

            except Exception as e:
                wait = 2 ** (attempt + 1)
                log.warning(
                    f"  Attempt {attempt + 1}/{max_retries} failed for {match_id}: {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
        else:
            log.error(f"  Failed to download {match_id} after {max_retries} attempts")
            stats["errors"] += 1

        time.sleep(rate_limit)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Download Court Vision tracking data")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tournament", help="Single tournament to download")
    parser.add_argument("--year", type=int, help="Single year to download")
    args = parser.parse_args()

    if CourtVision is None:
        log.error(
            "courtvisionpython not installed. Run: pip install courtvisionpython"
        )
        return

    config = load_config(args.config)
    cv_config = config["data_acquisition"]["court_vision"]
    output_dir = Path(config["paths"]["raw_data"]) / "court_vision"
    output_dir.mkdir(parents=True, exist_ok=True)

    cv = CourtVision()

    # Determine what to download
    if args.tournament and args.year:
        jobs = [(args.tournament, args.year)]
    else:
        jobs = []
        for year in cv_config["ao_years"]:
            jobs.append(("australian_open", year))
        for year in cv_config["rg_years"]:
            jobs.append(("roland_garros", year))

    all_stats = []
    for tournament, year in jobs:
        stats = download_tournament_year(
            cv=cv,
            tournament=tournament,
            year=year,
            output_dir=output_dir,
            rate_limit=cv_config["rate_limit_delay"],
            max_retries=cv_config["max_retries"],
        )
        all_stats.append(stats)
        log.info(
            f"Completed {tournament} {year}: "
            f"{stats['matches']} matches, {stats['errors']} errors"
        )

    # Write summary
    summary = {
        "download_date": datetime.utcnow().isoformat(),
        "stats": all_stats,
        "total_matches": sum(s["matches"] for s in all_stats),
        "total_errors": sum(s["errors"] for s in all_stats),
    }
    summary_path = output_dir / "download_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(
        f"Download complete. Total: {summary['total_matches']} matches, "
        f"{summary['total_errors']} errors. Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()
