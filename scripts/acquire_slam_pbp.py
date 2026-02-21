#!/usr/bin/env python3
"""
Phase 0: Clone and audit the Grand Slam Point-by-Point dataset.

Contains serve speed, rally length, ace/DF/winner/UE flags per point
for all 4 Grand Slams from 2011-present.

Usage:
    python scripts/acquire_slam_pbp.py
"""

import subprocess
import logging
import argparse
from pathlib import Path

import pandas as pd

from tennis_miner.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def clone_repo(repo_url: str, clone_dir: str) -> Path:
    """Clone a git repository if it doesn't already exist."""
    target = Path(clone_dir)
    if target.exists() and (target / ".git").exists():
        log.info(f"Repository already cloned: {target}")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Cloning {repo_url} -> {target}")
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(target)],
        check=True,
    )
    return target


def audit_slam_pbp(pbp_dir: Path):
    """Audit the Grand Slam point-by-point dataset."""
    log.info("Auditing Grand Slam point-by-point data...")

    csv_files = sorted(pbp_dir.rglob("*.csv"))
    log.info(f"Found {len(csv_files)} CSV files:")

    total_rows = 0
    for f in csv_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        if size_mb < 0.01:
            continue
        try:
            df = pd.read_csv(f, low_memory=False)
            total_rows += len(df)
            log.info(f"  {f.name}: {len(df):,} rows, {size_mb:.1f} MB")
            if len(df) > 0:
                log.info(f"    Columns: {list(df.columns)}")
        except Exception as e:
            log.warning(f"  Could not read {f.name}: {e}")

    log.info(f"Total rows across all files: {total_rows:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Clone and audit Grand Slam point-by-point data"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--skip-clone", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    pbp_config = config["data_acquisition"]["slam_pbp"]

    if not args.skip_clone:
        pbp_dir = clone_repo(pbp_config["repo_url"], pbp_config["clone_dir"])
    else:
        pbp_dir = Path(pbp_config["clone_dir"])

    if not args.skip_audit:
        audit_slam_pbp(pbp_dir)


if __name__ == "__main__":
    main()
