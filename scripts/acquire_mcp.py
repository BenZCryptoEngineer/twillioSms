#!/usr/bin/env python3
"""
Phase 0: Clone and audit the Match Charting Project (MCP) dataset.

The MCP contains 17,179 matches, 10.15M shots, 2.68M points with
shot type, direction (categorical), depth, error type, and point outcome.

Usage:
    python scripts/acquire_mcp.py
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


def audit_mcp(mcp_dir: Path):
    """Audit the MCP dataset and print summary statistics."""
    log.info("Auditing Match Charting Project data...")

    csv_files = sorted(mcp_dir.glob("*.csv")) + sorted(
        mcp_dir.glob("charting-*.csv")
    )
    if not csv_files:
        csv_files = sorted(mcp_dir.rglob("*.csv"))

    log.info(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        log.info(f"  {f.name}: {size_mb:.1f} MB")

        # Quick preview
        try:
            df = pd.read_csv(f, nrows=5, low_memory=False)
            log.info(f"    Columns: {list(df.columns)}")
            full_df = pd.read_csv(f, low_memory=False)
            log.info(f"    Rows: {len(full_df):,}")
        except Exception as e:
            log.warning(f"    Could not read {f.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Clone and audit MCP dataset")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--skip-clone", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    mcp_config = config["data_acquisition"]["mcp"]

    if not args.skip_clone:
        mcp_dir = clone_repo(mcp_config["repo_url"], mcp_config["clone_dir"])
    else:
        mcp_dir = Path(mcp_config["clone_dir"])

    if not args.skip_audit:
        audit_mcp(mcp_dir)


if __name__ == "__main__":
    main()
