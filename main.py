#!/usr/bin/env python3
"""
Tennis Miner CLI.

Usage:
    python main.py acquire --source all
    python main.py train --phase 1
    python main.py audit
"""

import sys
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("tennis_miner")


def cmd_acquire(args):
    from tennis_miner.orchestration.config import load_config
    from tennis_miner.ingestion.base import clone_git_repo

    cfg = load_config(args.config)

    sources = [args.source] if args.source != "all" else ["court_vision", "mcp", "slam_pbp"]

    for src in sources:
        if src == "mcp":
            c = cfg["data_acquisition"]["mcp"]
            clone_git_repo(c["repo_url"], c["clone_dir"])
        elif src == "slam_pbp":
            c = cfg["data_acquisition"]["slam_pbp"]
            clone_git_repo(c["repo_url"], c["clone_dir"])
        elif src == "court_vision":
            log.info("Court Vision: run scripts/acquire_court_vision.py separately")


def cmd_train(args):
    if args.phase == 1:
        from tennis_miner.orchestration.pipeline import run_phase1
        result = run_phase1(args.config)
        kp = result.get("kill_point_1", {})
        if kp.get("OVERALL_PASS"):
            log.info("Phase 1 PASSED. Proceed to Phase 2A.")
        else:
            log.warning("Phase 1 FAILED Kill Point #1.")
    else:
        log.error(f"Phase {args.phase} not implemented yet.")
        sys.exit(1)


def cmd_audit(args):
    from tennis_miner.orchestration.config import load_config
    cfg = load_config(args.config)

    for name, key in [("MCP", "mcp"), ("Slam PBP", "slam_pbp")]:
        d = Path(cfg["data_acquisition"][key]["clone_dir"])
        if d.exists():
            csvs = list(d.rglob("*.csv"))
            sz = sum(f.stat().st_size for f in csvs) / (1024 * 1024)
            log.info(f"{name}: {len(csvs)} CSVs, {sz:.1f} MB")
        else:
            log.info(f"{name}: NOT DOWNLOADED")

    cv = Path(cfg["paths"]["raw_data"]) / "court_vision"
    if cv.exists():
        jsons = [f for f in cv.rglob("*.json") if f.name != "download_summary.json"]
        log.info(f"Court Vision: {len(jsons)} match files")
    else:
        log.info("Court Vision: NOT DOWNLOADED")


def main():
    p = argparse.ArgumentParser(description="Tennis Miner CLI")
    p.add_argument("--config", default="configs/default.yaml")
    sub = p.add_subparsers(dest="command")

    acq = sub.add_parser("acquire")
    acq.add_argument("--source", choices=["court_vision", "mcp", "slam_pbp", "all"], default="all")

    tr = sub.add_parser("train")
    tr.add_argument("--phase", type=int, default=1, choices=[1, 2])

    sub.add_parser("audit")

    args = p.parse_args()
    if args.command == "acquire":
        cmd_acquire(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "audit":
        cmd_audit(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
