#!/usr/bin/env python3
"""
Tennis Miner — Per-Shot Win Rate Prediction System

CLI entry point for all phases.

Usage:
    # Phase 0: Acquire data
    python main.py acquire --source court_vision
    python main.py acquire --source mcp
    python main.py acquire --source slam_pbp
    python main.py acquire --source all

    # Phase 1: Train and evaluate
    python main.py train --phase 1

    # Phase 1: Evaluate Kill Point #1
    python main.py evaluate --kill-point 1
"""

import sys
import logging
import argparse
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("tennis_miner")


def cmd_acquire(args):
    """Run data acquisition scripts."""
    sources = [args.source] if args.source != "all" else ["court_vision", "mcp", "slam_pbp"]

    for source in sources:
        if source == "court_vision":
            log.info("Acquiring Court Vision data...")
            from scripts.acquire_court_vision import main as acquire_cv
            acquire_cv()

        elif source == "mcp":
            log.info("Acquiring Match Charting Project data...")
            from scripts.acquire_mcp import main as acquire_mcp
            acquire_mcp()

        elif source == "slam_pbp":
            log.info("Acquiring Grand Slam point-by-point data...")
            from scripts.acquire_slam_pbp import main as acquire_slam
            acquire_slam()

        else:
            log.error(f"Unknown source: {source}")
            sys.exit(1)


def cmd_train(args):
    """Run training pipeline for specified phase."""
    if args.phase == 1:
        log.info("Running Phase 1 pipeline...")
        from tennis_miner.pipeline import run_phase1
        result = run_phase1(args.config)

        kp = result.get("kill_point_1", {})
        if kp.get("OVERALL_PASS"):
            log.info("Phase 1 PASSED. Proceed to Phase 2A.")
        else:
            log.warning("Phase 1 FAILED Kill Point #1. See report for details.")
    else:
        log.error(f"Phase {args.phase} not yet implemented.")
        sys.exit(1)


def cmd_evaluate(args):
    """Run evaluation only (requires trained models)."""
    log.info(f"Evaluation for Kill Point #{args.kill_point}")

    if args.kill_point == 1:
        from tennis_miner.pipeline import run_phase1
        result = run_phase1(args.config)
        return result
    else:
        log.error(f"Kill Point #{args.kill_point} not yet implemented.")
        sys.exit(1)


def cmd_audit(args):
    """Audit downloaded data and print summary."""
    from tennis_miner.utils.config import load_config
    config = load_config(args.config)

    log.info("Data Inventory Audit")
    log.info("=" * 50)

    # Court Vision
    cv_dir = Path(config["paths"]["raw_data"]) / "court_vision"
    if cv_dir.exists():
        json_files = list(cv_dir.rglob("*.json"))
        json_files = [f for f in json_files if f.name != "download_summary.json"]
        log.info(f"Court Vision: {len(json_files)} match files")
        for tourney_dir in sorted(cv_dir.iterdir()):
            if tourney_dir.is_dir():
                year_dirs = sorted(tourney_dir.iterdir())
                for yd in year_dirs:
                    if yd.is_dir():
                        n = len(list(yd.glob("*.json")))
                        log.info(f"  {tourney_dir.name}/{yd.name}: {n} matches")
    else:
        log.info("Court Vision: NOT DOWNLOADED")

    # MCP
    mcp_dir = Path(config["data_acquisition"]["mcp"]["clone_dir"])
    if mcp_dir.exists():
        csv_files = list(mcp_dir.rglob("*.csv"))
        total_size = sum(f.stat().st_size for f in csv_files) / (1024 * 1024)
        log.info(f"MCP: {len(csv_files)} CSV files, {total_size:.1f} MB total")
    else:
        log.info("MCP: NOT CLONED")

    # Slam PBP
    pbp_dir = Path(config["data_acquisition"]["slam_pbp"]["clone_dir"])
    if pbp_dir.exists():
        csv_files = list(pbp_dir.rglob("*.csv"))
        total_size = sum(f.stat().st_size for f in csv_files) / (1024 * 1024)
        log.info(f"Slam PBP: {len(csv_files)} CSV files, {total_size:.1f} MB total")
    else:
        log.info("Slam PBP: NOT CLONED")


def main():
    parser = argparse.ArgumentParser(
        description="Tennis Miner — Per-Shot Win Rate Prediction System",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # acquire
    p_acq = sub.add_parser("acquire", help="Download data sources")
    p_acq.add_argument(
        "--source",
        choices=["court_vision", "mcp", "slam_pbp", "all"],
        default="all",
        help="Data source to acquire",
    )

    # train
    p_train = sub.add_parser("train", help="Run training pipeline")
    p_train.add_argument("--phase", type=int, default=1, choices=[1, 2], help="Phase number")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Run evaluation")
    p_eval.add_argument("--kill-point", type=int, default=1, choices=[1, 2, 3, 4])

    # audit
    sub.add_parser("audit", help="Audit downloaded data")

    args = parser.parse_args()

    if args.command == "acquire":
        cmd_acquire(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "audit":
        cmd_audit(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
