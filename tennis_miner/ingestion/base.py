"""
Shared utilities for all data loaders.

Eliminates the duplicated _find_column / _find_val pattern.
"""

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find first matching column name from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def get_value(row: pd.Series, candidates: list[str]):
    """Get value from first matching column in a row."""
    for c in candidates:
        if c in row.index and pd.notna(row[c]):
            return row[c]
    return None


def safe_int(val, default: int = 0) -> int:
    """Convert value to int, returning default on failure."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def safe_float(val, default: float | None = None) -> float | None:
    """Convert value to float, returning default on failure."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def clone_git_repo(repo_url: str, target_dir: str, depth: int = 1) -> Path:
    """Clone a git repo if not already present."""
    import subprocess

    target = Path(target_dir)
    if target.exists() and (target / ".git").exists():
        log.info(f"Repo already exists: {target}")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Cloning {repo_url} → {target}")
    subprocess.run(
        ["git", "clone", "--depth", str(depth), repo_url, str(target)],
        check=True,
    )
    return target
