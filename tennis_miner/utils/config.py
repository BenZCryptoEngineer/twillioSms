"""Configuration loading utilities."""

import yaml
from pathlib import Path


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)
