"""Config loading with validation."""

import yaml
from pathlib import Path


def load_config(path: str = "configs/default.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(p) as f:
        cfg = yaml.safe_load(f)

    # Validate required keys
    for key in ("paths", "data_acquisition", "phase1"):
        if key not in cfg:
            raise ValueError(f"Missing config key: {key}")
    return cfg
