"""Phase 2 spatial feature extraction (ball + player positions)."""

import numpy as np

from tennis_miner.core.schema import Shot

SPATIAL_FEATURE_DIM = 7


def extract_spatial_features(shot: Shot) -> np.ndarray:
    """7-dim vector: [ball_x, ball_y, speed, player_x, player_y, opp_x, opp_y].

    Missing values → 0.0. Caller should use a separate mask if needed.
    """
    return np.array([
        shot.ball_landing_x or 0.0,
        shot.ball_landing_y or 0.0,
        (shot.serve_speed_kmh or 0.0) / 250.0,
        shot.player_x or 0.0,
        shot.player_y or 0.0,
        shot.opponent_x or 0.0,
        shot.opponent_y or 0.0,
    ], dtype=np.float32)


def spatial_mask(shot: Shot) -> np.ndarray:
    """7-dim binary mask: 1.0 where spatial data exists, 0.0 where missing."""
    return np.array([
        1.0 if shot.ball_landing_x is not None else 0.0,
        1.0 if shot.ball_landing_y is not None else 0.0,
        1.0 if shot.serve_speed_kmh is not None else 0.0,
        1.0 if shot.player_x is not None else 0.0,
        1.0 if shot.player_y is not None else 0.0,
        1.0 if shot.opponent_x is not None else 0.0,
        1.0 if shot.opponent_y is not None else 0.0,
    ], dtype=np.float32)
