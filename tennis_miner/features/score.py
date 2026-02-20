"""Score-state feature extraction for baseline model."""

import numpy as np

from tennis_miner.core.schema import Point, Match
from tennis_miner.features.encoder import SURFACE_ENC

SCORE_FEATURE_DIM = 8


def extract_score_features(point: Point, match: Match) -> np.ndarray:
    """Extract 8-dim score-state vector.

    [set_num, game_num, point_num, is_server_p1, surface,
     server_sets_won, returner_sets_won, is_tiebreak]
    """
    return np.array([
        point.set_num,
        point.game_num,
        point.point_num,
        1.0 if point.server == match.player1 else 0.0,
        SURFACE_ENC.get(match.surface, 0),
        point.server_sets_won,
        point.returner_sets_won,
        1.0 if point.is_tiebreak else 0.0,
    ], dtype=np.float32)
