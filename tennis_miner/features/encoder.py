"""
Categorical encoding registry.

Centralized enum→int mappings so every module uses the same encoding.
Avoids the old bug where ordinal encoding mixed with continuous features.
"""

from tennis_miner.core.schema import (
    Surface, ShotType, ShotDirection, ShotDepth, ShotOutcome,
)

# Stable integer mappings (order matters — do not change after training)
SURFACE_ENC = {s: i for i, s in enumerate(Surface)}
SHOT_TYPE_ENC = {s: i for i, s in enumerate(ShotType)}
DIRECTION_ENC = {s: i for i, s in enumerate(ShotDirection)}
DEPTH_ENC = {s: i for i, s in enumerate(ShotDepth)}
OUTCOME_ENC = {s: i for i, s in enumerate(ShotOutcome)}

# Vocabulary sizes (for embedding layers)
N_SURFACES = len(Surface)
N_SHOT_TYPES = len(ShotType)
N_DIRECTIONS = len(ShotDirection)
N_DEPTHS = len(ShotDepth)
N_OUTCOMES = len(ShotOutcome)
