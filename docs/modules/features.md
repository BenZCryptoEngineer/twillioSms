# features/ — Feature Engineering

## Purpose

Converts validated domain objects (`list[Match]`) into numeric arrays (`DataBundle`) ready for model training.

## Files

| File | Responsibility |
|------|---------------|
| `encoder.py` | Centralized enum → int mappings for all categoricals |
| `score.py` | Score-state feature extraction (8-dim per point) |
| `sequence.py` | Shot-sequence feature extraction (6-dim per shot) + `build_dataset` |
| `spatial.py` | Phase 2 spatial feature extraction (7-dim per shot) |

## Encoder Registry (encoder.py)

All categorical encoding is centralized here to prevent the ordinal-encoding bugs from v0.1:

```python
SURFACE_ENC   = {Surface.HARD: 0, Surface.CLAY: 1, ...}
SHOT_TYPE_ENC = {ShotType.FIRST_SERVE: 0, ...}
DIRECTION_ENC = {ShotDirection.WIDE_AD: 0, ...}
DEPTH_ENC     = {ShotDepth.DEEP: 0, ...}
OUTCOME_ENC   = {ShotOutcome.IN_PLAY: 0, ...}
```

Also exports vocabulary sizes (`N_SURFACES`, `N_SHOT_TYPES`, etc.) for embedding layers.

**Rule**: Encoding order is fixed after first training. Do not reorder enums.

## Score Features (score.py)

`extract_score_features(point, match)` → `np.ndarray` shape (8,):

| Index | Feature | Type |
|-------|---------|------|
| 0 | set_num | int |
| 1 | game_num | int |
| 2 | point_num | int |
| 3 | is_server_p1 | binary |
| 4 | surface | encoded int |
| 5 | server_sets_won | int |
| 6 | returner_sets_won | int |
| 7 | is_tiebreak | binary |

Constant: `SCORE_FEATURE_DIM = 8`

## Shot Sequence Features (sequence.py)

`extract_shot_features(shot)` → `np.ndarray` shape (6,):

| Index | Feature | Type |
|-------|---------|------|
| 0 | is_server | binary |
| 1 | shot_type | encoded int |
| 2 | direction | encoded int |
| 3 | depth | encoded int |
| 4 | outcome | encoded int |
| 5 | shot_num_norm | float (shot_num/50, capped at 1.0) |

Constant: `SHOT_FEATURE_DIM = 6`

`extract_shot_sequence(point, max_length)` → `np.ndarray` shape (max_length, 6):
- Zero-padded after actual rally length
- Truncated if rally exceeds max_length

### build_dataset

The central pipeline function:

```python
build_dataset(matches, max_rally_length=50, require_shots=True) -> DataBundle
```

- Iterates all points across all matches
- `require_shots=True` (default): skip points without shot data
- `require_shots=False`: include all (shots will be zero-padded)
- Returns empty DataBundle with correct shapes when no data

## Spatial Features (spatial.py) — Phase 2

`extract_spatial_features(shot)` → `np.ndarray` shape (7,):

| Index | Feature | Normalization |
|-------|---------|--------------|
| 0 | ball_landing_x | raw (court coords) |
| 1 | ball_landing_y | raw (court coords) |
| 2 | serve_speed_kmh | / 250.0 |
| 3 | player_x | raw |
| 4 | player_y | raw |
| 5 | opponent_x | raw |
| 6 | opponent_y | raw |

`spatial_mask(shot)` → `np.ndarray` shape (7,):
- 1.0 where spatial data exists, 0.0 where None
- Used to mask loss for partially-observed spatial data

## Data Flow

```
list[Match]
    ↓
build_dataset()
    ├── extract_score_features()  → (N, 8)
    ├── extract_shot_sequence()   → (N, max_rally, 6)
    └── labels + match_ids
    ↓
DataBundle
```

## Dependencies

- Imports from: `core` only
- Imported by: `orchestration`
- External: `numpy`
