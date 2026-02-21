# core/ — Domain Schema & Interfaces

## Purpose

Defines the shared language of the entire project. Every other module imports `core`; `core` imports nothing.

## Files

| File | Responsibility |
|------|---------------|
| `schema.py` | Domain objects (Shot, Point, Match) + data transfer object (DataBundle) |
| `interfaces.py` | Abstract base classes (BaseLoader, BaseModel, BaseEvaluator) |

## Domain Objects

### Shot

A single stroke in a rally.

| Field | Type | Validation |
|-------|------|------------|
| `shot_num` | `int` | >= 1 |
| `player` | `str` | "server" or "returner" |
| `shot_type` | `ShotType` | Enum, default UNKNOWN |
| `direction` | `ShotDirection` | Enum, default UNKNOWN |
| `depth` | `ShotDepth` | Enum, default UNKNOWN |
| `outcome` | `ShotOutcome` | Enum, default IN_PLAY |
| `ball_landing_x/y` | `float?` | Phase 2 spatial (optional) |
| `serve_speed_kmh` | `float?` | >= 0 if present |
| `player_x/y` | `float?` | Phase 2 spatial (optional) |
| `opponent_x/y` | `float?` | Phase 2 spatial (optional) |

Properties:
- `has_spatial` — True only if both `ball_landing_x` and `ball_landing_y` are set
- `is_point_ending` — True for WINNER, ACE, FORCED_ERROR, UNFORCED_ERROR, DOUBLE_FAULT

### Point

A single point containing a rally of shots plus score context.

| Field | Type | Validation |
|-------|------|------------|
| `point_id` | `str` | Required |
| `match_id` | `str` | Required |
| `set_num` | `int` | >= 1 |
| `game_num` | `int` | >= 1 |
| `server` | `str` | Non-empty |
| `returner` | `str` | Non-empty |
| `server_won` | `bool` | Label |
| `shots` | `list[Shot]` | Default [] |
| `rally_length` | `int` | Auto-filled from shots if 0 |
| Score fields | `str/int/bool` | Server/returner score, sets, games, tiebreak, break point |

### Match

A complete tennis match.

| Field | Type | Validation |
|-------|------|------------|
| `match_id` | `str` | Non-empty |
| `year` | `int` | 1900-2100 |
| `player1/player2` | `str` | Non-empty |
| `surface` | `Surface` | Enum |
| `points` | `list[Point]` | Default [] |

Properties: `n_points`, `n_points_with_shots`, `has_spatial`

### DataBundle

Typed container between `features/` and `models/`. All arrays share the first dimension N.

| Field | Shape | Type |
|-------|-------|------|
| `score_features` | (N, 8) | float32 |
| `shot_sequences` | (N, max_rally, 6) | float32 |
| `sequence_lengths` | (N,) | int32 |
| `labels` | (N,) | float32 |
| `match_ids` | (N,) | list[str] |

Validates shape consistency in `__post_init__`. Supports `subset(mask)` for train/val/test splitting.

## Interfaces

### BaseLoader
```python
load() -> list[Match]
validate(matches) -> list[str]  # validation warnings
```

### BaseModel
```python
fit(train: DataBundle, val: DataBundle) -> dict
predict_proba(data: DataBundle) -> np.ndarray  # shape (N,)
save(path: str) -> None
load(path: str) -> None
```

### BaseEvaluator
```python
evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict
```

## Enums

| Enum | Values |
|------|--------|
| `Surface` | HARD, CLAY, GRASS, CARPET |
| `ShotType` | FIRST_SERVE, SECOND_SERVE, FOREHAND, BACKHAND, FH_VOLLEY, BH_VOLLEY, OVERHEAD, LOB, DROP_SHOT, HALF_VOLLEY, SWING_VOLLEY, TRICK, UNKNOWN |
| `ShotDirection` | WIDE_AD(1), BODY_AD(2), CENTER_AD(3), CENTER_DEUCE(4), BODY_DEUCE(5), WIDE_DEUCE(6), DTL(7), CROSSCOURT(8), MIDDLE(9), UNKNOWN(0) |
| `ShotDepth` | DEEP, MIDDLE, SHORT, UNKNOWN |
| `ShotOutcome` | IN_PLAY, WINNER, FORCED_ERROR, UNFORCED_ERROR, ACE, DOUBLE_FAULT, LET, UNKNOWN |

## Design Decisions

1. **Validation at construction** — `__post_init__` catches bad data immediately, not downstream
2. **No external imports** — `core` only uses stdlib + numpy (no pandas, no torch)
3. **DataBundle as contract** — typed alternative to raw dicts; enables safe splitting
4. **Enums for all categoricals** — prevents typos and enables exhaustive encoding
