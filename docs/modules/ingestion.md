# ingestion/ — Data Loaders & Validation

## Purpose

Converts raw data files (CSV, JSON) into validated `list[Match]` domain objects. Each data source has its own loader implementing `BaseLoader`.

## Files

| File | Responsibility |
|------|---------------|
| `base.py` | Shared utilities: `find_column`, `get_value`, `safe_int`, `safe_float`, `clone_git_repo` |
| `mcp.py` | Match Charting Project loader + shot-string parser |
| `court_vision.py` | Court Vision (Hawk-Eye) JSON loader with spatial coordinates |
| `slam_pbp.py` | Grand Slam point-by-point CSV loader |
| `validator.py` | Post-load quality checks (DataValidator) |

## Shared Utilities (base.py)

These eliminate the 3x duplicated `_find_column`/`_find_val` from v0.1:

| Function | Purpose |
|----------|---------|
| `find_column(df, candidates)` | Find first matching column name from a list of aliases |
| `get_value(row, candidates)` | Get value from first matching column in a Series |
| `safe_int(val, default)` | Cast to int without crashing on NaN/None |
| `safe_float(val, default)` | Cast to float without crashing on NaN/None |
| `clone_git_repo(url, target, depth)` | Shallow-clone a git repo if not already present |

## MCP Loader (mcp.py)

### Shot Encoding Format

MCP uses a compact string encoding for rallies:

```
"f27b3*" → forehand-zone2-deep, backhand-zone3-winner
```

| Char | Meaning |
|------|---------|
| f/b/r/s/v/z/o/u/l/p/j/k/t | Shot type (FH, BH, slice, volley, etc.) |
| 1-6 | Direction zone |
| 7/8/9 | Depth: deep/middle/short |
| * | Winner |
| # | Forced error |
| @ | Unforced error |

### Key Functions

- `tokenize_rally(str)` — Splits "f1b3f2*" → ["f1", "b3", "f2*"]
- `parse_shot_string(token, num, player)` — Single token → Shot object
- `parse_rally(str)` — Full string → alternating server/returner Shot list
- `_infer_server_won(shots)` — Infer point winner from last shot outcome when label is missing

### MCPLoader Class

- Reads `charting-{m,w}-matches.csv` and `charting-{m,w}-points.csv`
- Joins match metadata with point-level shot data
- Handles flexible column naming via `find_column`

## Court Vision Loader (court_vision.py)

- Reads Hawk-Eye JSON files with spatial coordinates
- Extracts `ball_landing_x/y`, `serve_speed_kmh`, player positions
- Maps keywords to `ShotType` and `ShotDirection`

## Slam PBP Loader (slam_pbp.py)

- Reads Grand Slam point-by-point CSV data
- Extracts score context (set, game, point, tiebreak, break point)
- No shot-level data (points only)

## DataValidator

Post-load quality checks on `list[Match]`:

| Check | Severity |
|-------|----------|
| Zero points in match | Error |
| rally_length != len(shots) | Warning |
| Last shot not point-ending | Warning |
| Non-alternating server/returner | Warning |
| Unusual score values | Stat counter |
| Server win rate < 0.3 or > 0.8 | Warning (label encoding issue) |

Returns `{"stats": {...}, "errors": [...], "warnings": [...], "is_clean": bool}`.

## Data Flow

```
Raw files (CSV/JSON)
    ↓
MCPLoader / CourtVisionLoader / SlamPBPLoader
    ↓
list[Match]  (validated domain objects)
    ↓
DataValidator.validate()
    ↓
list[Match]  (quality-checked)
```

## Dependencies

- Imports from: `core` only
- Imported by: `orchestration`
- External: `pandas`
