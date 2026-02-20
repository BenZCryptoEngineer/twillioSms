# Tennis Miner — Per-Shot Win Rate Prediction System

> Calculates P(win point) from any game state. Like AlphaGo's value network, but for tennis.

## Architecture Overview

```
tennis_miner/
├── core/              # Domain objects & shared contracts
│   ├── schema.py      # Match/Point/Shot dataclasses + validation
│   └── interfaces.py  # Abstract base classes (DataLoader, Model, Evaluator)
│
├── ingestion/         # Data acquisition & normalization (Phase 0)
│   ├── base.py        # BaseLoader with shared CSV/JSON utilities
│   ├── mcp.py         # Match Charting Project loader
│   ├── court_vision.py# Court Vision (Hawk-Eye) loader
│   ├── slam_pbp.py    # Grand Slam point-by-point loader
│   └── validator.py   # Post-load data quality checks
│
├── features/          # Feature engineering (Phase 0→1 bridge)
│   ├── encoder.py     # Categorical encoding registry
│   ├── score.py       # Score-state feature extraction
│   ├── sequence.py    # Shot-sequence feature extraction
│   └── spatial.py     # Phase 2 spatial feature extraction
│
├── models/            # Model training & inference (Phase 1+)
│   ├── registry.py    # Model registry (factory pattern)
│   ├── baseline.py    # Logistic regression baseline
│   ├── lstm.py        # LSTM shot-sequence model
│   └── transformer.py # Transformer shot-sequence model
│
├── evaluation/        # Metrics, testing, reporting (all phases)
│   ├── metrics.py     # AUC, log-loss, calibration, Brier
│   ├── significance.py# Paired bootstrap & statistical tests
│   ├── kill_point.py  # Kill-point gate evaluation
│   └── report.py      # Report generation (JSON + text + plots)
│
├── orchestration/     # Pipeline coordination
│   ├── pipeline.py    # Phase-aware pipeline runner
│   └── config.py      # Config loading + validation
│
└── scripts/           # Standalone CLI scripts
    ├── acquire.py     # Unified data acquisition
    └── train.py       # Training entry point
```

## Module Responsibilities

| Module | Single Responsibility | Input | Output |
|--------|----------------------|-------|--------|
| `core` | Domain language — what a Match/Point/Shot IS | None | Type definitions + contracts |
| `ingestion` | Raw data → validated domain objects | CSV/JSON files | `list[Match]` |
| `features` | Domain objects → numeric tensors | `list[Match]` | `dict[str, np.ndarray]` |
| `models` | Tensors → predictions | `np.ndarray` / `torch.Tensor` | `np.ndarray` probabilities |
| `evaluation` | Predictions → decisions | `(y_true, y_pred)` pairs | Metric dicts + reports |
| `orchestration` | Wires everything together | Config YAML | Pipeline results |

## Inter-Module Communication

```
                    ┌─────────────┐
                    │    core/    │  (shared types — every module imports this)
                    │  schema.py  │
                    │ interfaces  │
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
    ┌─────▼─────┐   ┌─────▼─────┐   ┌──────▼──────┐
    │ ingestion │──▶│ features  │──▶│   models    │
    │           │   │           │   │             │
    │ list[Match]   │ DataBundle │   │ np.ndarray  │
    └───────────┘   └───────────┘   └──────┬──────┘
                                           │
                                    ┌──────▼──────┐
                                    │ evaluation  │
                                    │             │
                                    │ KP decision │
                                    └──────┬──────┘
                                           │
                                    ┌──────▼──────┐
                                    │orchestration│
                                    │  (wiring)   │
                                    └─────────────┘
```

### Data Flow Contracts

1. **ingestion → features**: `list[Match]` (validated domain objects)
2. **features → models**: `DataBundle` (named dict with numpy arrays + metadata)
3. **models → evaluation**: `(y_true: np.ndarray, y_pred: np.ndarray)`
4. **evaluation → orchestration**: `KillPointResult` (typed decision object)

### Rules
- `core` imports nothing from other modules
- `ingestion` imports only `core`
- `features` imports only `core`
- `models` imports only `core`
- `evaluation` imports only `core`
- `orchestration` imports everything — it is the only module that wires dependencies

This ensures any module can be tested in complete isolation.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Phase 0: Download data
python -m tennis_miner.scripts.acquire --source all

# Phase 1: Train + evaluate Kill Point #1
python -m tennis_miner.scripts.train --phase 1

# Run tests
pytest tests/ -v
```

## Phase Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full phase plan with kill points.

See [docs/PROGRESS.md](docs/PROGRESS.md) for current project status.

## Module Documentation

Each module has its own README:
- [core/README.md](docs/modules/core.md) — Domain schema & interfaces
- [ingestion/README.md](docs/modules/ingestion.md) — Data loaders & validation
- [features/README.md](docs/modules/features.md) — Feature engineering
- [models/README.md](docs/modules/models.md) — Model architectures
- [evaluation/README.md](docs/modules/evaluation.md) — Metrics & kill-point gates
- [orchestration/README.md](docs/modules/orchestration.md) — Pipeline & config

## Test Plan

See [docs/TEST_PLAN.md](docs/TEST_PLAN.md) for the comprehensive test strategy.

---
*Copyright (c) 2026 EXPETA TECHNOLOGIES INC. All Rights Reserved.*
