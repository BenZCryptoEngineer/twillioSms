# CLAUDE.md — Tennis Miner Development Guide

## Project Overview

Tennis Miner is a per-shot win rate prediction system for tennis. Like AlphaGo's value network, it calculates P(win point) from any game state.

## Current Status

**Phase:** Phase 0 (Data Acquisition) — architecture complete, awaiting data
**Version:** v0.2.0
**Date:** 2026-02-21

All code, tests (122 cases), and documentation are in place. The only remaining Phase 0 work is acquiring the actual datasets.

---

## Tasks

When starting a new session, check which tasks are incomplete and execute them in order.

### Task 1: Install project dependencies
Check if the project is already installed. If not, run:
```bash
pip install -e ".[dev]"
```

### Task 2: Clone MCP dataset
Check if `external/tennis_MatchChartingProject/` exists. If not, run:
```bash
python scripts/acquire_mcp.py
```
- Source: `https://github.com/JeffSackmann/tennis_MatchChartingProject`
- Free, public, no auth needed
- Expected: 17,179 matches, 10.15M shots, 2.68M points
- Verify: directory exists and contains CSV files

### Task 3: Clone Slam PBP dataset
Check if `external/tennis_slam_pointbypoint/` exists. If not, run:
```bash
python scripts/acquire_slam_pbp.py
```
- Source: `https://github.com/JeffSackmann/tennis_slam_pointbypoint`
- Free, public, no auth needed
- Verify: directory exists and contains CSV files

### Task 4: Download Court Vision data (optional)
Check if `tennis_miner/data/raw/court_vision/` has JSON files. If not, attempt:
```bash
pip install courtvisionpython
python scripts/acquire_court_vision.py
```
- Source: Infosys Court Vision (Hawk-Eye) API
- Free, no API key, but **may become unavailable at any time**
- Covers: AO 2020-2025, RG 2019-2025
- If the API is unavailable or `courtvisionpython` fails to install, skip this task and note it as unavailable. This is expected and acceptable.

### Task 5: Run data audit
After datasets are acquired, verify everything loaded correctly:
```bash
python main.py audit
```
- Must print non-zero counts for MCP
- Must load at least 1 match from each available source
- Court Vision JSON files on disk (if API was accessible)
- If audit script fails, manually verify by checking that CSV/JSON files exist in the expected directories

### Task 6: Run test suite
Confirm nothing is broken after data acquisition:
```bash
pytest tests/ -v
```
- All 122 tests should pass
- If any tests fail, investigate and fix before proceeding

### Task 7: Update PROGRESS.md
After all above tasks complete, update `docs/PROGRESS.md`:
- Check off completed items under "Phase 0: Data Acquisition"
- Note Court Vision status (downloaded or confirmed unavailable)
- Update the "Last updated" date

---

## Phase 0 Completion Criteria
- [ ] MCP data cloned and audit shows non-zero counts
- [ ] Slam PBP data cloned
- [ ] Court Vision data downloaded (or confirmed unavailable)
- [ ] At least 1 match loadable from each available source
- [ ] All tests pass

Once all criteria are met, Phase 0 is complete. Proceed to Phase 1.

---

## Phase 1 Tasks (after Phase 0 is complete)

### Task 8: Load real data through pipeline
```bash
python -m tennis_miner.scripts.train --phase 1
```
1. Load real MCP data through the ingestion → features pipeline
2. Train logistic regression baseline on score-state features
3. Train LSTM on score-state + shot sequences
4. Run Kill Point #1 evaluation
5. Write Go/No-Go decision document

### Kill Point #1 Thresholds (ALL must pass)

| Metric | Threshold |
|--------|-----------|
| AUC improvement (LSTM over baseline) | > 0.03 |
| Log-loss improvement | > 0.02 |
| Calibration error | < 5% across deciles |
| Statistical significance (bootstrap) | p < 0.01 |

### Decision
- **GO:** All 4 metrics pass → proceed to Phase 2A (spatial features, $50-100 GPU)
- **NO-GO:** Any metric fails → project terminates, $0 sunk cost

---

## Key Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Install with Court Vision support
pip install -e ".[dev,cv]"

# Run tests
pytest tests/ -v

# Phase 0: Acquire data
python scripts/acquire_mcp.py
python scripts/acquire_slam_pbp.py
python scripts/acquire_court_vision.py

# Phase 1: Train + evaluate
python -m tennis_miner.scripts.train --phase 1
```

---

## Architecture

6 modules with strict dependency rules:
- `core/` — Domain types (imports nothing)
- `ingestion/` — Data loaders (imports only core)
- `features/` — Feature engineering (imports only core)
- `models/` — Model training/inference (imports only core)
- `evaluation/` — Metrics & kill-point gates (imports only core)
- `orchestration/` — Pipeline wiring (imports everything)

Config: `configs/default.yaml`

## Data Sources

| Source | Type | Cost | Auth | Risk |
|--------|------|------|------|------|
| MCP | Git clone (CSV) | $0 | None | Low — stable public repo |
| Slam PBP | Git clone (CSV) | $0 | None | Low — stable public repo |
| Court Vision | API (JSON) | $0 | None | High — may become unavailable |

## Phase Roadmap Summary

| Phase | Duration | Cost | Gate |
|-------|----------|------|------|
| Phase 0: Data Acquisition | 1-2 days | $0 | No kill point |
| Phase 1: Discrete V(state) | 5-7 days | $0 | KP-1 (Go/No-Go) |
| Phase 2A: Spatial V(state) | 2-3 weeks | $50-100 | KP-2 |
| Phase 2B: CV Pipeline | 3-4 weeks | $100-150 | KP-3 |
| Phase 3: Counterfactual | 2-3 weeks | $50-100 | KP-4 |
