# Progress Report — Tennis Miner

**Last updated:** 2026-02-20
**Current phase:** Phase 0 (Data Acquisition) — architecture complete, awaiting data

---

## Completed

### Architecture Refactoring (v0.2.0)
- [x] PRD analysis and modular architecture design
- [x] 6-module structure: core / ingestion / features / models / evaluation / orchestration
- [x] Abstract interfaces (BaseLoader, BaseModel, BaseEvaluator)
- [x] Domain objects with validation (Shot, Point, Match, DataBundle)
- [x] Model registry with factory pattern
- [x] Proper train/val/test split (fixed data leakage from v0.1)
- [x] Data validation layer (DataValidator)
- [x] Shared utilities (base.py) eliminating code duplication
- [x] Comprehensive test suite: 80+ test cases across 6 test modules
- [x] Module documentation and inter-module communication docs
- [x] Phase roadmap with quantitative kill-point criteria

### Data Loaders Implemented
- [x] MCP loader with shot-string parser
- [x] Court Vision (Hawk-Eye) JSON loader with spatial coordinates
- [x] Grand Slam point-by-point loader
- [x] Court Vision acquisition script

### Models Implemented
- [x] Logistic regression baseline (score-only)
- [x] LSTM shot-sequence model
- [x] Transformer shot-sequence model (alternative)

### Evaluation Framework
- [x] AUC, log-loss, Brier, calibration metrics
- [x] Paired bootstrap significance test
- [x] Kill Point #1 gate evaluator
- [x] Report generation (JSON + text + plots)

---

## In Progress

### Phase 0: Data Acquisition
- [ ] Download Court Vision data (requires `courtvisionpython` + API access)
- [ ] Clone MCP dataset
- [ ] Clone Slam PBP dataset
- [ ] Data inventory audit

---

## Not Started

### Phase 1: Discrete V(state) Validation
- [ ] Load real MCP data through pipeline
- [ ] Train baseline on real data
- [ ] Train LSTM on real data
- [ ] Run KP-1 evaluation
- [ ] Write Go/No-Go decision document

### Phase 2A: Spatial V(state)
- [ ] Court Vision data processing
- [ ] Spatial feature integration
- [ ] Ablation study
- [ ] Cross-surface transfer test

### Phase 3: Counterfactual Analysis
- [ ] Alternative shot generation
- [ ] One-step lookahead engine
- [ ] Expert validation

---

## Key Metrics (will be filled after Phase 1)

| Metric | Baseline | Sequence | Delta | Pass? |
|--------|----------|----------|-------|-------|
| AUC | — | — | — | — |
| Log-Loss | — | — | — | — |
| Brier | — | — | — | — |
| Bootstrap p | — | — | — | — |

---

## Architecture Audit Findings (v0.1 → v0.2)

| Issue | Severity | Fix |
|-------|----------|-----|
| No data validation | High | Added DataValidator + __post_init__ |
| Duplicated _find_column across 3 files | Medium | Shared base.py utilities |
| Pipeline trains on test data (data leakage) | Critical | Proper 3-way split (70/15/15) |
| No abstract interfaces | Medium | Added BaseLoader/BaseModel/BaseEvaluator |
| Models not registered | Low | Added registry with factory pattern |
| Tests shallow (~3 assertions each) | Medium | 80+ tests with edge cases |
| No inter-module docs | Medium | Added architecture README + module docs |

---

## Next Actions Guide

### Immediate: Complete Phase 0

All code is in place. Only data acquisition remains.

| Step | Command | Source | Risk |
|------|---------|--------|------|
| 1. Clone MCP | `python scripts/acquire_mcp.py` | GitHub (free, public) | Low |
| 2. Clone Slam PBP | `python scripts/acquire_slam_pbp.py` | GitHub (free, public) | Low |
| 3. Download Court Vision | `pip install courtvisionpython && python scripts/acquire_court_vision.py` | Infosys API (free, no key) | High — may become unavailable |
| 4. Verify | `python main.py audit` | — | — |

**Phase 0 exit criteria:**
- MCP audit prints non-zero counts
- Court Vision JSON files on disk (if API accessible)
- At least 1 match loadable from each source

### Then: Phase 1 — Kill Point #1

Train baseline (logistic regression) + LSTM on real data, then evaluate:

| Metric | Threshold | Pass = |
|--------|-----------|--------|
| AUC improvement | > 0.03 | LSTM beats baseline by 0.03+ |
| Log-loss improvement | > 0.02 | Prediction quality improves |
| Calibration | < 5% error | Probabilities are well-calibrated |
| Bootstrap p-value | < 0.01 | Improvement is statistically significant |

**GO (all pass):** Proceed to Phase 2A — add spatial features ($50-100 GPU)
**NO-GO (any fail):** Project terminates at $0 sunk cost

### Phase 1 Steps
1. Load real MCP data through ingestion → features pipeline
2. Train logistic regression baseline on score-state features
3. Train LSTM on score-state + shot sequences
4. Run KP-1 evaluation: `python -m tennis_miner.scripts.train --phase 1`
5. Write Go/No-Go decision document
