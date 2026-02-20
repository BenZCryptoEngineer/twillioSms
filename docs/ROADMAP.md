# Phase Roadmap — Tennis Miner

## Phase 0: Data Acquisition

| Attribute | Value |
|-----------|-------|
| Duration | 1-2 days |
| Cash Cost | $0 |
| Kill Point | None |

### Minimum Deliverable
1. Court Vision data cached locally (all AO 2020-2025 + RG 2019-2025)
2. MCP repo cloned, schema understood
3. Slam PBP repo cloned
4. Data inventory: total matches/points/shots per source

### Success Criteria
- [ ] `python main.py audit` prints non-zero counts for MCP
- [ ] Court Vision JSON files exist on disk (if API accessible)
- [ ] Can load 1 match from each source without errors

---

## Phase 1: Discrete V(state) Validation

| Attribute | Value |
|-----------|-------|
| Duration | 5-7 days |
| Cash Cost | $0 (CPU only) |
| Kill Point | KP-1 at Day 5 |

### Minimum Deliverable
1. Baseline model trained (logistic regression on score-state)
2. LSTM model trained (score-state + shot sequences)
3. KP-1 evaluation report generated
4. Go/No-Go decision documented

### Success Criteria (Kill Point #1)

| Metric | Threshold | How Measured |
|--------|-----------|--------------|
| AUC improvement | > 0.03 over baseline | ROC AUC on held-out test set |
| Log-loss improvement | > 0.02 over baseline | Binary cross-entropy on test set |
| Calibration | Within 5% across deciles | Predicted vs actual in probability bins |
| Statistical significance | p < 0.01 | Paired bootstrap test (10K samples) |

### If PASS → proceed to Phase 2A
### If FAIL → project terminates, $0 sunk cost

---

## Phase 2A: Spatial V(state) with Court Vision

| Attribute | Value |
|-----------|-------|
| Duration | 2-3 weeks |
| Cash Cost | $50-100 (GPU) |
| Kill Point | KP-2 at Week 4 |
| Prerequisite | Phase 1 PASS |

### Minimum Deliverable
1. Court Vision data processing pipeline (JSON → normalized schema)
2. Spatial V(state) model trained with ball landing XY
3. Ablation study: spatial features vs discrete-only
4. Cross-surface transfer test (AO → RG)

### Success Criteria (Kill Point #2)

| Metric | Threshold |
|--------|-----------|
| AUC improvement over Phase 1 | > 0.02 |
| Spatial feature ablation | Significant (p < 0.05) |
| Cross-surface AUC drop | < 0.05 |

### If PASS → Phase 2B (optional) or Phase 3
### If FAIL → pivot to content product with Phase 1 model

---

## Phase 2B: CV Pipeline (Conditional)

| Attribute | Value |
|-----------|-------|
| Duration | 3-4 weeks |
| Cash Cost | $100-150 |
| Kill Point | KP-3 at Week 8 |
| Prerequisite | Phase 2A PASS + user-upload needed |

### Minimum Deliverable
1. Video → structured shot data pipeline
2. Accuracy benchmark vs Court Vision ground truth
3. 5 fully processed demo matches

### Success Criteria (Kill Point #3)

| Metric | Threshold |
|--------|-----------|
| Landing point error | < 2m on compressed video |
| Processing speed | < 10 min per match |
| Ball tracking F1 | > 80% on novel video |

### If FAIL → abandon user-upload, use Court Vision only

---

## Phase 3: Counterfactual Analysis

| Attribute | Value |
|-----------|-------|
| Duration | 2-3 weeks |
| Cash Cost | $50-100 |
| Kill Point | KP-4 at Week 11 |
| Prerequisite | Phase 2A PASS |

### Minimum Deliverable
1. One-step lookahead counterfactual engine
2. Alternative shot generation (5-8 per actual shot)
3. 10 demo match analyses with counterfactual commentary
4. Commercial readiness assessment

### Success Criteria (Kill Point #4)

| Metric | Threshold |
|--------|-----------|
| Expert agreement | > 70% |
| Hindsight accuracy | Recommended > actual in > 60% of cases |
| Latency | < 2s per shot |

### If FAIL → release Phase 2 as analytics tool

---

## Cost Summary

| Path | Duration | Cash | Engineering |
|------|----------|------|-------------|
| Phase 0+1 only (FAIL) | ~1 week | $0 | 35h |
| Phase 0+1+2A+3 (no CV) | ~7-9 weeks | $100-200 | 170h |
| Full path with 2B | ~10-14 weeks | $200-350 | 290h |
