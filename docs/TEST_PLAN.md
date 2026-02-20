# Test Plan — Tennis Miner

## Strategy

| Layer | Scope | Tooling | Coverage Target |
|-------|-------|---------|-----------------|
| Unit | Single function/class | pytest | Every public method |
| Edge Case | Boundary inputs | pytest | All enums, empty/null, overflow |
| Integration | Module boundaries | pytest + synthetic data | All data flow contracts |
| Smoke | Full pipeline | pytest | 1 end-to-end run |

## Test Matrix by Module

### core/schema.py

| Test Case | Category | Description |
|-----------|----------|-------------|
| Shot: valid creation | Happy path | All valid enum combinations |
| Shot: shot_num=0 | Edge: boundary | Must raise ValueError |
| Shot: shot_num=-1 | Edge: negative | Must raise ValueError |
| Shot: player="spectator" | Edge: invalid enum | Must raise ValueError |
| Shot: speed=-10 | Edge: negative float | Must raise ValueError |
| Shot: speed=0 | Edge: boundary | Must be valid |
| Shot: has_spatial partial | Edge: incomplete | x without y = False |
| Shot: is_point_ending | Logic | Each outcome type checked |
| Point: auto rally_length | Logic | When 0 + shots present → auto-fill |
| Point: set_num=0 | Edge: boundary | Must raise ValueError |
| Point: empty server | Edge: empty string | Must raise ValueError |
| Match: empty match_id | Edge: empty string | Must raise ValueError |
| Match: year=1800 | Edge: out of range | Must raise ValueError |
| Match: n_points_with_shots | Logic | Mixed points with/without shots |
| DataBundle: shape mismatch | Edge: invalid | score_features rows != labels |
| DataBundle: match_ids mismatch | Edge: invalid | len(match_ids) != len(labels) |
| DataBundle: subset | Logic | Boolean mask produces correct subset |
| DataBundle: empty | Edge: zero samples | All operations still work |

### ingestion/mcp.py

| Test Case | Category | Description |
|-----------|----------|-------------|
| tokenize: basic rally | Happy path | "f1b3f2*" → 3 tokens |
| tokenize: empty string | Edge | Returns [] |
| tokenize: None | Edge | Returns [] |
| tokenize: whitespace | Edge | Stripped correctly |
| tokenize: letters only | Edge | "fb" → ["f", "b"] |
| parse_shot: each type | Coverage | f,b,r,s,v,z,o,u,l,p,j,k,t |
| parse_shot: each outcome | Coverage | *, #, @ |
| parse_shot: each direction | Coverage | Zones 1-6 |
| parse_shot: depth 7,8,9 | Coverage | Deep, middle, short |
| parse_shot: empty/None | Edge | Returns UNKNOWN |
| parse_shot: unknown char | Edge | Returns UNKNOWN type |
| parse_rally: alternation | Logic | Even=server, odd=returner |
| parse_rally: shot_num | Logic | Sequential 1-indexed |
| parse_rally: long (10+) | Stress | 10-shot rally |
| infer_winner: winner by server | Logic | Server hits winner → True |
| infer_winner: error by server | Logic | Server UE → False |
| infer_winner: ace | Logic | → True |
| infer_winner: double fault | Logic | → False |
| infer_winner: empty | Edge | → True (default) |

### features/

| Test Case | Category | Description |
|-----------|----------|-------------|
| score_features shape | Contract | Always (8,) float32 |
| shot_features shape | Contract | Always (6,) float32 |
| shot_sequence padding | Logic | Zeros after rally_length |
| shot_sequence truncation | Edge | max_length < rally → truncated |
| build_dataset: skip no shots | Logic | require_shots=True |
| build_dataset: include no shots | Logic | require_shots=False |
| build_dataset: empty matches | Edge | Returns empty bundle |
| build_dataset: multiple matches | Logic | Correct match_ids |
| spatial: with data | Logic | Values correctly placed |
| spatial: missing | Logic | All zeros |
| spatial_mask | Logic | 1.0 where present, 0.0 where None |

### models/

| Test Case | Category | Description |
|-----------|----------|-------------|
| Logistic: fit + predict | Happy path | Valid probabilities [0,1] |
| Logistic: predict before fit | Edge | RuntimeError |
| Logistic: save/load roundtrip | Logic | Predictions identical |
| LSTM: forward shape | Contract | (batch,) output |
| LSTM: single sample | Edge | batch=1 works |
| LSTM: all length 1 | Edge | Minimum sequences |
| LSTM: fit + predict | Happy path | 3 epochs, valid output |
| LSTM: predict before fit | Edge | RuntimeError |
| Transformer: forward shape | Contract | (batch,) output |
| Transformer: min length | Edge | Length 1 works |
| Registry: list models | Coverage | All 3 models registered |
| Registry: create by name | Logic | Returns correct type |
| Registry: unknown name | Edge | ValueError |

### evaluation/

| Test Case | Category | Description |
|-----------|----------|-------------|
| AUC: perfect | Logic | == 1.0 |
| AUC: random | Logic | ≈ 0.5 |
| AUC: single class | Edge | Returns 0.5 (degenerate) |
| AUC: inverted | Logic | == 0.0 |
| LogLoss: good predictions | Logic | < 0.5 |
| LogLoss: clips extremes | Edge | Pred=0.0/1.0 → no crash |
| Brier: perfect/worst | Logic | 0.0 and 1.0 |
| Calibration: bin counts | Contract | n_bins entries |
| Calibration: empty bins | Edge | NaN handled |
| Calibration: perfect | Logic | Max error < 0.05 |
| Bootstrap: better model | Logic | Positive diff, low p |
| Bootstrap: equal models | Logic | Near-zero diff, high p |
| Bootstrap: degenerate labels | Edge | No crash |
| KP1: clear pass | Logic | OVERALL_PASS=True |
| KP1: clear fail | Logic | OVERALL_PASS=False |
| KP1: result structure | Contract | All expected keys present |

### integration/

| Test Case | Category | Description |
|-----------|----------|-------------|
| Synthetic → features | Integration | Valid DataBundle |
| Synthetic → baseline | Integration | Trains + predicts |
| Synthetic → LSTM | Integration | Trains + predicts |
| Synthetic → KP1 eval | Integration | Returns valid result |
| Full flow | E2E | data → features → 2 models → KP1 |

## Running Tests

```bash
# All tests
pytest tests/ -v

# By module
pytest tests/test_core/ -v
pytest tests/test_ingestion/ -v
pytest tests/test_features/ -v
pytest tests/test_models/ -v
pytest tests/test_evaluation/ -v
pytest tests/test_integration/ -v

# With coverage
pytest tests/ --cov=tennis_miner --cov-report=term-missing
```
