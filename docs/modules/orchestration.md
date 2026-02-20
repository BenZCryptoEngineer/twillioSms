# orchestration/ — Pipeline & Config

## Purpose

The **only module** that imports from all other modules. Wires ingestion → features → models → evaluation into phase-specific pipelines.

## Files

| File | Responsibility |
|------|---------------|
| `pipeline.py` | Phase-aware pipeline runner |
| `config.py` | YAML config loading + validation |

## Config (config.py)

`load_config(path)` loads a YAML file and validates required top-level keys: `paths`, `data_acquisition`, `phase1`.

### Config Structure (configs/default.yaml)

```yaml
paths:
  raw_data: "data/raw/"
  processed: "data/processed/"
  models: "tennis_miner/models/saved/"

data_acquisition:
  mcp:
    repo_url: "https://github.com/JeffSackmann/tennis_MatchChartingProject.git"
    clone_dir: "data/raw/mcp"
  court_vision:
    base_url: "https://courtvisiondata.ausopen.com"
    output_dir: "data/raw/court_vision"
    tournaments: [...]
    years: [...]

phase1:
  random_seed: 42
  sequence_model:
    model_type: "lstm"
    embedding_dim: 32
    hidden_dim: 128
    num_layers: 2
    dropout: 0.2
    learning_rate: 0.001
    max_epochs: 100
    batch_size: 512
    max_rally_length: 50
  kill_point:
    min_auc_improvement: 0.03
    min_logloss_improvement: 0.02
    calibration_tolerance: 0.05
    significance_level: 0.01
```

## Pipeline (pipeline.py)

### run_phase1(config_path) → dict

End-to-end Phase 1 execution:

```
Step 1: Load data      → MCPLoader + CourtVisionLoader
Step 2: Validate        → DataValidator
Step 3: Feature eng.    → build_dataset() → DataBundle
Step 4: Split           → train/val/test by match ID (70/15/15)
Step 5: Train baseline  → LogisticBaseline
Step 6: Train sequence  → LSTMModel or TransformerModel
Step 7: Evaluate KP-1   → evaluate_kill_point_1()
Step 8: Report          → generate_report()
```

### Data Splitting (Critical Fix from v0.1)

The v0.1 pipeline had a **critical bug**: it used test data as validation data during training (data leakage). v0.2 fixes this with proper 3-way split:

```python
unique_ids = list(set(full_bundle.match_ids))
train_ids, temp_ids = train_test_split(unique_ids, test_size=0.3, random_state=seed)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=seed)
```

Split is done **by match ID** (not by point) to ensure no match appears in multiple splits. This prevents the model from seeing related points during both training and evaluation.

### Error Handling

Returns early with `{"error": "no_data"}` or `{"error": "empty_dataset"}` if:
- No data files found (need to run acquisition first)
- Feature engineering produces 0 samples

### Return Structure

```python
{
    "baseline": {...},
    "sequence_model": {...},
    "comparison": {...},
    "kill_point_1": {"OVERALL_PASS": bool, ...},
    "report_path": str,
    "data_stats": {...},
}
```

## Dependency Wiring

This is the only place where cross-module imports happen:

```
orchestration imports:
  ├── ingestion.mcp.MCPLoader
  ├── ingestion.court_vision.CourtVisionLoader
  ├── ingestion.validator.DataValidator
  ├── features.sequence.build_dataset
  ├── models.baseline.LogisticBaseline
  ├── models.lstm.LSTMModel
  ├── models.transformer.TransformerModel
  ├── evaluation.kill_point.evaluate_kill_point_1
  └── evaluation.report.generate_report
```

Every other module only imports from `core`.

## Dependencies

- Imports from: ALL modules
- Imported by: `main.py` (CLI entry point)
- External: `sklearn.model_selection`, `yaml`, `numpy`
