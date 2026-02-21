# models/ — Model Architectures

## Purpose

Trains models and produces P(server wins) predictions. All models implement `BaseModel` and are registered via the factory pattern.

## Files

| File | Responsibility |
|------|---------------|
| `registry.py` | Factory pattern: `@register` decorator + `create_model(name)` |
| `baseline.py` | Logistic regression on score-state only (KP-1 control) |
| `lstm.py` | LSTM shot-sequence V(state) model |
| `transformer.py` | Transformer shot-sequence V(state) model (alternative) |

## Registry (registry.py)

```python
@register("logistic_baseline")
class LogisticBaseline(BaseModel): ...

@register("lstm")
class LSTMModel(BaseModel): ...

@register("transformer")
class TransformerModel(BaseModel): ...
```

Usage:
```python
from tennis_miner.models.registry import create_model, list_models

model = create_model("lstm", embed_dim=32, hidden_dim=128)
list_models()  # ["logistic_baseline", "lstm", "transformer"]
```

## LogisticBaseline (baseline.py)

The **control model** for Kill Point #1. Uses only score-state features — no shot information.

| Attribute | Value |
|-----------|-------|
| Input | `DataBundle.score_features` (8-dim) |
| Algorithm | sklearn LogisticRegression (L2, LBFGS) |
| Normalization | StandardScaler (fit on train only) |
| Output | P(server wins) via `predict_proba[:, 1]` |
| Serialization | joblib (scaler + model) |

Hyperparameters: `C=1.0`, `max_iter=1000`

## LSTMModel (lstm.py)

The **experimental model** — tests whether shot-sequence information improves V(state).

### Architecture (LSTMNetwork)

```
shots (N, T, 6) → Linear(6→embed_dim) → ReLU → LSTM → h_n[-1]
                                                           ↓
scores (N, 8)  → Linear(8→32) → ReLU → Linear(32→32) → concat → Linear(hidden+32→64) → ReLU → Linear(64→1)
```

| Component | Default |
|-----------|---------|
| Shot embedding | Linear(6 → 32) + ReLU |
| LSTM | 2-layer, hidden_dim=128, dropout=0.2 |
| Score encoder | 2-layer MLP (8 → 32 → 32) |
| Prediction head | Linear(160 → 64) → ReLU → Dropout → Linear(64 → 1) |

### Training

| Attribute | Value |
|-----------|-------|
| Loss | BCEWithLogitsLoss |
| Optimizer | Adam (lr=0.001) |
| Gradient clipping | max_norm=1.0 |
| Early stopping | patience=15 (on val loss) |
| Normalization | Z-score on score features (computed from train) |
| Packed sequences | Yes (via `pack_padded_sequence`) |
| Device | Auto (CUDA if available) |

### Save/Load

Saves: `state_dict`, `config`, `score_mean`, `score_std` as a single `.pt` file.

## TransformerModel (transformer.py)

Alternative to LSTM with self-attention over shot sequences.

### Architecture (TransformerNetwork)

```
shots (N, T, 6) → Linear(6→d_model) + positional_encoding → TransformerEncoder → mean_pool
                                                                                      ↓
scores (N, 8) → Linear(8→d_model) ──────────────────────────────────────→ concat → Linear → 1
```

| Component | Default |
|-----------|---------|
| Input projection | Linear(6 → 64) |
| Positional encoding | Learnable (max_seq_len × d_model) |
| Transformer | 2 encoder layers, 4 heads, dim_feedforward=256 |
| Pooling | Mean over non-padded positions |
| Prediction head | Linear(d_model × 2 → d_model) → ReLU → Linear(d_model → 1) |

## Common Interface

All models share:

```python
model.fit(train_bundle, val_bundle) -> dict   # training history
model.predict_proba(bundle) -> np.ndarray     # shape (N,), values in [0, 1]
model.save(path)
model.load(path)
```

## Dependencies

- Imports from: `core` only
- Imported by: `orchestration`
- External: `sklearn`, `torch`, `joblib`, `numpy`
