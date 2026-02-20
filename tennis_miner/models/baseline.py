"""
Phase 1 Baseline Model: Score-only win probability prediction.

Two variants:
  1. Logistic regression (sklearn) — simplest baseline
  2. Shallow MLP (PyTorch) — slightly more expressive

Both take only score-state features (no shot information) and predict
P(server wins point). This is the control model that shot-sequence V
must beat by AUC > 0.03 to pass Kill Point #1.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


class LogisticBaseline:
    """Logistic regression baseline on score-state features."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train on score features."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        log.info(f"Logistic baseline trained on {len(y)} samples")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict P(server wins) for each point."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def save(self, path: str):
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "model": self.model}, path)
        log.info(f"Logistic baseline saved to {path}")

    def load(self, path: str):
        import joblib
        data = joblib.load(path)
        self.scaler = data["scaler"]
        self.model = data["model"]


class MLPBaseline(nn.Module):
    """Shallow MLP baseline on score-state features."""

    def __init__(self, input_dim: int = 8, hidden_sizes: list[int] = None):
        super().__init__()
        hidden_sizes = hidden_sizes or [64, 32]

        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits (not probabilities)."""
        return self.net(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


def train_mlp_baseline(
    model: MLPBaseline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float = 0.001,
    epochs: int = 50,
    batch_size: int = 1024,
    device: str = "cpu",
) -> dict:
    """Train MLP baseline with early stopping."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Normalize features
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    X_t = torch.tensor(X_train_norm, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_v = torch.tensor(X_val_norm, dtype=torch.float32, device=device)
    y_v = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_val_loss = float("inf")
    best_state = None
    patience = 10
    patience_counter = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_t))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_t), batch_size):
            idx = perm[i : i + batch_size]
            logits = model(X_t[idx])
            loss = criterion(logits, y_t[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_v)
            val_loss = criterion(val_logits, y_v).item()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            log.info(
                f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, "
                f"val_loss={val_loss:.4f}"
            )

        if patience_counter >= patience:
            log.info(f"Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    history["feature_mean"] = mean
    history["feature_std"] = std

    return history
