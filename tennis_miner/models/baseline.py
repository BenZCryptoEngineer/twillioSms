"""
Score-only baseline model (Kill Point #1 control).

Uses logistic regression — no shot information, just score state.
"""

import logging
from pathlib import Path

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from tennis_miner.core.schema import DataBundle
from tennis_miner.core.interfaces import BaseModel
from tennis_miner.models.registry import register

log = logging.getLogger(__name__)


@register("logistic_baseline")
class LogisticBaseline(BaseModel):

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=max_iter, solver="lbfgs", C=C)
        self._fitted = False

    def fit(self, train: DataBundle, val: DataBundle) -> dict:
        X = self.scaler.fit_transform(train.score_features)
        self.model.fit(X, train.labels)
        self._fitted = True

        val_preds = self.predict_proba(val)
        log.info(f"Logistic baseline: trained on {train.n_samples} samples")
        return {"val_predictions": val_preds}

    def predict_proba(self, data: DataBundle) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = self.scaler.transform(data.score_features)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "model": self.model}, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.scaler = data["scaler"]
        self.model = data["model"]
        self._fitted = True
