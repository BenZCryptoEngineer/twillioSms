"""
LSTM-based shot-sequence V(state) model.

Architecture: shot embedding → LSTM → concat with score encoding → prediction head.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from tennis_miner.core.schema import DataBundle
from tennis_miner.core.interfaces import BaseModel
from tennis_miner.models.registry import register

log = logging.getLogger(__name__)


class LSTMNetwork(nn.Module):

    def __init__(
        self,
        shot_dim: int = 6,
        score_dim: int = 8,
        embed_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.shot_embed = nn.Sequential(nn.Linear(shot_dim, embed_dim), nn.ReLU())
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.score_enc = nn.Sequential(
            nn.Linear(score_dim, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 32, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def forward(self, shots, lengths, scores):
        emb = self.shot_embed(shots)
        packed = pack_padded_sequence(
            emb, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)
        seq_repr = h_n[-1]
        score_repr = self.score_enc(scores)
        return self.head(torch.cat([seq_repr, score_repr], dim=1)).squeeze(-1)


@register("lstm")
class LSTMModel(BaseModel):

    def __init__(
        self,
        embed_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 0.001,
        epochs: int = 100,
        batch_size: int = 512,
        patience: int = 15,
    ):
        self.config = dict(
            embed_dim=embed_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout,
        )
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.net: LSTMNetwork | None = None
        self.score_mean: np.ndarray | None = None
        self.score_std: np.ndarray | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, train: DataBundle, val: DataBundle) -> dict:
        self.score_mean = train.score_features.mean(axis=0)
        self.score_std = train.score_features.std(axis=0) + 1e-8

        self.net = LSTMNetwork(**self.config).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        train_t = self._to_tensors(train)
        val_t = self._to_tensors(val)

        best_loss = float("inf")
        best_state = None
        wait = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(train_t["labels"]), device=self.device)
            epoch_loss, n_batch = 0.0, 0

            for i in range(0, len(perm), self.batch_size):
                idx = perm[i:i + self.batch_size]
                logits = self.net(
                    train_t["shots"][idx], train_t["lengths"][idx], train_t["scores"][idx],
                )
                loss = criterion(logits, train_t["labels"][idx])
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batch += 1

            train_loss = epoch_loss / max(n_batch, 1)

            self.net.eval()
            with torch.no_grad():
                val_logits = self.net(val_t["shots"], val_t["lengths"], val_t["scores"])
                val_loss = criterion(val_logits, val_t["labels"]).item()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if epoch % 10 == 0:
                log.info(f"LSTM epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}")
            if wait >= self.patience:
                log.info(f"LSTM early stop at epoch {epoch}")
                break

        if best_state:
            self.net.load_state_dict(best_state)

        return history

    def predict_proba(self, data: DataBundle) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Model not fitted")
        self.net.eval()
        t = self._to_tensors(data)
        with torch.no_grad():
            logits = self.net(t["shots"], t["lengths"], t["scores"])
            return torch.sigmoid(logits).cpu().numpy()

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "config": self.config,
            "score_mean": self.score_mean,
            "score_std": self.score_std,
        }, path)

    def load(self, path: str):
        data = torch.load(path, weights_only=False)
        self.config = data["config"]
        self.score_mean = data["score_mean"]
        self.score_std = data["score_std"]
        self.net = LSTMNetwork(**self.config).to(self.device)
        self.net.load_state_dict(data["state_dict"])

    def _to_tensors(self, bundle: DataBundle) -> dict:
        scores = (bundle.score_features - self.score_mean) / self.score_std
        return {
            "scores": torch.tensor(scores, dtype=torch.float32, device=self.device),
            "shots": torch.tensor(bundle.shot_sequences, dtype=torch.float32, device=self.device),
            "lengths": torch.tensor(bundle.sequence_lengths, dtype=torch.long, device=self.device),
            "labels": torch.tensor(bundle.labels, dtype=torch.float32, device=self.device),
        }
