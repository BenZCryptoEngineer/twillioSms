"""
Transformer-based shot-sequence V(state) model.

Uses self-attention over shot sequences. May capture long-range
rally dependencies better than LSTM.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tennis_miner.core.schema import DataBundle
from tennis_miner.core.interfaces import BaseModel
from tennis_miner.models.registry import register

log = logging.getLogger(__name__)


class TransformerNetwork(nn.Module):

    def __init__(
        self,
        shot_dim: int = 6,
        score_dim: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        max_seq_len: int = 50,
    ):
        super().__init__()
        self.shot_embed = nn.Linear(shot_dim, d_model)
        self.pos_enc = nn.Embedding(max_seq_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers)

        self.score_enc = nn.Sequential(
            nn.Linear(score_dim, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model + 32, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def forward(self, shots, lengths, scores):
        B, T, _ = shots.shape
        device = shots.device
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        emb = self.shot_embed(shots) + self.pos_enc(pos)
        mask = torch.arange(T, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        enc = self.transformer(emb, src_key_padding_mask=mask)
        valid = (~mask).unsqueeze(-1).float()
        pool = (enc * valid).sum(1) / valid.sum(1).clamp(min=1)
        score_repr = self.score_enc(scores)
        return self.head(torch.cat([pool, score_repr], 1)).squeeze(-1)


@register("transformer")
class TransformerModel(BaseModel):

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        max_seq_len: int = 50,
        lr: float = 0.001,
        epochs: int = 100,
        batch_size: int = 512,
        patience: int = 15,
    ):
        self.net_config = dict(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dropout=dropout, max_seq_len=max_seq_len,
        )
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.net: TransformerNetwork | None = None
        self.score_mean: np.ndarray | None = None
        self.score_std: np.ndarray | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, train: DataBundle, val: DataBundle) -> dict:
        self.score_mean = train.score_features.mean(axis=0)
        self.score_std = train.score_features.std(axis=0) + 1e-8

        self.net = TransformerNetwork(**self.net_config).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        train_t = self._to_tensors(train)
        val_t = self._to_tensors(val)

        best_loss, best_state, wait = float("inf"), None, 0
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

            t_loss = epoch_loss / max(n_batch, 1)
            self.net.eval()
            with torch.no_grad():
                v_loss = criterion(
                    self.net(val_t["shots"], val_t["lengths"], val_t["scores"]),
                    val_t["labels"],
                ).item()

            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)

            if v_loss < best_loss:
                best_loss = v_loss
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if epoch % 10 == 0:
                log.info(f"Transformer epoch {epoch}: train={t_loss:.4f} val={v_loss:.4f}")
            if wait >= self.patience:
                log.info(f"Transformer early stop at epoch {epoch}")
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
            return torch.sigmoid(self.net(t["shots"], t["lengths"], t["scores"])).cpu().numpy()

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "config": self.net_config,
            "score_mean": self.score_mean, "score_std": self.score_std,
        }, path)

    def load(self, path: str):
        d = torch.load(path, weights_only=False)
        self.net_config = d["config"]
        self.score_mean, self.score_std = d["score_mean"], d["score_std"]
        self.net = TransformerNetwork(**self.net_config).to(self.device)
        self.net.load_state_dict(d["state_dict"])

    def _to_tensors(self, b: DataBundle) -> dict:
        s = (b.score_features - self.score_mean) / self.score_std
        return {
            "scores": torch.tensor(s, dtype=torch.float32, device=self.device),
            "shots": torch.tensor(b.shot_sequences, dtype=torch.float32, device=self.device),
            "lengths": torch.tensor(b.sequence_lengths, dtype=torch.long, device=self.device),
            "labels": torch.tensor(b.labels, dtype=torch.float32, device=self.device),
        }
