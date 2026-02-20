"""
Phase 1 Shot-Sequence Model: V(state) using LSTM on shot sequences.

Takes score state + sequential shot information and predicts P(server wins point).
This is the model that must outperform the baseline to pass Kill Point #1.

The LSTM processes the shot sequence and its final hidden state (combined with
score features) produces the win probability prediction.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

log = logging.getLogger(__name__)


class ShotSequenceLSTM(nn.Module):
    """LSTM-based value function V(state) on shot sequences.

    Architecture:
      1. Shot embedding: linear projection of per-shot features
      2. LSTM: processes variable-length shot sequence
      3. Score encoder: MLP on score features
      4. Fusion: concatenate LSTM output + score encoding -> prediction head
    """

    def __init__(
        self,
        shot_input_dim: int = 6,
        score_input_dim: int = 8,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Shot embedding
        self.shot_embed = nn.Sequential(
            nn.Linear(shot_input_dim, embedding_dim),
            nn.ReLU(),
        )

        # LSTM for shot sequence
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Score state encoder
        self.score_encoder = nn.Sequential(
            nn.Linear(score_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Prediction head: fuse LSTM output + score encoding
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        shot_seq: torch.Tensor,
        seq_lengths: torch.Tensor,
        score_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            shot_seq: (batch, max_len, shot_dim) padded shot sequences
            seq_lengths: (batch,) actual lengths
            score_features: (batch, score_dim) score state features

        Returns:
            (batch,) logits for P(server wins)
        """
        batch_size = shot_seq.size(0)

        # Embed shots
        embedded = self.shot_embed(shot_seq)  # (B, T, embed_dim)

        # Pack for efficient LSTM processing
        lengths_clamped = seq_lengths.clamp(min=1).cpu()
        packed = pack_padded_sequence(
            embedded, lengths_clamped, batch_first=True, enforce_sorted=False
        )

        # LSTM forward
        packed_out, (h_n, _) = self.lstm(packed)

        # Use final hidden state of top layer as sequence representation
        seq_repr = h_n[-1]  # (B, hidden_dim)

        # Encode score state
        score_repr = self.score_encoder(score_features)  # (B, 32)

        # Fuse and predict
        fused = torch.cat([seq_repr, score_repr], dim=1)
        logits = self.head(fused).squeeze(-1)  # (B,)

        return logits

    def predict_proba(
        self,
        shot_seq: torch.Tensor,
        seq_lengths: torch.Tensor,
        score_features: torch.Tensor,
    ) -> torch.Tensor:
        """Predict win probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(shot_seq, seq_lengths, score_features)
            return torch.sigmoid(logits)


class ShotSequenceTransformer(nn.Module):
    """Transformer-based alternative for shot sequence modeling.

    Uses self-attention over shot sequence + score features.
    May capture long-range dependencies better than LSTM for long rallies.
    """

    def __init__(
        self,
        shot_input_dim: int = 6,
        score_input_dim: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        max_seq_len: int = 50,
    ):
        super().__init__()

        self.shot_embed = nn.Linear(shot_input_dim, d_model)
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.score_encoder = nn.Sequential(
            nn.Linear(score_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(d_model + 32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        shot_seq: torch.Tensor,
        seq_lengths: torch.Tensor,
        score_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_len, _ = shot_seq.shape
        device = shot_seq.device

        # Embed + positional encoding
        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
        embedded = self.shot_embed(shot_seq) + self.pos_encoding(positions)

        # Create attention mask for padding
        mask = torch.arange(max_len, device=device).unsqueeze(0) >= seq_lengths.unsqueeze(1)

        # Transformer
        encoded = self.transformer(embedded, src_key_padding_mask=mask)

        # Mean pooling over non-padded positions
        mask_expanded = (~mask).unsqueeze(-1).float()
        seq_repr = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        # Score encoding + fusion
        score_repr = self.score_encoder(score_features)
        fused = torch.cat([seq_repr, score_repr], dim=1)
        logits = self.head(fused).squeeze(-1)

        return logits

    def predict_proba(
        self,
        shot_seq: torch.Tensor,
        seq_lengths: torch.Tensor,
        score_features: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(shot_seq, seq_lengths, score_features)
            return torch.sigmoid(logits)


def train_sequence_model(
    model: nn.Module,
    train_data: dict,
    val_data: dict,
    lr: float = 0.001,
    epochs: int = 100,
    batch_size: int = 512,
    device: str = "cpu",
) -> dict:
    """Train the shot-sequence model with early stopping.

    Args:
        model: ShotSequenceLSTM or ShotSequenceTransformer
        train_data: dict with score_features, shot_sequences, sequence_lengths, labels
        val_data: same structure
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Normalize score features
    score_mean = train_data["score_features"].mean(axis=0)
    score_std = train_data["score_features"].std(axis=0) + 1e-8

    def prepare(data):
        sf = (data["score_features"] - score_mean) / score_std
        return {
            "score": torch.tensor(sf, dtype=torch.float32, device=device),
            "shots": torch.tensor(data["shot_sequences"], dtype=torch.float32, device=device),
            "lengths": torch.tensor(data["sequence_lengths"], dtype=torch.long, device=device),
            "labels": torch.tensor(data["labels"], dtype=torch.float32, device=device),
        }

    train = prepare(train_data)
    val = prepare(val_data)
    n_train = len(train["labels"])

    best_val_loss = float("inf")
    best_state = None
    patience = 15
    patience_counter = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            logits = model(
                train["shots"][idx],
                train["lengths"][idx],
                train["score"][idx],
            )
            loss = criterion(logits, train["labels"][idx])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val["shots"], val["lengths"], val["score"])
            val_loss = criterion(val_logits, val["labels"]).item()

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

    history["score_mean"] = score_mean
    history["score_std"] = score_std

    return history
