"""Tests for model architectures."""

import numpy as np
import torch

from tennis_miner.models.baseline import LogisticBaseline, MLPBaseline
from tennis_miner.models.sequence_model import ShotSequenceLSTM, ShotSequenceTransformer


def test_logistic_baseline():
    X = np.random.randn(100, 8).astype(np.float32)
    y = (np.random.rand(100) > 0.5).astype(np.float32)
    model = LogisticBaseline()
    model.fit(X, y)
    preds = model.predict_proba(X)
    assert preds.shape == (100,)
    assert 0 <= preds.min() <= preds.max() <= 1


def test_mlp_baseline():
    model = MLPBaseline(input_dim=8, hidden_sizes=[32, 16])
    x = torch.randn(10, 8)
    logits = model(x)
    assert logits.shape == (10,)
    probs = model.predict_proba(x)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_lstm_forward():
    model = ShotSequenceLSTM(
        shot_input_dim=6, score_input_dim=8,
        embedding_dim=16, hidden_dim=32, num_layers=1,
    )
    shots = torch.randn(4, 10, 6)
    lengths = torch.tensor([3, 5, 10, 1])
    scores = torch.randn(4, 8)
    logits = model(shots, lengths, scores)
    assert logits.shape == (4,)


def test_lstm_predict_proba():
    model = ShotSequenceLSTM(
        shot_input_dim=6, score_input_dim=8,
        embedding_dim=16, hidden_dim=32, num_layers=1,
    )
    shots = torch.randn(4, 10, 6)
    lengths = torch.tensor([3, 5, 10, 1])
    scores = torch.randn(4, 8)
    probs = model.predict_proba(shots, lengths, scores)
    assert probs.shape == (4,)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_transformer_forward():
    model = ShotSequenceTransformer(
        shot_input_dim=6, score_input_dim=8,
        d_model=32, nhead=4, num_layers=1, max_seq_len=10,
    )
    shots = torch.randn(4, 10, 6)
    lengths = torch.tensor([3, 5, 10, 1])
    scores = torch.randn(4, 8)
    logits = model(shots, lengths, scores)
    assert logits.shape == (4,)
