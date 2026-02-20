"""Comprehensive tests for model architectures."""

import numpy as np
import pytest
import torch

from tennis_miner.core.schema import DataBundle
from tennis_miner.models.baseline import LogisticBaseline
from tennis_miner.models.lstm import LSTMModel, LSTMNetwork
from tennis_miner.models.transformer import TransformerModel, TransformerNetwork
from tennis_miner.models.registry import create_model, list_models


def _make_bundle(n=50, max_rally=10):
    return DataBundle(
        score_features=np.random.randn(n, 8).astype(np.float32),
        shot_sequences=np.random.randn(n, max_rally, 6).astype(np.float32),
        sequence_lengths=np.random.randint(1, max_rally + 1, n).astype(np.int32),
        labels=(np.random.rand(n) > 0.5).astype(np.float32),
        match_ids=[f"m{i % 5}" for i in range(n)],
    )


class TestLogisticBaseline:
    def test_fit_predict(self):
        m = LogisticBaseline()
        train, val = _make_bundle(80), _make_bundle(20)
        m.fit(train, val)
        preds = m.predict_proba(val)
        assert preds.shape == (20,)
        assert 0 <= preds.min() <= preds.max() <= 1

    def test_predict_before_fit_raises(self):
        m = LogisticBaseline()
        with pytest.raises(RuntimeError, match="not fitted"):
            m.predict_proba(_make_bundle(5))

    def test_save_load(self, tmp_path):
        m = LogisticBaseline()
        train, val = _make_bundle(80), _make_bundle(20)
        m.fit(train, val)
        p1 = m.predict_proba(val)

        path = str(tmp_path / "model.joblib")
        m.save(path)

        m2 = LogisticBaseline()
        m2.load(path)
        p2 = m2.predict_proba(val)
        np.testing.assert_array_almost_equal(p1, p2)


class TestLSTMNetwork:
    def test_forward_shape(self):
        net = LSTMNetwork(shot_dim=6, score_dim=8, embed_dim=16, hidden_dim=32, num_layers=1)
        shots = torch.randn(4, 10, 6)
        lengths = torch.tensor([3, 5, 10, 1])
        scores = torch.randn(4, 8)
        out = net(shots, lengths, scores)
        assert out.shape == (4,)

    def test_single_sample(self):
        net = LSTMNetwork(embed_dim=8, hidden_dim=16, num_layers=1)
        out = net(torch.randn(1, 5, 6), torch.tensor([3]), torch.randn(1, 8))
        assert out.shape == (1,)

    def test_all_length_one(self):
        """Edge case: all sequences have length 1."""
        net = LSTMNetwork(embed_dim=8, hidden_dim=16, num_layers=1)
        out = net(torch.randn(3, 10, 6), torch.tensor([1, 1, 1]), torch.randn(3, 8))
        assert out.shape == (3,)


class TestLSTMModel:
    def test_fit_predict(self):
        m = LSTMModel(embed_dim=8, hidden_dim=16, num_layers=1, epochs=3, batch_size=20)
        train, val = _make_bundle(40), _make_bundle(10)
        history = m.fit(train, val)
        assert "train_loss" in history
        preds = m.predict_proba(val)
        assert preds.shape == (10,)
        assert 0 <= preds.min() <= preds.max() <= 1

    def test_predict_before_fit(self):
        m = LSTMModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            m.predict_proba(_make_bundle(5))


class TestTransformerNetwork:
    def test_forward_shape(self):
        net = TransformerNetwork(d_model=32, nhead=4, num_layers=1, max_seq_len=10)
        out = net(torch.randn(4, 10, 6), torch.tensor([3, 5, 10, 1]), torch.randn(4, 8))
        assert out.shape == (4,)

    def test_min_length(self):
        net = TransformerNetwork(d_model=16, nhead=4, num_layers=1, max_seq_len=5)
        out = net(torch.randn(2, 5, 6), torch.tensor([1, 1]), torch.randn(2, 8))
        assert out.shape == (2,)


class TestRegistry:
    def test_list_models(self):
        models = list_models()
        assert "logistic_baseline" in models
        assert "lstm" in models
        assert "transformer" in models

    def test_create_baseline(self):
        m = create_model("logistic_baseline")
        assert isinstance(m, LogisticBaseline)

    def test_unknown_model(self):
        with pytest.raises(ValueError, match="Unknown model"):
            create_model("nonexistent_model")
