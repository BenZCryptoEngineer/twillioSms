"""Integration test: end-to-end pipeline with synthetic data."""

import numpy as np
import pytest

from tennis_miner.core.schema import (
    Match, Point, Shot, Surface, DataBundle,
    ShotType, ShotDirection, ShotOutcome,
)
from tennis_miner.features.sequence import build_dataset
from tennis_miner.models.baseline import LogisticBaseline
from tennis_miner.models.lstm import LSTMModel
from tennis_miner.evaluation.kill_point import evaluate_kill_point_1


def _generate_synthetic_matches(n_matches=10, points_per_match=20):
    """Generate synthetic match data for integration testing."""
    matches = []
    for mi in range(n_matches):
        m = Match(
            match_id=f"syn_{mi}", tournament="test", year=2023,
            surface=Surface.HARD, player1="A", player2="B",
            winner="A", score="6-4",
        )
        for pi in range(points_per_match):
            rally_len = np.random.randint(1, 8)
            shots = []
            for si in range(rally_len):
                player = "server" if si % 2 == 0 else "returner"
                outcome = ShotOutcome.IN_PLAY
                if si == rally_len - 1:
                    outcome = np.random.choice([
                        ShotOutcome.WINNER, ShotOutcome.UNFORCED_ERROR,
                    ])
                shots.append(Shot(
                    shot_num=si + 1,
                    player=player,
                    shot_type=np.random.choice(list(ShotType)),
                    direction=np.random.choice(list(ShotDirection)),
                    outcome=outcome,
                ))

            last = shots[-1]
            if last.outcome == ShotOutcome.WINNER:
                server_won = last.player == "server"
            else:
                server_won = last.player != "server"

            m.points.append(Point(
                point_id=f"syn_{mi}_p{pi}", match_id=f"syn_{mi}",
                set_num=1, game_num=(pi // 4) + 1, point_num=(pi % 4) + 1,
                server="A", returner="B",
                server_won=server_won, shots=shots,
            ))
        matches.append(m)
    return matches


class TestEndToEndPipeline:
    """Tests the full data → features → model → evaluation flow."""

    def test_feature_engineering_produces_valid_bundle(self):
        matches = _generate_synthetic_matches(5, 10)
        bundle = build_dataset(matches, max_rally_length=20)
        assert bundle.n_samples == 50
        assert bundle.score_features.shape[1] == 8
        assert bundle.shot_sequences.shape[1] == 20
        assert 0.0 <= bundle.server_win_rate <= 1.0

    def test_baseline_trains_on_bundle(self):
        matches = _generate_synthetic_matches(5, 20)
        bundle = build_dataset(matches)
        mask = np.array([i < 80 for i in range(bundle.n_samples)])
        train = bundle.subset(mask)
        val = bundle.subset(~mask)

        model = LogisticBaseline()
        model.fit(train, val)
        preds = model.predict_proba(val)
        assert len(preds) == val.n_samples
        assert all(0 <= p <= 1 for p in preds)

    def test_lstm_trains_on_bundle(self):
        matches = _generate_synthetic_matches(5, 20)
        bundle = build_dataset(matches, max_rally_length=15)
        mask = np.array([i < 80 for i in range(bundle.n_samples)])
        train = bundle.subset(mask)
        val = bundle.subset(~mask)

        model = LSTMModel(
            embed_dim=8, hidden_dim=16, num_layers=1,
            epochs=3, batch_size=20,
        )
        history = model.fit(train, val)
        assert len(history["train_loss"]) > 0

        preds = model.predict_proba(val)
        assert len(preds) == val.n_samples

    def test_kill_point_evaluation_runs(self):
        np.random.seed(42)
        n = 100
        y = np.random.binomial(1, 0.5, n).astype(float)
        a = np.clip(0.5 + np.random.randn(n) * 0.2, 0.01, 0.99)
        b = np.clip(0.5 + np.random.randn(n) * 0.2, 0.01, 0.99)
        result = evaluate_kill_point_1(y, a, b)
        assert "kill_point_1" in result
        assert isinstance(result["kill_point_1"]["OVERALL_PASS"], bool)

    def test_full_flow(self):
        """Full pipeline: synthetic data → features → baseline + LSTM → KP1 eval."""
        matches = _generate_synthetic_matches(8, 30)
        bundle = build_dataset(matches, max_rally_length=15)

        # Split by match
        unique = list(set(bundle.match_ids))
        train_ids = set(unique[:6])
        test_ids = set(unique[6:])
        train_mask = np.array([m in train_ids for m in bundle.match_ids])
        test_mask = ~train_mask

        train = bundle.subset(train_mask)
        test = bundle.subset(test_mask)

        # Baseline
        baseline = LogisticBaseline()
        baseline.fit(train, test)
        b_preds = baseline.predict_proba(test)

        # LSTM
        lstm = LSTMModel(
            embed_dim=8, hidden_dim=16, num_layers=1,
            epochs=5, batch_size=32,
        )
        lstm.fit(train, test)
        s_preds = lstm.predict_proba(test)

        # KP1
        result = evaluate_kill_point_1(test.labels, b_preds, s_preds)
        assert "kill_point_1" in result
        assert "baseline" in result
        assert "comparison" in result
