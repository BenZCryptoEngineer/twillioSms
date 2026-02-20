"""
Abstract base classes defining contracts between modules.

Every module implements one of these interfaces. Orchestration wires them.
"""

from abc import ABC, abstractmethod

import numpy as np

from tennis_miner.core.schema import Match, DataBundle


class BaseLoader(ABC):
    """Contract: raw files → list[Match]."""

    @abstractmethod
    def load(self) -> list[Match]:
        """Load and return validated Match objects."""
        ...

    @abstractmethod
    def validate(self, matches: list[Match]) -> list[str]:
        """Return list of validation warnings (empty = clean)."""
        ...


class BaseModel(ABC):
    """Contract: numeric arrays → probability predictions."""

    @abstractmethod
    def fit(self, train: DataBundle, val: DataBundle) -> dict:
        """Train on data. Return training history dict."""
        ...

    @abstractmethod
    def predict_proba(self, data: DataBundle) -> np.ndarray:
        """Return P(server wins) for each sample. Shape: (N,)."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...


class BaseEvaluator(ABC):
    """Contract: (y_true, y_pred) → metric dict."""

    @abstractmethod
    def evaluate(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict:
        ...
