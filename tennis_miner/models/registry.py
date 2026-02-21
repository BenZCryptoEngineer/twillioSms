"""
Model registry — factory for creating model instances from config.
"""

from tennis_miner.core.interfaces import BaseModel


_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Decorator to register a model class."""
    def wrapper(cls):
        _REGISTRY[name] = cls
        return cls
    return wrapper


def create_model(name: str, **kwargs) -> BaseModel:
    """Create a model instance by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return _REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    return sorted(_REGISTRY.keys())
