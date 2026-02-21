# Import all models so they register with the registry
from tennis_miner.models.baseline import LogisticBaseline  # noqa: F401
from tennis_miner.models.lstm import LSTMModel  # noqa: F401
from tennis_miner.models.transformer import TransformerModel  # noqa: F401
