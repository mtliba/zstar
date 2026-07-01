from omegaconf import DictConfig
from .base import BaseEncoder


_REGISTRY = {}


def register_encoder(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_encoder(name: str, input_dim: int, latent_dim: int, config: DictConfig) -> BaseEncoder:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown encoder '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](input_dim=input_dim, latent_dim=latent_dim, config=config)


from .mlp import MLPEncoder
from .gru import GRUEncoder
from .lstm import LSTMEncoder
from .transformer import TransformerTemporalEncoder
from .tcn import TCNEncoder
from .set_transformer import SetTransformerEncoder
