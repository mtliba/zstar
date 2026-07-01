from omegaconf import DictConfig
from .base import BaseDecoder


_REGISTRY = {}


def register_decoder(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_decoder(name: str, latent_dim: int, output_dim: int, config: DictConfig) -> BaseDecoder:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown decoder '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](latent_dim=latent_dim, output_dim=output_dim, config=config)


from .mlp import MLPDecoder
from .temporal import TemporalDecoder
