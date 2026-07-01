from omegaconf import DictConfig
from .base import BaseFusion


_REGISTRY = {}


def register_fusion(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_fusion(name: str, modality_names: list, latent_dim: int, config: DictConfig = None) -> BaseFusion:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown fusion '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](modality_names=modality_names, latent_dim=latent_dim, config=config)


from .poe import PoEFusion
from .concat import ConcatFusionWrapper
from .moe import MoEFusion
from .attention import AttentionFusion
