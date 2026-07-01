import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from omegaconf import DictConfig

from . import register_fusion
from .base import BaseFusion


@register_fusion("concat")
class ConcatFusionWrapper(BaseFusion):

    def __init__(self, modality_names: list, latent_dim: int, config: Optional[DictConfig] = None):
        super().__init__(modality_names, latent_dim, config)
        n = len(modality_names)
        self.mlp = nn.Sequential(
            nn.Linear(n * latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim * 2),
        )

    def forward(
        self,
        mus: Dict[str, torch.Tensor],
        log_vars: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = next(iter(mus.values())).shape[0]
        device = next(iter(mus.values())).device

        parts = []
        for name in self.modality_names:
            if name in mus:
                mask_i = masks[name].unsqueeze(-1)
                parts.append(mus[name] * mask_i)
            else:
                parts.append(torch.zeros(B, self.latent_dim, device=device))

        out = self.mlp(torch.cat(parts, dim=-1))
        mu = out[:, :self.latent_dim]
        log_var = out[:, self.latent_dim:]
        return mu, log_var
