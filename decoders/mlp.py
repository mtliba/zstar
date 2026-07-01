import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'mmvae_pharm'))

import torch
from typing import Optional
from omegaconf import DictConfig

from models.modality_vae import build_mlp
from . import register_decoder
from .base import BaseDecoder


@register_decoder("mlp")
class MLPDecoder(BaseDecoder):

    def __init__(self, latent_dim: int, output_dim: int, config: DictConfig):
        super().__init__(latent_dim, output_dim, config)
        hidden_dims = list(config.get("hidden_dims", [128, 256]))
        dropout = float(config.get("dropout", 0.1))
        self.net = build_mlp(latent_dim, hidden_dims, output_dim, dropout)

    def forward(
        self,
        z: torch.Tensor,
        target_timestamps: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.net(z)
