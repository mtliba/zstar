import torch
import torch.nn as nn
from typing import Optional, Tuple
from omegaconf import DictConfig

from zstar._mlp_utils import build_mlp
from . import register_encoder
from .base import BaseEncoder


@register_encoder("mlp")
class MLPEncoder(BaseEncoder):

    def __init__(self, input_dim: int, latent_dim: int, config: DictConfig):
        super().__init__(input_dim, latent_dim, config)
        hidden_dims = list(config.get("hidden_dims", [256, 128]))
        dropout = float(config.get("dropout", 0.1))

        self.net = build_mlp(input_dim, hidden_dims, hidden_dims[-1], dropout)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.fc_mu(h), self.fc_log_var(h)
