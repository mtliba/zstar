import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from omegaconf import DictConfig


class BaseEncoder(nn.Module, ABC):

    def __init__(self, input_dim: int, latent_dim: int, config: DictConfig):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.config = config

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, D] for static, [B, T, D] for temporal/event
        timestamps: [B, T] for irregular sampling (None for static)
        lengths: [B] actual sequence lengths (None for static)
        Returns: (mu, log_var) each [B, latent_dim]
        """
        ...
