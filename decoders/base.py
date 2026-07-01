import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional
from omegaconf import DictConfig


class BaseDecoder(nn.Module, ABC):

    def __init__(self, latent_dim: int, output_dim: int, config: DictConfig):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.config = config

    @abstractmethod
    def forward(
        self,
        z: torch.Tensor,
        target_timestamps: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        z: [B, latent_dim]
        Returns: [B, D] for static, [B, T, D] for temporal
        """
        ...
