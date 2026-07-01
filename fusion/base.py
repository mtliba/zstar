import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
from omegaconf import DictConfig


class BaseFusion(nn.Module, ABC):

    def __init__(self, modality_names: list, latent_dim: int, config: Optional[DictConfig] = None):
        super().__init__()
        self.modality_names = modality_names
        self.latent_dim = latent_dim

    @abstractmethod
    def forward(
        self,
        mus: Dict[str, torch.Tensor],
        log_vars: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu_fused, log_var_fused) each [B, latent_dim]."""
        ...
