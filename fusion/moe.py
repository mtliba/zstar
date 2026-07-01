import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from omegaconf import DictConfig

from . import register_fusion
from .base import BaseFusion


@register_fusion("moe")
class MoEFusion(BaseFusion):

    def __init__(self, modality_names: list, latent_dim: int, config: Optional[DictConfig] = None):
        super().__init__(modality_names, latent_dim, config)
        n = len(modality_names)
        self.gate = nn.Sequential(
            nn.Linear(n * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, n),
        )

    def forward(
        self,
        mus: Dict[str, torch.Tensor],
        log_vars: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = next(iter(mus.values())).shape[0]
        device = next(iter(mus.values())).device

        mu_list, lv_list, mask_list = [], [], []
        gate_input_parts = []

        for name in self.modality_names:
            if name in mus:
                m = masks[name].unsqueeze(-1)
                mu_list.append(mus[name])
                lv_list.append(log_vars[name])
                mask_list.append(masks[name])
                gate_input_parts.append(mus[name] * m)
            else:
                mu_list.append(torch.zeros(B, self.latent_dim, device=device))
                lv_list.append(torch.zeros(B, self.latent_dim, device=device))
                mask_list.append(torch.zeros(B, device=device))
                gate_input_parts.append(torch.zeros(B, self.latent_dim, device=device))

        mu_stack = torch.stack(mu_list, dim=1)      # [B, N, D]
        lv_stack = torch.stack(lv_list, dim=1)       # [B, N, D]
        mask_stack = torch.stack(mask_list, dim=1)   # [B, N]

        gate_input = torch.cat(gate_input_parts, dim=-1)
        weights = self.gate(gate_input)  # [B, N]
        weights = weights.masked_fill(mask_stack == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)  # [B, N]

        w = weights.unsqueeze(-1)  # [B, N, 1]
        mu_fused = (w * mu_stack).sum(dim=1)
        log_var_fused = (w * lv_stack).sum(dim=1)
        return mu_fused, log_var_fused
