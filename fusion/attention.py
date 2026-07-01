import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from omegaconf import DictConfig

from . import register_fusion
from .base import BaseFusion


@register_fusion("attention")
class AttentionFusion(BaseFusion):

    def __init__(self, modality_names: list, latent_dim: int, config: Optional[DictConfig] = None):
        super().__init__(modality_names, latent_dim, config)
        nhead = 4
        self.query = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.attn = nn.MultiheadAttention(latent_dim, nhead, batch_first=True)
        self.norm = nn.LayerNorm(latent_dim)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_log_var = nn.Linear(latent_dim, latent_dim)

    def forward(
        self,
        mus: Dict[str, torch.Tensor],
        log_vars: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = next(iter(mus.values())).shape[0]
        device = next(iter(mus.values())).device

        mu_list, mask_list = [], []
        for name in self.modality_names:
            if name in mus:
                mu_list.append(mus[name])
                mask_list.append(masks[name])
            else:
                mu_list.append(torch.zeros(B, self.latent_dim, device=device))
                mask_list.append(torch.zeros(B, device=device))

        kv = torch.stack(mu_list, dim=1)         # [B, N, D]
        key_padding_mask = torch.stack(mask_list, dim=1) == 0  # [B, N], True=ignore

        query = self.query.expand(B, -1, -1)
        out, _ = self.attn(query, kv, kv, key_padding_mask=key_padding_mask)
        out = self.norm(out[:, 0])  # [B, D]

        return self.fc_mu(out), self.fc_log_var(out)
