import torch
from typing import Dict, Tuple, Optional
from omegaconf import DictConfig

from . import register_fusion
from .base import BaseFusion


@register_fusion("poe")
class PoEFusion(BaseFusion):

    def __init__(self, modality_names: list, latent_dim: int, config: Optional[DictConfig] = None):
        super().__init__(modality_names, latent_dim, config)

    def forward(
        self,
        mus: Dict[str, torch.Tensor],
        log_vars: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = next(iter(mus.values())).shape[0]
        device = next(iter(mus.values())).device

        prec_sum = torch.ones(B, self.latent_dim, device=device)
        mu_prec_sum = torch.zeros(B, self.latent_dim, device=device)

        for name in mus:
            prec_i = torch.exp(-log_vars[name])
            mask_i = masks[name].unsqueeze(-1)
            prec_sum = prec_sum + mask_i * prec_i
            mu_prec_sum = mu_prec_sum + mask_i * prec_i * mus[name]

        mu_combined = mu_prec_sum / prec_sum
        log_var_combined = -torch.log(prec_sum)
        return mu_combined, log_var_combined
