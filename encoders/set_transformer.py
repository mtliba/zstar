import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from omegaconf import DictConfig

from . import register_encoder
from .base import BaseEncoder


class MultiheadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)
        q = self.norm1(q + self.dropout(out))
        q = self.norm2(q + self.dropout(self.ff(q)))
        return q


@register_encoder("set_transformer")
class SetTransformerEncoder(BaseEncoder):

    def __init__(self, input_dim: int, latent_dim: int, config: DictConfig):
        super().__init__(input_dim, latent_dim, config)
        d_model = int(config.get("d_model", 64))
        num_heads = int(config.get("num_heads", 4))
        num_inducing = int(config.get("num_inducing", 32))
        num_layers = int(config.get("num_layers", 2))
        dropout = float(config.get("dropout", 0.1))

        self.input_proj = nn.Linear(input_dim, d_model)
        self.inducing_points = nn.Parameter(torch.randn(1, num_inducing, d_model))

        self.enc_layers = nn.ModuleList([
            MultiheadAttentionBlock(d_model, num_heads, dropout) for _ in range(num_layers)
        ])
        self.pool_attn = MultiheadAttentionBlock(d_model, num_heads, dropout)
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))

        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_log_var = nn.Linear(d_model, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        h = self.input_proj(x)  # [B, T, d_model]

        mask = None
        if lengths is not None:
            T = h.size(1)
            safe_lengths = lengths.clamp(min=1)
            mask = torch.arange(T, device=h.device).unsqueeze(0) >= safe_lengths.unsqueeze(1)

        inducing = self.inducing_points.expand(B, -1, -1)
        for layer in self.enc_layers:
            inducing = layer(inducing, h, key_padding_mask=mask)

        query = self.pool_query.expand(B, -1, -1)
        pooled = self.pool_attn(query, inducing)[:, 0]  # [B, d_model]

        return self.fc_mu(pooled), self.fc_log_var(pooled)
