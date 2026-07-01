import torch
import torch.nn as nn
from typing import Optional, Tuple
from omegaconf import DictConfig

from . import register_encoder
from .base import BaseEncoder
from .time_encoding import get_time_encoding


@register_encoder("transformer")
class TransformerTemporalEncoder(BaseEncoder):

    def __init__(self, input_dim: int, latent_dim: int, config: DictConfig):
        super().__init__(input_dim, latent_dim, config)
        d_model = int(config.get("d_model", 128))
        nhead = int(config.get("nhead", 4))
        num_layers = int(config.get("num_layers", 3))
        dim_feedforward = int(config.get("dim_feedforward", 256))
        dropout = float(config.get("dropout", 0.1))
        self.pooling = str(config.get("pooling", "cls"))

        self.input_proj = nn.Linear(input_dim, d_model)
        self.time_enc = get_time_encoding(
            str(config.get("time_encoding", "sinusoidal")), d_model
        )

        if self.pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_log_var = nn.Linear(d_model, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        h = self.input_proj(x)
        if timestamps is not None:
            h = h + self.time_enc(timestamps)

        if self.pooling == "cls":
            cls = self.cls_token.expand(B, -1, -1)
            h = torch.cat([cls, h], dim=1)  # [B, 1+T, d_model]
            T_ext = T + 1
        else:
            T_ext = T

        mask = None
        if lengths is not None:
            seq_len = lengths.clamp(min=1) + (1 if self.pooling == "cls" else 0)
            mask = torch.arange(T_ext, device=x.device).unsqueeze(0) >= seq_len.unsqueeze(1)

        h = self.transformer(h, src_key_padding_mask=mask)
        h = self.layer_norm(h)

        pooled = self._pool(h, lengths, T_ext)
        return self.fc_mu(pooled), self.fc_log_var(pooled)

    def _pool(self, h: torch.Tensor, lengths: Optional[torch.Tensor], T_ext: int) -> torch.Tensor:
        if self.pooling == "cls":
            return h[:, 0]
        elif self.pooling == "mean":
            if lengths is not None:
                mask = torch.arange(T_ext, device=h.device).unsqueeze(0) < lengths.unsqueeze(1)
                return (h * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(-1).clamp(min=1)
            return h.mean(dim=1)
        elif self.pooling == "last":
            if lengths is not None:
                idx = (lengths - 1).long().clamp(min=0)
                return h[torch.arange(h.size(0), device=h.device), idx]
            return h[:, -1]
        raise ValueError(f"Unknown pooling '{self.pooling}'")
