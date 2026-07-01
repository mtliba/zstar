import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from omegaconf import DictConfig

from . import register_encoder
from .base import BaseEncoder
from .time_encoding import get_time_encoding


class CausalConv1dBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        self.chomp = padding
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.chomp > 0:
            out = out[:, :, :-self.chomp]
        out = self.dropout(self.act(self.norm(out)))
        return out + self.residual(x)


@register_encoder("tcn")
class TCNEncoder(BaseEncoder):

    def __init__(self, input_dim: int, latent_dim: int, config: DictConfig):
        super().__init__(input_dim, latent_dim, config)
        num_channels: List[int] = list(config.get("num_channels", [64, 64, 128]))
        kernel_size = int(config.get("kernel_size", 3))
        dropout = float(config.get("dropout", 0.1))

        d_model = num_channels[0]
        self.input_proj = nn.Linear(input_dim, d_model)
        self.time_enc = get_time_encoding(
            str(config.get("time_encoding", "sinusoidal")), d_model
        )

        layers = []
        in_ch = d_model
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(CausalConv1dBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(num_channels[-1], latent_dim)
        self.fc_log_var = nn.Linear(num_channels[-1], latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        if timestamps is not None:
            h = h + self.time_enc(timestamps)

        # TCN expects [B, C, T]
        h = h.transpose(1, 2)
        h = self.tcn(h)  # [B, C_out, T]

        # Pool: take last valid position
        if lengths is not None:
            idx = (lengths - 1).long().clamp(min=0)
            pooled = h[torch.arange(h.size(0), device=h.device), :, idx]
        else:
            pooled = h[:, :, -1]

        return self.fc_mu(pooled), self.fc_log_var(pooled)
