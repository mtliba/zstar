import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional, Tuple
from omegaconf import DictConfig

from . import register_encoder
from .base import BaseEncoder
from .time_encoding import get_time_encoding


@register_encoder("gru")
class GRUEncoder(BaseEncoder):

    def __init__(self, input_dim: int, latent_dim: int, config: DictConfig):
        super().__init__(input_dim, latent_dim, config)
        hidden_dim = int(config.get("hidden_dim", 128))
        num_layers = int(config.get("num_layers", 2))
        bidirectional = bool(config.get("bidirectional", False))
        dropout = float(config.get("dropout", 0.1))
        self.pooling = str(config.get("pooling", "last"))

        d_model = hidden_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        self.time_enc = get_time_encoding(
            str(config.get("time_encoding", "sinusoidal")), d_model
        )

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc_mu = nn.Linear(out_dim, latent_dim)
        self.fc_log_var = nn.Linear(out_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        if timestamps is not None:
            h = h + self.time_enc(timestamps)

        if lengths is not None:
            lengths_cpu = lengths.clamp(min=1).cpu()
            packed = pack_padded_sequence(h, lengths_cpu, batch_first=True, enforce_sorted=False)
            output, _ = self.gru(packed)
            output, _ = pad_packed_sequence(output, batch_first=True)
        else:
            output, _ = self.gru(h)

        pooled = self._pool(output, lengths)
        return self.fc_mu(pooled), self.fc_log_var(pooled)

    def _pool(self, output: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
        if self.pooling == "last":
            if lengths is not None:
                idx = (lengths - 1).long().clamp(min=0)
                return output[torch.arange(output.size(0), device=output.device), idx]
            return output[:, -1]
        elif self.pooling == "mean":
            if lengths is not None:
                mask = torch.arange(output.size(1), device=output.device).unsqueeze(0) < lengths.unsqueeze(1)
                return (output * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(-1).clamp(min=1)
            return output.mean(dim=1)
        elif self.pooling == "cls":
            return output[:, 0]
        raise ValueError(f"Unknown pooling '{self.pooling}'")
