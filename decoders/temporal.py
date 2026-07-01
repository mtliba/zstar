import torch
import torch.nn as nn
from typing import Optional
from omegaconf import DictConfig

from . import register_decoder
from .base import BaseDecoder


@register_decoder("temporal")
class TemporalDecoder(BaseDecoder):

    def __init__(self, latent_dim: int, output_dim: int, config: DictConfig):
        super().__init__(latent_dim, output_dim, config)
        hidden_dim = int(config.get("hidden_dim", 128))
        num_layers = int(config.get("num_layers", 2))
        max_seq_len = int(config.get("max_seq_len", 200))
        self.max_seq_len = max_seq_len
        decoder_type = str(config.get("type", "gru"))

        self.z_proj = nn.Linear(latent_dim, hidden_dim)

        if decoder_type == "gru":
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            self.rnn = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        z: torch.Tensor,
        target_timestamps: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = z.size(0)
        h = self.z_proj(z)  # [B, hidden_dim]

        if target_lengths is not None:
            T = int(target_lengths.max().item())
        elif target_timestamps is not None:
            T = target_timestamps.size(1)
        else:
            T = self.max_seq_len

        # Repeat z projection as input at each timestep
        decoder_input = h.unsqueeze(1).expand(B, T, -1)  # [B, T, hidden_dim]
        output, _ = self.rnn(decoder_input)  # [B, T, hidden_dim]
        return self.output_proj(output)  # [B, T, output_dim]
