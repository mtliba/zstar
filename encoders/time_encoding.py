import math
import torch
import torch.nn as nn


class SinusoidalTimeEncoding(nn.Module):

    def __init__(self, d_model: int, max_period: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """timestamps [B, T] → [B, T, d_model]"""
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=timestamps.device, dtype=torch.float32) / half
        )
        args = timestamps.unsqueeze(-1) * freqs  # [B, T, half]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class LearnableTimeEncoding(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """timestamps [B, T] → [B, T, d_model]"""
        return self.net(timestamps.unsqueeze(-1))


def get_time_encoding(name: str, d_model: int) -> nn.Module:
    if name == "sinusoidal":
        return SinusoidalTimeEncoding(d_model)
    elif name == "learnable":
        return LearnableTimeEncoding(d_model)
    raise ValueError(f"Unknown time encoding '{name}'. Choose: sinusoidal | learnable")
