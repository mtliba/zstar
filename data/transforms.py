import torch
from typing import Optional


class TemporalJitter:

    def __init__(self, sigma: float = 0.01):
        self.sigma = sigma

    def __call__(self, timestamps: torch.Tensor) -> torch.Tensor:
        return timestamps + torch.randn_like(timestamps) * self.sigma


class FeatureDropout:

    def __init__(self, p: float = 0.05):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not x.requires_grad:
            x = x.clone()
        mask = torch.rand_like(x) > self.p
        return x * mask


class TimeWarp:

    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma

    def __call__(self, timestamps: torch.Tensor) -> torch.Tensor:
        warp = 1.0 + torch.randn(1, device=timestamps.device) * self.sigma
        return timestamps * warp.clamp(min=0.5, max=2.0)
