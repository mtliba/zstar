import torch
from typing import Tuple


class MaskGenerator:

    def __init__(self, mask_ratio: float = 0.15, strategy: str = "random"):
        self.mask_ratio = mask_ratio
        self.strategy = strategy

    def __call__(
        self, x: torch.Tensor, mod_type: str = "static"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mod_type == "static":
            return self._mask_static(x)
        return self._mask_temporal(x)

    def _mask_static(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, D]
        mask = torch.rand_like(x) < self.mask_ratio
        x_masked = x.clone()
        x_masked[mask] = 0.0
        return x_masked, mask

    def _mask_temporal(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, D]
        B, T, D = x.shape
        x_masked = x.clone()

        if self.strategy == "random":
            mask = torch.rand(B, T, D, device=x.device) < self.mask_ratio
        elif self.strategy == "block":
            mask = torch.zeros(B, T, D, device=x.device, dtype=torch.bool)
            block_size = max(1, int(T * self.mask_ratio))
            for i in range(B):
                start = torch.randint(0, max(1, T - block_size), (1,)).item()
                mask[i, start:start + block_size, :] = True
        elif self.strategy == "feature_wise":
            mask = torch.zeros(B, T, D, device=x.device, dtype=torch.bool)
            n_masked = max(1, int(D * self.mask_ratio))
            for i in range(B):
                idx = torch.randperm(D)[:n_masked]
                mask[i, :, idx] = True
        else:
            raise ValueError(f"Unknown masking strategy '{self.strategy}'")

        x_masked[mask] = 0.0
        return x_masked, mask
