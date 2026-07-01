import torch
import torch.nn.functional as F
from typing import Dict
from .contrastive import nt_xent_loss


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    return torch.exp(-(diff ** 2).sum(-1) / (2 * sigma ** 2))


def mmd_loss(z1: torch.Tensor, z2: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    K11 = _rbf_kernel(z1, z1, sigma)
    K22 = _rbf_kernel(z2, z2, sigma)
    K12 = _rbf_kernel(z1, z2, sigma)
    n, m = z1.shape[0], z2.shape[0]
    mmd = (
        (K11.sum() - K11.trace()) / (n * (n - 1) + 1e-8)
        + (K22.sum() - K22.trace()) / (m * (m - 1) + 1e-8)
        - 2 * K12.mean()
    )
    return mmd.clamp(min=0.0)


def alignment_loss(
    zs: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    strategy: str = "mmd",
    temperature: float = 0.07,
) -> torch.Tensor:
    if strategy == "none" or len(zs) < 2:
        return torch.tensor(0.0, device=next(iter(zs.values())).device)

    names = list(zs.keys())
    total = torch.tensor(0.0, device=next(iter(zs.values())).device)
    n_pairs = 0

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            both = masks[names[i]].bool() & masks[names[j]].bool()
            if both.sum() < 2:
                continue
            zi, zj = zs[names[i]][both], zs[names[j]][both]
            if strategy == "mmd":
                total = total + mmd_loss(zi, zj)
            elif strategy == "contrastive":
                total = total + nt_xent_loss(zi, zj, temperature)
            else:
                raise ValueError(f"Unknown alignment strategy '{strategy}'")
            n_pairs += 1

    return total / max(n_pairs, 1)
