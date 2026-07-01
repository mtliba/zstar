import torch
import torch.nn.functional as F
from typing import Dict


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    sim.masked_fill_(torch.eye(2 * B, device=z.device, dtype=torch.bool), float("-inf"))
    labels = torch.cat([torch.arange(B, 2 * B, device=z.device), torch.arange(0, B, device=z.device)])
    return F.cross_entropy(sim, labels)


def contrastive_loss(
    zs: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    temperature: float = 0.07,
) -> torch.Tensor:
    if len(zs) < 2:
        return torch.tensor(0.0, device=next(iter(zs.values())).device)

    names = list(zs.keys())
    total = torch.tensor(0.0, device=next(iter(zs.values())).device)
    n_pairs = 0

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            both = masks[names[i]].bool() & masks[names[j]].bool()
            if both.sum() < 2:
                continue
            total = total + nt_xent_loss(zs[names[i]][both], zs[names[j]][both], temperature)
            n_pairs += 1

    return total / max(n_pairs, 1)
