import torch
from typing import Dict


def vq_aggregate_loss(vq_losses: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
    total = torch.tensor(0.0)
    for name, loss_dict in vq_losses.items():
        vq_l = loss_dict["vq_loss"]
        if total.device != vq_l.device:
            total = total.to(vq_l.device)
        total = total + vq_l
    return total / max(len(vq_losses), 1)
