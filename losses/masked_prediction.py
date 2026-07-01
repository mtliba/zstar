import torch
import torch.nn.functional as F
from typing import Dict


def masked_prediction_loss(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    mask_positions: Dict[str, torch.Tensor],
) -> torch.Tensor:
    total = torch.tensor(0.0)
    count = 0
    for name in predictions:
        pos = mask_positions[name]
        if pos.sum() == 0:
            continue
        pred = predictions[name]
        tgt = targets[name]
        if total.device != pred.device:
            total = total.to(pred.device)
        total = total + F.mse_loss(pred[pos], tgt[pos])
        count += 1
    return total / max(count, 1)
