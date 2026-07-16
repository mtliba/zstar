import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def temporal_prediction_loss(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    valid: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    MSE between forecast and the held-out future.

    `valid` marks subjects whose sequence was long enough to supply both a past
    to encode and a full horizon to predict. Subjects without one contribute a
    target gathered from clamped indices, which would be meaningless
    supervision -- they are excluded rather than fitted.
    """
    total = torch.tensor(0.0)
    count = 0
    for name in predictions:
        pred, tgt = predictions[name], targets[name]
        if total.device != pred.device:
            total = total.to(pred.device)

        if valid is not None and name in valid:
            m = valid[name]
            if not bool(m.any()):
                continue
            pred, tgt = pred[m], tgt[m]

        total = total + F.mse_loss(pred, tgt)
        count += 1
    return total / max(count, 1)


class TemporalPredictionHead(nn.Module):

    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 128, prediction_horizon: int = 5):
        super().__init__()
        self.prediction_horizon = prediction_horizon
        self.z_proj = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.z_proj(z).unsqueeze(1).expand(-1, self.prediction_horizon, -1)
        out, _ = self.gru(h)
        return self.output_proj(out)  # [B, H, output_dim]
