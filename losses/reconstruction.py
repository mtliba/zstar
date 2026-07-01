import torch
import torch.nn.functional as F
from typing import Optional, Dict


def reconstruction_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mask: torch.Tensor,
    loss_fn: str = "mse",
    mod_type: str = "static",
    lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    present = mask.bool()
    if present.sum() == 0:
        return torch.tensor(0.0, device=x.device)

    if x_recon.dim() != x.dim():
        # Flatten both to [B, -1] for shape-agnostic comparison
        x_flat = x[present].reshape(present.sum(), -1)
        r_flat = x_recon[present].reshape(present.sum(), -1)
        min_d = min(x_flat.shape[1], r_flat.shape[1])
        return _compute_loss_fn(r_flat[:, :min_d], x_flat[:, :min_d], loss_fn).mean()

    if mod_type == "static" or x.dim() == 2:
        per_sample = _compute_loss_fn(x_recon, x, loss_fn).mean(dim=-1)
        return per_sample[present].mean()
    else:
        if lengths is not None:
            total = torch.tensor(0.0, device=x.device)
            count = 0
            T_recon = x_recon.size(1)
            for i in range(x.size(0)):
                if not present[i]:
                    continue
                L = min(int(lengths[i].item()), T_recon, x.size(1))
                if L == 0:
                    continue
                total = total + _compute_loss_fn(x_recon[i, :L], x[i, :L], loss_fn).mean()
                count += 1
            return total / max(count, 1)
        else:
            T_min = min(x_recon.size(1), x.size(1))
            per_sample = _compute_loss_fn(x_recon[:, :T_min], x[:, :T_min], loss_fn).mean(dim=(-2, -1))
            return per_sample[present].mean()


def _compute_loss_fn(pred: torch.Tensor, target: torch.Tensor, loss_fn: str) -> torch.Tensor:
    if loss_fn == "mse":
        return F.mse_loss(pred, target, reduction="none")
    elif loss_fn == "bce":
        return F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    elif loss_fn == "huber":
        return F.smooth_l1_loss(pred, target, reduction="none")
    raise ValueError(f"Unknown loss_fn '{loss_fn}'")


def masked_recon_loss(
    masked_recons: Dict[str, torch.Tensor],
    mask_positions: Dict[str, torch.Tensor],
    batch: Dict,
) -> torch.Tensor:
    total = torch.tensor(0.0, device=next(iter(masked_recons.values())).device)
    count = 0
    for name, recon in masked_recons.items():
        positions = mask_positions[name]
        x_orig = batch[name]["x"]
        if positions.sum() > 0:
            total = total + F.mse_loss(recon[positions], x_orig[positions])
            count += 1
    return total / max(count, 1)
