import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict
from zstar.data.collate import zstar_collate


@torch.no_grad()
def cross_modal_reconstruction(
    model,
    dataset,
    source_modalities: list,
    target_modality: str,
    batch_size: int = 256,
    device=None,
) -> Dict[str, float]:
    from sklearn.metrics import r2_score

    if device is None:
        device = next(model.parameters()).device
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=zstar_collate)

    y_true_list, y_pred_list = [], []

    for batch in loader:
        batch_dev = {
            name: {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
            for name, item in batch.items()
        }

        # Only keep source modalities for encoding
        source_batch = {name: batch_dev[name] for name in source_modalities if name in batch_dev}
        if not source_batch or target_modality not in batch_dev:
            continue

        target_mask = batch_dev[target_modality]["mask"].bool()
        source_masks = [batch_dev[s]["mask"].bool() for s in source_modalities if s in batch_dev]
        if not source_masks:
            continue
        valid = target_mask
        for sm in source_masks:
            valid = valid & sm
        if valid.sum() == 0:
            continue

        z_star = model.extract_zstar(source_batch)
        pred = model.modules_dict[target_modality].decode(z_star)

        target_x = batch_dev[target_modality]["x"]
        # Handle both static [B, D] and temporal [B, T, D]
        if target_x.dim() == 2:
            y_true_list.append(target_x[valid].cpu().numpy())
            y_pred_list.append(pred[valid].cpu().numpy())
        else:
            for i in range(target_x.size(0)):
                if valid[i]:
                    y_true_list.append(target_x[i].cpu().numpy().reshape(1, -1))
                    y_pred_list.append(pred[i].cpu().numpy().reshape(1, -1))

    if not y_true_list:
        return {"mse": float("nan"), "r2": float("nan")}

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    mse = float(np.mean((y_true - y_pred) ** 2))
    r2 = float(r2_score(y_true.ravel(), y_pred.ravel()))
    return {"mse": mse, "r2": r2}
