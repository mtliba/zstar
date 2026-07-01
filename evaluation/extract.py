import numpy as np
import torch
from torch.utils.data import DataLoader
from zstar.data.collate import zstar_collate


@torch.no_grad()
def extract_zstar(
    model,
    dataset,
    batch_size: int = 256,
    device=None,
    save_path: str = None,
) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=zstar_collate)

    zs = []
    for batch in loader:
        batch_dev = {}
        for name, item in batch.items():
            batch_dev[name] = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in item.items()
            }
        z = model.extract_zstar(batch_dev)
        zs.append(z.cpu().numpy())

    embeddings = np.concatenate(zs, axis=0)

    if save_path:
        np.save(save_path, embeddings)
        print(f"Saved z-star embeddings: {embeddings.shape} → {save_path}")

    return embeddings
