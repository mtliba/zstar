import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional


class GraftLossPredictor(nn.Module):

    def __init__(self, latent_dim: int, hidden_dims: List[int] = [64, 32]):
        super().__init__()
        layers = []
        prev = latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class EGFRTrajectoryPredictor(nn.Module):

    def __init__(self, latent_dim: int, prediction_points: List[int], hidden_dims: List[int] = [64, 32]):
        super().__init__()
        self.prediction_points = prediction_points
        layers = []
        prev = latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, len(prediction_points)))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)  # [B, n_points]


def train_downstream_head(
    head: nn.Module,
    zstar: np.ndarray,
    labels: np.ndarray,
    task: str = "binary_classification",
    epochs: int = 100,
    lr: float = 1e-3,
    val_split: float = 0.2,
    batch_size: int = 64,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = head.to(device)

    z_tensor = torch.tensor(zstar, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.float32)

    n = len(z_tensor)
    n_val = max(1, int(n * val_split))
    perm = torch.randperm(n)
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    train_ds = TensorDataset(z_tensor[train_idx], y_tensor[train_idx])
    val_ds = TensorDataset(z_tensor[val_idx], y_tensor[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    if task == "binary_classification":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        head.train()
        for z_batch, y_batch in train_loader:
            z_batch, y_batch = z_batch.to(device), y_batch.to(device)
            pred = head(z_batch)
            loss = loss_fn(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for z_batch, y_batch in val_loader:
                z_batch, y_batch = z_batch.to(device), y_batch.to(device)
                val_loss += loss_fn(head(z_batch), y_batch).item() * len(z_batch)
        val_loss /= len(val_ds)
        best_val = min(best_val, val_loss)

    # Final evaluation
    head.eval()
    results = {"best_val_loss": best_val}

    if task == "binary_classification":
        from sklearn.metrics import roc_auc_score, average_precision_score
        with torch.no_grad():
            z_val = z_tensor[val_idx].to(device)
            y_val = y_tensor[val_idx].numpy()
            preds = torch.sigmoid(head(z_val)).cpu().numpy()
            try:
                results["auroc"] = float(roc_auc_score(y_val, preds))
                results["auprc"] = float(average_precision_score(y_val, preds))
            except ValueError:
                pass

    return results
