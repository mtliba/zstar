import numpy as np
from typing import Dict
from .downstream import GraftLossPredictor, train_downstream_head


def compare_approaches(
    zstar: np.ndarray,
    graft_loss_labels: np.ndarray,
    latent_dim: int,
) -> Dict[str, Dict[str, float]]:
    results = {}

    # 1. Linear probe on frozen z-star
    linear = GraftLossPredictor(latent_dim, hidden_dims=[])
    results["linear_probe"] = train_downstream_head(
        linear, zstar, graft_loss_labels, task="binary_classification"
    )

    # 2. MLP on frozen z-star
    mlp = GraftLossPredictor(latent_dim, hidden_dims=[64, 32])
    results["mlp_frozen_zstar"] = train_downstream_head(
        mlp, zstar, graft_loss_labels, task="binary_classification"
    )

    print("\n=== Approach Comparison ===")
    for name, metrics in results.items():
        print(f"  {name}:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

    return results
