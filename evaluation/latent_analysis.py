import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional


def umap_plot(
    latents: np.ndarray,
    labels: Optional[np.ndarray] = None,
    label_name: str = "label",
    title: str = "z-star Latent Space (UMAP)",
    save_path: Optional[str] = None,
):
    try:
        import umap as umap_lib
    except ImportError:
        print("umap-learn not installed. pip install umap-learn")
        return

    embedding = umap_lib.UMAP(n_components=2, random_state=42).fit_transform(latents)
    _scatter_plot(embedding, labels, label_name, title, save_path)


def tsne_plot(
    latents: np.ndarray,
    labels: Optional[np.ndarray] = None,
    label_name: str = "label",
    title: str = "z-star Latent Space (t-SNE)",
    save_path: Optional[str] = None,
):
    from sklearn.manifold import TSNE
    embedding = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents) - 1)).fit_transform(latents)
    _scatter_plot(embedding, labels, label_name, title, save_path)


def cluster_analysis(
    latents: np.ndarray,
    n_clusters: int = 5,
    method: str = "kmeans",
) -> dict:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif method == "hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        raise ValueError(f"Unknown clustering method '{method}'")

    labels = model.fit_predict(latents)
    sil = silhouette_score(latents, labels) if n_clusters > 1 else 0.0

    return {"labels": labels, "silhouette": sil, "n_clusters": n_clusters}


def latent_interpolation(
    model,
    z1: np.ndarray,
    z2: np.ndarray,
    target_modality: str,
    n_steps: int = 10,
    device=None,
) -> np.ndarray:
    import torch
    if device is None:
        device = next(model.parameters()).device

    alphas = np.linspace(0, 1, n_steps)
    results = []
    model.eval()

    with torch.no_grad():
        for alpha in alphas:
            z_interp = torch.tensor(
                (1 - alpha) * z1 + alpha * z2, dtype=torch.float32, device=device
            ).unsqueeze(0)
            recon = model.modules_dict[target_modality].decode(z_interp)
            results.append(recon.cpu().numpy())

    return np.concatenate(results, axis=0)


def _scatter_plot(embedding, labels, label_name, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=labels if labels is not None else "steelblue",
        cmap="tab20" if labels is not None else None,
        alpha=0.6, s=8,
    )
    if labels is not None:
        plt.colorbar(sc, ax=ax, label=label_name)
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
