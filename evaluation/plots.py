"""
Plotting utilities for explaining and evaluating the z-star pipeline:
training dynamics, the embedding process itself, reconstruction quality,
downstream classification performance, and raw-data quality diagnostics.

Every function takes a `save_path`; if omitted the figure is shown instead
of saved. All functions are independent -- call only the ones relevant to
what you have, or use `generate_full_report()` to run whatever is available.
"""

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig, save_path: Optional[str]):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ── Training dynamics ──────────────────────────────────────────────────────

def plot_training_curves(history: Dict[str, List[dict]], save_path: Optional[str] = None):
    """One panel per loss component logged during training (total, recon, kl, ...), train vs val."""
    keys = []
    for split in ("train", "val"):
        for epoch_metrics in history.get(split, []):
            for k in epoch_metrics:
                if k not in keys:
                    keys.append(k)
    if "total" in keys:
        keys.remove("total")
        keys = ["total"] + keys

    ncols = min(len(keys), 4)
    nrows = max(1, (len(keys) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False)
    axes = axes.flatten()

    for i, key in enumerate(keys):
        ax = axes[i]
        for split, ls, color in [("train", "-", "steelblue"), ("val", "--", "tomato")]:
            vals = [m.get(key, np.nan) for m in history.get(split, [])]
            if vals:
                ax.plot(vals, linestyle=ls, color=color, label=split)
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        if all(v > 0 for v in vals if not np.isnan(v)):
            ax.set_yscale("log")
    for j in range(len(keys), len(axes)):
        axes[j].axis("off")

    plt.suptitle("z-star Training Curves", y=1.02)
    plt.tight_layout()
    _save(fig, save_path)


# ── The embedding process itself ───────────────────────────────────────────

def plot_embedding_pipeline_diagram(
    modality_names: List[str],
    modality_types: Optional[Dict[str, str]] = None,
    fusion: str = "poe",
    latent_dim: int = 0,
    save_path: Optional[str] = None,
):
    """
    Schematic of the actual pipeline used: per-modality encoders -> fusion -> z-star.
    A visual aid for explaining what z-star is, not a data plot.
    """
    modality_types = modality_types or {}
    n = max(len(modality_names), 1)
    fig_h = max(3.0, 1.1 * n + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n)
    ax.axis("off")

    box = dict(boxstyle="round,pad=0.5", linewidth=1.3)
    arrow_style = dict(arrowstyle="-|>", color="#8b96b3", linewidth=1.4,
                        shrinkA=0, shrinkB=0, mutation_scale=14)

    # Fixed column centers and half-widths (data units), so arrows anchor to
    # actual box edges rather than approximate text positions.
    data_x, data_hw = 0.8, 0.75
    enc_x, enc_hw = 3.1, 0.55
    fusion_x, fusion_hw = 5.6, 0.7
    z_x, z_hw = 7.8, 0.55
    dec_x, dec_hw = 9.4, 0.55

    y_positions = [n - 0.5 - i for i in range(len(modality_names))] or [n / 2]
    mid_y = n / 2

    for y, name in zip(y_positions, modality_names):
        mtype = modality_types.get(name, "")
        label = f"{name}\n({mtype})" if mtype else name
        ax.text(data_x, y, label, ha="center", va="center", fontsize=9,
                bbox=dict(**box, facecolor="#eaf0f7", edgecolor="#253E6B"))
        ax.annotate("", xy=(enc_x - enc_hw, y), xytext=(data_x + data_hw, y),
                    arrowprops=arrow_style)

        ax.text(enc_x, y, "encoder", ha="center", va="center", fontsize=8,
                bbox=dict(**box, facecolor="#f2e9f5", edgecolor="#733E85"))
        ax.annotate("", xy=(fusion_x - fusion_hw, mid_y), xytext=(enc_x + enc_hw, y),
                    arrowprops={**arrow_style, "alpha": 0.6})

    ax.text(fusion_x, mid_y, f"fusion\n({fusion})", ha="center", va="center", fontsize=9,
            bbox=dict(**box, facecolor="#eaf5f0", edgecolor="#377860"))
    ax.annotate("", xy=(z_x - z_hw, mid_y), xytext=(fusion_x + fusion_hw, mid_y), arrowprops=arrow_style)

    ax.text(z_x, mid_y, f"z*\n[{latent_dim}]", ha="center", va="center", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#dbeafe", edgecolor="#7c9fff", linewidth=2))
    ax.annotate("", xy=(dec_x - dec_hw, mid_y), xytext=(z_x + z_hw, mid_y), arrowprops=arrow_style)

    ax.text(dec_x, mid_y, "decoders /\ndownstream\nheads", ha="center", va="center", fontsize=8,
            bbox=dict(**box, facecolor="#fcf4e3", edgecolor="#A56327"))

    ax.set_title("z-star embedding pipeline", fontsize=13, fontweight="bold", loc="left")
    plt.tight_layout()
    _save(fig, save_path)


def plot_latent_variance(zstar_embeddings: np.ndarray, save_path: Optional[str] = None):
    """PCA scree plot of z-star: how much variance concentrates in how many dimensions."""
    from sklearn.decomposition import PCA

    n_components = min(zstar_embeddings.shape)
    pca = PCA(n_components=n_components)
    pca.fit(zstar_embeddings)
    var_ratio = pca.explained_variance_ratio_
    cum = np.cumsum(var_ratio)

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.bar(range(1, len(var_ratio) + 1), var_ratio, color="steelblue", alpha=0.85)
    ax1.set_xlabel("Principal component of z*")
    ax1.set_ylabel("Variance explained", color="steelblue")
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(cum) + 1), cum, color="tomato", marker="o", markersize=3)
    ax2.set_ylabel("Cumulative variance explained", color="tomato")
    ax2.set_ylim(0, 1.02)
    ax1.set_title("z-star latent variance (PCA)")
    plt.tight_layout()
    _save(fig, save_path)


# ── Reconstruction quality ─────────────────────────────────────────────────

def plot_reconstruction_scatter(
    y_true: np.ndarray, y_pred: np.ndarray, title: str = "Reconstruction", save_path: Optional[str] = None
):
    """True vs. reconstructed/imputed values, with R^2 annotated. Flattens multi-dim input."""
    from sklearn.metrics import r2_score

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    r2 = r2_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.35, s=10, color="steelblue")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1)
    ax.set_xlabel("True")
    ax.set_ylabel("Reconstructed")
    ax.set_title(f"{title}  (R²={r2:.3f})")
    plt.tight_layout()
    _save(fig, save_path)


# ── Downstream classification metrics ──────────────────────────────────────

def plot_roc_pr_curves(
    y_true: np.ndarray, y_scores: np.ndarray, title: str = "Downstream classifier", save_path: Optional[str] = None
):
    """Side-by-side ROC and precision-recall curves, with AUROC/AUPRC annotated."""
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    if len(np.unique(y_true)) < 2:
        print(f"[plot_roc_pr_curves] Only one class present in y_true; skipping '{title}'.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)
    base_rate = y_true.mean()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(fpr, tpr, color="steelblue", linewidth=2)
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC (AUROC={auroc:.3f})")

    axes[1].plot(rec, prec, color="tomato", linewidth=2)
    axes[1].axhline(base_rate, linestyle="--", color="gray", label=f"base rate={base_rate:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall (AUPRC={auprc:.3f})")
    axes[1].legend(fontsize=8)

    plt.suptitle(title)
    plt.tight_layout()
    _save(fig, save_path)


def plot_calibration_curve(
    y_true: np.ndarray, y_scores: np.ndarray, n_bins: int = 10,
    title: str = "Calibration", save_path: Optional[str] = None,
):
    """Reliability diagram: mean predicted probability vs. observed event frequency, by bin."""
    from sklearn.calibration import calibration_curve

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    if len(np.unique(y_true)) < 2:
        print(f"[plot_calibration_curve] Only one class present in y_true; skipping '{title}'.")
        return

    n_bins = min(n_bins, max(2, len(y_true) // 5))
    frac_pos, mean_pred = calibration_curve(y_true, y_scores, n_bins=n_bins, strategy="quantile")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, frac_pos, "o-", color="steelblue")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    plt.tight_layout()
    _save(fig, save_path)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred_binary: np.ndarray,
    class_names: Sequence[str] = ("negative", "positive"),
    title: str = "Confusion Matrix", save_path: Optional[str] = None,
):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred_binary)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(title)
    plt.tight_layout()
    _save(fig, save_path)


def plot_downstream_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: Sequence[str] = ("auroc", "auprc"),
    title: str = "Downstream Approach Comparison",
    save_path: Optional[str] = None,
):
    """Grouped bar chart comparing multiple downstream approaches (e.g. linear probe vs. MLP vs. raw baseline)."""
    names = list(results.keys())
    n_metrics = len(metrics)
    width = 0.8 / n_metrics
    x = np.arange(len(names))
    colors = ["#7c9fff", "#b47cff", "#4fd1c5", "#ff9d7c"]

    fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(names)), 4.5))
    for i, m in enumerate(metrics):
        vals = [results[n].get(m, np.nan) for n in names]
        ax.bar(x + i * width - width * (n_metrics - 1) / 2, vals, width=width,
               label=m.upper(), color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=1)
    ax.legend()
    ax.set_title(title)
    plt.tight_layout()
    _save(fig, save_path)


# ── Raw data quality (before/independent of the model) ─────────────────────

def plot_missingness(df, columns: Optional[List[str]] = None, title: str = "Missingness pattern", save_path: Optional[str] = None):
    """Per-column missing-value rate for a raw table -- run this before building the dataset."""
    cols = columns or list(df.columns)
    rates = df[cols].isna().mean()
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(cols)), 4.5))
    ax.bar(range(len(cols)), rates.values, color="#ff9d7c")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_ylabel("Fraction missing")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    plt.tight_layout()
    _save(fig, save_path)


# ── One-call report ─────────────────────────────────────────────────────────

def generate_full_report(
    output_dir: str,
    history: Optional[dict] = None,
    modality_names: Optional[List[str]] = None,
    modality_types: Optional[Dict[str, str]] = None,
    fusion: Optional[str] = None,
    latent_dim: Optional[int] = None,
    zstar_embeddings: Optional[np.ndarray] = None,
    reconstruction: Optional[dict] = None,   # {"y_true", "y_pred", "title"}
    downstream: Optional[dict] = None,       # {"y_true", "y_scores", "title"}
    comparison_results: Optional[dict] = None,
    raw_tables: Optional[Dict[str, "object"]] = None,  # {name: pd.DataFrame}
):
    """
    Generates every plot for which inputs were provided, into output_dir.
    Nothing you omit causes an error -- pass whatever you have.
    """
    os.makedirs(output_dir, exist_ok=True)

    if history is not None:
        plot_training_curves(history, save_path=os.path.join(output_dir, "training_curves.png"))

    if modality_names is not None:
        plot_embedding_pipeline_diagram(
            modality_names, modality_types or {}, fusion or "?", latent_dim or 0,
            save_path=os.path.join(output_dir, "pipeline_diagram.png"),
        )

    if zstar_embeddings is not None:
        plot_latent_variance(zstar_embeddings, save_path=os.path.join(output_dir, "latent_variance.png"))

    if reconstruction is not None:
        plot_reconstruction_scatter(
            reconstruction["y_true"], reconstruction["y_pred"],
            title=reconstruction.get("title", "Reconstruction"),
            save_path=os.path.join(output_dir, "reconstruction_scatter.png"),
        )

    if downstream is not None:
        plot_roc_pr_curves(
            downstream["y_true"], downstream["y_scores"],
            title=downstream.get("title", "Downstream"),
            save_path=os.path.join(output_dir, "downstream_roc_pr.png"),
        )
        plot_calibration_curve(
            downstream["y_true"], downstream["y_scores"],
            title=downstream.get("title", "Calibration"),
            save_path=os.path.join(output_dir, "downstream_calibration.png"),
        )

    if comparison_results is not None:
        plot_downstream_comparison(comparison_results, save_path=os.path.join(output_dir, "downstream_comparison.png"))

    if raw_tables is not None:
        for name, df in raw_tables.items():
            plot_missingness(df, title=f"'{name}' missingness", save_path=os.path.join(output_dir, f"missingness_{name}.png"))

    print(f"\nFull report saved to: {output_dir}")
