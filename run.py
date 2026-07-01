"""
z-star: Self-Supervised Patient Representation Framework

Usage
-----
  python run.py                                    # default config + synthetic data
  python run.py model.fusion=attention             # switch fusion strategy
  python run.py model.latent_type=discrete         # switch to VQ-VAE
  python run.py modalities.creatinine.encoder=lstm # switch encoder

Replace make_synthetic_data() with your real data loading logic.
"""

import os
import sys
import numpy as np
import torch
from omegaconf import OmegaConf

# Add parent dir so zstar package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zstar.models import ZStarModel
from zstar.data import ZStarDataset
from zstar.training import ZStarTrainer
from zstar.evaluation import (
    extract_zstar,
    umap_plot,
    cross_modal_reconstruction,
    compare_approaches,
)


def make_synthetic_data(cfg) -> dict:
    rng = np.random.default_rng(42)
    N = 300

    data_dict = {}
    for name, mcfg in cfg.modalities.items():
        if not mcfg.get("enabled", True):
            continue

        mod_type = str(mcfg.type)
        input_dim = int(mcfg.input_dim)

        if mod_type == "static":
            arr = rng.standard_normal((N, input_dim)).astype(np.float32)
            missing = rng.choice(N, size=int(0.15 * N), replace=False)
            arr[missing] = np.nan
            data_dict[name] = {"type": "static", "data": arr}

        elif mod_type in ("temporal", "event"):
            max_len = int(mcfg.get("max_seq_len", 100))
            samples = []
            for i in range(N):
                if rng.random() < 0.15:
                    samples.append(None)
                else:
                    seq_len = rng.integers(10, min(max_len, 80))
                    timestamps = np.sort(rng.uniform(0, 365, size=seq_len)).astype(np.float32)
                    values = rng.standard_normal((seq_len, input_dim)).astype(np.float32)
                    samples.append((timestamps, values))
            data_dict[name] = {"type": mod_type, "data": samples}

    return data_dict


def main():
    cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config.yaml"))
    if len(sys.argv) > 1:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(sys.argv[1:]))

    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # Auto-enable VQ commitment loss for discrete/hybrid
    if str(cfg.model.get("latent_type", "continuous")) in ("discrete", "hybrid"):
        cfg.objectives.vq_commitment.enabled = True

    print("[INFO] Using synthetic data for demonstration.\n")
    raw = make_synthetic_data(cfg)

    dataset = ZStarDataset(raw, normalize=cfg.data.normalize)
    print("\nModality info:")
    for name, info in dataset.modality_info.items():
        print(f"  {name:20s}: type={info['type']:10s}  availability={info['availability'] * 100:.1f}%")

    model = ZStarModel(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_params:,} trainable parameters")
    print(f"Fusion       : {cfg.model.fusion}")
    print(f"Latent type  : {cfg.model.get('latent_type', 'continuous')}")
    print(f"Latent dim   : {cfg.model.latent_dim}")
    print(f"β (target)   : {cfg.model.beta}\n")

    trainer = ZStarTrainer(model, dataset, cfg)
    history = trainer.train()

    # Load best checkpoint
    device = next(model.parameters()).device
    ckpt = os.path.join(cfg.logging.save_dir, "best_zstar.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    # Extract z-star
    print("\nExtracting z-star embeddings...")
    os.makedirs("outputs", exist_ok=True)
    zstar_embeddings = extract_zstar(model, dataset, device=device, save_path="outputs/zstar_embeddings.npy")
    print(f"z-star shape: {zstar_embeddings.shape}")

    umap_plot(zstar_embeddings, title="z-star Latent Space", save_path="outputs/zstar_umap.png")

    # Cross-modal reconstruction
    enabled = [n for n, m in cfg.modalities.items() if m.get("enabled", True)]
    static_mods = [n for n in enabled if cfg.modalities[n].type == "static"]
    if len(enabled) >= 2 and static_mods:
        src = [m for m in enabled if m != static_mods[0]][:1]
        tgt = static_mods[0]
        if src:
            metrics = cross_modal_reconstruction(model, dataset, src, tgt, device=device)
            print(f"\nCross-modal {src} → {tgt}:")
            print(f"  MSE = {metrics['mse']:.4f}")
            print(f"  R²  = {metrics['r2']:.4f}")

    # Downstream comparison (with synthetic labels)
    if cfg.downstream.get("graft_loss", {}).get("enabled", False):
        print("\n--- Downstream: Graft Loss Prediction (synthetic labels) ---")
        rng = np.random.default_rng(99)
        synthetic_labels = rng.integers(0, 2, size=len(dataset)).astype(np.float32)
        compare_approaches(zstar_embeddings, synthetic_labels, cfg.model.latent_dim)

    print("\nDone.")


if __name__ == "__main__":
    main()
