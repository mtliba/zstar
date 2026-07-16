"""
One-command pipeline for the REXETRIS popf_v3 (baseline) + apgf (checkup)
tables, as specified in BASELINE_AND_LONGITUDINAL_README.md and
BASELINE_AND_LONGITUDINAL_GENERATION_NOTES.md.

Usage (after `pip install transplant-zstar1`)
---------------------------------------------
    zstar-popf-apgf --popf popf_v3.csv --apgf apgf.csv --output-dir results/

Or from Python:
    from zstar.pipelines.popf_apgf import run
    run(popf_path="popf_v3.csv", apgf_path="apgf.csv", output_dir="results/")

What it does
------------
1. Loads and joins the two tables by `natt1` (zstar.data.build_from_tables),
   encoding boolean columns, log-transforming the right-skewed columns the
   README flags, and excluding the survival/outcome columns from the input
   features (they are the labels, not inputs).
2. Builds a ZStarDataset with per-feature missing-mask concatenation on
   `popf` and iterative (MICE-style) imputation for its structured
   missingness (e.g. donor comorbidities recorded only for living donors).
3. Trains a ZStarModel (PoE fusion, MLP encoder for popf, GRU encoder for
   apgf; VAE + contrastive + MMD objectives) via ZStarTrainer.
4. Extracts z-star for the full cohort and saves it to
   <output_dir>/zstar_embeddings.npy.
5. Runs a cross-modal reconstruction check (apgf -> popf) and a downstream
   graft-loss head trained on frozen z-star (linear probe vs. MLP).
6. Generates a full plotting report (training curves, pipeline diagram,
   latent variance, reconstruction scatter, ROC/PR, calibration,
   missingness) into <output_dir>/plots/.

This codebase (and this pipeline) has so far only been exercised on
synthetic data generated to match the table specs above -- see the
project README's Status & Scope section before treating any metric this
pipeline prints as evidence of real predictive performance.
"""

import argparse
import os

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from zstar.data import build_from_tables, ZStarDataset
from zstar.models import ZStarModel
from zstar.training import ZStarTrainer
from zstar.evaluation import (
    extract_zstar,
    cross_modal_reconstruction,
    generate_full_report,
)
from zstar.evaluation.downstream import GraftLossPredictor, train_downstream_head

ID_COL = "natt1"

LABEL_COLS = [
    "GraftSurvivalDays", "FailureWithinStudyPeriod",
    "PatientSurvivalDays", "DeathWithinStudyPeriod",
]

BOOL_COLS_POPF = [
    "sex", "sex_donor", "donor_living", "cmv", "ebv", "cmv_donor_2l",
    "ebv_donor_2l", "crossmatch_2l", "cirrhosis_regist", "diabetes_regist",
    "dyslipidemia_regist", "history_hbp_regist", "neuropath_regist",
    "uro_patho_regist", "smoking_ever_regist", "cvd_regist", "diabetes_donor",
    "hbp_donor", "kidney_disease_donor", "kidney_machine",
    "functionnal_discharge_hosp",
]
BOOL_COLS_APGF = [
    "proteinuria_imputed", "dsa_dn", "cancer", "lymphoma", "hla", "bkv",
    "cmv", "ebv", "daily_activity", "pregnancy",
]

# Right-skewed columns the README recommends log-transforming
LOG_COLS_POPF = ["cit", "creat_dc_hosp", "creat_donor"]
LOG_COLS_APGF = ["creat", "proteinuria"]

BINARY_LABEL_COLS = {"FailureWithinStudyPeriod", "DeathWithinStudyPeriod"}


def build_config(
    popf_input_dim: int,
    apgf_input_dim: int,
    latent_dim: int,
    fusion: str,
    epochs: int,
    batch_size: int,
    apgf_encoder: str,
    seed: int,
):
    return OmegaConf.create({
        "project": {"name": "popf_apgf", "seed": seed},
        "model": {
            "latent_dim": latent_dim, "fusion": fusion,
            "latent_type": "continuous", "beta": 2.0, "cond_dim": 0,
        },
        "modalities": {
            "popf": {
                "enabled": True, "type": "static", "input_dim": popf_input_dim,
                "encoder": "mlp", "decoder": "mlp",
                "encoder_config": {"hidden_dims": [128, 64], "dropout": 0.1},
                "decoder_config": {"hidden_dims": [64, 128], "dropout": 0.1},
            },
            "apgf": {
                "enabled": True, "type": "temporal", "input_dim": apgf_input_dim,
                "max_seq_len": 60,
                "encoder": apgf_encoder, "decoder": "temporal",
                "encoder_config": {
                    "hidden_dim": 64, "num_layers": 2, "bidirectional": True,
                    "dropout": 0.1, "time_encoding": "sinusoidal", "pooling": "last",
                    "d_model": 64, "nhead": 4,
                },
                "decoder_config": {"type": "gru", "hidden_dim": 64, "num_layers": 2, "max_seq_len": 60},
            },
        },
        "objectives": {
            "reconstruction": {"enabled": True, "weight": 1.0, "loss_fn": "mse"},
            "kl": {"enabled": True, "weight": 1.0},
            "vq_commitment": {"enabled": False, "weight": 0.25},
            "contrastive": {"enabled": True, "weight": 0.5, "temperature": 0.07},
            "masked_reconstruction": {"enabled": False, "weight": 0.3},
            "temporal_prediction": {"enabled": False, "weight": 0.3, "prediction_horizon": 3, "modalities": []},
            "alignment": {"enabled": True, "weight": 0.5, "strategy": "mmd", "temperature": 0.07},
        },
        "training": {
            "batch_size": batch_size, "epochs": epochs, "lr": 1e-3, "weight_decay": 1e-5,
            "val_split": 0.2, "optimizer": "adam", "grad_clip": 1.0,
            "kl_warmup_epochs": max(1, epochs // 3), "kl_schedule": "linear",
            "scheduler": {"type": "reduce_on_plateau", "patience": 8, "factor": 0.5},
        },
        "data": {"path": "data/", "normalize": True},
        "logging": {"log_every": max(1, epochs // 10), "save_dir": "checkpoints/"},
    })


def run(
    popf_path: str,
    apgf_path: str,
    output_dir: str = "zstar_output",
    label_col: str = "FailureWithinStudyPeriod",
    latent_dim: int = 32,
    fusion: str = "poe",
    apgf_encoder: str = "gru",
    imputation: str = "iterative",
    include_missing_mask: bool = True,
    epochs: int = 100,
    batch_size: int = 32,
    seed: int = 7,
):
    if label_col not in LABEL_COLS:
        raise ValueError(f"label_col must be one of {LABEL_COLS}, got '{label_col}'")

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("1. Loading and joining tables")
    popf_df = pd.read_csv(popf_path)
    apgf_df = pd.read_csv(apgf_path)
    print(f"  popf_v3: {popf_df.shape}   apgf: {apgf_df.shape}")

    tables = {
        "popf": {
            "df": popf_df, "type": "static",
            "bool_cols": BOOL_COLS_POPF, "exclude_cols": LABEL_COLS,
            "log_columns": LOG_COLS_POPF, "include_missing_mask": include_missing_mask,
            "imputation": imputation,
        },
        "apgf": {
            "df": apgf_df, "type": "temporal", "timestamp_col": "days_since_tx",
            "bool_cols": BOOL_COLS_APGF, "log_columns": LOG_COLS_APGF,
        },
    }
    data_dict, ids, feature_columns = build_from_tables(tables, id_col=ID_COL)
    labels_df = popf_df.set_index(ID_COL).reindex(ids)[LABEL_COLS]

    print("\n2. Building dataset")
    dataset = ZStarDataset(data_dict, normalize=True)
    info = dataset.modality_info
    for name, mod_info in info.items():
        print(f"  {name}: type={mod_info['type']}, availability={mod_info['availability']:.1%}, n_features={mod_info['n_features']}")

    cfg = build_config(
        popf_input_dim=info["popf"]["n_features"],
        apgf_input_dim=info["apgf"]["n_features"],
        latent_dim=latent_dim, fusion=fusion, epochs=epochs, batch_size=batch_size,
        apgf_encoder=apgf_encoder, seed=seed,
    )

    print("\n3. Building model")
    model = ZStarModel(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {n_params:,} trainable parameters | fusion={fusion} | latent_dim={latent_dim}")

    print(f"\n4. Training ({epochs} epochs)")
    cfg.logging.save_dir = os.path.join(output_dir, "checkpoints")
    trainer = ZStarTrainer(model, dataset, cfg)
    history = trainer.train()

    device = next(model.parameters()).device
    ckpt = os.path.join(cfg.logging.save_dir, "best_zstar.pt")
    import torch
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    print("\n5. Extracting z-star")
    zstar_path = os.path.join(output_dir, "zstar_embeddings.npy")
    zstar_embeddings = extract_zstar(model, dataset, device=device, save_path=zstar_path)
    np.save(os.path.join(output_dir, "ids.npy"), ids)
    labels_df.to_csv(os.path.join(output_dir, "labels.csv"))

    print("\n6. Cross-modal reconstruction (apgf -> popf)")
    recon_metrics = cross_modal_reconstruction(model, dataset, source_modalities=["apgf"], target_modality="popf", device=device)
    print(f"  MSE={recon_metrics['mse']:.4f}  R2={recon_metrics['r2']:.4f}")

    print(f"\n7. Downstream: predict '{label_col}' from frozen z-star")
    y = labels_df[label_col].to_numpy(dtype=np.float32)
    is_binary = label_col in BINARY_LABEL_COLS
    task = "binary_classification" if is_binary else "regression"

    head_linear = GraftLossPredictor(latent_dim=latent_dim, hidden_dims=[])
    head_mlp = GraftLossPredictor(latent_dim=latent_dim, hidden_dims=[32, 16])
    results_linear = train_downstream_head(head_linear, zstar_embeddings, y, task=task, epochs=50)
    results_mlp = train_downstream_head(head_mlp, zstar_embeddings, y, task=task, epochs=50)
    print(f"  linear probe: {results_linear}")
    print(f"  mlp:          {results_mlp}")

    print("\n8. Generating plot report")
    downstream_plot_kwargs = None
    if is_binary:
        import torch as _torch
        head_mlp.eval()
        head_device = next(head_mlp.parameters()).device
        with _torch.no_grad():
            z_input = _torch.tensor(zstar_embeddings, dtype=_torch.float32, device=head_device)
            scores = _torch.sigmoid(head_mlp(z_input)).cpu().numpy()
        downstream_plot_kwargs = {"y_true": y, "y_scores": scores, "title": label_col}

    generate_full_report(
        output_dir=os.path.join(output_dir, "plots"),
        history=history,
        modality_names=list(cfg.modalities.keys()),
        modality_types={n: cfg.modalities[n].type for n in cfg.modalities},
        fusion=fusion, latent_dim=latent_dim,
        zstar_embeddings=zstar_embeddings,
        downstream=downstream_plot_kwargs,
        comparison_results={"linear_probe": results_linear, "mlp": results_mlp} if is_binary else None,
        raw_tables={"popf_v3": popf_df, "apgf": apgf_df},
    )

    print("\n" + "=" * 60)
    print(f"Done. Outputs in: {output_dir}/")
    print("  - zstar_embeddings.npy, ids.npy, labels.csv")
    print("  - checkpoints/best_zstar.pt")
    print("  - plots/ (training curves, pipeline diagram, ROC/PR, missingness, ...)")
    return {
        "zstar_embeddings": zstar_embeddings, "history": history,
        "reconstruction": recon_metrics, "downstream": {"linear_probe": results_linear, "mlp": results_mlp},
    }


def main():
    parser = argparse.ArgumentParser(description="Run the z-star pipeline on real popf_v3 + apgf tables.")
    parser.add_argument("--popf", required=True, help="Path to popf_v3.csv")
    parser.add_argument("--apgf", required=True, help="Path to apgf.csv")
    parser.add_argument("--output-dir", default="zstar_output")
    parser.add_argument("--label-col", default="FailureWithinStudyPeriod", choices=LABEL_COLS)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--fusion", default="poe", choices=["poe", "moe", "attention", "concat"])
    parser.add_argument("--apgf-encoder", default="gru", choices=["gru", "lstm", "transformer", "tcn"])
    parser.add_argument("--imputation", default="iterative", choices=["zero", "mean", "median", "iterative"])
    parser.add_argument("--no-missing-mask", action="store_true", help="Disable per-feature missing-mask concatenation for popf")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    run(
        popf_path=args.popf, apgf_path=args.apgf, output_dir=args.output_dir,
        label_col=args.label_col, latent_dim=args.latent_dim, fusion=args.fusion,
        apgf_encoder=args.apgf_encoder, imputation=args.imputation,
        include_missing_mask=not args.no_missing_mask,
        epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
    )


if __name__ == "__main__":
    main()
