"""
One-command pipeline for the REXETRIS popf_v3 (baseline) + apgf (checkup)
tables, as specified in BASELINE_AND_LONGITUDINAL_README.md and
BASELINE_AND_LONGITUDINAL_GENERATION_NOTES.md.

Usage (after `pip install transplant-zstar2`)
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
from typing import Optional

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from zstar.data import build_from_tables, ZStarDataset, apply_landmark
from zstar.models import ZStarModel
from zstar.training import ZStarTrainer
from zstar.evaluation import (
    extract_zstar,
    cross_modal_reconstruction,
    generate_full_report,
    concordance_index,
    plot_survival_overview,
    plot_followup_distribution,
    plot_km_by_zstar_risk,
    plot_landmark_diagnostic,
    derive_competing_events,
    plot_competing_risks_overview,
    plot_cumulative_incidence,
    plot_cif_by_group,
    plot_head_training_dynamics,
    plot_cif_calibration,
    plot_finetuning_comparison,
    train_competing_risks_head,
    compare_finetuning_strategies,
    CAUSE_NAMES,
)
from zstar.evaluation.downstream import GraftLossPredictor, train_downstream_head

ID_COL = "natt1"

LABEL_COLS = [
    "GraftSurvivalDays", "FailureWithinStudyPeriod",
    "PatientSurvivalDays", "DeathWithinStudyPeriod",
]

# The labels are time-to-event pairs: a duration column plus its event mask.
# `*Days` is the observed duration; `*WithinStudyPeriod` is True when the event
# was observed at that duration and False when observation was censored there.
SURVIVAL_PAIRS = {
    "GraftSurvivalDays": ("GraftSurvivalDays", "FailureWithinStudyPeriod"),
    "FailureWithinStudyPeriod": ("GraftSurvivalDays", "FailureWithinStudyPeriod"),
    "PatientSurvivalDays": ("PatientSurvivalDays", "DeathWithinStudyPeriod"),
    "DeathWithinStudyPeriod": ("PatientSurvivalDays", "DeathWithinStudyPeriod"),
}

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
    progress: bool = True,
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
        "logging": {
            "log_every": max(1, epochs // 10),
            "save_dir": "checkpoints/",
            "progress": progress,
        },
    })


def run(
    popf_path: str,
    apgf_path: str,
    output_dir: str = "zstar_output",
    label_col: str = "FailureWithinStudyPeriod",
    landmark_day: Optional[float] = None,
    latent_dim: int = 32,
    fusion: str = "poe",
    apgf_encoder: str = "gru",
    imputation: str = "iterative",
    include_missing_mask: bool = True,
    epochs: int = 100,
    batch_size: int = 32,
    seed: int = 7,
    progress: bool = True,
    cr_epochs: int = 200,
    cr_bins: int = 20,
    compare_finetuning: bool = False,
    ft_epochs: int = 30,
):
    if label_col not in LABEL_COLS:
        raise ValueError(f"label_col must be one of {LABEL_COLS}, got '{label_col}'")

    duration_col, event_col = SURVIVAL_PAIRS[label_col]

    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")

    print("=" * 60)
    print("1. Loading and joining tables")
    popf_df = pd.read_csv(popf_path)
    apgf_df = pd.read_csv(apgf_path)
    print(f"  popf_v3: {popf_df.shape}   apgf: {apgf_df.shape}")

    # Leakage diagnostic on the RAW data, before any landmark is applied --
    # this is what shows whether a landmark is needed in the first place.
    print("\n1b. Leakage diagnostic (raw data, pre-landmark)")
    leak_report = plot_landmark_diagnostic(
        apgf_df, popf_df, landmark_day=None, id_col=ID_COL,
        timestamp_col="days_since_tx", duration_col=duration_col, event_col=event_col,
        save_path=os.path.join(plots_dir, "landmark_diagnostic_raw.png"),
    )
    if leak_report:
        print(f"  events: {leak_report['n_events']:,} | median gap to last observation: "
              f"{leak_report['median_gap_days']:.0f}d")
        print(f"  observations recorded AFTER the event: {leak_report['n_observation_after_event']:,}")
        print(f"  events with an observation within 90d: {leak_report['n_within_90d']:,}")
        if landmark_day is None and leak_report["n_within_90d"] > 0.1 * leak_report["n_events"]:
            print("  [WARNING] A substantial share of events have observations recorded")
            print("            close to (or after) the event. Without --landmark-day, downstream")
            print("            metrics will likely reflect detecting the event already in")
            print("            progress rather than forecasting it.")

    if landmark_day is not None:
        print(f"\n1c. Applying landmark cutoff at day {landmark_day:.0f}")
        popf_df, temporal_out, lm_report = apply_landmark(
            static_df=popf_df,
            temporal_tables={"apgf": (apgf_df, "days_since_tx")},
            landmark_day=landmark_day,
            duration_col=duration_col, event_col=event_col, id_col=ID_COL,
        )
        apgf_df = temporal_out["apgf"]

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

    print("\n1d. Outcome overview (competing risks: graft loss vs. death)")
    cr_time, cr_cause, cr_report = derive_competing_events(labels_df)
    plot_competing_risks_overview(
        cr_time, cr_cause, cr_report,
        save_path=os.path.join(plots_dir, "competing_risks_overview.png"),
    )
    plot_cumulative_incidence(
        cr_time, cr_cause, compare_with_km=True,
        save_path=os.path.join(plots_dir, "cumulative_incidence.png"),
    )
    plot_survival_overview(labels_df, save_path=os.path.join(plots_dir, "survival_overview_km.png"))
    plot_followup_distribution(labels_df, save_path=os.path.join(plots_dir, "followup_distribution.png"))

    print("\n2. Building dataset")
    dataset = ZStarDataset(data_dict, normalize=True)
    info = dataset.modality_info
    for name, mod_info in info.items():
        print(f"  {name}: type={mod_info['type']}, availability={mod_info['availability']:.1%}, n_features={mod_info['n_features']}")

    cfg = build_config(
        popf_input_dim=info["popf"]["n_features"],
        apgf_input_dim=info["apgf"]["n_features"],
        latent_dim=latent_dim, fusion=fusion, epochs=epochs, batch_size=batch_size,
        apgf_encoder=apgf_encoder, seed=seed, progress=progress,
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
    results_linear = train_downstream_head(head_linear, zstar_embeddings, y, task=task, epochs=50, seed=seed)
    results_mlp = train_downstream_head(head_mlp, zstar_embeddings, y, task=task, epochs=50, seed=seed)
    print(f"  linear probe: {results_linear}")
    print(f"  mlp:          {results_mlp}")

    if is_binary:
        print("  [NOTE] These heads treat the event mask as a plain binary target, which")
        print("         ignores censoring: a subject censored early counts as a negative")
        print("         identical to one observed event-free for years. The C-index below")
        print("         is the censoring-aware metric and should be preferred over AUROC.")

    print("\n7b. Competing-risks model (discrete-time neural hazards, both outcomes)")
    print("     Predicts graft loss and death jointly as competing events.")
    print("     No proportional-hazards assumption: each (time bin, cause) has its")
    print("     own hazard output, so effects may vary freely over time.")
    cr = train_competing_risks_head(
        zstar_embeddings, cr_time, cr_cause,
        n_bins=cr_bins, epochs=cr_epochs, seed=seed, verbose=True,
    )
    va = cr["val_idx"]

    print("\n  Held-out C-index (val split only):")
    for cause_id, cause_name in CAUSE_NAMES.items():
        cv = cr[f"c_index_val_cause{cause_id}"]
        ct = cr[f"c_index_train_cause{cause_id}"]
        print(f"    {cause_name:<12}: val={cv:.4f}  train={ct:.4f}   (0.5 = chance)")

    # Training dynamics per outcome
    plot_head_training_dynamics(
        cr["history"],
        save_path=os.path.join(plots_dir, "competing_risks_training_dynamics.png"),
    )

    # Per-cause: CIF stratified by predicted risk, and CIF calibration -- val only
    for cause_id, cause_name in CAUSE_NAMES.items():
        risk = cr[f"risk_cause{cause_id}"]
        slug = cause_name.replace(" ", "_")

        q = np.quantile(risk[va], [0.25, 0.5, 0.75])
        groups_va = np.digitize(risk[va], q)
        plot_cif_by_group(
            cr_time[va], cr_cause[va], groups_va, cause=cause_id, cause_name=cause_name,
            group_names={0: "Q1 (lowest risk)", 1: "Q2", 2: "Q3", 3: "Q4 (highest risk)"},
            title=f"{cause_name}: cumulative incidence by predicted risk quartile (held-out)",
            save_path=os.path.join(plots_dir, f"cif_by_predicted_risk_{slug}.png"),
        )
        plot_cif_calibration(
            risk[va], cr_time[va], cr_cause[va], cause=cause_id, cause_name=cause_name,
            save_path=os.path.join(plots_dir, f"cif_calibration_{slug}.png"),
        )

    # Unsupervised: does z-star separate risk without ever seeing a label?
    from sklearn.cluster import KMeans
    clusters = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(zstar_embeddings)
    for cause_id, cause_name in CAUSE_NAMES.items():
        slug = cause_name.replace(" ", "_")
        plot_cif_by_group(
            cr_time, cr_cause, clusters, cause=cause_id, cause_name=cause_name,
            title=f"{cause_name}: cumulative incidence by z-star cluster (unsupervised)",
            save_path=os.path.join(plots_dir, f"cif_by_zstar_cluster_{slug}.png"),
        )

    ft_results = None
    if compare_finetuning:
        print("\n7c. Frozen vs. fine-tuned vs. from-scratch encoder")
        print("     Same split, head, schedule and seed for all three arms, so any")
        print("     difference is attributable to the encoder treatment alone.")
        print("     'scratch' uses NO self-supervised pretraining -- if it matches")
        print("     'finetune', the pretraining contributed nothing.")
        ft_results = compare_finetuning_strategies(
            model, dataset, cr_time, cr_cause, cfg=cfg,
            n_bins=cr_bins, epochs=ft_epochs, batch_size=batch_size, seed=seed,
        )
        print("\n  Held-out C-index by encoder treatment:")
        for m, r in ft_results.items():
            parts = " ".join(
                f"{nm}={r[f'c_index_val_cause{c}']:.4f}" for c, nm in CAUSE_NAMES.items()
            )
            print(f"    {m:<9}: {parts}   ({r['n_trainable_params']:,} trainable params)")
        plot_finetuning_comparison(
            ft_results,
            save_path=os.path.join(plots_dir, "finetuning_comparison.png"),
        )

    c_index = cr["c_index_val_cause1"]  # graft loss, held-out

    # Binary-head risk scores, for the legacy ROC/PR plots below (val split only)
    import torch as _torch
    head_mlp.eval()
    head_device = next(head_mlp.parameters()).device
    with _torch.no_grad():
        z_input = _torch.tensor(zstar_embeddings, dtype=_torch.float32, device=head_device)
        raw_out = head_mlp(z_input)
        scores = _torch.sigmoid(raw_out).cpu().numpy() if is_binary else raw_out.cpu().numpy()

    print("\n8. Generating plot report")
    downstream_plot_kwargs = None
    if is_binary:
        # val split only -- these heads were fitted on the rest
        from zstar.evaluation.downstream import make_split
        _, bin_val_idx = make_split(len(y), seed=seed)
        bv = bin_val_idx.numpy()
        downstream_plot_kwargs = {
            "y_true": y[bv], "y_scores": scores[bv],
            "title": f"{label_col} (held-out split)",
        }

    generate_full_report(
        output_dir=plots_dir,
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
    print("  - plots/ (survival overview, KM by z-star, landmark diagnostic,")
    print("            training curves, ROC/PR, missingness, ...)")
    if landmark_day is None:
        print("\n  Reminder: no landmark was applied. See plots/landmark_diagnostic_raw.png")
        print("  before interpreting downstream performance -- observations recorded at or")
        print("  near the event let a model detect rather than forecast it.")
    return {
        "zstar_embeddings": zstar_embeddings, "history": history,
        "reconstruction": recon_metrics,
        "downstream": {"linear_probe": results_linear, "mlp": results_mlp},
        "c_index": c_index,
        "competing_risks": {
            k: cr[k] for k in cr
            if k.startswith("c_index_") or k == "best_val_loss"
        },
        "finetuning_comparison": (
            {m: {k: v for k, v in r.items() if k.startswith("c_index_")
                 or k in ("best_val_loss", "n_trainable_params")}
             for m, r in ft_results.items()} if ft_results else None
        ),
        "leakage_report": leak_report,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run the z-star pipeline on real popf_v3 + apgf tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Labels are time-to-event pairs (duration + censoring mask):\n"
            "  GraftSurvivalDays   + FailureWithinStudyPeriod\n"
            "  PatientSurvivalDays + DeathWithinStudyPeriod\n\n"
            "Use --landmark-day to prevent leakage from observations recorded at or\n"
            "near the event. Without it, downstream metrics likely reflect detecting\n"
            "the event in progress rather than forecasting it."
        ),
    )
    parser.add_argument("--popf", required=True, help="Path to popf_v3.csv")
    parser.add_argument("--apgf", required=True, help="Path to apgf.csv")
    parser.add_argument("--output-dir", default="zstar_output")
    parser.add_argument("--label-col", default="FailureWithinStudyPeriod", choices=LABEL_COLS)
    parser.add_argument(
        "--landmark-day", type=float, default=None,
        help="Landmark cutoff in days. Keeps only subjects still at risk at this "
             "time, truncates apgf to observations at or before it, and predicts "
             "the event after it. Strongly recommended (e.g. 365).",
    )
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--fusion", default="poe", choices=["poe", "moe", "attention", "concat"])
    parser.add_argument("--apgf-encoder", default="gru", choices=["gru", "lstm", "transformer", "tcn"])
    parser.add_argument("--imputation", default="iterative", choices=["zero", "mean", "median", "iterative"])
    parser.add_argument("--no-missing-mask", action="store_true", help="Disable per-feature missing-mask concatenation for popf")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cr-epochs", type=int, default=200,
                        help="Epochs for the competing-risks head.")
    parser.add_argument("--cr-bins", type=int, default=20,
                        help="Time bins for the discrete-time competing-risks model.")
    parser.add_argument("--compare-finetuning", action="store_true",
                        help="Also train fine-tuned and from-scratch encoders and compare "
                             "against frozen z-star. ~3x the cost; the from-scratch arm is "
                             "what tells you whether SSL pretraining helped.")
    parser.add_argument("--ft-epochs", type=int, default=30,
                        help="Epochs per arm for --compare-finetuning.")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable tqdm progress bars (useful when logs are scraped).")
    args = parser.parse_args()

    run(
        popf_path=args.popf, apgf_path=args.apgf, output_dir=args.output_dir,
        label_col=args.label_col, landmark_day=args.landmark_day,
        latent_dim=args.latent_dim, fusion=args.fusion,
        apgf_encoder=args.apgf_encoder, imputation=args.imputation,
        include_missing_mask=not args.no_missing_mask,
        epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
        progress=not args.no_progress,
        cr_epochs=args.cr_epochs, cr_bins=args.cr_bins,
        compare_finetuning=args.compare_finetuning, ft_epochs=args.ft_epochs,
    )


if __name__ == "__main__":
    main()
