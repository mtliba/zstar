# z-star

A self-supervised, multimodal representation learning framework for kidney transplant patient
data. It encodes static risk factors, longitudinal lab/drug time series, and clinical events
through configurable neural encoders, fuses them into a single latent vector (**z-star**), and
supports downstream tasks such as graft-loss prediction, eGFR trajectory forecasting, and
missing-modality imputation.

**Documentation:** https://mtliba.github.io/zstar/

## Status

This codebase has been verified end-to-end on synthetic (randomly generated) data only:
correct tensor shapes, no NaNs, decreasing training loss, and working save/load across all
encoder/fusion/latent-type combinations. It has **not** been validated on real clinical data,
and no hyperparameters have been tuned for any real outcome. See the
[Status & Scope](https://mtliba.github.io/zstar/#status) section of the documentation for
details on what has and has not been verified.

## Quickstart

```bash
pip install torch omegaconf scikit-learn matplotlib numpy umap-learn
python -m zstar.run
```

This trains on synthetic static/temporal/event data, extracts z-star embeddings, and runs a
downstream comparison. Replace `make_synthetic_data()` in `run.py` with a loader over your own
tables to use real data.

## Architecture

- **Encoders** (`encoders/`): MLP, GRU, LSTM, Transformer, TCN, Set-Transformer — one per
  modality, config-selectable.
- **Fusion** (`fusion/`): Product of Experts, Mixture of Experts, Attention, Concat.
- **Quantization** (`quantization/`): VQ-VAE codebook with EMA updates and straight-through
  estimator, for discrete/hybrid latents.
- **Losses** (`losses/`): reconstruction, KL (β-VAE), VQ commitment, contrastive (NT-Xent),
  MMD alignment, temporal prediction. See the docs for which are auto-wired vs. require manual
  integration.
- **Models** (`models/`): `ZStarModel` orchestrates per-modality encoders, fusion, and z-star
  extraction.
- **Data** (`data/`): dataset/collate for static + temporal + event modalities, masking utilities.
- **Training** (`training/`): trainer with KL annealing, LR scheduling, staged objectives.
- **Evaluation** (`evaluation/`): z-star extraction, latent analysis (UMAP/t-SNE/clustering),
  cross-modal reconstruction, downstream heads (graft loss, eGFR), approach comparison.

Full documentation, API reference, and guides: **https://mtliba.github.io/zstar/**

## License

Research code, no license file added yet — all rights reserved by default until a license is
chosen.
