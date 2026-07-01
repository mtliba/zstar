# z-star

A self-supervised, multimodal representation learning framework for kidney transplant patient
data. It encodes static risk factors, longitudinal lab/drug time series, and clinical events
through configurable neural encoders, fuses them into a single latent vector — **z-star (`z*`)**
— and supports downstream tasks such as graft-loss prediction, eGFR trajectory forecasting,
and missing-modality imputation.

**Full documentation site:** https://mtliba.github.io/zstar/

---

## Table of contents

- [Status & scope](#status--scope)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Architecture](#architecture)
- [Modality types](#modality-types)
- [Dataset & collation](#dataset--collation)
- [Encoders](#encoders)
- [Fusion strategies](#fusion-strategies)
- [VQ-VAE quantization](#vq-vae-quantization)
- [Using the model (API)](#using-the-model-api)
- [Self-supervised objectives](#self-supervised-objectives)
- [Using SSL losses directly](#using-ssl-losses-directly)
- [Staged training](#staged-training)
- [Configuration reference](#configuration-reference)
- [Reconstruction guide](#reconstruction-guide)
- [Comparison guide](#comparison-guide)
- [Downstream tasks](#downstream-tasks)
- [Latent space analysis](#latent-space-analysis)
- [Directory structure](#directory-structure)
- [Extending the framework](#extending-the-framework)
- [License](#license)

---

## Status & scope

**This codebase has been exercised only with synthetic (randomly generated) data.** Training
runs, architecture combinations, and evaluation code paths have been checked for correctness —
correct tensor shapes, no NaNs, decreasing training loss, successful checkpointing — not for
predictive validity. It has not been run on real clinical data and no hyperparameters have been
tuned for any real outcome.

### What has been verified

| Check | Result |
|---|---|
| Forward/backward pass, all 6 encoders | Runs without shape errors or NaNs |
| Fusion: PoE, MoE, Attention, Concat | Each runs end-to-end with mixed missing modalities |
| VQ-VAE (discrete / hybrid latent) | Runs; codebook perplexity > 1 observed (codebook is used, not collapsed) |
| Training loop, 300 synthetic patients, 100 epochs | Train/val total loss decreases across epochs |
| z-star extraction and save/load round-trip | Produces a `[N, latent_dim]` matrix; loads back correctly |
| Downstream heads (linear probe, MLP) | Train without error on frozen z-star |

### What has not been verified

- Whether z-star captures anything clinically meaningful — synthetic data is uncorrelated
  Gaussian noise, so there is nothing meaningful for it to capture.
- Any AUROC / AUPRC / R² number produced by the code in its current state is **not evidence of
  predictive skill** — synthetic labels are drawn independently of synthetic features. Values
  near 0.5 AUROC are expected and correct.
- Runtime and memory behavior at realistic cohort sizes and sequence lengths.
- Whether the default architectures (e.g. Transformer for creatinine, GRU for medications) are
  appropriate choices for a specific dataset — these are configurable defaults, not validated
  recommendations.

Treat every code example in this document as a description of **how to call the framework**,
not a claim about what results it will produce on your data. Results depend entirely on the
dataset you provide.

---

## Installation

```bash
git clone https://github.com/mtliba/zstar.git
cd zstar
pip install -r requirements.txt
```

`requirements.txt` pins: `torch`, `omegaconf`, `numpy`, `scikit-learn`, `matplotlib`, `umap-learn`.

---

## Quickstart

Run the framework end-to-end with synthetic data to confirm your environment is set up
correctly, before connecting real data.

```bash
python -m zstar.run
```

This generates synthetic static/temporal/event data, trains for the configured number of
epochs, extracts z-star, saves it to `outputs/zstar_embeddings.npy`, and runs the downstream
comparison described below. Expect this run to demonstrate that the code executes correctly —
not to produce a meaningful predictive result, since the labels and features are both synthetic
and independent of each other.

Faster iteration while developing:

```bash
python -m zstar.run training.epochs=5 training.batch_size=16 logging.log_every=1
```

Override any config value from the command line with dot-notation (OmegaConf merges it over the
file defaults):

```bash
python -m zstar.run model.fusion=attention
python -m zstar.run model.latent_type=discrete
python -m zstar.run modalities.creatinine.encoder=lstm
python -m zstar.run model.beta=8.0 training.epochs=200 model.fusion=moe
```

### Connecting real data

Replace `make_synthetic_data()` in `run.py` with a function returning the same dictionary shape
from your actual tables:

```python
def load_real_data(cfg) -> dict:
    return {
        "donor_recipient": {"type": "static",   "data": donor_recipient_df.values},
        "creatinine":      {"type": "temporal", "data": creatinine_per_patient},
        "medications":     {"type": "temporal", "data": drug_dose_series},
        "events":          {"type": "event",    "data": rejection_episodes},
    }
```

Each `modalities.<name>.input_dim` in `config.yaml` must be updated to match the actual feature
count of your real data before running.

---

## Architecture

Every modality — regardless of type — is processed through the same three-stage pipeline:
independent per-modality encoding, posterior fusion into one shared latent, and per-modality
decoding used for the reconstruction-based training signal.

```
Patient record (static + temporal + event tables)
        │
        ▼
Per-modality encoders  (MLP / GRU / LSTM / Transformer / TCN / SetTransformer)
        │
        ▼
Fusion  (PoE / MoE / Attention / Concat)
        │
        ▼
z-star  (mean of the fused posterior)
        │
        ▼
Per-modality decoders  (reconstruction / imputation)
```

### Forward pass

| Step | Operation | Output |
|---|---|---|
| 1 | Each present modality is encoded independently | `(mu_i, log_var_i)` per modality |
| 2 | Optional vector quantization (if `latent_type` is discrete/hybrid) | `z_q`, codebook indices, commitment loss |
| 3 | Reparameterization samples a latent per modality (training only; deterministic at eval) | `z_i` per modality |
| 4 | Fusion combines all present-modality posteriors | `mu_shared, log_var_shared` |
| 5 | z-star is read off as the mean of the fused posterior | `z* = mu_shared` |
| 6 | Each modality's decoder reconstructs from the shared latent | per-modality reconstruction |

**Missing modalities are handled by masking, not imputation-before-training.** Each sample
carries a per-modality binary mask. With PoE fusion, a missing modality contributes zero
precision to the fused posterior — it has no effect on the result — and the prior `N(0,I)` is
always included as a baseline expert, so a fully-missing patient still receives a defined
(uninformative) z-star rather than an error.

---

## Modality types

Table schemas are not hardcoded. Each variable group is declared as one of three modality types
in `config.yaml`; the modality type determines which encoder families are applicable.

| Type | Description | Example | Applicable encoders |
|---|---|---|---|
| `static` | Flat tabular features that don't change over time | donor age, cold ischaemia time, HLA mismatches | `mlp` |
| `temporal` | Irregularly-sampled longitudinal series | creatinine trajectory, medication doses over time | `gru`, `lstm`, `transformer`, `tcn` |
| `event` | Discrete, sparsely-timed occurrences | rejection episodes, hospitalizations, drug switches | `set_transformer` |

Example modality declaration:

```yaml
modalities:
  creatinine:
    enabled: true
    type: temporal
    input_dim: 1
    max_seq_len: 200
    encoder: transformer
    decoder: temporal
    encoder_config:
      d_model: 128
      nhead: 4
      num_layers: 3
      time_encoding: sinusoidal  # sinusoidal | learnable
      pooling: cls               # cls | mean | last
```

Adding a modality requires no code changes to the orchestrator — declare it in the config and
point it at a registered encoder/decoder name. Choosing sensible `encoder_config` values
(sequence length, hidden size) for your actual data is still up to you; the defaults shown here
are illustrative, not tuned.

---

## Dataset & collation

`ZStarDataset` (`data/dataset.py`) normalizes each modality using only its non-missing samples.
`zstar_collate` (`data/collate.py`) pads variable-length sequences within a batch and produces a
`lengths` tensor used for packed-sequence RNNs and attention masks.

- **Static handling:** a row that is entirely `NaN` is flagged missing (`mask=0`). Z-score
  normalization statistics are computed only from valid rows; missing values are set to zero
  after normalization.
- **Temporal / event handling:** each patient supplies a `(timestamps, values)` tuple, or
  `None` if the modality is entirely absent. Within a batch, sequences are padded to the
  batch's longest sequence.

```python
from zstar.data import ZStarDataset, zstar_collate

data_dict = {
    "donor_recipient": {"type": "static",   "data": static_array},       # np.ndarray [N, D], NaN rows = missing
    "creatinine":      {"type": "temporal", "data": creatinine_sequences}, # list length N of (timestamps, values) or None
    "events":          {"type": "event",    "data": rejection_events},
}

dataset = ZStarDataset(data_dict, normalize=True)
print(dataset.modality_info)
# {'creatinine': {'type': 'temporal', 'availability': 0.847}, ...}
# availability is the fraction of patients for which that modality is present
```

Exact shape produced per sample by `zstar_collate`:

| Modality type | Batch dict entry |
|---|---|
| static | `{"x": [B, D], "mask": [B], "type": "static"}` |
| temporal / event | `{"x": [B, T_max, D], "mask": [B], "timestamps": [B, T_max], "lengths": [B], "type": "temporal"}` |

`T_max` is the maximum sequence length within that specific batch, not a global constant — it
varies from batch to batch.

---

## Encoders

Every encoder implements the same call signature —
`forward(x, timestamps, lengths) → (mu, log_var)` — so the model orchestrator never branches on
modality type. Static encoders ignore the temporal arguments.

| Encoder | File | Applicable modality type | Notes |
|---|---|---|---|
| MLP | `encoders/mlp.py` | static | Feed-forward encoder for static tabular features |
| GRU | `encoders/gru.py` | temporal | Time injected via sinusoidal/learnable encoding added at each step; padded steps excluded via `pack_padded_sequence` |
| LSTM | `encoders/lstm.py` | temporal | Same time-injection scheme as GRU |
| Transformer | `encoders/transformer.py` | temporal / event | Self-attention with continuous time encoding in place of integer positional encoding — does not assume evenly spaced samples; CLS/mean/last pooling |
| TCN | `encoders/tcn.py` | temporal | Causal dilated convolutions with residual connections |
| Set Transformer | `encoders/set_transformer.py` | event | Inducing-point attention pooling over an unordered set of events — output does not depend on event order |
| Time encoding | `encoders/time_encoding.py` | shared utility | `sinusoidal` is parameter-free; `learnable` is a small MLP from scalar time to a vector |

```python
class BaseEncoder(nn.Module, ABC):
    def forward(
        self,
        x: Tensor,                          # [B, D] static  |  [B, T, D] temporal/event
        timestamps: Optional[Tensor] = None, # [B, T]; ignored by static encoders
        lengths: Optional[Tensor] = None,    # [B]; true sequence length before padding
    ) -> Tuple[Tensor, Tensor]:
        """Returns (mu, log_var), each [B, latent_dim]."""
```

---

## Fusion strategies

Fusion combines each modality's `(mu_i, log_var_i)` and presence mask into a single fused
posterior. z-star is derived from this fused posterior's mean.

| Strategy | File | Description |
|---|---|---|
| Product of Experts (default) | `fusion/poe.py` | Analytically combines diagonal Gaussians by precision-weighted averaging (closed-form, no learned parameters). A missing modality contributes zero precision. The prior `N(0,I)` always participates as an additional expert. |
| Mixture of Experts | `fusion/moe.py` | A small learned gating network produces softmax weights over available modalities. Missing modalities are masked out of the softmax before weighting. |
| Attention Fusion | `fusion/attention.py` | A learned query vector attends over the stacked per-modality means, with a padding mask for absent modalities. |
| Concat Fusion | `fusion/concat.py` | Concatenates masked means, passes them through an MLP that outputs the fused mean and log-variance directly. No distributional assumption about the individual experts. |

**z\* is defined as `mu_shared` at evaluation time**, i.e. `model.eval()` +
`extract_zstar()`. During training, `z_shared` is a stochastic sample from the fused posterior
via the reparameterization trick — this is what the decoders and downstream losses actually see
while the model is being fit.

---

## VQ-VAE quantization

Setting `model.latent_type: discrete` (or `hybrid`) routes each modality's continuous latent
through a learned codebook (`quantization/vq.py`).

| Mechanism | Description |
|---|---|
| Codebook lookup | Nearest-neighbor search over `num_embeddings` learned vectors via `torch.cdist` |
| Straight-through estimator | Gradients are copied from `z_q` to `z_e` so the encoder remains trainable through the non-differentiable argmin |
| EMA codebook update | Exponential moving average of cluster assignments updates the codebook directly when `use_ema: true`, avoiding a separate codebook loss term |
| Dead code restart | Codes with EMA usage below `restart_threshold` are reinitialized from randomly sampled encoder outputs in the current batch |
| Commitment loss | `commitment_cost · ‖z_e − sg(z_q)‖²`, added to the total loss under the `vq_commitment` objective |

```yaml
model:
  latent_type: discrete      # continuous | discrete | hybrid
  vq:
    num_embeddings: 512       # codebook size (K)
    embedding_dim: 64         # must equal model.latent_dim
    commitment_cost: 0.25
    use_ema: true
    ema_decay: 0.99
    restart_threshold: 1.0
```

`run.py` auto-sets `objectives.vq_commitment.enabled=true` whenever `model.latent_type` is
`discrete` or `hybrid`. In `hybrid` mode, the continuous `(mu, log_var)` are still what fusion
and the KL term operate on; the discrete code exists in parallel per modality.

> Codebook perplexity was checked to be greater than 1 on synthetic data (more than one code is
> actually used), which rules out immediate collapse. This is a weak sanity check, not evidence
> that the codebook is capturing anything semantically useful — that can only be assessed on
> real data.

---

## Using the model (API)

`ZStarModel` (`models/zstar_model.py`) is a standard `torch.nn.Module`.

### 1. Instantiating from config

```python
from omegaconf import OmegaConf
from zstar.models import ZStarModel

cfg = OmegaConf.load("zstar/config.yaml")
model = ZStarModel(cfg)   # builds one ModalityModule per enabled modality + the fusion module
```

The `input_dim` declared per modality in `cfg.modalities` must match the feature dimension of
the data you pass in. Mismatches raise a shape error at the first linear layer of the
corresponding encoder.

### 2. Training forward pass

```python
model.train()
outputs = model(batch)  # batch: output of zstar_collate()
```

`outputs` keys, and when each one is present:

| Key | Shape / type | Always present? |
|---|---|---|
| `recons` | `{modality: tensor}`, per-modality reconstruction | yes |
| `mus` / `log_vars` | `{modality: [B, latent_dim]}`, per-modality posterior params | yes |
| `zs` | `{modality: [B, latent_dim]}`, per-modality sampled latent | yes |
| `masks` | `{modality: [B]}`, presence mask per modality | yes |
| `mu_shared` / `log_var_shared` | `[B, latent_dim]`, fused posterior | yes |
| `z_shared` | `[B, latent_dim]`; equals `mu_shared` in eval mode | yes |
| `vq_losses` | `{modality: {"vq_loss": ..., "perplexity": ...}}` | only if `latent_type` is discrete/hybrid |
| `temporal_preds` / `temporal_targets` | `{modality: [B, horizon, D]}` | only if `objectives.temporal_prediction.enabled` and modality is listed under it |

### 3. Extracting z-star (deterministic inference)

```python
model.eval()
z_star = model.extract_zstar(batch)   # [B, latent_dim], no gradient, calls model.eval() internally
```

`extract_zstar` only encodes and fuses — it does not run any decoder, so it is cheaper than a
full forward pass when you only need the representation.

### 4. Decoding from an arbitrary latent vector

```python
# Decode a specific modality from any [B, latent_dim] tensor,
# not necessarily one produced by encoding real data
recon = model.modules_dict["creatinine"].decode(z_star)

# For temporal decoders you can control the output sequence length:
recon = model.modules_dict["creatinine"].decode(
    z_star,
    target_timestamps=timestamps,   # [B, T] — if provided, output has T steps
    target_lengths=lengths,         # [B] — used if timestamps is None
)
# If neither is given, the temporal decoder falls back to modalities.<name>.max_seq_len
```

### 5. Imputing a missing modality

```python
# partial_batch contains only the modalities that ARE available for this patient
imputed = model.impute(partial_batch, target_modality="creatinine")
# Internally: z = extract_zstar(partial_batch); return modules_dict["creatinine"].decode(z, ...)
```

`model.modules_dict` is an `nn.ModuleDict` keyed by modality name — this is how you reach a
specific encoder or decoder directly, for example to freeze one modality's encoder or to
inspect its parameters.

---

## Self-supervised objectives

Objectives are toggled and weighted independently in `config.yaml` under
`objectives.<name>.enabled` / `.weight`.

| Objective | File | Description |
|---|---|---|
| Reconstruction | `losses/reconstruction.py` | MSE / BCE / Huber between input and decoder output, computed only over present samples, summed across enabled modalities |
| KL divergence (β-VAE) | `losses/kl.py` | Analytical KL between the fused posterior and `N(0,I)`, scaled by an annealed β (linear / cyclical / monotonic schedule) |
| VQ commitment | `losses/vq_loss.py` | Active only when `latent_type` is discrete/hybrid — pulls encoder outputs toward their assigned codebook entry |
| Contrastive (NT-Xent) | `losses/contrastive.py` | Treats one patient's latents from two different modalities as a positive pair against all other patients in the batch |
| Masked reconstruction | `losses/masked_prediction.py` | Loss function exists. **Not called automatically** — see [Using SSL losses directly](#using-ssl-losses-directly) |
| Temporal prediction | `losses/temporal_prediction.py` | A small GRU head forecasts the last `prediction_horizon` steps of a chosen temporal modality from z-star |
| Alignment (MMD) | `losses/alignment.py` | Kernel-based (RBF) distributional distance between modality-pair latents; `strategy: mmd` or `strategy: contrastive` |

---

## Using SSL losses directly

Each loss is a plain, independently importable function under `zstar/losses/`. The orchestrator
`compute_total_loss()` calls a subset of them automatically based on `cfg.objectives`; you can
also call any of them directly for debugging, unit testing, or a custom training loop.

### What is wired automatically vs. what requires manual code

| Objective | Auto-wired in `compute_total_loss`? | Condition |
|---|:---:|---|
| `reconstruction` | ✅ | always, when enabled |
| `kl` | ✅ | always, when enabled |
| `vq_commitment` | ✅ | requires `outputs["vq_losses"]`, i.e. discrete/hybrid latent |
| `contrastive` | ✅ | needs ≥2 modalities present in the same sample |
| `alignment` | ✅ | needs ≥2 modalities present in the same sample |
| `temporal_prediction` | ✅ | requires listing target modalities under `objectives.temporal_prediction.modalities` |
| `masked_reconstruction` | ⚠️ manual | `ZStarModel.forward()` does not apply masking or populate `outputs["masked_recons"]` — see below |

> Setting `objectives.masked_reconstruction.enabled: true` in the config currently has **no
> effect** on training, because `compute_total_loss` only adds this term when
> `outputs.get("masked_recons")` is truthy, and the default forward pass never sets that key.
> The loss function and the masking utility both exist and are usable — they are just not
> connected to the default training loop.

### Calling the orchestrator (standard path)

```python
from zstar.losses import compute_total_loss

outputs = model(batch)
losses = compute_total_loss(batch, outputs, cfg, beta=1.0, stage_name=None)
# losses is a dict: {"recon": ..., "kl": ..., "contrastive": ..., ..., "total": ...}
losses["total"].backward()
```

### Calling individual loss functions directly

```python
from zstar.losses.reconstruction import reconstruction_loss
from zstar.losses.kl import kl_divergence
from zstar.losses.contrastive import contrastive_loss, nt_xent_loss
from zstar.losses.alignment import alignment_loss, mmd_loss
from zstar.losses.vq_loss import vq_aggregate_loss

# Reconstruction for one modality (handles static [B,D] and temporal [B,T,D])
r_loss = reconstruction_loss(
    x=batch["donor_recipient"]["x"],
    x_recon=outputs["recons"]["donor_recipient"],
    mask=batch["donor_recipient"]["mask"],
    loss_fn="mse",             # mse | bce | huber
    mod_type="static",
)

# KL between the fused posterior and N(0, I)
kl = kl_divergence(outputs["mu_shared"], outputs["log_var_shared"])

# Contrastive alignment across all present modality pairs in the batch
c_loss = contrastive_loss(outputs["zs"], outputs["masks"], temperature=0.07)

# Or just one pair of modalities directly:
pair_loss = nt_xent_loss(outputs["zs"]["creatinine"], outputs["zs"]["medications"], temperature=0.07)

# MMD alignment (strategy="mmd") or NT-Xent alignment (strategy="contrastive")
a_loss = alignment_loss(outputs["zs"], outputs["masks"], strategy="mmd")
```

### Manually wiring masked reconstruction (not built into the default forward pass)

To use masked reconstruction as it is implemented today, mask the input yourself before
encoding, run the model with the masked input, and compute the loss against the masked
positions. This means writing a small custom step rather than relying on `ZStarTrainer` as-is:

```python
from zstar.data.masking import MaskGenerator
from zstar.losses.masked_prediction import masked_prediction_loss

masker = MaskGenerator(mask_ratio=0.15, strategy="random")  # random | block | feature_wise

# 1. Mask one modality's input before encoding
x_orig = batch["donor_recipient"]["x"]
x_masked, mask_positions = masker(x_orig, mod_type="static")

# 2. Substitute the masked tensor into a copy of the batch and run the model
masked_batch = dict(batch)
masked_batch["donor_recipient"] = dict(batch["donor_recipient"])
masked_batch["donor_recipient"]["x"] = x_masked
outputs = model(masked_batch)

# 3. Compare the reconstruction only at the masked positions
recon = outputs["recons"]["donor_recipient"]
extra_loss = masked_prediction_loss(
    predictions={"donor_recipient": recon},
    targets={"donor_recipient": x_orig},
    mask_positions={"donor_recipient": mask_positions},
)
# Add extra_loss into your total loss alongside compute_total_loss(...)
```

If you rely on masked reconstruction for a real experiment, either extend
`ZStarModel.forward()` to apply `MaskGenerator` internally and populate
`outputs["masked_recons"]` / `outputs["mask_positions"]`, or run the manual step above inside a
custom training loop copied from `ZStarTrainer._run_batch`.

---

## Staged training

Objectives can optionally be phased in over the course of training via `training.stages`,
rather than all being active from epoch 1.

```yaml
training:
  stages:
    - name: "reconstruction_only"
      epochs: 50
      objectives: [reconstruction, kl]
    - name: "add_contrastive"
      epochs: 75
      objectives: [reconstruction, kl, contrastive, alignment]
    - name: "full"
      epochs: 75
      objectives: all
```

Stage epoch counts are cumulative — in this example, epochs 1–50 use stage one, 51–125 use
stage two, 126–200 use stage three. Omit `training.stages` entirely to train all
`enabled: true` objectives jointly from epoch 1, which is the default.

---

## Configuration reference

Everything is driven by `zstar/config.yaml`, loaded with OmegaConf. Values can be overridden
from the command line with dot-notation, which OmegaConf merges over the file defaults.

### Top-level sections

| Section | Contains |
|---|---|
| `project` | `name`, `seed` |
| `model` | `latent_dim`, `fusion`, `latent_type`, `beta`, `vq{}` |
| `modalities` | per-modality `type` / `encoder` / `decoder` / dims |
| `objectives` | `enabled` + `weight` per SSL loss |
| `training` | `batch_size`, `epochs`, `lr`, `scheduler`, `stages` |
| `data` | `path`, `normalize` |
| `logging` | `log_every`, `save_dir` |
| `downstream` | `graft_loss{}`, `egfr_trajectory{}` |

### Command-line overrides

```bash
# switch fusion strategy
python -m zstar.run model.fusion=attention

# switch to discrete VQ-VAE latents
python -m zstar.run model.latent_type=discrete

# swap the encoder used for one modality
python -m zstar.run modalities.creatinine.encoder=lstm

# combine multiple overrides
python -m zstar.run model.beta=8.0 training.epochs=200 model.fusion=moe
```

---

## Reconstruction guide

"Reconstruction" covers three distinct operations in this framework: reconstructing a modality
the model already saw (training signal), imputing a modality that is missing for a specific
patient, and decoding an arbitrary point in latent space (for interpolation or inspection).

### 1. Reconstruction as training signal

During a normal forward pass, every present modality is decoded from `z_shared` and compared to
its own input — this is what `outputs["recons"]` contains and what the reconstruction loss is
computed on.

```python
outputs = model(batch)
recon = outputs["recons"]["creatinine"]   # [B, T, 1], reconstructed from the SAME creatinine data
```

This tells you how well the model can reproduce a modality it was allowed to see — useful for
checking training convergence, not for imputation.

### 2. Imputing a modality that is missing for a patient

```python
# partial_batch must NOT contain the "creatinine" key at all, or must have
# mask=0 for it — either way the encoder for "creatinine" is skipped
imputed_creatinine = model.impute(partial_batch, target_modality="creatinine")
```

Internally this calls `extract_zstar(partial_batch)` — which fuses only the modalities present
in `partial_batch` — then decodes the target modality from that z-star. The quality of this
imputation depends entirely on how much the fused latent actually correlates with the target
modality, which is only measurable on real data.

### 3. Evaluating cross-modal reconstruction quality across a dataset

```python
from zstar.evaluation import cross_modal_reconstruction

metrics = cross_modal_reconstruction(
    model, dataset,
    source_modalities=["medications", "events"],
    target_modality="creatinine",
    device=device,
)
# {"mse": float, "r2": float}
```

This function iterates the full dataset, keeps only samples where **both** the source
modalities and the target modality are present (so it can compute ground truth), encodes using
only the source modalities, decodes the target, and reports MSE and R² against the real target
values.

| Detail | Behavior |
|---|---|
| Evaluation subset | Only samples where all listed sources *and* the target are present |
| Static target | Compared directly, `[N_valid, D]` |
| Temporal target | Compared per-timestep after flattening; sequence-length mismatches between prediction and target are not automatically truncated in this function — ensure the decoder's output length matches the target's length for the samples being compared |
| R² definition | `sklearn.metrics.r2_score` on the flattened arrays; negative values mean worse than predicting the mean |

> On synthetic data, MSE ≈ 1.0 and R² ≈ 0 to slightly negative is the expected and correct
> result — the source and target modalities are independently generated random noise, so there
> is no signal to transfer between them. A positive R² here would indicate a bug, not a good
> model.

### 4. Decoding an arbitrary latent point

Useful for interpolation between two patients or manual latent traversal — bypasses the encoder
entirely:

```python
from zstar.evaluation import latent_interpolation

path = latent_interpolation(
    model,
    z1=zstar_embeddings[0],
    z2=zstar_embeddings[1],
    target_modality="donor_recipient",
    n_steps=10,
)
# [10, D] — linear interpolation in latent space, each point decoded independently
```

---

## Comparison guide

This section covers four distinct kinds of comparison the framework supports, and how to run
each one. None of them replace a proper held-out evaluation or cross-validation on real data
with real outcomes.

### 1. Comparing architecture configurations

Because everything is config-driven, comparing two architectures means running training twice
with different overrides and comparing the resulting validation loss or reconstruction metrics:

```bash
python -m zstar.run model.fusion=poe        logging.save_dir=checkpoints/poe
python -m zstar.run model.fusion=attention  logging.save_dir=checkpoints/attention
```

Compare the `best_val_loss` printed at the end of each run, and/or run
`cross_modal_reconstruction` with the same source/target pair against both checkpoints.
`ZStarTrainer.train()` returns a `history` dict with per-epoch `train`/`val` loss components if
you want to plot the curves rather than compare a single number.

### 2. Ablating SSL objectives

Toggle one objective at a time and compare the downstream result, holding the architecture
fixed:

```bash
python -m zstar.run objectives.contrastive.enabled=false
python -m zstar.run objectives.alignment.enabled=false
python -m zstar.run objectives.contrastive.enabled=false objectives.alignment.enabled=false
```

Because SSL pretraining has no single scalar "success" metric, ablations should be judged by
whatever downstream task you actually care about (Section 3 below), not by pretraining loss
alone — a lower reconstruction loss does not necessarily correspond to a more useful z-star.

### 3. Comparing a linear probe against a nonlinear head on frozen z-star

```python
from zstar.evaluation import compare_approaches

results = compare_approaches(
    zstar=zstar_embeddings,          # [N, latent_dim], already extracted and frozen
    graft_loss_labels=labels,        # [N], 0/1
    latent_dim=cfg.model.latent_dim,
)
# {"linear_probe": {...}, "mlp_frozen_zstar": {...}}
# each entry: {"best_val_loss": float, "auroc": float, "auprc": float}
```

`compare_approaches` trains two heads on the same frozen z-star and a single random train/val
split (80/20): a `GraftLossPredictor` with no hidden layers (linear probe) and one with
`[64, 32]` hidden units (MLP). The gap between them indicates how much of whatever signal
exists in z-star is linearly separable versus requiring a nonlinear decision boundary.

> This function uses a single random split and does not fix a fold seed across calls other than
> via `torch.randperm` without an explicit generator — re-running it will produce a different
> split and therefore different AUROC values. For a real evaluation, wrap this in repeated
> k-fold cross-validation and report mean ± standard deviation, rather than trusting one run.

### 4. Comparing frozen z-star against a raw-feature baseline

The framework does not include a built-in raw-feature baseline model — this comparison needs to
be assembled manually, but uses the same downstream head and training function so the
comparison is apples-to-apples:

```python
import numpy as np
from zstar.evaluation.downstream import GraftLossPredictor, train_downstream_head

# Baseline: concatenate raw per-modality static means (no learned encoder at all)
raw_features = np.concatenate([
    dataset.modalities["donor_recipient"]["data"].numpy(),
    # add other static modalities as needed; temporal ones need manual summarization
    # (e.g. mean/last-value per patient) since this baseline has no sequence encoder
], axis=1)

baseline_head = GraftLossPredictor(latent_dim=raw_features.shape[1], hidden_dims=[64, 32])
baseline_results = train_downstream_head(baseline_head, raw_features, labels, task="binary_classification")

# Compare baseline_results["auroc"] against results["mlp_frozen_zstar"]["auroc"] from Section 3
```

This comparison tells you whether the self-supervised pretraining is adding information beyond
what a simple feature-concatenation baseline already provides. It is the comparison most
directly relevant to justifying the framework's added complexity, and it is the one most in
need of real data and proper cross-validation before drawing conclusions.

> **None of the comparisons above are meaningful on synthetic data** — every number they
> produce there is noise measuring noise. They are documented here as procedures to run once
> real labeled outcomes (graft loss, eGFR) are available, not as results to report from the
> current codebase.

---

## Downstream tasks

Downstream heads are simple MLPs trained on the frozen z-star matrix; the encoder and fusion
module are not updated during this phase.

| Head | File | Task |
|---|---|---|
| `GraftLossPredictor(latent_dim, hidden_dims)` | `evaluation/downstream.py` | Binary classification. Trained via `train_downstream_head(..., task="binary_classification")`, reports AUROC and AUPRC on a held-out split |
| `EGFRTrajectoryPredictor(latent_dim, prediction_points, hidden_dims)` | `evaluation/downstream.py` | Regresses graft function at configured future time points (e.g. 30 / 90 / 180 / 365 days) in a single forward pass |

```python
from zstar.evaluation.downstream import EGFRTrajectoryPredictor, train_downstream_head

head = EGFRTrajectoryPredictor(
    latent_dim=cfg.model.latent_dim,
    prediction_points=[30, 90, 180, 365],
    hidden_dims=[64, 32],
)
results = train_downstream_head(
    head, zstar_embeddings, egfr_targets,   # egfr_targets: [N, 4]
    task="regression",
)
```

`train_downstream_head` always instantiates a fresh 80/20 train/val split, trains with Adam and
early best-checkpoint tracking by validation loss, and for `task="binary_classification"`
additionally computes AUROC/AUPRC on the validation split at the end of training.

---

## Latent space analysis

Tools for inspecting the structure of z-star before relying on it downstream
(`evaluation/latent_analysis.py`).

```python
from zstar.evaluation import umap_plot, tsne_plot, cluster_analysis

umap_plot(zstar_embeddings, labels=graft_loss_labels, save_path="outputs/umap.png")
tsne_plot(zstar_embeddings, labels=graft_loss_labels, save_path="outputs/tsne.png")

clusters = cluster_analysis(zstar_embeddings, n_clusters=5, method="kmeans")
# {"labels": [N], "silhouette": float, "n_clusters": 5}
```

A UMAP or t-SNE plot showing visually separated clusters is not, by itself, evidence that the
clusters correspond to clinically distinct patient groups — cross-reference cluster assignment
against known outcomes or chart review before drawing that conclusion.

---

## Directory structure

```
zstar/
├── run.py                    # entry point; contains make_synthetic_data()
├── config.yaml                # default configuration
├── requirements.txt
│
├── encoders/                # mlp · gru · lstm · transformer · tcn · set_transformer
├── decoders/                # mlp · temporal
├── quantization/            # vq.py — codebook, straight-through, EMA
├── fusion/                  # poe · moe · attention · concat
├── losses/                  # reconstruction · kl · vq · contrastive · alignment · masked · temporal
├── models/
│   ├── modality_module.py   # encoder + optional VQ + decoder per modality
│   └── zstar_model.py       # orchestrates all modules + fusion → z-star
├── data/                    # dataset · collate · masking · augmentations
├── training/                # trainer · KL / LR schedulers
├── evaluation/              # extract · latent analysis · downstream heads · comparison
└── docs/
    └── index.html            # full documentation site (GitHub Pages)
```

---

## Extending the framework

Encoders, decoders, and fusion strategies use a name-based registry pattern.

```python
# encoders/my_encoder.py
from . import register_encoder
from .base import BaseEncoder

@register_encoder("my_encoder")
class MyEncoder(BaseEncoder):
    def forward(self, x, timestamps=None, lengths=None):
        # ... your architecture ...
        return mu, log_var
```

Reference it from any modality block in `config.yaml`:

```yaml
modalities:
  my_modality:
    encoder: my_encoder
```

The same pattern applies to `register_decoder` (in `decoders/`) and `register_fusion` (in
`fusion/`). New objectives are added by writing a loss function under `losses/` and adding a
corresponding branch in `losses/__init__.py:compute_total_loss` that reads from `outputs`.

---

## License

MIT — see [LICENSE](LICENSE).
