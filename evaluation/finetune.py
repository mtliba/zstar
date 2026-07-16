"""
Frozen vs. fine-tuned vs. from-scratch comparison.

The question this answers
-------------------------
Using z-star frozen (train a head on fixed embeddings) is cheap and is what the
rest of this framework does by default. But it is not obviously the best use of
a pretrained encoder, and on its own it cannot tell you whether the
self-supervised pretraining contributed anything at all.

Three arms, identical in every other respect (same split, same head, same
schedule), so the differences are attributable:

  frozen    : encoder weights fixed; only the head learns.
              Tests: is the outcome linearly/nonlinearly decodable from z-star
              as pretrained?
  finetune  : encoder starts from the pretrained weights and keeps training,
              jointly with the head, at a lower learning rate.
              Tests: does adapting the representation to the outcome help?
  scratch   : encoder re-initialised randomly, trained jointly with the head.
              NO self-supervised pretraining is used at all.
              Tests: did the SSL pretraining actually buy anything?

The `scratch` arm is the one that makes the comparison meaningful. If
`scratch` matches `finetune`, the pretraining added nothing and the encoder
architecture (plus the labels) is doing all the work -- which is a result worth
knowing before claiming a foundational model is useful.

Expect `finetune` and `scratch` to overfit more readily than `frozen`: they
have orders of magnitude more trainable parameters fitting the same small
number of events. Compare on the held-out split, not the training split.
"""

import copy
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from zstar.data.collate import zstar_collate
from .discrete_survival import (
    DiscreteTimeCompetingRisksHead,
    discrete_time_competing_risks_loss,
    predict_cif,
    make_time_bins,
    discretize,
)

MODES = ("frozen", "finetune", "scratch")


class _IndexedDataset(Dataset):
    """Wraps a dataset so each item carries its original index (to look up labels)."""

    def __init__(self, base, indices):
        self.base = base
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = int(self.indices[i])
        return self.base[j], j


def _indexed_collate(items):
    batch = zstar_collate([x[0] for x in items])
    idx = torch.tensor([x[1] for x in items], dtype=torch.long)
    return batch, idx


def _to_device(batch, device):
    return {
        name: {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in item.items()}
        for name, item in batch.items()
    }


def train_end_to_end(
    zstar_model,
    dataset,
    durations: np.ndarray,
    causes: np.ndarray,
    cfg=None,
    mode: str = "finetune",
    n_bins: int = 20,
    hidden_dims: List[int] = [64, 32],
    epochs: int = 30,
    head_lr: float = 1e-3,
    encoder_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    val_split: float = 0.2,
    seed: int = 42,
    grad_clip: float = 1.0,
    verbose: bool = True,
) -> Dict:
    """
    Train the competing-risks head on top of the encoder in one of three modes.

    `encoder_lr` is deliberately lower than `head_lr` for `finetune`: the head
    is randomly initialised and needs to move fast, while the encoder already
    holds a useful representation that large updates would destroy.

    For `mode="scratch"`, `cfg` must be supplied so a fresh ZStarModel can be
    built with identical architecture but random weights.
    """
    from .competing_risks import cause_specific_concordance
    from .downstream import make_split
    from zstar.models import ZStarModel

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got '{mode}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    durations = np.asarray(durations, dtype=float)
    causes = np.asarray(causes, dtype=int)
    n_causes = int(causes.max()) if causes.max() > 0 else 1

    cuts = make_time_bins(durations, causes, n_bins=n_bins)
    bins = discretize(durations, cuts)
    K = len(cuts) + 1

    # Same split for every mode -- otherwise the arms are not comparable
    train_idx, val_idx = make_split(len(dataset), val_split=val_split, seed=seed)
    train_idx = train_idx.numpy()
    val_idx = val_idx.numpy()

    if mode == "scratch":
        if cfg is None:
            raise ValueError("mode='scratch' requires cfg to rebuild a randomly-initialised encoder")
        torch.manual_seed(seed)
        encoder = ZStarModel(cfg).to(device)
    else:
        encoder = copy.deepcopy(zstar_model).to(device)

    frozen = mode == "frozen"
    for p in encoder.parameters():
        p.requires_grad = not frozen

    head = DiscreteTimeCompetingRisksHead(
        latent_dim=encoder.latent_dim, n_bins=K, n_causes=n_causes, hidden_dims=hidden_dims,
    ).to(device)

    param_groups = [{"params": head.parameters(), "lr": head_lr}]
    if not frozen:
        param_groups.append({"params": encoder.parameters(), "lr": encoder_lr})
    optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)

    t_all = torch.tensor(bins, dtype=torch.long, device=device)
    k_all = torch.tensor(causes, dtype=torch.long, device=device)

    train_loader = DataLoader(
        _IndexedDataset(dataset, train_idx), batch_size=batch_size,
        shuffle=True, collate_fn=_indexed_collate,
    )
    val_loader = DataLoader(
        _IndexedDataset(dataset, val_idx), batch_size=batch_size,
        shuffle=False, collate_fn=_indexed_collate,
    )

    n_trainable = sum(p.numel() for p in head.parameters())
    if not frozen:
        n_trainable += sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    history = {"epoch": [], "train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        encoder.train(not frozen)
        head.train()
        tot, seen = 0.0, 0
        for batch, idx in train_loader:
            batch = _to_device(batch, device)
            idx = idx.to(device)

            if frozen:
                with torch.no_grad():
                    z = encoder.encode_and_fuse(batch)
            else:
                z = encoder.encode_and_fuse(batch)

            loss = discrete_time_competing_risks_loss(head(z), t_all[idx], k_all[idx])
            optimizer.zero_grad()
            loss.backward()
            if not frozen:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
            optimizer.step()
            tot += loss.item() * len(idx)
            seen += len(idx)
        train_loss = tot / max(seen, 1)

        encoder.eval()
        head.eval()
        vtot, vseen = 0.0, 0
        with torch.no_grad():
            for batch, idx in val_loader:
                batch = _to_device(batch, device)
                idx = idx.to(device)
                z = encoder.encode_and_fuse(batch)
                vloss = discrete_time_competing_risks_loss(head(z), t_all[idx], k_all[idx])
                vtot += vloss.item() * len(idx)
                vseen += len(idx)
        val_loss = vtot / max(vseen, 1)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "head": {k: v.detach().clone() for k, v in head.state_dict().items()},
                "encoder": {k: v.detach().clone() for k, v in encoder.state_dict().items()},
            }

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == 1):
            print(f"    [{mode}] epoch {epoch:3d}: train={train_loss:.4f} val={val_loss:.4f}")

    if best_state is not None:
        head.load_state_dict(best_state["head"])
        encoder.load_state_dict(best_state["encoder"])

    # Predict CIF for every subject
    encoder.eval()
    head.eval()
    all_loader = DataLoader(
        _IndexedDataset(dataset, np.arange(len(dataset))), batch_size=256,
        shuffle=False, collate_fn=_indexed_collate,
    )
    cif_chunks = []
    with torch.no_grad():
        for batch, idx in all_loader:
            batch = _to_device(batch, device)
            z = encoder.encode_and_fuse(batch)
            cif_chunks.append(predict_cif(head(z)).cpu().numpy())
    cif_all = np.concatenate(cif_chunks, axis=0)

    results = {
        "mode": mode,
        "history": history,
        "best_val_loss": best_val,
        "cif": cif_all,
        "val_idx": val_idx,
        "train_idx": train_idx,
        "n_trainable_params": n_trainable,
    }
    for c in range(1, n_causes + 1):
        risk = cif_all[:, -1, c - 1]
        results[f"risk_cause{c}"] = risk
        results[f"c_index_val_cause{c}"] = cause_specific_concordance(
            durations[val_idx], causes[val_idx], risk[val_idx], cause=c
        )
        results[f"c_index_train_cause{c}"] = cause_specific_concordance(
            durations[train_idx], causes[train_idx], risk[train_idx], cause=c
        )
    return results


def compare_finetuning_strategies(
    zstar_model,
    dataset,
    durations: np.ndarray,
    causes: np.ndarray,
    cfg,
    modes: tuple = MODES,
    verbose: bool = True,
    **kwargs,
) -> Dict[str, Dict]:
    """
    Run every arm and return {mode: results}. All arms share the split, head
    architecture, schedule and seed, so differences are attributable to the
    encoder treatment alone.
    """
    out = {}
    for mode in modes:
        if verbose:
            print(f"\n  --- {mode} ---")
        out[mode] = train_end_to_end(
            zstar_model, dataset, durations, causes, cfg=cfg, mode=mode,
            verbose=verbose, **kwargs,
        )
    return out
