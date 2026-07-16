"""
Discrete-time competing-risks neural network over frozen z-star.

Model
-----
Follow-up is discretised into K bins. For every subject the network emits, for
each bin, a distribution over {no event, cause 1, ..., cause C}:

    p(bin s, x) = softmax( f_theta(x)[s, :] )    over  C+1  outcomes

so  h_k(s|x) = P(event of cause k in bin s | event-free at start of bin s, x).

Why this rather than Cox
------------------------
There is no proportional-hazards assumption anywhere. Cox constrains every
subject's hazard to be a *fixed multiple* of a shared baseline over all time --
so two patients can never cross, and a covariate's effect cannot change with
time. Here each (bin, cause) has its own output, so the hazard's shape over
time is free to differ arbitrarily between subjects, and a covariate is allowed
to matter early and not late (or reverse sign).

Why this is competing-risks-correct
-----------------------------------
The C+1 outcomes are mutually exclusive within a bin and share one softmax, so
the causes compete for probability mass by construction. Dying in bin s removes
a subject from ever accruing graft-loss incidence afterwards, exactly as
reality requires. CIFs derived from these hazards therefore sum (with the
event-free probability) to 1 -- the property that per-cause 1-KM violates.

Censoring is handled in the likelihood, not by relabelling: a subject censored
in bin t contributes "survived" (no-event) terms for bins 0..t and nothing
after. They are never scored as having failed, and their censoring time is
never treated as an event time. See `discrete_time_competing_risks_loss` for
why the censored subject must be included at bin t itself.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Time discretisation ────────────────────────────────────────────────────

def make_time_bins(
    durations: np.ndarray, causes: np.ndarray, n_bins: int = 20
) -> np.ndarray:
    """
    Bin edges from quantiles of the observed *event* times, so bins carry
    roughly equal numbers of events rather than equal calendar width (which,
    with heavy censoring, would leave most bins empty).
    """
    d = np.asarray(durations, dtype=float)
    c = np.asarray(causes, dtype=int)
    event_times = d[(c > 0) & np.isfinite(d)]
    if len(event_times) < 2:
        event_times = d[np.isfinite(d)]
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    cuts = np.unique(np.quantile(event_times, qs))
    return cuts


def discretize(durations: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Map each duration to its bin index in [0, len(cuts)]."""
    return np.digitize(np.asarray(durations, dtype=float), cuts).astype(int)


# ── Model ──────────────────────────────────────────────────────────────────

class DiscreteTimeCompetingRisksHead(nn.Module):
    """
    z-star -> per-bin, per-cause hazard logits.

    Output shape [B, n_bins, n_causes + 1]; index 0 of the last axis is
    "no event in this bin".
    """

    def __init__(
        self,
        latent_dim: int,
        n_bins: int,
        n_causes: int = 2,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.n_causes = n_causes

        layers = []
        prev = latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_bins * (n_causes + 1)))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        return out.view(-1, self.n_bins, self.n_causes + 1)


def discrete_time_competing_risks_loss(
    logits: torch.Tensor,
    time_bin: torch.Tensor,
    cause: torch.Tensor,
) -> torch.Tensor:
    """
    Negative log-likelihood of the discrete-time competing-risks model.

    For subject i observed until bin t_i with cause k_i (0 = censored):
        bins s < t_i : -log P(no event at s)
        bin  t_i     : -log P(cause k_i at t_i)   if k_i > 0
                       -log P(no event at t_i)    if censored

    Note the censored branch at bin t_i. Excluding it (on the argument that we
    cannot know what happened inside the interval where observation stopped)
    looks more conservative but is badly wrong in practice: with heavy
    censoring at long follow-up, almost every censored subject lands in the
    FINAL bin, so that bin ends up supervised only by the handful of subjects
    who had an event there. The model then learns hazard ~= 1 in the last bin,
    every subject's CIF is pinned near 1.0, and the only remaining variation is
    the probability of *surviving* to that bin -- which inverts the risk score
    and drives C-index below 0.5. Including the censored subject as "no event"
    at t_i is the standard discrete-time (person-period) convention and avoids
    this.

    logits    : [B, K, C+1]
    time_bin  : [B] long, the bin in which observation ended
    cause     : [B] long, 0 = censored, 1..C = cause of the observed event
    """
    B, K, _ = logits.shape
    log_p = F.log_softmax(logits, dim=-1)

    arange = torch.arange(K, device=logits.device)
    t_clamped = time_bin.clamp(max=K - 1)
    survived_mask = (arange[None, :] < t_clamped[:, None]).float()   # [B, K]

    # Censored subjects also contribute "no event" at their final bin
    censored = cause == 0
    if censored.any():
        survived_mask[censored, t_clamped[censored]] = 1.0

    loss_survive = -(log_p[:, :, 0] * survived_mask).sum()

    has_event = cause > 0
    if has_event.any():
        idx = torch.nonzero(has_event, as_tuple=True)[0]
        t_e = t_clamped[idx]
        k_e = cause[idx]
        loss_event = -log_p[idx, t_e, k_e].sum()
    else:
        loss_event = logits.sum() * 0.0

    return (loss_survive + loss_event) / B


@torch.no_grad()
def predict_cif(logits: torch.Tensor) -> torch.Tensor:
    """
    Cumulative incidence per cause from hazard logits.

        S(s)      = prod_{u <= s} P(no event at u)
        CIF_k(s)  = sum_{u <= s} S(u-1) * P(cause k at u)

    Returns [B, K, C] -- CIF for each cause at each bin (cause index 0 here
    corresponds to model cause 1).
    """
    p = F.softmax(logits, dim=-1)              # [B, K, C+1]
    p_no_event = p[:, :, 0]                    # [B, K]
    p_causes = p[:, :, 1:]                     # [B, K, C]

    survival = torch.cumprod(p_no_event, dim=1)                    # S(s)
    survival_prev = torch.cat(
        [torch.ones_like(survival[:, :1]), survival[:, :-1]], dim=1
    )                                                              # S(s-1)

    increments = survival_prev.unsqueeze(-1) * p_causes             # [B, K, C]
    return torch.cumsum(increments, dim=1)


# ── Training ───────────────────────────────────────────────────────────────

def train_competing_risks_head(
    zstar: np.ndarray,
    durations: np.ndarray,
    causes: np.ndarray,
    n_bins: int = 20,
    hidden_dims: List[int] = [64, 32],
    dropout: float = 0.1,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    val_split: float = 0.2,
    seed: int = 42,
    eval_every: int = 5,
    verbose: bool = True,
) -> Dict:
    """
    Train the discrete-time competing-risks head on frozen z-star.

    Uses one seeded split, and reports every metric on the held-out portion
    only. Also records per-epoch train/val loss and per-cause validation
    C-index so the training dynamics can be plotted per outcome.
    """
    from .competing_risks import cause_specific_concordance
    from .downstream import make_split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    durations = np.asarray(durations, dtype=float)
    causes = np.asarray(causes, dtype=int)
    n_causes = int(causes.max()) if causes.max() > 0 else 1

    cuts = make_time_bins(durations, causes, n_bins=n_bins)
    bins = discretize(durations, cuts)
    K = len(cuts) + 1

    z = torch.tensor(zstar, dtype=torch.float32, device=device)
    t = torch.tensor(bins, dtype=torch.long, device=device)
    k = torch.tensor(causes, dtype=torch.long, device=device)

    train_idx, val_idx = make_split(len(z), val_split=val_split, seed=seed)
    train_idx, val_idx = train_idx.to(device), val_idx.to(device)

    head = DiscreteTimeCompetingRisksHead(
        latent_dim=z.shape[1], n_bins=K, n_causes=n_causes,
        hidden_dims=hidden_dims, dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)

    z_tr, t_tr, k_tr = z[train_idx], t[train_idx], k[train_idx]
    z_va, t_va, k_va = z[val_idx], t[val_idx], k[val_idx]

    va_np = val_idx.cpu().numpy()
    tr_np = train_idx.cpu().numpy()

    history = {"epoch": [], "train_loss": [], "val_loss": []}
    for c in range(1, n_causes + 1):
        history[f"val_cindex_cause{c}"] = []

    best_val = float("inf")
    best_state = None
    n_train = len(z_tr)

    for epoch in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss, seen = 0.0, 0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            loss = discrete_time_competing_risks_loss(head(z_tr[idx]), t_tr[idx], k_tr[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
            seen += len(idx)
        train_loss = epoch_loss / max(seen, 1)

        head.eval()
        with torch.no_grad():
            val_loss = discrete_time_competing_risks_loss(head(z_va), t_va, k_va).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {kk: v.detach().clone() for kk, v in head.state_dict().items()}

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % eval_every == 0 or epoch == 1:
            with torch.no_grad():
                cif_va = predict_cif(head(z_va)).cpu().numpy()      # [Bva, K, C]
            for c in range(1, n_causes + 1):
                risk = cif_va[:, -1, c - 1]     # CIF at end of follow-up = overall risk
                ci = cause_specific_concordance(durations[va_np], causes[va_np], risk, cause=c)
                history[f"val_cindex_cause{c}"].append((epoch, ci))
        if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == 1):
            print(f"    epoch {epoch:4d}: train={train_loss:.4f}  val={val_loss:.4f}")

    if best_state is not None:
        head.load_state_dict(best_state)

    head.eval()
    with torch.no_grad():
        cif_all = predict_cif(head(z)).cpu().numpy()               # [N, K, C]

    results = {
        "head": head,
        "history": history,
        "cuts": cuts,
        "n_bins": K,
        "cif": cif_all,
        "val_idx": va_np,
        "train_idx": tr_np,
        "best_val_loss": best_val,
    }

    # Held-out and training C-index per cause, using CIF at end of follow-up
    for c in range(1, n_causes + 1):
        risk_all = cif_all[:, -1, c - 1]
        results[f"c_index_val_cause{c}"] = cause_specific_concordance(
            durations[va_np], causes[va_np], risk_all[va_np], cause=c
        )
        results[f"c_index_train_cause{c}"] = cause_specific_concordance(
            durations[tr_np], causes[tr_np], risk_all[tr_np], cause=c
        )
        results[f"risk_cause{c}"] = risk_all

    return results
