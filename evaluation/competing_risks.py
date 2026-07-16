"""
Competing-risks handling for graft loss vs. death.

Why this module exists
----------------------
Graft loss and death are *competing* events: a patient who dies with a
functioning graft can never subsequently experience graft failure. Treating
death as ordinary censoring for the graft outcome breaks the independent-
censoring assumption that Kaplan-Meier relies on, and 1-KM then *overestimates*
the incidence of graft loss -- it implicitly credits dead patients with the
possibility of failing later.

The correct nonparametric estimator is the Aalen-Johansen cumulative incidence
function (CIF), which accounts for the fact that a competing event removes a
subject from ever experiencing the other one.

Event coding used throughout this module:
    0 = censored (neither event observed)
    1 = graft loss occurred first
    2 = death occurred first
"""

import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAUSE_CENSORED = 0
CAUSE_GRAFT_LOSS = 1
CAUSE_DEATH = 2

CAUSE_NAMES = {CAUSE_GRAFT_LOSS: "graft loss", CAUSE_DEATH: "death"}
_PALETTE = {CAUSE_GRAFT_LOSS: "#253E6B", CAUSE_DEATH: "#A56327"}
_GROUP_PALETTE = ["#253E6B", "#377860", "#A56327", "#733E85", "#7c9fff", "#4fd1c5"]


def _save(fig, save_path: Optional[str]):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ── Deriving the competing-risks encoding ──────────────────────────────────

def derive_competing_events(
    labels_df,
    graft_duration_col: str = "GraftSurvivalDays",
    graft_event_col: str = "FailureWithinStudyPeriod",
    death_duration_col: str = "PatientSurvivalDays",
    death_event_col: str = "DeathWithinStudyPeriod",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Collapse the two (duration, mask) pairs into a single competing-risks
    encoding: time to the FIRST event, and which event it was.

        time  = min(graft duration, death duration)
        cause = 1 if graft loss happened first and was observed
                2 if death happened first and was observed
                0 otherwise (censored -- neither event observed by `time`)

    Returns (time, cause, report).
    """
    t_g = labels_df[graft_duration_col].to_numpy(dtype=float)
    e_g = np.asarray(labels_df[graft_event_col].to_numpy()).astype(bool)
    t_d = labels_df[death_duration_col].to_numpy(dtype=float)
    e_d = np.asarray(labels_df[death_event_col].to_numpy()).astype(bool)

    time = np.minimum(t_g, t_d)
    cause = np.zeros(len(time), dtype=int)

    # Graft loss first (ties broken toward graft loss: the graft failed at or
    # before death, so it is the first event for the graft outcome)
    graft_first = e_g & (t_g <= t_d)
    # Death first, and strictly before any observed graft failure
    death_first = e_d & (t_d < t_g) | (e_d & (t_d == t_g) & ~e_g)

    cause[graft_first] = CAUSE_GRAFT_LOSS
    cause[death_first & ~graft_first] = CAUSE_DEATH

    report = {
        "n": len(time),
        "n_graft_loss": int((cause == CAUSE_GRAFT_LOSS).sum()),
        "n_death": int((cause == CAUSE_DEATH).sum()),
        "n_censored": int((cause == CAUSE_CENSORED).sum()),
    }
    report["pct_graft_loss"] = 100 * report["n_graft_loss"] / report["n"]
    report["pct_death"] = 100 * report["n_death"] / report["n"]
    report["pct_censored"] = 100 * report["n_censored"] / report["n"]

    if verbose:
        print(f"  competing-risks encoding over {report['n']:,} subjects:")
        print(f"    graft loss first : {report['n_graft_loss']:,} ({report['pct_graft_loss']:.1f}%)")
        print(f"    death first      : {report['n_death']:,} ({report['pct_death']:.1f}%)")
        print(f"    censored         : {report['n_censored']:,} ({report['pct_censored']:.1f}%)")

    return time, cause, report


# ── Aalen-Johansen cumulative incidence ────────────────────────────────────

def aalen_johansen(
    durations: np.ndarray, causes: np.ndarray, cause: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aalen-Johansen estimator of the cumulative incidence function for `cause`.

        CIF_k(t) = sum_{s <= t}  S(s-) * d_k(s) / n(s)

    where S is the all-cause Kaplan-Meier survival and d_k(s) counts events of
    type k at time s. Subjects experiencing a competing event leave the risk
    set and correctly stop accruing incidence for this cause -- which is the
    whole difference from 1-KM.

    Returns (times, cif), both starting at t=0 with CIF=0.
    """
    d = np.asarray(durations, dtype=float)
    c = np.asarray(causes, dtype=int)

    finite = np.isfinite(d)
    d, c = d[finite], c[finite]
    n = len(d)
    if n == 0:
        return np.array([0.0]), np.array([0.0])

    unique_times, inverse = np.unique(d, return_inverse=True)
    n_at_time = np.bincount(inverse, minlength=len(unique_times))
    d_any = np.bincount(inverse, weights=(c > 0).astype(float), minlength=len(unique_times))
    d_k = np.bincount(inverse, weights=(c == cause).astype(float), minlength=len(unique_times))

    cum_before = np.concatenate([[0], np.cumsum(n_at_time)[:-1]])
    at_risk = n - cum_before

    with np.errstate(divide="ignore", invalid="ignore"):
        km_factors = 1.0 - d_any / at_risk
        hazard_k = d_k / at_risk
    km_factors = np.where(at_risk > 0, km_factors, 1.0)
    hazard_k = np.where(at_risk > 0, hazard_k, 0.0)

    survival = np.cumprod(km_factors)
    survival_prev = np.concatenate([[1.0], survival[:-1]])   # S(s-)
    cif = np.cumsum(survival_prev * hazard_k)

    return np.concatenate([[0.0], unique_times]), np.concatenate([[0.0], cif])


def cause_specific_concordance(
    durations: np.ndarray,
    causes: np.ndarray,
    risk_scores: np.ndarray,
    cause: int,
    chunk_size: int = 256,
) -> float:
    """
    Cause-specific concordance for `cause`.

    Comparable pairs: subject i experienced `cause` at T_i, and subject j was
    still under observation at T_i (T_j > T_i). Concordant if i was assigned
    the higher risk. Competing events are treated as censoring at their time,
    which is the cause-specific (as opposed to subdistribution) convention --
    it answers "among those still alive and grafted, who fails from this cause
    first", not "what fraction of everyone eventually fails from this cause".
    """
    from .survival import concordance_index

    events_k = (np.asarray(causes) == cause)
    return concordance_index(durations, events_k, risk_scores, chunk_size=chunk_size)


# ── Plots ──────────────────────────────────────────────────────────────────

def plot_cumulative_incidence(
    durations: np.ndarray,
    causes: np.ndarray,
    cause_names: Optional[Dict[int, str]] = None,
    title: str = "Cumulative incidence (Aalen-Johansen)",
    xlabel: str = "Days since transplant",
    compare_with_km: bool = True,
    save_path: Optional[str] = None,
):
    """
    CIF for each competing cause.

    With compare_with_km=True, overlays 1-KM for each cause (treating the other
    cause as plain censoring) as a dashed line. The gap between the two is the
    bias you would introduce by ignoring competing risks -- 1-KM sits above the
    CIF, overstating incidence.
    """
    from .survival import kaplan_meier

    cause_names = cause_names or CAUSE_NAMES
    durations = np.asarray(durations, dtype=float)
    causes = np.asarray(causes, dtype=int)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for k, name in cause_names.items():
        times, cif = aalen_johansen(durations, causes, cause=k)
        color = _PALETTE.get(k, _GROUP_PALETTE[k % len(_GROUP_PALETTE)])
        n_k = int((causes == k).sum())
        ax.step(times, cif, where="post", color=color, linewidth=2.2,
                label=f"{name} — CIF (n={n_k:,}, {cif[-1]:.3f} at end)")

        if compare_with_km:
            km_times, km_surv, _ = kaplan_meier(durations, causes == k)
            ax.step(km_times, 1 - km_surv, where="post", color=color, linewidth=1.3,
                    linestyle="--", alpha=0.75,
                    label=f"{name} — 1−KM (biased: {1 - km_surv[-1]:.3f})")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative incidence")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="upper left")

    if compare_with_km:
        ax.text(
            0.99, 0.02,
            "dashed 1−KM treats the competing event as censoring and overstates incidence",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            color="#6b7385", style="italic",
        )

    plt.tight_layout()
    _save(fig, save_path)


def plot_stacked_incidence(
    durations: np.ndarray,
    causes: np.ndarray,
    cause_names: Optional[Dict[int, str]] = None,
    title: str = "Competing-risks state occupancy",
    save_path: Optional[str] = None,
):
    """
    Stacked view: at any time, what fraction of the cohort is event-free, has
    lost the graft, or has died. The three bands sum to 1 by construction,
    which is the property 1-KM per cause violates.
    """
    cause_names = cause_names or CAUSE_NAMES
    durations = np.asarray(durations, dtype=float)
    causes = np.asarray(causes, dtype=int)

    all_times = np.unique(np.concatenate([[0.0], durations[np.isfinite(durations)]]))
    stacked = []
    labels = []
    colors = []
    for k, name in cause_names.items():
        t_k, cif_k = aalen_johansen(durations, causes, cause=k)
        stacked.append(np.interp(all_times, t_k, cif_k))
        labels.append(name)
        colors.append(_PALETTE.get(k, "#999999"))

    total_incidence = np.sum(stacked, axis=0)
    event_free = 1.0 - total_incidence

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.stackplot(
        all_times, event_free, *stacked,
        labels=["event-free"] + labels,
        colors=["#dbe2f2"] + colors,
        alpha=0.9,
    )
    ax.set_xlim(0, all_times.max())
    ax.set_ylim(0, 1)
    ax.set_xlabel("Days since transplant")
    ax.set_ylabel("Fraction of cohort")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    _save(fig, save_path)


def plot_cif_by_group(
    durations: np.ndarray,
    causes: np.ndarray,
    groups: np.ndarray,
    cause: int = CAUSE_GRAFT_LOSS,
    cause_name: Optional[str] = None,
    group_names: Optional[Dict] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    CIF for one cause, stratified by group (e.g. z-star clusters or predicted-
    risk quartiles). Separation here is evidence the representation carries
    outcome-relevant structure for *that specific cause*.
    """
    cause_name = cause_name or CAUSE_NAMES.get(cause, f"cause {cause}")
    durations = np.asarray(durations, dtype=float)
    causes = np.asarray(causes, dtype=int)
    groups = np.asarray(groups)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, g in enumerate(np.unique(groups)):
        mask = groups == g
        times, cif = aalen_johansen(durations[mask], causes[mask], cause=cause)
        name = (group_names or {}).get(g, f"group {g}")
        n_ev = int((causes[mask] == cause).sum())
        ax.step(times, cif, where="post", linewidth=2,
                color=_GROUP_PALETTE[i % len(_GROUP_PALETTE)],
                label=f"{name} (n={int(mask.sum()):,}, events={n_ev:,})")

    ax.set_xlabel("Days since transplant")
    ax.set_ylabel(f"Cumulative incidence of {cause_name}")
    ax.set_title(title or f"{cause_name}: cumulative incidence by group")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    _save(fig, save_path)


def plot_head_training_dynamics(
    history: Dict,
    cause_names: Optional[Dict[int, str]] = None,
    title: str = "Competing-risks head: training dynamics",
    save_path: Optional[str] = None,
):
    """
    Training diagnostics for the survival head, per outcome.

    Left  : train vs. val negative log-likelihood. A val curve that turns up
            while train keeps falling is overfitting -- and marks the epoch the
            restored best checkpoint came from.
    Right : held-out C-index per cause across epochs. Loss and concordance can
            move independently: the likelihood is dominated by the (large)
            censored population, while C-index only depends on ranking the
            events, so a falling loss does not guarantee better discrimination.
    """
    cause_names = cause_names or CAUSE_NAMES

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    ax.plot(history["epoch"], history["train_loss"], color="#253E6B", linewidth=1.8, label="train")
    ax.plot(history["epoch"], history["val_loss"], color="#c0392b", linewidth=1.8,
            linestyle="--", label="val")
    if history["val_loss"]:
        best_i = int(np.argmin(history["val_loss"]))
        ax.axvline(history["epoch"][best_i], color="#377860", linestyle=":", linewidth=1.5,
                   label=f"best val (epoch {history['epoch'][best_i]})")
        ax.plot(history["epoch"][best_i], history["val_loss"][best_i], "o",
                color="#377860", markersize=6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative log-likelihood")
    ax.set_title("Discrete-time competing-risks loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1]
    plotted = False
    for c, name in cause_names.items():
        key = f"val_cindex_cause{c}"
        if key not in history or not history[key]:
            continue
        eps = [e for e, _ in history[key]]
        cis = [v for _, v in history[key]]
        ax.plot(eps, cis, marker="o", markersize=3, linewidth=1.8,
                color=_PALETTE.get(c, "#999999"), label=name)
        plotted = True
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=1, label="chance (0.5)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Held-out C-index")
    ax.set_ylim(0, 1)
    ax.set_title("Held-out discrimination per cause")
    if plotted:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)


def plot_cif_calibration(
    predicted_cif: np.ndarray,
    durations: np.ndarray,
    causes: np.ndarray,
    cause: int,
    n_bins: int = 10,
    cause_name: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Calibration for a competing-risks CIF prediction.

    Subjects are grouped by predicted risk; each group's *observed* incidence is
    the Aalen-Johansen CIF at the end of follow-up for that group (not a raw
    event rate -- that would be biased by censoring and by the competing event).
    """
    cause_name = cause_name or CAUSE_NAMES.get(cause, f"cause {cause}")
    predicted_cif = np.asarray(predicted_cif, dtype=float)
    durations = np.asarray(durations, dtype=float)
    causes = np.asarray(causes, dtype=int)

    qs = np.quantile(predicted_cif, np.linspace(0, 1, n_bins + 1)[1:-1])
    groups = np.digitize(predicted_cif, qs)

    mean_pred, observed = [], []
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() < 2:
            continue
        mean_pred.append(float(predicted_cif[mask].mean()))
        _, cif = aalen_johansen(durations[mask], causes[mask], cause=cause)
        observed.append(float(cif[-1]))

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot(mean_pred, observed, "o-", color=_PALETTE.get(cause, "#253E6B"))
    lim = max(max(mean_pred + observed + [0.01]) * 1.15, 0.05)
    ax.plot([0, lim], [0, lim], "--", color="gray", linewidth=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(f"Mean predicted CIF ({cause_name})")
    ax.set_ylabel(f"Observed incidence (Aalen-Johansen)")
    ax.set_title(f"CIF calibration: {cause_name}")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    _save(fig, save_path)


def plot_competing_risks_overview(
    durations: np.ndarray,
    causes: np.ndarray,
    report: Optional[dict] = None,
    save_path: Optional[str] = None,
):
    """Single figure: CIF per cause, stacked occupancy, and the event breakdown."""
    durations = np.asarray(durations, dtype=float)
    causes = np.asarray(causes, dtype=int)

    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 0.8])

    # CIF per cause
    ax = fig.add_subplot(gs[0, 0])
    for k, name in CAUSE_NAMES.items():
        times, cif = aalen_johansen(durations, causes, cause=k)
        ax.step(times, cif, where="post", color=_PALETTE[k], linewidth=2, label=name)
    ax.set_xlabel("Days since transplant")
    ax.set_ylabel("Cumulative incidence")
    ax.set_title("CIF per competing cause")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # Stacked occupancy
    ax = fig.add_subplot(gs[0, 1])
    all_times = np.unique(np.concatenate([[0.0], durations[np.isfinite(durations)]]))
    stacked, colors, labels = [], [], []
    for k, name in CAUSE_NAMES.items():
        t_k, cif_k = aalen_johansen(durations, causes, cause=k)
        stacked.append(np.interp(all_times, t_k, cif_k))
        colors.append(_PALETTE[k]); labels.append(name)
    ax.stackplot(all_times, 1.0 - np.sum(stacked, axis=0), *stacked,
                 labels=["event-free"] + labels, colors=["#dbe2f2"] + colors, alpha=0.9)
    ax.set_xlim(0, all_times.max()); ax.set_ylim(0, 1)
    ax.set_xlabel("Days since transplant")
    ax.set_ylabel("Fraction of cohort")
    ax.set_title("State occupancy (sums to 1)")
    ax.legend(fontsize=7, loc="lower left")

    # Breakdown
    ax = fig.add_subplot(gs[0, 2])
    counts = [int((causes == CAUSE_GRAFT_LOSS).sum()),
              int((causes == CAUSE_DEATH).sum()),
              int((causes == CAUSE_CENSORED).sum())]
    names = ["graft loss", "death", "censored"]
    cols = [_PALETTE[CAUSE_GRAFT_LOSS], _PALETTE[CAUSE_DEATH], "#c9d3ec"]
    ax.barh(names, counts, color=cols)
    for i, v in enumerate(counts):
        ax.text(v, i, f"  {v:,}\n  ({100*v/len(causes):.1f}%)", va="center", fontsize=8)
    ax.set_xlim(0, max(counts) * 1.4)
    ax.set_title("First-event breakdown")
    ax.grid(alpha=0.2, axis="x")

    plt.suptitle("Competing risks: graft loss vs. death", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)
