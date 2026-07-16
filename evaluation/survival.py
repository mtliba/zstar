"""
Survival-aware evaluation and visualization.

The REXETRIS labels come in time-to-event / censoring-indicator pairs:

    GraftSurvivalDays   + FailureWithinStudyPeriod   (graft outcome)
    PatientSurvivalDays + DeathWithinStudyPeriod     (patient outcome)

The `*Days` column is the observed duration; the `*WithinStudyPeriod` flag is
the event mask -- True means the event was observed at that duration, False
means the observation was *censored* at that duration (the graft/patient was
still event-free when observation stopped, and what happened afterwards is
unknown).

This distinction is why plain binary classification on the mask is wrong: it
treats "censored at day 90" and "event-free through day 4000" as the same
negative label. The estimators here (Kaplan-Meier, concordance index) handle
censoring correctly instead.

Kaplan-Meier and the concordance index are implemented directly on numpy
rather than via lifelines/scikit-survival, to avoid adding a dependency that
would need to clear a package mirror before this module could be used.
"""

import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PALETTE = ["#253E6B", "#377860", "#A56327", "#733E85", "#7c9fff", "#4fd1c5"]


def _save(fig, save_path: Optional[str]):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ── Estimators ─────────────────────────────────────────────────────────────

def kaplan_meier(
    durations: np.ndarray, events: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Kaplan-Meier product-limit estimator.

    durations : observed time to event or censoring, per subject
    events    : True = event observed at `durations`, False = censored there

    Returns (times, survival_probability, n_at_risk), each aligned and starting
    at t=0 with S(0)=1. Censored subjects correctly leave the risk set without
    causing a drop in the survival curve.
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events).astype(bool)

    finite = np.isfinite(durations)
    durations, events = durations[finite], events[finite]
    n = len(durations)
    if n == 0:
        return np.array([0.0]), np.array([1.0]), np.array([0])

    unique_times, inverse = np.unique(durations, return_inverse=True)
    n_at_each_time = np.bincount(inverse, minlength=len(unique_times))
    n_events_at_time = np.bincount(
        inverse, weights=events.astype(float), minlength=len(unique_times)
    )

    # At risk at time t = everyone whose duration >= t
    cum_before = np.concatenate([[0], np.cumsum(n_at_each_time)[:-1]])
    at_risk = n - cum_before

    with np.errstate(divide="ignore", invalid="ignore"):
        factors = 1.0 - n_events_at_time / at_risk
    factors = np.where(at_risk > 0, factors, 1.0)
    survival = np.cumprod(factors)

    times = np.concatenate([[0.0], unique_times])
    survival = np.concatenate([[1.0], survival])
    at_risk = np.concatenate([[n], at_risk])
    return times, survival, at_risk


def median_survival_time(times: np.ndarray, survival: np.ndarray) -> float:
    """First time at which the KM curve drops to or below 0.5; NaN if it never does."""
    below = np.where(survival <= 0.5)[0]
    if len(below) == 0:
        return float("nan")
    return float(times[below[0]])


def concordance_index(
    durations: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
    chunk_size: int = 256,
) -> float:
    """
    Harrell's concordance index (C-index) -- the survival analogue of AUROC.

    Measures, over all comparable pairs, how often the subject who failed
    earlier was assigned the higher risk score. A pair is comparable only if
    the earlier subject's event was actually observed (not censored), which is
    what makes this valid under censoring where AUROC is not.

    0.5 = no better than chance, 1.0 = perfect risk ordering.

    Computed exactly (not sampled), chunked over the event subjects to keep
    memory bounded on large cohorts.
    """
    d = np.asarray(durations, dtype=float)
    e = np.asarray(events).astype(bool)
    r = np.asarray(risk_scores, dtype=float)

    valid = np.isfinite(d) & np.isfinite(r)
    d, e, r = d[valid], e[valid], r[valid]

    event_idx = np.where(e)[0]
    if len(event_idx) == 0:
        return float("nan")

    concordant = 0.0
    tied = 0.0
    comparable_total = 0.0

    for start in range(0, len(event_idx), chunk_size):
        idx = event_idx[start:start + chunk_size]
        t_i = d[idx][:, None]
        r_i = r[idx][:, None]

        # Comparable: subject i had an observed event strictly before subject j
        comparable = d[None, :] > t_i
        r_j = r[None, :]

        concordant += np.sum((r_i > r_j) & comparable)
        tied += np.sum((r_i == r_j) & comparable)
        comparable_total += np.sum(comparable)

    if comparable_total == 0:
        return float("nan")
    return float((concordant + 0.5 * tied) / comparable_total)


# ── Plots ──────────────────────────────────────────────────────────────────

def plot_kaplan_meier(
    durations: np.ndarray,
    events: np.ndarray,
    groups: Optional[np.ndarray] = None,
    group_names: Optional[Dict] = None,
    title: str = "Kaplan-Meier survival",
    xlabel: str = "Days since transplant",
    show_at_risk: bool = True,
    save_path: Optional[str] = None,
):
    """
    Kaplan-Meier curve, optionally stratified by `groups`.

    Censoring is handled correctly: censored subjects leave the risk set
    without producing a step down. Tick marks show censoring events.
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events).astype(bool)

    if show_at_risk:
        fig, (ax, ax_risk) = plt.subplots(
            2, 1, figsize=(9, 6.4), gridspec_kw={"height_ratios": [4, 1]}, sharex=True
        )
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax_risk = None

    if groups is None:
        strata = [(None, np.ones(len(durations), dtype=bool))]
    else:
        groups = np.asarray(groups)
        strata = [(g, groups == g) for g in np.unique(groups)]

    risk_table_rows = []
    for i, (g, mask) in enumerate(strata):
        d_g, e_g = durations[mask], events[mask]
        if len(d_g) == 0:
            continue
        times, surv, at_risk = kaplan_meier(d_g, e_g)
        color = _PALETTE[i % len(_PALETTE)]

        label = None
        if g is not None:
            label = (group_names or {}).get(g, f"group {g}")
            med = median_survival_time(times, surv)
            n_ev = int(e_g.sum())
            med_str = f"{med:.0f}d" if np.isfinite(med) else "not reached"
            label = f"{label} (n={len(d_g)}, events={n_ev}, median={med_str})"

        ax.step(times, surv, where="post", color=color, linewidth=2, label=label)

        # Censoring tick marks
        cens_times = d_g[~e_g]
        if len(cens_times) > 0:
            # Subsample ticks on large cohorts so the plot stays readable
            if len(cens_times) > 300:
                cens_times = np.random.default_rng(0).choice(cens_times, 300, replace=False)
            cens_surv = np.interp(cens_times, times, surv)
            ax.plot(cens_times, cens_surv, "|", color=color, markersize=5, alpha=0.5)

        risk_table_rows.append((label or "all", times, at_risk, color))

    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    if groups is not None:
        ax.legend(fontsize=8, loc="lower left")

    if ax_risk is not None:
        xticks = np.linspace(0, durations.max(), 6)
        ax_risk.set_xlim(ax.get_xlim())
        ax_risk.set_yticks(range(len(risk_table_rows)))
        ax_risk.set_yticklabels(
            [r[0].split(" (")[0] if r[0] else "all" for r in risk_table_rows], fontsize=7
        )
        for row_i, (_, times, at_risk, color) in enumerate(risk_table_rows):
            for xt in xticks:
                idx = np.searchsorted(times, xt, side="right") - 1
                idx = np.clip(idx, 0, len(at_risk) - 1)
                ax_risk.text(
                    xt, row_i, str(int(at_risk[idx])),
                    ha="center", va="center", fontsize=7, color=color,
                )
        ax_risk.set_ylim(-0.5, len(risk_table_rows) - 0.5)
        ax_risk.set_xlabel(xlabel)
        ax_risk.set_title("Number at risk", fontsize=8, loc="left")
        ax_risk.grid(False)
        for spine in ax_risk.spines.values():
            spine.set_visible(False)
        ax_risk.tick_params(axis="x", length=0)
    else:
        ax.set_xlabel(xlabel)

    plt.tight_layout()
    _save(fig, save_path)


def plot_survival_overview(
    labels_df,
    outcomes: Sequence[Tuple[str, str, str]] = (
        ("GraftSurvivalDays", "FailureWithinStudyPeriod", "Graft survival"),
        ("PatientSurvivalDays", "DeathWithinStudyPeriod", "Patient survival"),
    ),
    save_path: Optional[str] = None,
):
    """
    Side-by-side overview of every time-to-event outcome: KM curve, plus the
    event/censoring breakdown that the curve is estimated from.
    """
    n_outcomes = len(outcomes)
    fig, axes = plt.subplots(2, n_outcomes, figsize=(6 * n_outcomes, 8),
                             gridspec_kw={"height_ratios": [3, 1.5]}, squeeze=False)

    for col, (dur_col, event_col, title) in enumerate(outcomes):
        if dur_col not in labels_df.columns or event_col not in labels_df.columns:
            axes[0][col].axis("off")
            axes[1][col].axis("off")
            continue

        d = labels_df[dur_col].to_numpy(dtype=float)
        e = labels_df[event_col].to_numpy()
        e = np.asarray(e).astype(bool)

        times, surv, _ = kaplan_meier(d, e)
        med = median_survival_time(times, surv)

        ax = axes[0][col]
        ax.step(times, surv, where="post", color=_PALETTE[col], linewidth=2)
        if np.isfinite(med):
            ax.axhline(0.5, linestyle=":", color="gray", linewidth=1)
            ax.axvline(med, linestyle=":", color="gray", linewidth=1)
            ax.text(med, 0.52, f" median {med:.0f}d", fontsize=8, color="gray")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Survival probability")
        ax.set_xlabel("Days since transplant")
        ax.set_title(f"{title}\n(Kaplan-Meier, censoring-aware)")
        ax.grid(alpha=0.2)

        # Event / censoring breakdown -- what the KM curve is actually built from
        ax2 = axes[1][col]
        n_event = int(e.sum())
        n_cens = int((~e).sum())
        ax2.barh(["event\nobserved", "censored"], [n_event, n_cens],
                 color=[_PALETTE[col], "#c9d3ec"])
        for i, v in enumerate([n_event, n_cens]):
            pct = 100 * v / len(e) if len(e) else 0
            ax2.text(v, i, f"  {v:,} ({pct:.1f}%)", va="center", fontsize=9)
        ax2.set_xlim(0, max(n_event, n_cens) * 1.35)
        ax2.set_title(f"{event_col} breakdown", fontsize=9)
        ax2.grid(alpha=0.2, axis="x")

    plt.suptitle("Survival outcomes overview", y=1.00, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)


def plot_followup_distribution(
    labels_df,
    outcomes: Sequence[Tuple[str, str, str]] = (
        ("GraftSurvivalDays", "FailureWithinStudyPeriod", "Graft"),
        ("PatientSurvivalDays", "DeathWithinStudyPeriod", "Patient"),
    ),
    save_path: Optional[str] = None,
):
    """Observed-duration distributions, split by event vs. censored."""
    fig, axes = plt.subplots(1, len(outcomes), figsize=(6 * len(outcomes), 4.2), squeeze=False)

    for col, (dur_col, event_col, title) in enumerate(outcomes):
        ax = axes[0][col]
        if dur_col not in labels_df.columns or event_col not in labels_df.columns:
            ax.axis("off")
            continue
        d = labels_df[dur_col].to_numpy(dtype=float)
        e = np.asarray(labels_df[event_col].to_numpy()).astype(bool)

        bins = np.linspace(0, np.nanmax(d), 40)
        ax.hist(d[e], bins=bins, alpha=0.75, label=f"event (n={int(e.sum()):,})",
                color=_PALETTE[col])
        ax.hist(d[~e], bins=bins, alpha=0.55, label=f"censored (n={int((~e).sum()):,})",
                color="#c9d3ec")
        ax.set_xlabel("Observed duration (days)")
        ax.set_ylabel("Count")
        ax.set_title(f"{title}: follow-up distribution")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    _save(fig, save_path)


def plot_km_by_zstar_risk(
    zstar_embeddings: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    n_groups: int = 4,
    method: str = "kmeans",
    risk_scores: Optional[np.ndarray] = None,
    title: str = "Graft survival stratified by z-star",
    save_path: Optional[str] = None,
):
    """
    Kaplan-Meier curves stratified by z-star -- the plot that actually shows
    whether the learned representation separates real risk groups.

    method="kmeans"   : cluster z-star into n_groups unsupervised clusters.
    method="quantile" : split by `risk_scores` quantiles (supply a fitted
                        model's predicted risk).

    Well-separated, correctly-ordered curves are evidence z-star carries
    outcome-relevant structure; overlapping curves are evidence it does not.
    """
    if method == "kmeans":
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
        groups = km.fit_predict(zstar_embeddings)
        group_names = {g: f"cluster {g}" for g in np.unique(groups)}
    elif method == "quantile":
        if risk_scores is None:
            raise ValueError("method='quantile' requires risk_scores")
        qs = np.quantile(risk_scores, np.linspace(0, 1, n_groups + 1)[1:-1])
        groups = np.digitize(risk_scores, qs)
        labels = ["lowest risk", "low", "high", "highest risk"]
        group_names = {
            g: (labels[g] if n_groups == 4 and g < len(labels) else f"risk Q{g + 1}")
            for g in np.unique(groups)
        }
    else:
        raise ValueError(f"Unknown method '{method}'. Choose: kmeans | quantile")

    plot_kaplan_meier(
        durations, events, groups=groups, group_names=group_names,
        title=title, save_path=save_path,
    )
    return groups


def plot_landmark_diagnostic(
    temporal_df,
    static_df,
    landmark_day: Optional[float] = None,
    id_col: str = "natt1",
    timestamp_col: str = "days_since_tx",
    duration_col: str = "GraftSurvivalDays",
    event_col: str = "FailureWithinStudyPeriod",
    save_path: Optional[str] = None,
):
    """
    Leakage diagnostic: how close does each subject's LAST observation sit to
    their event?

    If observations run right up to the event, a model can "predict" the event
    by simply observing it already in progress -- inflating apparent
    performance without any real forecasting ability. Mass near zero (or
    negative) on this plot is the warning sign; a landmark cutoff is the fix.

    Negative values mean observations recorded *after* the event time, which
    is a data-consistency problem in its own right.
    """
    merged = temporal_df.merge(
        static_df[[id_col, duration_col, event_col]], on=id_col, how="inner"
    )
    events_only = merged[merged[event_col].astype(bool)]
    if len(events_only) == 0:
        print("[plot_landmark_diagnostic] No observed events; skipping.")
        return

    last_obs = events_only.groupby(id_col)[timestamp_col].max()
    duration = events_only.groupby(id_col)[duration_col].first()
    gap = (duration - last_obs).to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    finite_gap = gap[np.isfinite(gap)]
    ax.hist(finite_gap, bins=60, color=_PALETTE[0], alpha=0.85)
    ax.axvline(0, color="#c0392b", linestyle="--", linewidth=1.5,
               label="observation at event time")
    if landmark_day is not None:
        ax.axvline(landmark_day, color=_PALETTE[2], linestyle="-", linewidth=1.5,
                   label=f"landmark gap ({landmark_day:.0f}d)")
    ax.set_xlabel(f"{duration_col} − last observation (days)")
    ax.set_ylabel("Subjects with observed event")
    ax.set_title("Gap between last observation and event")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1]
    n_neg = int((finite_gap < 0).sum())
    n_close = int(((finite_gap >= 0) & (finite_gap <= 90)).sum())
    n_far = int((finite_gap > 90).sum())
    bars = ["after event\n(inconsistent)", "within 90d\n(leak risk)", ">90d before\n(usable)"]
    vals = [n_neg, n_close, n_far]
    colors = ["#c0392b", "#A56327", _PALETTE[1]]
    ax.bar(bars, vals, color=colors)
    for i, v in enumerate(vals):
        pct = 100 * v / len(finite_gap) if len(finite_gap) else 0
        ax.text(i, v, f"{v:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Subjects with observed event")
    ax.set_ylim(0, max(vals) * 1.25 if max(vals) else 1)
    ax.set_title("Leakage exposure breakdown")
    ax.grid(alpha=0.2, axis="y")

    plt.suptitle("Landmark / leakage diagnostic", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)

    return {
        "n_events": len(finite_gap),
        "n_observation_after_event": n_neg,
        "n_within_90d": n_close,
        "median_gap_days": float(np.median(finite_gap)),
    }
