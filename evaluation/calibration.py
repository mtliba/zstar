"""
Horizon-based calibration and recalibration for competing-risks CIF predictions.

Why evaluate at a fixed horizon
-------------------------------
Comparing predicted CIF against the Aalen-Johansen CIF *at the end of
follow-up* is unreliable: in the far tail almost nobody is still at risk, so
the AJ estimate there is unstable and drifts upward. Any apparent
miscalibration is then partly an artifact of the estimator rather than the
model. Evaluating at a fixed, clinically meaningful horizon (1 year, 5 years)
where a substantial number of subjects remain at risk gives a trustworthy
reference. `n_at_risk` is reported alongside so a thin horizon is visible
rather than silently believed.

Why recalibration needs IPCW
----------------------------
Isotonic regression needs a per-subject binary outcome, but censoring means
you do not have one for everybody. At horizon tau, a subject's cause-k status
is:

    T <= tau, cause == k        -> 1   (had the event)
    T <= tau, cause == other    -> 0   (competing event; can NEVER have cause k)
    T >  tau                    -> 0   (event-free at tau)
    T <= tau, censored          -> UNKNOWN

Simply dropping the unknowns biases the fit, because subjects censored early
are not a random subset. Instead each usable subject is weighted by the
inverse probability of remaining uncensored to its relevant time (IPCW),
estimated by a reverse Kaplan-Meier on the censoring distribution. Unknowns
get weight 0; the rest are up-weighted to stand in for those they resemble.

Note the competing-event row: a subject who dies without graft loss is scored
0 for graft loss *forever*, not censored. That is the subdistribution
convention and is what makes the target consistent with a CIF.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .competing_risks import aalen_johansen, CAUSE_NAMES, _PALETTE, _save


def cif_at_horizon(
    cif_matrix: np.ndarray, cuts: np.ndarray, horizon: float, cause: int
) -> np.ndarray:
    """
    Predicted CIF for `cause` at `horizon`, from the discrete-time CIF.

    cif_matrix : [N, K, C] as returned by predict_cif / train_competing_risks_head
    cuts       : bin edges from make_time_bins
    cause      : 1-based cause id
    """
    cif_matrix = np.asarray(cif_matrix)
    bin_idx = int(np.digitize([horizon], cuts)[0])
    bin_idx = min(bin_idx, cif_matrix.shape[1] - 1)
    return cif_matrix[:, bin_idx, cause - 1]


def censoring_survival(durations: np.ndarray, causes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reverse Kaplan-Meier: G(t) = P(censoring time > t).

    Fitted by treating *censoring* as the event of interest and any real event
    as censoring -- the mirror image of the usual estimator.
    """
    from .survival import kaplan_meier
    return kaplan_meier(durations, np.asarray(causes) == 0)[:2]


def ipcw_binary_outcome(
    durations: np.ndarray,
    causes: np.ndarray,
    cause: int,
    horizon: float,
    eps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Binary cause-k status at `horizon`, with IPCW weights.

    Returns (y, w). Subjects whose status at `horizon` is unknown (censored
    before it) receive w = 0 and are effectively excluded; everyone else is
    weighted by 1 / G(.) to compensate for them.
    """
    d = np.asarray(durations, dtype=float)
    c = np.asarray(causes, dtype=int)

    g_times, g_surv = censoring_survival(d, c)

    def G(t):
        return np.clip(np.interp(t, g_times, g_surv), eps, None)

    y = np.zeros(len(d), dtype=float)
    w = np.zeros(len(d), dtype=float)

    had_event_by_h = (d <= horizon) & (c > 0)
    survived_past_h = d > horizon
    # censored before the horizon -> status unknown -> weight stays 0

    y[had_event_by_h & (c == cause)] = 1.0
    w[had_event_by_h] = 1.0 / G(d[had_event_by_h])
    w[survived_past_h] = 1.0 / G(horizon)

    return y, w


class CIFRecalibrator:
    """
    Isotonic recalibration of CIF predictions at a fixed horizon.

    Isotonic is monotone by construction, so it can only rescale the predicted
    probabilities -- it cannot reorder subjects. Discrimination (C-index) is
    therefore unchanged; only calibration moves. That is exactly what is wanted
    when ranking is already good and the absolute level is off.

    Fit on the TRAINING split and apply to the held-out split. Fitting and
    evaluating on the same subjects would report the recalibrator's own fit
    back to you.
    """

    def __init__(self, cause: int, horizon: float):
        self.cause = cause
        self.horizon = horizon
        self.iso = None

    def fit(self, predicted: np.ndarray, durations: np.ndarray, causes: np.ndarray):
        from sklearn.isotonic import IsotonicRegression

        y, w = ipcw_binary_outcome(durations, causes, self.cause, self.horizon)
        usable = w > 0
        if usable.sum() < 10:
            raise ValueError(
                f"Only {int(usable.sum())} subjects have known cause-{self.cause} status at "
                f"horizon {self.horizon}; too few to recalibrate. Use an earlier horizon."
            )
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.iso.fit(np.asarray(predicted)[usable], y[usable], sample_weight=w[usable])
        self.n_fit = int(usable.sum())
        return self

    def transform(self, predicted: np.ndarray) -> np.ndarray:
        if self.iso is None:
            raise RuntimeError("CIFRecalibrator must be fit before transform")
        return self.iso.predict(np.asarray(predicted))


def calibration_curve_at_horizon(
    predicted: np.ndarray,
    durations: np.ndarray,
    causes: np.ndarray,
    cause: int,
    horizon: float,
    n_groups: int = 10,
) -> Dict:
    """
    Observed vs. predicted incidence at `horizon`, by predicted-risk group.

    Observed incidence is the Aalen-Johansen CIF evaluated at `horizon` within
    each group -- censoring-aware, and (unlike an end-of-follow-up read) taken
    where subjects are still at risk.
    """
    predicted = np.asarray(predicted, dtype=float)
    durations = np.asarray(durations, dtype=float)
    causes = np.asarray(causes, dtype=int)

    qs = np.quantile(predicted, np.linspace(0, 1, n_groups + 1)[1:-1])
    groups = np.digitize(predicted, qs)

    mean_pred, observed, n_at_risk, n_group = [], [], [], []
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() < 2:
            continue
        times, cif = aalen_johansen(durations[mask], causes[mask], cause=cause)
        mean_pred.append(float(predicted[mask].mean()))
        observed.append(float(np.interp(horizon, times, cif)))
        n_at_risk.append(int((durations[mask] >= horizon).sum()))
        n_group.append(int(mask.sum()))

    mean_pred = np.array(mean_pred)
    observed = np.array(observed)
    ici = float(np.mean(np.abs(observed - mean_pred))) if len(observed) else float("nan")

    return {
        "mean_predicted": mean_pred,
        "observed": observed,
        "n_at_risk": np.array(n_at_risk),
        "n_group": np.array(n_group),
        "ici": ici,   # integrated calibration index: mean |observed - predicted|
        "horizon": horizon,
    }


def plot_cif_calibration_at_horizons(
    cif_matrix: np.ndarray,
    cuts: np.ndarray,
    durations: np.ndarray,
    causes: np.ndarray,
    cause: int,
    horizons: Sequence[float] = (365, 1825),
    horizon_labels: Optional[Sequence[str]] = None,
    recalibrated: Optional[Dict[float, np.ndarray]] = None,
    cause_name: Optional[str] = None,
    n_groups: int = 10,
    save_path: Optional[str] = None,
):
    """
    Calibration at fixed horizons, one panel each.

    `recalibrated` optionally maps horizon -> recalibrated predictions, drawn as
    a second series so the before/after is directly comparable.

    Each panel reports ICI (mean |observed - predicted|) and the number still at
    risk at that horizon. A horizon with few at risk should not be trusted no
    matter how the curve looks.
    """
    cause_name = cause_name or CAUSE_NAMES.get(cause, f"cause {cause}")
    horizon_labels = horizon_labels or [f"{h:.0f} days" for h in horizons]
    color = _PALETTE.get(cause, "#253E6B")

    fig, axes = plt.subplots(1, len(horizons), figsize=(5.2 * len(horizons), 5.2), squeeze=False)

    for i, (h, hl) in enumerate(zip(horizons, horizon_labels)):
        ax = axes[0][i]
        pred_h = cif_at_horizon(cif_matrix, cuts, h, cause)
        cc = calibration_curve_at_horizon(pred_h, durations, causes, cause, h, n_groups)

        if len(cc["mean_predicted"]) == 0:
            ax.axis("off")
            continue

        ax.plot(cc["mean_predicted"], cc["observed"], "o-", color=color,
                label=f"raw (ICI={cc['ici']:.3f})")

        if recalibrated is not None and h in recalibrated:
            cc2 = calibration_curve_at_horizon(
                recalibrated[h], durations, causes, cause, h, n_groups
            )
            ax.plot(cc2["mean_predicted"], cc2["observed"], "s--", color="#377860",
                    label=f"recalibrated (ICI={cc2['ici']:.3f})")

        vals = list(cc["mean_predicted"]) + list(cc["observed"])
        lim = max(max(vals) * 1.2, 0.02)
        ax.plot([0, lim], [0, lim], "--", color="gray", linewidth=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel(f"Mean predicted CIF")
        ax.set_ylabel("Observed incidence (Aalen-Johansen)")
        min_at_risk = int(cc["n_at_risk"].min()) if len(cc["n_at_risk"]) else 0
        ax.set_title(f"{hl}\n(min {min_at_risk:,} still at risk per group)", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.2)

        if min_at_risk < 20:
            ax.text(0.5, 0.02, "few at risk -- unreliable", transform=ax.transAxes,
                    ha="center", fontsize=8, color="#c0392b", style="italic")

    plt.suptitle(f"CIF calibration at fixed horizons: {cause_name}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)


def recalibrate_and_report(
    cif_matrix: np.ndarray,
    cuts: np.ndarray,
    durations: np.ndarray,
    causes: np.ndarray,
    cause: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    horizons: Sequence[float] = (365, 1825),
    verbose: bool = True,
) -> Dict:
    """
    Fit isotonic recalibration on the training split, evaluate on the held-out
    split, and report ICI before and after at each horizon.

    Returns {horizon: {"recalibrated_val": array, "ici_before": .., "ici_after": ..}}.
    """
    out = {}
    for h in horizons:
        pred_all = cif_at_horizon(cif_matrix, cuts, h, cause)
        try:
            rc = CIFRecalibrator(cause=cause, horizon=h).fit(
                pred_all[train_idx], durations[train_idx], causes[train_idx]
            )
        except ValueError as e:
            if verbose:
                print(f"    horizon {h:.0f}d: skipped -- {e}")
            continue

        recal_val = rc.transform(pred_all[val_idx])
        before = calibration_curve_at_horizon(
            pred_all[val_idx], durations[val_idx], causes[val_idx], cause, h
        )
        after = calibration_curve_at_horizon(
            recal_val, durations[val_idx], causes[val_idx], cause, h
        )
        out[h] = {
            "recalibrated_val": recal_val,
            "ici_before": before["ici"],
            "ici_after": after["ici"],
            "n_fit": rc.n_fit,
        }
        if verbose:
            print(f"    horizon {h:>5.0f}d: ICI {before['ici']:.4f} -> {after['ici']:.4f} "
                  f"(fit on {rc.n_fit:,} subjects with known status)")
    return out
