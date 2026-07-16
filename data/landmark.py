"""
Landmark filtering for time-to-event modelling.

The problem this solves
-----------------------
If a longitudinal table's observations run right up to a subject's event, a
model doesn't have to *predict* anything -- it can read the event off inputs
that already show it happening (e.g. a creatinine measurement taken days
before graft failure, showing function already collapsing). Downstream metrics
then look excellent while the model has no real forecasting ability.

Landmarking removes this structurally:
  1. Pick a landmark time L (e.g. 365 days post-transplant).
  2. Keep only subjects still event-free and under observation at L.
  3. Truncate every longitudinal input to observations at or before L.
  4. Predict the event occurring *after* L, with durations re-based to L.

The model then only ever sees information genuinely available at the landmark,
so downstream performance reflects forecasting rather than detection.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def apply_landmark(
    static_df: pd.DataFrame,
    temporal_tables: Dict[str, Tuple[pd.DataFrame, str]],
    landmark_day: float,
    duration_col: str,
    event_col: str,
    id_col: str = "natt1",
    rebase_durations: bool = True,
    drop_observations_after_event: bool = True,
    additional_duration_cols: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], dict]:
    """
    Apply a landmark cutoff to a static table plus its longitudinal tables.

    Parameters
    ----------
    static_df : one row per subject, containing `id_col`, `duration_col`,
        `event_col`.
    temporal_tables : {name: (dataframe, timestamp_col)} -- long-format tables
        with many rows per subject.
    landmark_day : the landmark time L, in the same units as the duration and
        timestamp columns (days here).
    duration_col / event_col : the time-to-event pair (e.g. GraftSurvivalDays /
        FailureWithinStudyPeriod). `event_col` is the censoring mask: True =
        event observed at `duration_col`, False = censored there.
    rebase_durations : if True, durations of the retained cohort are shifted to
        be measured *from the landmark* rather than from time zero, which is
        what a landmark model should predict.
    drop_observations_after_event : drop longitudinal rows timestamped after
        the subject's recorded event/censoring time. These are data
        inconsistencies (an observation of a graft after it failed) and are
        dropped rather than silently kept.
    additional_duration_cols : OTHER duration columns on the same time axis
        (e.g. PatientSurvivalDays when landmarking on GraftSurvivalDays). These
        are rebased alongside `duration_col`, and a subject must be at risk on
        ALL of them at the landmark to be retained.

        This matters for competing risks: rebasing only `duration_col` leaves
        the other duration still measured from time zero, so it is inflated by
        `landmark_day` relative to the one that moved. Any downstream
        min(t_graft, t_death) then always picks the rebased column, and events
        recorded in the untouched one are silently reclassified as censored --
        which quietly empties an entire competing cause.

    Returns
    -------
    (static_filtered, {name: temporal_filtered}, report)
    """
    report = {"landmark_day": landmark_day, "n_before": len(static_df)}

    if duration_col not in static_df.columns or event_col not in static_df.columns:
        raise ValueError(
            f"static_df must contain '{duration_col}' and '{event_col}'; "
            f"got columns: {list(static_df.columns)}"
        )

    additional_duration_cols = [
        c for c in (additional_duration_cols or []) if c in static_df.columns and c != duration_col
    ]
    rebase_cols = [duration_col] + additional_duration_cols
    report["rebased_columns"] = rebase_cols

    static = static_df.copy()
    durations = static[duration_col].to_numpy(dtype=float)
    events = static[event_col].to_numpy()
    events = np.asarray(events).astype(bool)

    # 1. Cohort: only subjects still under observation and event-free at the
    #    landmark, on EVERY duration column -- a subject who died before the
    #    landmark is not at risk for graft loss after it, even if the graft
    #    column alone suggests otherwise.
    at_risk_at_landmark = durations > landmark_day
    for c in additional_duration_cols:
        at_risk_at_landmark &= static[c].to_numpy(dtype=float) > landmark_day
    n_dropped_early = int((~at_risk_at_landmark).sum())
    n_dropped_early_events = int((~at_risk_at_landmark & events).sum())

    static = static[at_risk_at_landmark].copy()
    eligible_ids = set(static[id_col].tolist())

    report["n_dropped_not_at_risk_at_landmark"] = n_dropped_early
    report["n_dropped_events_before_landmark"] = n_dropped_early_events
    report["n_after"] = len(static)

    if len(static) == 0:
        raise ValueError(
            f"Landmark day {landmark_day} leaves no subjects at risk "
            f"(max observed duration is {durations.max():.0f}). Choose a smaller landmark."
        )

    # 2. Truncate longitudinal tables to <= landmark, restricted to the cohort
    temporal_out = {}
    for name, (df, ts_col) in temporal_tables.items():
        before_rows = len(df)
        sub = df[df[id_col].isin(eligible_ids)].copy()

        if drop_observations_after_event:
            # Drop rows recorded after the subject's own event/censoring time
            dur_map = static.set_index(id_col)[duration_col]
            subj_duration = sub[id_col].map(dur_map)
            inconsistent = sub[ts_col].to_numpy(dtype=float) > subj_duration.to_numpy(dtype=float)
            n_inconsistent = int(np.nansum(inconsistent))
            sub = sub[~inconsistent].copy()
            report[f"{name}_rows_after_event_dropped"] = n_inconsistent

        sub = sub[sub[ts_col] <= landmark_day].copy()
        temporal_out[name] = sub
        report[f"{name}_rows_before"] = before_rows
        report[f"{name}_rows_after"] = len(sub)
        report[f"{name}_subjects_with_data"] = int(sub[id_col].nunique())

    # 3. Re-base every duration to measure time from the landmark forward.
    #    All of them, not just duration_col -- see additional_duration_cols.
    if rebase_durations:
        for c in rebase_cols:
            static[c] = static[c] - landmark_day
        report["durations_rebased_to_landmark"] = True

    n_events_after = int(np.asarray(static[event_col].to_numpy()).astype(bool).sum())
    report["n_events_after_landmark"] = n_events_after
    report["event_rate_after_landmark"] = (
        n_events_after / len(static) if len(static) else float("nan")
    )

    if verbose:
        print(f"  Landmark day {landmark_day:.0f}:")
        print(f"    cohort: {report['n_before']:,} -> {report['n_after']:,} subjects "
              f"({n_dropped_early:,} not at risk at landmark, of which "
              f"{n_dropped_early_events:,} had already had the event)")
        for name in temporal_tables:
            dropped_key = f"{name}_rows_after_event_dropped"
            extra = ""
            if report.get(dropped_key):
                extra = f", {report[dropped_key]:,} rows dropped as recorded after event"
            print(f"    {name}: {report[f'{name}_rows_before']:,} -> "
                  f"{report[f'{name}_rows_after']:,} rows{extra}")
        print(f"    post-landmark event rate: {report['event_rate_after_landmark']:.3f} "
              f"({n_events_after:,} events)")
        print(f"    rebased duration columns: {', '.join(rebase_cols)}")

    return static, temporal_out, report


def suggest_landmark_days(
    static_df: pd.DataFrame,
    duration_col: str = "GraftSurvivalDays",
    event_col: str = "FailureWithinStudyPeriod",
    candidates: Optional[list] = None,
) -> pd.DataFrame:
    """
    Summarise the cohort/event trade-off across candidate landmark times.

    Later landmarks let the model see more history but retain fewer subjects
    and fewer post-landmark events -- this table makes that trade-off explicit
    rather than leaving the choice arbitrary.
    """
    candidates = candidates or [30, 90, 180, 365, 730, 1095]
    durations = static_df[duration_col].to_numpy(dtype=float)
    events = np.asarray(static_df[event_col].to_numpy()).astype(bool)

    rows = []
    for L in candidates:
        at_risk = durations > L
        n_at_risk = int(at_risk.sum())
        n_events_after = int((at_risk & events).sum())
        rows.append({
            "landmark_day": L,
            "n_at_risk": n_at_risk,
            "pct_cohort_retained": 100 * n_at_risk / len(durations) if len(durations) else 0,
            "n_events_after": n_events_after,
            "event_rate_after": n_events_after / n_at_risk if n_at_risk else float("nan"),
        })
    return pd.DataFrame(rows)
