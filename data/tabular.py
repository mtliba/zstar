"""
Utilities for turning raw pandas tables into a ZStarDataset-ready data_dict,
joined and aligned by a shared patient/graft id column.

This is the piece a real dataset needs that synthetic-data smoke tests never
exercise: multiple tables, joined by id, with mixed static/temporal/event
types, boolean columns, and label columns that must be excluded from the
model's input features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def encode_bool_columns(df: pd.DataFrame, bool_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convert boolean-like columns to float32 (0.0/1.0), preserving NaN.
    Any column already of pandas/numpy bool dtype is converted automatically;
    `bool_cols` covers columns stored as object/nullable-boolean dtype.
    """
    df = df.copy()
    for c in (bool_cols or []):
        if c in df.columns:
            df[c] = df[c].astype("float32")
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype("float32")
    return df


def build_from_tables(
    tables: Dict[str, dict],
    id_col: str = "natt1",
    reference_ids: Optional[np.ndarray] = None,
) -> Tuple[dict, np.ndarray, Dict[str, List[str]]]:
    """
    Join and align multiple raw tables into a ZStarDataset-ready data_dict.

    Parameters
    ----------
    tables : {modality_name: spec}, where spec is:
        {
            "df": pd.DataFrame,
            "type": "static" | "temporal" | "event",
            "timestamp_col": str,            # required for temporal/event
            "bool_cols": [str, ...],         # optional
            "exclude_cols": [str, ...],      # optional, e.g. label/outcome columns
            "normalize_mode": "zscore" | "maxabs" | "none",  # optional
            "include_missing_mask": bool,    # optional, static only
            "imputation": "zero" | "mean" | "median" | "iterative",  # optional, static only
            "log_columns": [str, ...],       # optional, column NAMES to log-transform
        }
    id_col : shared join key across all tables.
    reference_ids : canonical id ordering. If None, the sorted union of every
        table's ids is used. An id absent from a given table becomes a fully
        missing sample for that modality (all-NaN static row / None temporal
        entry) rather than dropping the id from the dataset.

    Returns
    -------
    data_dict : ready to pass to ZStarDataset(data_dict, ...)
    ids : the canonical id array, in the row order used throughout data_dict
    feature_columns : {modality_name: [column names actually used as features]}
        -- inspect this to size `modalities.<name>.input_dim` in config.yaml
        (remember include_missing_mask doubles the static width).
    """
    if reference_ids is None:
        all_ids = set()
        for spec in tables.values():
            all_ids.update(spec["df"][id_col].tolist())
        ids = np.array(sorted(all_ids))
    else:
        ids = np.asarray(reference_ids)

    data_dict = {}
    feature_columns = {}

    for name, spec in tables.items():
        df = spec["df"]
        mod_type = spec["type"]
        bool_cols = spec.get("bool_cols", [])
        exclude_cols = set(spec.get("exclude_cols", [])) | {id_col}
        log_col_names = spec.get("log_columns")

        if mod_type == "static":
            dup = df[id_col].duplicated().sum()
            if dup:
                raise ValueError(
                    f"Static modality '{name}' has {dup} duplicate '{id_col}' rows; "
                    f"static tables must have one row per id."
                )
            indexed = df.set_index(id_col).reindex(ids)
            feat_cols = [c for c in indexed.columns if c not in exclude_cols]
            encoded = encode_bool_columns(indexed[feat_cols], bool_cols)
            arr = encoded.to_numpy(dtype=np.float32)

            entry = {"type": "static", "data": arr}
            if "normalize_mode" in spec:
                entry["normalize_mode"] = spec["normalize_mode"]
            if spec.get("include_missing_mask"):
                entry["include_missing_mask"] = True
            if "imputation" in spec:
                entry["imputation"] = spec["imputation"]
            if log_col_names:
                entry["log_columns"] = [feat_cols.index(c) for c in log_col_names]
            if bool_cols:
                # bool_cols already marks which columns are binary -- reuse it so
                # iterative imputation knows to clip those columns back to [0,1]
                entry["binary_columns"] = [feat_cols.index(c) for c in bool_cols if c in feat_cols]

            data_dict[name] = entry
            feature_columns[name] = feat_cols

        elif mod_type in ("temporal", "event"):
            ts_col = spec["timestamp_col"]
            exclude_cols |= {ts_col}
            feat_cols = [c for c in df.columns if c not in exclude_cols]
            encoded = encode_bool_columns(df, bool_cols)
            groups = dict(tuple(encoded.groupby(id_col)))

            samples = []
            for _id in ids:
                g = groups.get(_id)
                if g is None or len(g) == 0:
                    samples.append(None)
                else:
                    g = g.sort_values(ts_col)
                    ts = g[ts_col].to_numpy(dtype=np.float32)
                    vals = g[feat_cols].to_numpy(dtype=np.float32)
                    samples.append((ts, vals))

            entry = {"type": mod_type, "data": samples}
            if "normalize_mode" in spec:
                entry["normalize_mode"] = spec["normalize_mode"]
            if log_col_names:
                entry["log_columns"] = [feat_cols.index(c) for c in log_col_names]

            data_dict[name] = entry
            feature_columns[name] = feat_cols

        else:
            raise ValueError(f"Unknown modality type '{mod_type}' for '{name}'")

    return data_dict, ids, feature_columns
