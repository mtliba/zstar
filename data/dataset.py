import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional


class ZStarDataset(Dataset):
    """
    Handles three modality types:
    - static:   {name: array [N, D]}
    - temporal:  {name: list of N tuples (timestamps [T_i], values [T_i, D])}
    - event:     {name: list of N tuples (timestamps [T_i], features [T_i, D])}

    Missing: for static, entire row is NaN. For temporal/event, entry is None.

    Per-modality spec keys (all optional beyond "type"/"data"):
    - normalize_mode: "zscore" | "maxabs" | "none". Defaults to "zscore" if the
      dataset-level `normalize=True`, else "none". "maxabs" divides by the
      per-feature max absolute value with no centering, so 0 stays 0 --
      appropriate for columns where 0 is a meaningful baseline (e.g. dose=0,
      event count=0) rather than a scaling artifact.
    - log_columns: list of feature-column indices to log-transform (via
      log(clip(x, eps, None))) before normalization. NaNs pass through
      unaffected.
    - include_missing_mask (static only): if True, concatenates a per-feature
      observed/missing indicator (1=observed, 0=missing) to the feature
      vector, doubling its width. Use this when a modality has real per-field
      missingness (not just whole-row absence) that the model should see
      rather than have silently zero-imputed. Check `dataset.modality_info`
      after construction for the resulting encoded width before setting
      `modalities.<name>.input_dim` in config.
    - imputation (static only): "zero" | "mean" | "median" | "iterative".
      Defaults to "zero" -- fills NaN with 0 *after* normalization, which
      equals the per-feature mean under normalize_mode="zscore" but literal
      zero under "maxabs"/"none" (unchanged legacy behavior). "mean"/"median"
      fill with the per-feature statistic in raw space, before normalization.
      "iterative" runs scikit-learn's IterativeImputer (MICE-style: each
      missing feature is predicted from the other observed features in the
      same row via a fitted regression, refined over a few rounds) -- more
      informative than a population mean when missingness correlates with
      other recorded fields, though it degenerates toward the population
      mean when a field's missingness is fully determined by a single other
      column with no overlapping observed cases to learn from.
    - binary_columns (static only, "iterative" imputation only): list of
      feature-column indices that are binary/boolean. IterativeImputer's
      default regressor (BayesianRidge) treats every column as continuous,
      so a binary column's imputed values can land outside [0,1] or take
      non-discrete values. Declared binary columns are clipped to [0,1]
      after imputation (kept as a soft implied probability, not rounded).
    """

    _LOG_EPS = 1e-6

    def __init__(
        self,
        data_dict: Dict[str, dict],
        normalize: bool = True,
    ):
        self.modalities = {}
        self._n_samples: Optional[int] = None

        for name, spec in data_dict.items():
            mod_type = spec["type"]

            if mod_type == "static":
                self._init_static(name, spec, normalize)
            elif mod_type in ("temporal", "event"):
                self._init_temporal(name, spec, mod_type, normalize)
            else:
                raise ValueError(f"Unknown modality type '{mod_type}' for '{name}'")

    def _resolve_normalize_mode(self, spec: dict, global_normalize: bool) -> str:
        mode = spec.get("normalize_mode")
        if mode is not None:
            if mode not in ("zscore", "maxabs", "none"):
                raise ValueError(f"Unknown normalize_mode '{mode}'. Choose: zscore | maxabs | none")
            return mode
        return "zscore" if global_normalize else "none"

    def _apply_log(self, arr: np.ndarray, log_columns: Optional[List[int]]) -> np.ndarray:
        if not log_columns:
            return arr
        arr = arr.copy()
        for c in log_columns:
            arr[..., c] = np.log(np.clip(arr[..., c], self._LOG_EPS, None))
        return arr

    @staticmethod
    def _impute_static(
        arr: np.ndarray, mode: str, binary_columns: Optional[List[int]] = None
    ) -> np.ndarray:
        if not np.isnan(arr).any():
            return arr

        if mode == "mean":
            stat = np.nanmean(arr, axis=0)
        elif mode == "median":
            stat = np.nanmedian(arr, axis=0)
        elif mode == "iterative":
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer
            imputer = IterativeImputer(random_state=0, max_iter=10)
            arr = imputer.fit_transform(arr).astype(np.float32)
            # BayesianRidge (the imputer's default estimator) treats every
            # column as continuous regression, so a binary column can come
            # back as an out-of-[0,1] or non-discrete value. Clip declared
            # binary columns back to a valid soft-probability range.
            if binary_columns:
                for c in binary_columns:
                    arr[:, c] = np.clip(arr[:, c], 0.0, 1.0)
            return arr
        else:
            raise ValueError(f"Unknown imputation '{mode}'. Choose: zero | mean | median | iterative")

        stat = np.nan_to_num(stat, nan=0.0)  # a fully-NaN column has no statistic; fall back to 0
        inds = np.where(np.isnan(arr))
        arr = arr.copy()
        arr[inds] = np.take(stat, inds[1])
        return arr

    def _init_static(self, name: str, spec: dict, global_normalize: bool):
        data = spec["data"]
        normalize_mode = self._resolve_normalize_mode(spec, global_normalize)
        log_columns = spec.get("log_columns")
        include_missing_mask = bool(spec.get("include_missing_mask", False))
        imputation = spec.get("imputation", "zero")
        binary_columns = spec.get("binary_columns")

        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Static modality '{name}' must be 2-D, got shape {arr.shape}")
        self._check_n(arr.shape[0], name)

        mask = ~np.all(np.isnan(arr), axis=1)          # per-sample: modality present at all
        feature_observed = ~np.isnan(arr)               # per-feature: this value observed

        arr = self._apply_log(arr, log_columns)

        if imputation != "zero":
            arr = self._impute_static(arr, imputation, binary_columns)

        valid = arr[mask]
        if normalize_mode == "zscore" and len(valid) > 0:
            mean = np.nanmean(valid, axis=0)
            std = np.nanstd(valid, axis=0)
            std[std < 1e-8] = 1.0
            arr = (arr - mean) / std
        elif normalize_mode == "maxabs" and len(valid) > 0:
            maxabs = np.nanmax(np.abs(valid), axis=0)
            maxabs[maxabs < 1e-8] = 1.0
            arr = arr / maxabs

        arr = np.nan_to_num(arr, nan=0.0)  # covers imputation="zero", and any residual edge-case NaN

        if include_missing_mask:
            arr = np.concatenate([arr, feature_observed.astype(np.float32)], axis=1)

        self.modalities[name] = {
            "type": "static",
            "data": torch.tensor(arr, dtype=torch.float32),
            "mask": torch.tensor(mask.astype(np.float32)),
            "n_features": arr.shape[1],
        }

    def _init_temporal(self, name: str, spec: dict, mod_type: str, global_normalize: bool):
        data = spec["data"]
        normalize_mode = self._resolve_normalize_mode(spec, global_normalize)
        log_columns = spec.get("log_columns")

        N = len(data)
        self._check_n(N, name)

        raw_vals = [
            self._apply_log(np.asarray(item[1], dtype=np.float32), log_columns)
            for item in data if item is not None
        ]

        if normalize_mode == "zscore" and raw_vals:
            cat = np.concatenate(raw_vals, axis=0)
            mean = np.nanmean(cat, axis=0)
            std = np.nanstd(cat, axis=0)
            std[std < 1e-8] = 1.0
        elif normalize_mode == "maxabs" and raw_vals:
            cat = np.concatenate(raw_vals, axis=0)
            mean = 0.0
            std = np.nanmax(np.abs(cat), axis=0)
            std[std < 1e-8] = 1.0
        else:
            mean, std = 0.0, 1.0

        processed = []
        masks = []
        n_features = None
        for item in data:
            if item is None:
                processed.append(None)
                masks.append(0.0)
            else:
                ts, vals = item
                vals = self._apply_log(np.asarray(vals, dtype=np.float32), log_columns)
                vals = (vals - mean) / std
                vals = np.nan_to_num(vals, nan=0.0)
                n_features = vals.shape[-1]
                ts = np.asarray(ts, dtype=np.float32)
                processed.append((torch.tensor(ts), torch.tensor(vals)))
                masks.append(1.0)

        self.modalities[name] = {
            "type": mod_type,
            "data": processed,
            "mask": torch.tensor(masks, dtype=torch.float32),
            "n_features": n_features,
        }

    def _check_n(self, n: int, name: str):
        if self._n_samples is None:
            self._n_samples = n
        elif n != self._n_samples:
            raise ValueError(f"Modality '{name}' has {n} samples, expected {self._n_samples}")

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Dict:
        result = {}
        for name, mod in self.modalities.items():
            mask = mod["mask"][idx]
            if mod["type"] == "static":
                result[name] = {
                    "x": mod["data"][idx],
                    "mask": mask,
                    "type": "static",
                }
            else:
                item = mod["data"][idx]
                if item is not None:
                    timestamps, values = item
                    result[name] = {
                        "x": values,
                        "mask": mask,
                        "timestamps": timestamps,
                        "length": torch.tensor(len(timestamps)),
                        "type": mod["type"],
                    }
                else:
                    result[name] = {
                        "x": torch.zeros(1, self._get_dim(name)),
                        "mask": mask,
                        "timestamps": torch.zeros(1),
                        "length": torch.tensor(0),
                        "type": mod["type"],
                    }
        return result

    def _get_dim(self, name: str) -> int:
        mod = self.modalities[name]
        if mod.get("n_features"):
            return mod["n_features"]
        for item in mod["data"]:
            if item is not None:
                _, vals = item
                return vals.shape[-1]
        return 1

    @property
    def modality_info(self) -> Dict[str, dict]:
        info = {}
        for name, mod in self.modalities.items():
            avail = float(mod["mask"].mean())
            info[name] = {
                "type": mod["type"],
                "availability": avail,
                "n_features": mod.get("n_features") or self._get_dim(name),
            }
        return info
