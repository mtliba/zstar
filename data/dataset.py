import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union


class ZStarDataset(Dataset):
    """
    Handles three modality types:
    - static:   {name: array [N, D]}
    - temporal:  {name: list of N tuples (timestamps [T_i], values [T_i, D])}
    - event:     {name: list of N tuples (timestamps [T_i], features [T_i, D])}

    Missing: for static, entire row is NaN. For temporal/event, entry is None.
    """

    def __init__(
        self,
        data_dict: Dict[str, dict],
        normalize: bool = True,
    ):
        self.modalities = {}
        self._n_samples: Optional[int] = None

        for name, spec in data_dict.items():
            mod_type = spec["type"]
            data = spec["data"]

            if mod_type == "static":
                self._init_static(name, data, normalize)
            elif mod_type in ("temporal", "event"):
                self._init_temporal(name, data, mod_type, normalize)
            else:
                raise ValueError(f"Unknown modality type '{mod_type}' for '{name}'")

    def _init_static(self, name: str, data: np.ndarray, normalize: bool):
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Static modality '{name}' must be 2-D, got shape {arr.shape}")
        self._check_n(arr.shape[0], name)

        mask = ~np.all(np.isnan(arr), axis=1)
        if normalize:
            valid = arr[mask]
            if len(valid) > 0:
                mean = np.nanmean(valid, axis=0)
                std = np.nanstd(valid, axis=0)
                std[std < 1e-8] = 1.0
                arr = (arr - mean) / std

        arr = np.nan_to_num(arr, nan=0.0)
        self.modalities[name] = {
            "type": "static",
            "data": torch.tensor(arr, dtype=torch.float32),
            "mask": torch.tensor(mask.astype(np.float32)),
        }

    def _init_temporal(self, name: str, data: list, mod_type: str, normalize: bool):
        N = len(data)
        self._check_n(N, name)

        if normalize:
            all_vals = [item[1] for item in data if item is not None]
            if all_vals:
                cat = np.concatenate(all_vals, axis=0)
                mean = np.nanmean(cat, axis=0)
                std = np.nanstd(cat, axis=0)
                std[std < 1e-8] = 1.0
            else:
                mean, std = 0.0, 1.0
        else:
            mean, std = 0.0, 1.0

        processed = []
        masks = []
        for item in data:
            if item is None:
                processed.append(None)
                masks.append(0.0)
            else:
                ts, vals = item
                vals = (np.asarray(vals, dtype=np.float32) - mean) / std
                vals = np.nan_to_num(vals, nan=0.0)
                ts = np.asarray(ts, dtype=np.float32)
                processed.append((torch.tensor(ts), torch.tensor(vals)))
                masks.append(1.0)

        self.modalities[name] = {
            "type": mod_type,
            "data": processed,
            "mask": torch.tensor(masks, dtype=torch.float32),
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
            info[name] = {"type": mod["type"], "availability": avail}
        return info
