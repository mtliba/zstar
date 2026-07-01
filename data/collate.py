import torch
from typing import Dict, List


def zstar_collate(batch: List[Dict]) -> Dict:
    names = list(batch[0].keys())
    result = {}

    for name in names:
        mod_type = batch[0][name]["type"]

        if mod_type == "static":
            result[name] = {
                "x": torch.stack([item[name]["x"] for item in batch]),
                "mask": torch.stack([item[name]["mask"] for item in batch]),
                "type": "static",
            }
        else:
            xs = [item[name]["x"] for item in batch]
            timestamps_list = [item[name]["timestamps"] for item in batch]
            lengths = torch.stack([item[name]["length"] for item in batch])

            T_max = int(lengths.max().item())
            if T_max == 0:
                T_max = 1

            D = xs[0].shape[-1]
            B = len(batch)

            x_padded = torch.zeros(B, T_max, D)
            ts_padded = torch.zeros(B, T_max)

            for i in range(B):
                L = int(lengths[i].item())
                if L > 0:
                    x_padded[i, :L] = xs[i][:L]
                    ts_padded[i, :L] = timestamps_list[i][:L]

            result[name] = {
                "x": x_padded,
                "mask": torch.stack([item[name]["mask"] for item in batch]),
                "timestamps": ts_padded,
                "lengths": lengths,
                "type": mod_type,
            }

    return result
