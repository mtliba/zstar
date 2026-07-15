import torch.nn as nn
from typing import List


def build_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)
