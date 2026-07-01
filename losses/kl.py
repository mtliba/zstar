import torch


def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp()).mean()
