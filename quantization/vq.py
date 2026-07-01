import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from omegaconf import DictConfig


class VectorQuantizer(nn.Module):

    def __init__(self, config: DictConfig):
        super().__init__()
        self.num_embeddings = int(config.get("num_embeddings", 512))
        self.embedding_dim = int(config.get("embedding_dim", 64))
        self.commitment_cost = float(config.get("commitment_cost", 0.25))
        self.use_ema = bool(config.get("use_ema", True))
        self.ema_decay = float(config.get("ema_decay", 0.99))
        self.restart_threshold = float(config.get("restart_threshold", 1.0))

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

        if self.use_ema:
            self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
            self.register_buffer("ema_embed_sum", self.embedding.weight.clone())

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        distances = torch.cdist(z_e, self.embedding.weight)  # [B, K]
        indices = distances.argmin(dim=-1)  # [B]
        z_q = self.embedding(indices)  # [B, D]

        if self.training and self.use_ema:
            self._ema_update(z_e, indices)

        if self.use_ema:
            loss = self.commitment_cost * F.mse_loss(z_e, z_q.detach())
        else:
            loss = F.mse_loss(z_q, z_e.detach()) + self.commitment_cost * F.mse_loss(z_e, z_q.detach())

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        perplexity = self._perplexity(indices)

        return z_q_st, indices, {"vq_loss": loss, "perplexity": perplexity}

    def _ema_update(self, z_e: torch.Tensor, indices: torch.Tensor):
        one_hot = F.one_hot(indices, self.num_embeddings).float()  # [B, K]
        cluster_size = one_hot.sum(0)  # [K]
        embed_sum = one_hot.t() @ z_e  # [K, D]

        self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
        self.ema_embed_sum.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)

        n = self.ema_cluster_size.sum()
        cluster_size_stable = (
            (self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
        )
        self.embedding.weight.data.copy_(self.ema_embed_sum / cluster_size_stable.unsqueeze(-1))

        # Restart dead codes
        dead = self.ema_cluster_size < self.restart_threshold
        if dead.any():
            n_dead = dead.sum().item()
            random_idx = torch.randint(0, z_e.size(0), (n_dead,), device=z_e.device)
            self.embedding.weight.data[dead] = z_e[random_idx].detach()
            self.ema_cluster_size[dead] = self.restart_threshold
            self.ema_embed_sum[dead] = z_e[random_idx].detach() * self.restart_threshold

    def _perplexity(self, indices: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(indices, self.num_embeddings).float()
        avg_probs = one_hot.mean(0)
        return torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
