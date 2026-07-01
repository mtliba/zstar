import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from omegaconf import DictConfig

from zstar.encoders import get_encoder
from zstar.decoders import get_decoder
from zstar.quantization import VectorQuantizer


class ModalityModule(nn.Module):

    def __init__(self, name: str, modality_cfg: DictConfig, model_cfg: DictConfig):
        super().__init__()
        self.name = name
        self.modality_type = str(modality_cfg.type)
        latent_dim = int(model_cfg.latent_dim)

        self.encoder = get_encoder(
            name=str(modality_cfg.encoder),
            input_dim=int(modality_cfg.input_dim),
            latent_dim=latent_dim,
            config=modality_cfg.get("encoder_config", DictConfig({})),
        )
        self.decoder = get_decoder(
            name=str(modality_cfg.get("decoder", "mlp")),
            latent_dim=latent_dim,
            output_dim=int(modality_cfg.input_dim),
            config=modality_cfg.get("decoder_config", DictConfig({})),
        )

        latent_type = str(model_cfg.get("latent_type", "continuous"))
        if latent_type in ("discrete", "hybrid"):
            vq_cfg = model_cfg.get("vq", DictConfig({"num_embeddings": 512, "embedding_dim": latent_dim}))
            if int(vq_cfg.get("embedding_dim", latent_dim)) != latent_dim:
                raise ValueError(f"VQ embedding_dim ({vq_cfg.embedding_dim}) must match latent_dim ({latent_dim})")
            self.quantizer = VectorQuantizer(vq_cfg)
        else:
            self.quantizer = None

    def encode(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        mu, log_var = self.encoder(x, timestamps, lengths)
        result = {"mu": mu, "log_var": log_var}

        if self.quantizer is not None:
            z_q, indices, vq_info = self.quantizer(mu)
            result["z_q"] = z_q
            result["vq_indices"] = indices
            result["vq_loss"] = vq_info["vq_loss"]
            result["vq_perplexity"] = vq_info["perplexity"]

        return result

    def decode(
        self,
        z: torch.Tensor,
        target_timestamps: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.decoder(z, target_timestamps, target_lengths)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor, training: bool = True) -> torch.Tensor:
        if training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        return mu
