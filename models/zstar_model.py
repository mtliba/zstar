import torch
import torch.nn as nn
from typing import Dict, Optional
from omegaconf import DictConfig

from .modality_module import ModalityModule
from zstar.fusion import get_fusion
from zstar.losses.temporal_prediction import TemporalPredictionHead


class ZStarModel(nn.Module):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = int(cfg.model.latent_dim)

        self.modality_names = [
            name for name, mcfg in cfg.modalities.items() if mcfg.get("enabled", True)
        ]
        if not self.modality_names:
            raise ValueError("No modalities enabled.")

        self.modules_dict = nn.ModuleDict({
            name: ModalityModule(name, cfg.modalities[name], cfg.model)
            for name in self.modality_names
        })

        self.fusion = get_fusion(
            name=str(cfg.model.get("fusion", "poe")),
            modality_names=self.modality_names,
            latent_dim=self.latent_dim,
            config=cfg.model,
        )

        # Temporal prediction heads (if configured)
        self.temporal_pred_heads = nn.ModuleDict()
        if hasattr(cfg, "objectives") and hasattr(cfg.objectives, "temporal_prediction"):
            tp_cfg = cfg.objectives.temporal_prediction
            if tp_cfg.get("enabled", False):
                horizon = int(tp_cfg.get("prediction_horizon", 5))
                for mod_name in tp_cfg.get("modalities", []):
                    if mod_name in self.modality_names:
                        out_dim = int(cfg.modalities[mod_name].input_dim)
                        self.temporal_pred_heads[mod_name] = TemporalPredictionHead(
                            self.latent_dim, out_dim, prediction_horizon=horizon,
                        )

    def _reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        return mu

    def forward(self, batch: Dict) -> Dict:
        mus, log_vars, zs, masks = {}, {}, {}, {}
        vq_losses = {}
        recons = {}

        for name in self.modality_names:
            if name not in batch:
                continue

            item = batch[name]
            x = item["x"]
            mask = item["mask"]
            timestamps = item.get("timestamps")
            lengths = item.get("lengths")

            enc_out = self.modules_dict[name].encode(x, timestamps, lengths)
            mu = enc_out["mu"]
            log_var = enc_out["log_var"]

            mus[name] = mu
            log_vars[name] = log_var
            zs[name] = self.modules_dict[name].reparameterize(mu, log_var, self.training)
            masks[name] = mask.float()

            if "vq_loss" in enc_out:
                vq_losses[name] = {"vq_loss": enc_out["vq_loss"], "perplexity": enc_out["vq_perplexity"]}

        if not mus:
            raise RuntimeError("Batch contains no recognised modalities.")

        mu_shared, log_var_shared = self.fusion(mus, log_vars, masks)
        z_shared = self._reparameterize(mu_shared, log_var_shared)

        for name in mus:
            item = batch[name]
            recons[name] = self.modules_dict[name].decode(
                z_shared,
                target_timestamps=item.get("timestamps"),
                target_lengths=item.get("lengths"),
            )

        # Temporal predictions
        temporal_preds = {}
        temporal_targets = {}
        for name, head in self.temporal_pred_heads.items():
            if name in batch:
                temporal_preds[name] = head(z_shared)
                # Target: last `prediction_horizon` steps of the actual sequence
                x = batch[name]["x"]
                H = head.prediction_horizon
                if x.dim() == 3 and x.size(1) > H:
                    temporal_targets[name] = x[:, -H:, :]

        result = {
            "recons": recons,
            "mus": mus,
            "log_vars": log_vars,
            "zs": zs,
            "masks": masks,
            "mu_shared": mu_shared,
            "log_var_shared": log_var_shared,
            "z_shared": z_shared,
        }
        if vq_losses:
            result["vq_losses"] = vq_losses
        if temporal_preds:
            result["temporal_preds"] = temporal_preds
            result["temporal_targets"] = temporal_targets

        return result

    @torch.no_grad()
    def extract_zstar(self, batch: Dict) -> torch.Tensor:
        self.eval()
        mus, log_vars, masks = {}, {}, {}

        for name in self.modality_names:
            if name not in batch:
                continue
            item = batch[name]
            enc_out = self.modules_dict[name].encode(
                item["x"], item.get("timestamps"), item.get("lengths")
            )
            mus[name] = enc_out["mu"]
            log_vars[name] = enc_out["log_var"]
            masks[name] = item["mask"].float()

        mu_shared, _ = self.fusion(mus, log_vars, masks)
        return mu_shared

    @torch.no_grad()
    def impute(self, batch: Dict, target_modality: str) -> torch.Tensor:
        z_star = self.extract_zstar(batch)
        target_item = batch.get(target_modality, {})
        return self.modules_dict[target_modality].decode(
            z_star,
            target_timestamps=target_item.get("timestamps"),
            target_lengths=target_item.get("lengths"),
        )
