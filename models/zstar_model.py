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

        # Temporal prediction: forecast the future from the PAST ONLY.
        #
        # The sequence is cut `horizon` steps before each subject's true end;
        # only the past is encoded, and the head must predict the held-out tail
        # from that. Encoding the full sequence and then "predicting" its last
        # steps would be a copy task -- the encoder would already have seen the
        # answer -- and would apply no pressure to learn extrapolation.
        temporal_preds = {}
        temporal_targets = {}
        temporal_valid = {}
        for name, head in self.temporal_pred_heads.items():
            if name not in batch:
                continue
            item = batch[name]
            x, lengths = item["x"], item.get("lengths")
            H = head.prediction_horizon
            if x.dim() != 3 or lengths is None:
                continue

            # Need at least one past step plus a full horizon to supervise
            valid = lengths > H
            if not bool(valid.any()):
                continue

            past_batch = self._truncate_for_forecast(batch, name, H)
            z_past = self.encode_and_fuse(past_batch)

            temporal_preds[name] = head(z_past)
            temporal_targets[name] = self._gather_future(x, lengths, H)
            temporal_valid[name] = valid

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
            result["temporal_valid"] = temporal_valid

        return result

    @staticmethod
    def _truncate_for_forecast(batch: Dict, name: str, horizon: int) -> Dict:
        """
        Copy of `batch` with modality `name` cut `horizon` steps short.

        Both the lengths and the tensor tail are cut. Reducing `lengths` alone
        is enough for the packed-RNN and attention-masked encoders, but the
        tail is also zeroed so the truncation does not depend on each encoder
        honouring `lengths` -- a leak here would silently make the whole
        objective a copy task again.
        """
        item = batch[name]
        x, lengths = item["x"], item["lengths"]
        new_lengths = (lengths - horizon).clamp(min=1)

        T = x.size(1)
        keep = torch.arange(T, device=x.device)[None, :] < new_lengths[:, None]
        new_item = dict(item)
        new_item["x"] = x * keep.unsqueeze(-1)
        new_item["lengths"] = new_lengths
        if item.get("timestamps") is not None:
            new_item["timestamps"] = item["timestamps"] * keep

        out = dict(batch)
        out[name] = new_item
        return out

    @staticmethod
    def _gather_future(x: torch.Tensor, lengths: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        The last `horizon` real timesteps per subject -> [B, H, D].

        Indexed per subject from its own length, not from the padded width;
        `x[:, -H:]` would pick up padding for every subject shorter than the
        batch maximum.
        """
        B, T, D = x.shape
        starts = (lengths - horizon).clamp(min=0)
        idx = starts[:, None] + torch.arange(horizon, device=x.device)[None, :]
        idx = idx.clamp(max=T - 1)
        return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, D))

    def encode_and_fuse(self, batch: Dict) -> torch.Tensor:
        """
        Differentiable z-star: encode each present modality, fuse, return the
        shared posterior mean. No decoders are run.

        Unlike `extract_zstar`, this keeps the graph, so gradients flow back
        into the encoders -- required to fine-tune the pretrained encoder
        jointly with a downstream head.
        """
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
    def extract_zstar(self, batch: Dict) -> torch.Tensor:
        """Frozen inference: deterministic z-star, no gradient."""
        self.eval()
        return self.encode_and_fuse(batch)

    @torch.no_grad()
    def impute(self, batch: Dict, target_modality: str) -> torch.Tensor:
        z_star = self.extract_zstar(batch)
        target_item = batch.get(target_modality, {})
        return self.modules_dict[target_modality].decode(
            z_star,
            target_timestamps=target_item.get("timestamps"),
            target_lengths=target_item.get("lengths"),
        )
