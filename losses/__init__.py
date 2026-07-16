import torch
from typing import Dict, List, Optional
from omegaconf import DictConfig

from .reconstruction import reconstruction_loss, masked_recon_loss
from .kl import kl_divergence
from .vq_loss import vq_aggregate_loss
from .contrastive import contrastive_loss
from .alignment import alignment_loss
from .masked_prediction import masked_prediction_loss
from .temporal_prediction import temporal_prediction_loss


def get_active_objectives(cfg: DictConfig, stage_name: Optional[str] = None) -> List[str]:
    if stage_name and hasattr(cfg.training, "stages"):
        for s in cfg.training.stages:
            if s.name == stage_name:
                if s.get("objectives") == "all" or s.get("objectives") is None:
                    break
                return list(s.objectives)
    return [name for name, ocfg in cfg.objectives.items() if ocfg.get("enabled", False)]


def compute_total_loss(
    batch: Dict,
    outputs: Dict,
    cfg: DictConfig,
    beta: float = 1.0,
    stage_name: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    device = outputs["mu_shared"].device
    active = get_active_objectives(cfg, stage_name)
    losses = {}

    if "reconstruction" in active:
        recon = torch.tensor(0.0, device=device)
        for name, x_recon in outputs["recons"].items():
            x_orig = batch[name]["x"]
            mask = batch[name]["mask"]
            mod_type = batch[name].get("type", "static")
            recon = recon + reconstruction_loss(
                x_orig, x_recon, mask,
                loss_fn=str(cfg.objectives.reconstruction.get("loss_fn", "mse")),
                mod_type=mod_type,
                lengths=batch[name].get("lengths"),
            )
        losses["recon"] = cfg.objectives.reconstruction.weight * recon

    if "kl" in active:
        kl = kl_divergence(outputs["mu_shared"], outputs["log_var_shared"])
        losses["kl"] = beta * kl

    if "vq_commitment" in active and outputs.get("vq_losses"):
        losses["vq"] = cfg.objectives.vq_commitment.weight * vq_aggregate_loss(outputs["vq_losses"])

    if "contrastive" in active:
        losses["contrastive"] = cfg.objectives.contrastive.weight * contrastive_loss(
            outputs["zs"], outputs["masks"],
            temperature=float(cfg.objectives.contrastive.get("temperature", 0.07)),
        )

    if "alignment" in active:
        losses["alignment"] = cfg.objectives.alignment.weight * alignment_loss(
            outputs["zs"], outputs["masks"],
            strategy=str(cfg.objectives.alignment.get("strategy", "mmd")),
            temperature=float(cfg.objectives.alignment.get("temperature", 0.07)),
        )

    if "masked_reconstruction" in active and outputs.get("masked_recons"):
        losses["masked_recon"] = cfg.objectives.masked_reconstruction.weight * masked_recon_loss(
            outputs["masked_recons"], outputs["mask_positions"], batch,
        )

    if "temporal_prediction" in active and outputs.get("temporal_preds"):
        losses["temporal_pred"] = cfg.objectives.temporal_prediction.weight * temporal_prediction_loss(
            outputs["temporal_preds"], outputs["temporal_targets"],
            outputs.get("temporal_valid"),
        )

    total = sum(losses.values())
    losses["total"] = total
    return losses
