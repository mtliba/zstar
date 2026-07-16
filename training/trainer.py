import os
import torch
from torch.utils.data import DataLoader, random_split
from omegaconf import DictConfig
from typing import Dict, List, Optional

from zstar.models.zstar_model import ZStarModel
from zstar.data.collate import zstar_collate
from zstar.losses import compute_total_loss
from .schedulers import get_kl_beta
from .progress import progress_bar, tqdm_write, TQDM_AVAILABLE


class ZStarTrainer:

    def __init__(self, model: ZStarModel, dataset, cfg: DictConfig):
        self.model = model
        self.cfg = cfg
        # logging.progress: set false to silence bars (e.g. in a log-scraped job)
        self.show_progress = bool(cfg.get("logging", {}).get("progress", True))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        n = len(dataset)
        n_val = max(1, int(n * cfg.training.val_split))
        n_train = n - n_val
        train_set, val_set = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(cfg.get("project", {}).get("seed", 42)),
        )

        loader_kwargs = dict(
            batch_size=cfg.training.batch_size,
            collate_fn=zstar_collate,
            num_workers=0,
        )
        self.train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
        self.val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

        opt_name = str(cfg.training.get("optimizer", "adam"))
        if opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
            )
        else:
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
            )

        sched_cfg = cfg.training.get("scheduler", {})
        sched_type = str(sched_cfg.get("type", "reduce_on_plateau"))
        if sched_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.training.epochs
            )
        elif sched_type == "warmup_cosine":
            warmup = int(sched_cfg.get("warmup_epochs", 10))
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                [
                    torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.01, total_iters=warmup),
                    torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.training.epochs - warmup),
                ],
                milestones=[warmup],
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                patience=int(sched_cfg.get("patience", 15)),
                factor=float(sched_cfg.get("factor", 0.5)),
            )

        self.sched_type = sched_type
        os.makedirs(cfg.logging.save_dir, exist_ok=True)
        self.best_val_loss = float("inf")
        self.history: Dict[str, List[dict]] = {"train": [], "val": []}

    def _to_device(self, batch: Dict) -> Dict:
        result = {}
        for name, item in batch.items():
            dev_item = {}
            for k, v in item.items():
                if isinstance(v, torch.Tensor):
                    dev_item[k] = v.to(self.device)
                else:
                    dev_item[k] = v
            result[name] = dev_item
        return result

    def _get_stage(self, epoch: int) -> Optional[str]:
        if not hasattr(self.cfg.training, "stages") or not self.cfg.training.stages:
            return None
        cumulative = 0
        for stage in self.cfg.training.stages:
            cumulative += stage.epochs
            if epoch <= cumulative:
                return stage.name
        return None

    def _run_batch(self, batch: Dict, beta: float, stage_name: Optional[str]) -> Dict[str, torch.Tensor]:
        batch = self._to_device(batch)
        outputs = self.model(batch)
        return compute_total_loss(batch, outputs, self.cfg, beta=beta, stage_name=stage_name)

    def _epoch(
        self,
        loader,
        beta: float,
        train: bool,
        stage_name: Optional[str],
        epoch: Optional[int] = None,
    ) -> Dict[str, float]:
        self.model.train(train)
        totals = {}
        n = 0

        desc = f"  epoch {epoch} {'train' if train else 'val':<5}" if epoch else None
        bar = progress_bar(
            loader,
            enabled=self.show_progress,
            desc=desc,
            leave=False,          # batch bars disappear once the epoch is done
            unit="batch",
            dynamic_ncols=True,
        )

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in bar:
                losses = self._run_batch(batch, beta, stage_name)
                if train:
                    self.optimizer.zero_grad()
                    losses["total"].backward()
                    grad_clip = float(self.cfg.training.get("grad_clip", 1.0))
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.optimizer.step()

                bs = next(iter(batch.values()))["x"].shape[0]
                for k, v in losses.items():
                    totals[k] = totals.get(k, 0.0) + v.item() * bs
                n += bs

                bar.set_postfix(loss=f"{losses['total'].item():.4f}")

        return {k: v / n for k, v in totals.items()}

    def train(self) -> Dict[str, List[dict]]:
        epochs = self.cfg.training.epochs
        log_every = self.cfg.logging.log_every
        ckpt_path = os.path.join(self.cfg.logging.save_dir, "best_zstar.pt")

        print(f"Device : {self.device}")
        print(f"Epochs : {epochs}")
        print(f"Train  : {len(self.train_loader.dataset)} samples")
        print(f"Val    : {len(self.val_loader.dataset)} samples")
        if self.show_progress and not TQDM_AVAILABLE:
            print("Note   : install tqdm for progress bars (pip install tqdm)")
        print()

        epoch_bar = progress_bar(
            range(1, epochs + 1),
            enabled=self.show_progress,
            desc="Training",
            unit="epoch",
            dynamic_ncols=True,
        )

        for epoch in epoch_bar:
            beta = get_kl_beta(epoch, self.cfg)
            stage_name = self._get_stage(epoch)

            train_m = self._epoch(self.train_loader, beta, train=True,
                                  stage_name=stage_name, epoch=epoch)
            val_m = self._epoch(self.val_loader, beta, train=False,
                                stage_name=stage_name, epoch=epoch)

            self.history["train"].append(train_m)
            self.history["val"].append(val_m)

            if self.sched_type == "reduce_on_plateau":
                self.scheduler.step(val_m["total"])
            else:
                self.scheduler.step()

            # Live loss + ETA on the epoch bar itself
            epoch_bar.set_postfix(
                train=f"{train_m['total']:.4f}",
                val=f"{val_m['total']:.4f}",
                beta=f"{beta:.2f}",
                best=f"{min(self.best_val_loss, val_m['total']):.4f}",
            )

            if epoch % log_every == 0 or epoch == 1:
                stage_str = f" [{stage_name}]" if stage_name else ""
                loss_parts = " ".join(f"{k}={v:.4f}" for k, v in train_m.items() if k != "total")
                msg = (
                    f"[{epoch:4d}/{epochs}]{stage_str} β={beta:.2f} | "
                    f"train total={train_m['total']:.4f} {loss_parts} | "
                    f"val total={val_m['total']:.4f}"
                )
                # tqdm.write keeps log lines from corrupting the live bar
                if self.show_progress and TQDM_AVAILABLE:
                    tqdm_write(msg)
                else:
                    print(msg)

            if val_m["total"] < self.best_val_loss:
                self.best_val_loss = val_m["total"]
                torch.save(self.model.state_dict(), ckpt_path)

        print(f"\nDone. Best val loss: {self.best_val_loss:.4f}")
        print(f"Checkpoint: {ckpt_path}")
        return self.history
