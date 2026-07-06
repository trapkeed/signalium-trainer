"""Generic training engine — Phase 14.9B upgrade.

Works with any BaseQuantModel subclass.

Phase 14.9B additions:
  - Gradient accumulation (accumulate_grad_batches)
  - Warmup scheduler (warmup_epochs)
  - New optimizers: Adagrad, Adamax
  - New schedulers: CosineAnnealingWarmRestarts, OneCycleLR, LinearLR
  - Gradient checkpointing (applied at backbone level via config)
  - CheckpointManager with RNG state, config serialisation, top-k rotation
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from loguru import logger
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from trainer.checkpoints.manager import CheckpointManager
from trainer.config.schema import TrainerConfig
from trainer.metrics.tracker import MetricsTracker
from trainer.models.base import BaseQuantModel
from trainer.training.early_stopping import EarlyStopping
from trainer.training.scheduler import WarmupScheduler
from trainer.training.seed import set_seed
from trainer.utils.device import resolve_device


class TrainingEngine:
    """Orchestrates the full training lifecycle for any BaseQuantModel.

    Responsibilities:
    - Seed reproducibility
    - Device placement
    - Optimiser and scheduler construction
    - Warmup LR scheduling
    - Gradient accumulation
    - Mixed-precision training (AMP)
    - Gradient clipping
    - Checkpoint saving / resuming (with RNG state)
    - Early stopping
    - Metrics tracking

    Args:
        model: Untrained model instance.
        config: Full TrainerConfig.
        criterion: Loss function (e.g. CrossEntropyLoss).
    """

    def __init__(
        self,
        model: BaseQuantModel,
        config: TrainerConfig,
        criterion: nn.Module,
    ) -> None:
        self._config = config
        self._criterion = criterion

        set_seed(config.training.seed, deterministic=config.gpu.cudnn_deterministic)
        self._device = resolve_device(config.gpu.device)

        self._model = model.to(self._device)
        self._optimizer = self._build_optimizer()
        self._scheduler: Any = self._build_scheduler()
        self._scaler = GradScaler(enabled=config.training.mixed_precision)

        self._checkpoint_manager = CheckpointManager(
            checkpoint_dir=Path(config.checkpoint.checkpoint_dir),
            keep_top_k=config.checkpoint.keep_top_k,
            monitor_metric=config.checkpoint.monitor_metric,
            monitor_mode=config.checkpoint.monitor_mode,
            save_optimizer_state=config.checkpoint.save_optimizer_state,
            save_scheduler_state=config.checkpoint.save_scheduler_state,
            save_rng_state=config.checkpoint.save_rng_state,
            save_training_config=config.checkpoint.save_training_config,
        )
        self._early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            min_delta=config.training.early_stopping_min_delta,
            mode=config.checkpoint.monitor_mode,
        )
        self._metrics = MetricsTracker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_checkpoint: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Execute the complete training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            resume_checkpoint: If provided, resume from this checkpoint file.

        Returns:
            Dict with "history", "best_metrics", and "duration_seconds".
        """
        start_epoch = 0
        if resume_checkpoint:
            start_epoch = self._resume(resume_checkpoint)
            logger.info("Resumed from epoch {}", start_epoch)

        total_start = time.perf_counter()
        history: list[dict[str, float]] = []

        for epoch in range(start_epoch, self._config.training.epochs):
            train_metrics = self._train_one_epoch(epoch, train_loader)
            val_metrics = self._validate_one_epoch(epoch, val_loader)

            combined = {**train_metrics, **val_metrics}
            history.append({"epoch": epoch, **combined})
            self._metrics.update(combined)

            self._log_epoch(epoch, combined)
            self._step_scheduler(val_metrics.get("val_loss", float("inf")))

            monitor_value = val_metrics.get(self._config.checkpoint.monitor_metric, float("inf"))

            self._checkpoint_manager.save(
                epoch=epoch,
                model=self._model,
                optimizer=self._optimizer,
                scheduler=self._scheduler,
                metrics=combined,
                config=self._config,
            )

            if self._early_stopping.step(monitor_value):
                logger.info("Early stopping at epoch {}", epoch)
                break

        duration = time.perf_counter() - total_start
        return {
            "history": history,
            "best_metrics": self._checkpoint_manager.best_metrics,
            "duration_seconds": duration,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int, loader: DataLoader) -> dict[str, float]:
        self._model.train()
        total_loss = 0.0
        n_batches = 0
        accum_steps = max(1, self._config.training.accumulate_grad_batches)
        n_loader = len(loader)

        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(self._device, non_blocking=True)
            y = y.to(self._device, non_blocking=True)

            # Zero grad at the start of each accumulation window
            if batch_idx % accum_steps == 0:
                self._optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=self._device.type,
                enabled=self._config.training.mixed_precision,
            ):
                logits = self._model(x)
                # Divide by accum_steps so effective gradient scale is invariant
                loss = self._criterion(logits, y) / accum_steps

            self._scaler.scale(loss).backward()

            is_last_in_window = (batch_idx + 1) % accum_steps == 0
            is_last_batch = (batch_idx + 1) == n_loader

            if is_last_in_window or is_last_batch:
                if self._config.training.gradient_clip_val > 0:
                    self._scaler.unscale_(self._optimizer)
                    nn.utils.clip_grad_norm_(
                        self._model.parameters(), self._config.training.gradient_clip_val
                    )
                self._scaler.step(self._optimizer)
                self._scaler.update()

            # Report un-scaled loss for logging
            total_loss += loss.item() * accum_steps
            n_batches += 1

            if batch_idx % self._config.training.log_every_n_steps == 0:
                logger.debug(
                    "Epoch {} | Batch {}/{} | Loss {:.4f}",
                    epoch,
                    batch_idx,
                    n_loader,
                    loss.item() * accum_steps,
                )

        return {"train_loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def _validate_one_epoch(self, epoch: int, loader: DataLoader) -> dict[str, float]:
        self._model.eval()
        total_loss = 0.0
        n_batches = 0

        for x, y in loader:
            x = x.to(self._device, non_blocking=True)
            y = y.to(self._device, non_blocking=True)

            with torch.autocast(
                device_type=self._device.type,
                enabled=self._config.training.mixed_precision,
            ):
                logits = self._model(x)
                loss = self._criterion(logits, y)

            total_loss += loss.item()
            n_batches += 1

        return {"val_loss": total_loss / max(n_batches, 1)}

    def _log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info("Epoch {:>4d} | {}", epoch, metrics_str)

    def _step_scheduler(self, val_loss: float) -> None:
        from trainer.config.schema import SchedulerType

        if isinstance(self._scheduler, WarmupScheduler):
            stype = self._config.scheduler.type
            if stype == SchedulerType.REDUCE_ON_PLATEAU:
                self._scheduler.step(metric=val_loss)
            else:
                self._scheduler.step()
            return

        stype = self._config.scheduler.type
        if stype == SchedulerType.REDUCE_ON_PLATEAU:
            self._scheduler.step(val_loss)
        elif stype != SchedulerType.NONE:
            self._scheduler.step()

    def _resume(self, checkpoint_path: Path) -> int:
        """Load a checkpoint and return the next epoch to train from."""
        state = self._checkpoint_manager.load(
            checkpoint_path, self._model, self._optimizer, self._scheduler
        )
        return state.get("epoch", 0) + 1

    def _build_optimizer(self) -> torch.optim.Optimizer:
        from trainer.config.schema import OptimizerType

        cfg = self._config.optimizer
        params = self._model.parameters()

        if cfg.type == OptimizerType.ADAM:
            return torch.optim.Adam(
                params,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                betas=cfg.betas,
                eps=cfg.eps,
            )
        if cfg.type == OptimizerType.ADAMW:
            return torch.optim.AdamW(
                params,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                betas=cfg.betas,
                eps=cfg.eps,
            )
        if cfg.type == OptimizerType.SGD:
            return torch.optim.SGD(
                params,
                lr=cfg.learning_rate,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            )
        if cfg.type == OptimizerType.RMSPROP:
            return torch.optim.RMSprop(
                params,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                momentum=cfg.momentum,
            )
        if cfg.type == OptimizerType.ADAGRAD:
            return torch.optim.Adagrad(
                params,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                eps=cfg.eps,
            )
        if cfg.type == OptimizerType.ADAMAX:
            return torch.optim.Adamax(
                params,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                betas=cfg.betas,
                eps=cfg.eps,
            )
        raise ValueError(f"Unknown optimizer type: {cfg.type}")

    def _build_scheduler(self) -> Any:
        from trainer.config.schema import SchedulerType

        cfg = self._config.scheduler
        opt = self._optimizer
        is_plateau = cfg.type == SchedulerType.REDUCE_ON_PLATEAU

        base_scheduler = self._build_base_scheduler(opt, cfg)

        if cfg.warmup_epochs > 0:
            return WarmupScheduler(
                optimizer=opt,
                warmup_epochs=cfg.warmup_epochs,
                base_scheduler=base_scheduler,
                is_plateau=is_plateau,
            )
        return base_scheduler

    @staticmethod
    def _build_base_scheduler(
        opt: torch.optim.Optimizer,
        cfg: Any,
    ) -> Any:
        from trainer.config.schema import SchedulerType

        if cfg.type == SchedulerType.COSINE:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=cfg.T_max, eta_min=cfg.eta_min
            )
        if cfg.type == SchedulerType.STEP:
            return torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)
        if cfg.type == SchedulerType.REDUCE_ON_PLATEAU:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="min",
                patience=cfg.patience,
                factor=cfg.factor,
            )
        if cfg.type == SchedulerType.EXPONENTIAL:
            return torch.optim.lr_scheduler.ExponentialLR(opt, gamma=cfg.gamma)
        if cfg.type == SchedulerType.COSINE_WARM_RESTARTS:
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=cfg.T_0, T_mult=cfg.T_mult, eta_min=cfg.eta_min
            )
        if cfg.type == SchedulerType.LINEAR:
            return torch.optim.lr_scheduler.LinearLR(
                opt,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=cfg.T_max,
            )
        if cfg.type == SchedulerType.ONE_CYCLE:
            total_steps = cfg.total_steps if cfg.total_steps else cfg.T_max
            return torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=cfg.max_lr,
                total_steps=total_steps,
                pct_start=cfg.pct_start,
                div_factor=cfg.div_factor,
                final_div_factor=cfg.final_div_factor,
            )
        # SchedulerType.NONE — no-op constant scheduler
        return torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
