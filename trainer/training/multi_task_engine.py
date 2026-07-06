"""MultiTaskTrainingEngine — Phase 14.9B upgrade.

Extends TrainingEngine for Phase 14.5 multi-task training.

Phase 14.9B additions (same as engine.py):
  - Gradient accumulation (accumulate_grad_batches)
  - Warmup scheduler (warmup_epochs)
  - New optimizers: Adagrad, Adamax
  - New schedulers: CosineAnnealingWarmRestarts, OneCycleLR, LinearLR
  - CheckpointManager with full institutional state capture
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from trainer.checkpoints.manager import CheckpointManager
from trainer.config.schema import TrainerConfig
from trainer.losses.multi_task import MultiTaskLoss
from trainer.metrics.tracker import MetricsTracker
from trainer.models.base import BaseQuantModel
from trainer.training.early_stopping import EarlyStopping
from trainer.training.scheduler import WarmupScheduler
from trainer.training.seed import set_seed
from trainer.utils.device import resolve_device


class MultiTaskTrainingEngine:
    """Orchestrates multi-task training for QuantModelV1 Phase 14.5.

    The engine handles:
    - Seeding and device placement
    - Optimizer and scheduler construction
    - Warmup LR scheduling
    - Gradient accumulation
    - Mixed-precision training with AMP
    - Gradient clipping
    - Per-head loss tracking
    - Checkpoint saving / resuming (with RNG state)
    - Early stopping

    DataLoader batches must yield ``(x, labels_dict)`` where ``labels_dict``
    is a dict with keys like ``y_direction``, ``y_move``, ``y_sl``, etc.
    Use ``MultiTaskParquetDataset`` which produces this format automatically.

    Args:
        model: Untrained QuantModelV1 with multi-task heads enabled.
        config: Full TrainerConfig.
        criterion: Configured MultiTaskLoss instance.
    """

    def __init__(
        self,
        model: BaseQuantModel,
        config: TrainerConfig,
        criterion: MultiTaskLoss,
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
        resume_checkpoint: Path | None = None,
    ) -> dict[str, Any]:
        """Execute the full multi-task training loop.

        Args:
            train_loader: DataLoader yielding (x, labels_dict) batches.
            val_loader: DataLoader yielding (x, labels_dict) batches.
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
            self._step_scheduler(val_metrics.get("val_loss_total", float("inf")))

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

    def _move_labels(self, labels: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Move all label tensors to the training device."""
        return {k: v.to(self._device, non_blocking=True) for k, v in labels.items()}

    def _train_one_epoch(self, epoch: int, loader: DataLoader) -> dict[str, float]:
        self._model.train()
        accum: dict[str, float] = {}
        n_batches = 0
        accum_steps = max(1, self._config.training.accumulate_grad_batches)
        n_loader = len(loader)

        for batch_idx, (x, labels) in enumerate(loader):
            x = x.to(self._device, non_blocking=True)
            labels = self._move_labels(labels)

            # Zero grad at the start of each accumulation window
            if batch_idx % accum_steps == 0:
                self._optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=self._device.type,
                enabled=self._config.training.mixed_precision,
            ):
                predictions = self._model.forward_multi_task(x)
                loss_output = self._criterion(predictions, labels)
                # Scale loss for accumulation
                scaled_total = loss_output.total / accum_steps

            self._scaler.scale(scaled_total).backward()

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

            per_head = loss_output.to_dict()
            for k, v in per_head.items():
                accum[k] = accum.get(k, 0.0) + v
            n_batches += 1

            if batch_idx % self._config.training.log_every_n_steps == 0:
                logger.debug(
                    "Epoch {} | Batch {}/{} | Loss {:.4f}",
                    epoch,
                    batch_idx,
                    n_loader,
                    loss_output.total.item(),
                )

        n = max(n_batches, 1)
        return {f"train_{k}": v / n for k, v in accum.items()}

    @torch.no_grad()
    def _validate_one_epoch(self, epoch: int, loader: DataLoader) -> dict[str, float]:
        self._model.eval()

        accum: dict[str, float] = {}
        direction_correct = 0
        direction_total = 0

        move_preds: list[torch.Tensor] = []
        move_targets: list[torch.Tensor] = []
        sl_preds: list[torch.Tensor] = []
        sl_targets: list[torch.Tensor] = []
        tp_preds: list[torch.Tensor] = []
        tp_targets: list[torch.Tensor] = []
        ht_preds: list[torch.Tensor] = []
        ht_targets: list[torch.Tensor] = []
        tp_prob_preds: list[torch.Tensor] = []
        tp_prob_targets: list[torch.Tensor] = []

        n_batches = 0

        for x, labels in loader:
            x = x.to(self._device, non_blocking=True)
            labels = self._move_labels(labels)

            with torch.autocast(
                device_type=self._device.type,
                enabled=self._config.training.mixed_precision,
            ):
                predictions = self._model.forward_multi_task(x)
                loss_output = self._criterion(predictions, labels)

            per_head = loss_output.to_dict()
            for k, v in per_head.items():
                accum[k] = accum.get(k, 0.0) + v
            n_batches += 1

            preds_cls = predictions["direction_logits"].argmax(dim=-1)
            direction_correct += (preds_cls == labels["y_direction"].long()).sum().item()
            direction_total += preds_cls.shape[0]

            if "expected_move" in predictions and "y_move" in labels:
                move_preds.append(predictions["expected_move"].cpu())
                move_targets.append(labels["y_move"].float().cpu())
            if "stop_loss" in predictions and "y_sl" in labels:
                sl_preds.append(predictions["stop_loss"].cpu())
                sl_targets.append(labels["y_sl"].float().cpu())
            if "take_profit" in predictions and "y_tp" in labels:
                tp_preds.append(predictions["take_profit"].cpu())
                tp_targets.append(labels["y_tp"].float().cpu())
            if "holding_time" in predictions and "y_holding_time" in labels:
                ht_preds.append(predictions["holding_time"].cpu())
                ht_targets.append(labels["y_holding_time"].float().cpu())
            if "tp_probability" in predictions and "y_tp_prob" in labels:
                tp_prob_preds.append(predictions["tp_probability"].cpu())
                tp_prob_targets.append(labels["y_tp_prob"].float().cpu())

        n = max(n_batches, 1)
        metrics: dict[str, float] = {f"val_{k}": v / n for k, v in accum.items()}

        if direction_total > 0:
            metrics["val_direction_accuracy"] = direction_correct / direction_total

        def _mae_rmse(preds: list, targets: list, prefix: str) -> None:
            if preds:
                p = torch.cat(preds).squeeze(-1)
                t = torch.cat(targets)
                if t.dim() > 1:
                    t = t.squeeze(-1)
                mae = (p - t).abs().mean().item()
                rmse = ((p - t) ** 2).mean().sqrt().item()
                metrics[f"val_{prefix}_mae"] = mae
                metrics[f"val_{prefix}_rmse"] = rmse

        _mae_rmse(move_preds, move_targets, "move")
        _mae_rmse(sl_preds, sl_targets, "sl")
        _mae_rmse(tp_preds, tp_targets, "tp")
        _mae_rmse(ht_preds, ht_targets, "holding_time")

        if tp_prob_preds:
            try:
                from sklearn.metrics import roc_auc_score

                p_arr = torch.cat(tp_prob_preds).squeeze(-1).numpy()
                t_arr = torch.cat(tp_prob_targets).numpy()
                if t_arr.dim() > 0 and len(set(t_arr.tolist())) > 1:
                    metrics["val_tp_prob_roc_auc"] = float(roc_auc_score(t_arr, p_arr))
            except Exception:
                pass

        return metrics

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
    def _build_base_scheduler(opt: torch.optim.Optimizer, cfg: Any) -> Any:
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
        return torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
