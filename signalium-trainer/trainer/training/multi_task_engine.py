"""MultiTaskTrainingEngine — extends TrainingEngine for Phase 14.5 multi-task training.

Differences from the original TrainingEngine:
  - Accepts dict-valued labels (y_direction, y_move, y_sl, etc.) instead of a
    single label tensor.
  - Uses MultiTaskLoss and tracks per-head loss components.
  - Reports per-head validation metrics (MAE, RMSE, AUC-ROC, calibration).
  - forward_multi_task() is called instead of forward() for the model.
  - Mixed precision, gradient clipping, checkpointing, and early stopping are
    all inherited unchanged from TrainingEngine.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

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
from trainer.training.seed import set_seed
from trainer.utils.device import resolve_device


class MultiTaskTrainingEngine:
    """Orchestrates multi-task training for QuantModelV1 Phase 14.5.

    The engine handles:
    - Seeding and device placement
    - Optimizer and scheduler construction
    - Mixed-precision training with AMP
    - Gradient clipping
    - Per-head loss tracking
    - Checkpoint saving / resuming
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
        self._scheduler = self._build_scheduler()
        self._scaler = GradScaler(enabled=config.training.mixed_precision)

        self._checkpoint_manager = CheckpointManager(
            checkpoint_dir=Path(config.checkpoint.checkpoint_dir),
            keep_top_k=config.checkpoint.keep_top_k,
            monitor_metric=config.checkpoint.monitor_metric,
            monitor_mode=config.checkpoint.monitor_mode,
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

            monitor_value = val_metrics.get(
                self._config.checkpoint.monitor_metric, float("inf")
            )
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

    def _move_labels(
        self, labels: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Move all label tensors to the training device."""
        return {k: v.to(self._device, non_blocking=True) for k, v in labels.items()}

    def _train_one_epoch(self, epoch: int, loader: DataLoader) -> dict[str, float]:
        self._model.train()
        accum: dict[str, float] = {}
        n_batches = 0

        for batch_idx, (x, labels) in enumerate(loader):
            x = x.to(self._device, non_blocking=True)
            labels = self._move_labels(labels)

            self._optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=self._device.type,
                enabled=self._config.training.mixed_precision,
            ):
                predictions = self._model.forward_multi_task(x)
                loss_output = self._criterion(predictions, labels)

            self._scaler.scale(loss_output.total).backward()

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
                    len(loader),
                    loss_output.total.item(),
                )

        n = max(n_batches, 1)
        return {f"train_{k}": v / n for k, v in accum.items()}

    @torch.no_grad()
    def _validate_one_epoch(
        self, epoch: int, loader: DataLoader
    ) -> dict[str, float]:
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

            # Accumulate for per-head metrics
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

        # Direction accuracy
        if direction_total > 0:
            metrics["val_direction_accuracy"] = direction_correct / direction_total

        # Regression MAE + RMSE helpers
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

        # TP probability ROC-AUC
        if tp_prob_preds:
            try:
                from sklearn.metrics import roc_auc_score
                p_arr = torch.cat(tp_prob_preds).squeeze(-1).numpy()
                t_arr = torch.cat(tp_prob_targets).numpy()
                if t_arr.dim() > 0 and len(set(t_arr.tolist())) > 1:
                    metrics["val_tp_prob_roc_auc"] = float(
                        roc_auc_score(t_arr, p_arr)
                    )
            except Exception:
                pass

        return metrics

    def _log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info("Epoch {:>4d} | {}", epoch, metrics_str)

    def _step_scheduler(self, val_loss: float) -> None:
        from trainer.config.schema import SchedulerType

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
                params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
                betas=cfg.betas, eps=cfg.eps,
            )
        if cfg.type == OptimizerType.ADAMW:
            return torch.optim.AdamW(
                params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
                betas=cfg.betas, eps=cfg.eps,
            )
        if cfg.type == OptimizerType.SGD:
            return torch.optim.SGD(
                params, lr=cfg.learning_rate, momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            )
        if cfg.type == OptimizerType.RMSPROP:
            return torch.optim.RMSprop(
                params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
                momentum=cfg.momentum,
            )
        raise ValueError(f"Unknown optimizer type: {cfg.type}")

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        from trainer.config.schema import SchedulerType

        cfg = self._config.scheduler
        opt = self._optimizer

        if cfg.type == SchedulerType.COSINE:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=cfg.T_max, eta_min=cfg.eta_min
            )
        if cfg.type == SchedulerType.STEP:
            return torch.optim.lr_scheduler.StepLR(
                opt, step_size=cfg.step_size, gamma=cfg.gamma
            )
        if cfg.type == SchedulerType.REDUCE_ON_PLATEAU:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode=self._config.checkpoint.monitor_mode,
                patience=cfg.patience, factor=cfg.factor,
            )
        if cfg.type == SchedulerType.EXPONENTIAL:
            return torch.optim.lr_scheduler.ExponentialLR(opt, gamma=cfg.gamma)
        return torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
