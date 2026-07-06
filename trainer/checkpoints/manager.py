"""CheckpointManager — Phase 14.9B institutional-grade checkpoint management.

Saves and rotates training checkpoints with complete state capture:
  - Model state dict
  - Optimizer state dict
  - Scheduler state dict
  - Full training metrics
  - RNG state (Python / NumPy / PyTorch / CUDA) for exact resume
  - Serialised training config
  - Epoch, timestamp, version tag

Rotation policy:
  - ``latest.pt`` — always the most recent checkpoint (symlink on POSIX,
    copy on Windows).
  - ``best.pt``   — checkpoint with the best monitored metric so far.
  - ``epoch_{n}.pt`` — per-epoch file; only the top-k are kept on disk.

Usage::

    manager = CheckpointManager(
        checkpoint_dir=Path("checkpoints/run_001"),
        keep_top_k=3,
        monitor_metric="val_loss",
        monitor_mode="min",
    )

    # During training
    manager.save(
        epoch=5,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics={"val_loss": 0.42, "val_accuracy": 0.87},
        config=trainer_config,
    )

    # Resuming
    state = manager.load(
        Path("checkpoints/run_001/best.pt"),
        model, optimizer, scheduler,
    )
    start_epoch = state["epoch"] + 1
"""

from __future__ import annotations

import heapq
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger

_CHECKPOINT_VERSION = "1.1.0"


class CheckpointManager:
    """Manages model checkpoints with best/latest/top-k rotation.

    Args:
        checkpoint_dir: Directory where checkpoint files are written.
        keep_top_k: Number of per-epoch checkpoints to retain on disk.
            The ``best.pt`` and ``latest.pt`` special files are never
            counted toward this limit and are never deleted.
        monitor_metric: Metric key used to rank checkpoints
            (e.g. "val_loss", "val_direction_accuracy").
        monitor_mode: "min" (lower is better) or "max" (higher is better).
        save_optimizer_state: Whether to include optimizer state in saves.
        save_scheduler_state: Whether to include scheduler state in saves.
        save_rng_state: Whether to capture RNG state for exact resume.
        save_training_config: Whether to serialise the training config.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        keep_top_k: int = 3,
        monitor_metric: str = "val_loss",
        monitor_mode: str = "min",
        save_optimizer_state: bool = True,
        save_scheduler_state: bool = True,
        save_rng_state: bool = True,
        save_training_config: bool = True,
    ) -> None:
        if monitor_mode not in ("min", "max"):
            raise ValueError(f"monitor_mode must be 'min' or 'max', got '{monitor_mode}'")

        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._keep_top_k = keep_top_k
        self._monitor_metric = monitor_metric
        self._monitor_mode = monitor_mode
        self._save_optimizer = save_optimizer_state
        self._save_scheduler = save_scheduler_state
        self._save_rng = save_rng_state
        self._save_config = save_training_config

        # Min-heap when mode=="max" (negate scores), max-heap proxy otherwise.
        # Each entry: (score, path_str) — lower score = worse when mode=="min".
        self._top_k_heap: list[tuple[float, str]] = []
        self._best_score: float | None = None
        self._best_metrics: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def best_metrics(self) -> dict[str, float]:
        """Metrics of the best checkpoint seen so far."""
        return dict(self._best_metrics)

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        metrics: dict[str, float],
        config: Any = None,
    ) -> Path:
        """Persist a training checkpoint to disk.

        Automatically updates ``latest.pt`` every call, and ``best.pt``
        when the monitored metric improves.  Enforces the top-k rotation
        policy, deleting the worst checkpoint when the queue is full.

        Args:
            epoch: Current epoch number (0-indexed).
            model: Model whose ``state_dict`` is saved.
            optimizer: Optimizer whose ``state_dict`` is saved.
            scheduler: LR scheduler whose ``state_dict`` is saved.
            metrics: Dict of float-valued metric values for this epoch.
            config: Optional training config (serialised to JSON).

        Returns:
            ``Path`` to the newly written epoch checkpoint file.
        """
        monitor_value = metrics.get(self._monitor_metric, float("inf"))
        is_best = self._is_improvement(monitor_value)

        state = self._build_state(epoch, model, optimizer, scheduler, metrics, config)

        # ── Write per-epoch file ─────────────────────────────────────
        ckpt_path = self._dir / f"epoch_{epoch:04d}.pt"
        torch.save(state, ckpt_path)
        logger.info(
            "Checkpoint saved → {} | {}={:.6f}{}",
            ckpt_path.name,
            self._monitor_metric,
            monitor_value,
            " ✓ new best" if is_best else "",
        )

        # ── Update latest ────────────────────────────────────────────
        self._link_or_copy(ckpt_path, self._dir / "latest.pt")

        # ── Update best ──────────────────────────────────────────────
        if is_best:
            self._best_score = monitor_value
            self._best_metrics = dict(metrics)
            self._link_or_copy(ckpt_path, self._dir / "best.pt")
            logger.info("New best {} = {:.6f} → best.pt", self._monitor_metric, monitor_value)

        # ── Top-k rotation ───────────────────────────────────────────
        # Store negated score so heappop() always removes the worst checkpoint.
        # For "min" mode: worse = higher value → negate to make it smallest.
        # For "max" mode: worse = lower value  → negate to make it smallest.
        heap_score = -monitor_value if self._monitor_mode == "min" else monitor_value
        heapq.heappush(self._top_k_heap, (heap_score, str(ckpt_path)))

        while len(self._top_k_heap) > self._keep_top_k:
            _, worst_path_str = heapq.heappop(self._top_k_heap)
            worst_path = Path(worst_path_str)
            # Never delete best.pt or latest.pt targets
            best_target = self._resolve_link(self._dir / "best.pt")
            latest_target = self._resolve_link(self._dir / "latest.pt")
            if worst_path not in {best_target, latest_target}:
                if worst_path.exists():
                    worst_path.unlink()
                    logger.debug("Rotated out checkpoint: {}", worst_path.name)

        return ckpt_path

    def load(
        self,
        checkpoint_path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
    ) -> dict[str, Any]:
        """Load a checkpoint and restore all states in-place.

        Restores model weights, optimizer state, scheduler state, and
        optionally RNG state for bit-exact resume.

        Args:
            checkpoint_path: Path to the ``*.pt`` checkpoint file.
            model: Model to restore weights into.
            optimizer: Optimizer to restore state into.
            scheduler: Scheduler to restore state into.

        Returns:
            Full checkpoint dict (contains ``"epoch"``, ``"metrics"``,
            ``"config"``, etc.).
        """
        logger.info("Loading checkpoint: {}", checkpoint_path)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        model.load_state_dict(state["model_state_dict"])
        logger.debug("Model weights restored from epoch {}", state.get("epoch", "?"))

        if "optimizer_state_dict" in state and optimizer is not None:
            try:
                optimizer.load_state_dict(state["optimizer_state_dict"])
                logger.debug("Optimizer state restored.")
            except Exception as exc:
                logger.warning("Could not restore optimizer state: {}", exc)

        if "scheduler_state_dict" in state and scheduler is not None:
            sched_state = state["scheduler_state_dict"]
            if sched_state is not None:
                try:
                    scheduler.load_state_dict(sched_state)
                    logger.debug("Scheduler state restored.")
                except Exception as exc:
                    logger.warning("Could not restore scheduler state: {}", exc)

        if state.get("rng_state"):
            try:
                from trainer.training.reproducibility import ReproducibilityManager

                ReproducibilityManager.restore_rng_state(state["rng_state"])
                logger.debug("RNG state restored.")
            except Exception as exc:
                logger.warning("Could not restore RNG state: {}", exc)

        # Refresh internal best-score tracking from checkpoint
        saved_metrics = state.get("metrics", {})
        monitor_val = saved_metrics.get(self._monitor_metric)
        if monitor_val is not None:
            if self._is_improvement(monitor_val):
                self._best_score = monitor_val
                self._best_metrics = dict(saved_metrics)

        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_state(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        metrics: dict[str, float],
        config: Any,
    ) -> dict[str, Any]:
        """Assemble the full checkpoint state dict."""
        state: dict[str, Any] = {
            "version": _CHECKPOINT_VERSION,
            "epoch": epoch,
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "monitor_metric": self._monitor_metric,
            "monitor_mode": self._monitor_mode,
            "metrics": dict(metrics),
            "model_state_dict": model.state_dict(),
        }

        if self._save_optimizer:
            try:
                state["optimizer_state_dict"] = optimizer.state_dict()
            except Exception as exc:
                logger.warning("Could not capture optimizer state: {}", exc)

        if self._save_scheduler and scheduler is not None:
            try:
                state["scheduler_state_dict"] = scheduler.state_dict()
            except Exception as exc:
                logger.debug(
                    "Could not capture scheduler state ({}): {}", type(scheduler).__name__, exc
                )
                state["scheduler_state_dict"] = None

        if self._save_rng:
            try:
                from trainer.training.reproducibility import ReproducibilityManager

                state["rng_state"] = ReproducibilityManager.capture_rng_state()
            except Exception as exc:
                logger.warning("Could not capture RNG state: {}", exc)

        if self._save_config and config is not None:
            try:
                state["config"] = _serialise_config(config)
            except Exception as exc:
                logger.warning("Could not serialise config: {}", exc)

        return state

    def _is_improvement(self, value: float) -> bool:
        if self._best_score is None:
            return True
        if self._monitor_mode == "min":
            return value < self._best_score
        return value > self._best_score

    @staticmethod
    def _link_or_copy(src: Path, dst: Path) -> None:
        """Create a symlink dst → src, or copy if symlinks are unsupported."""
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.name)  # relative symlink
        except (OSError, NotImplementedError):
            # Windows or restricted filesystem — fall back to copy
            shutil.copy2(src, dst)

    @staticmethod
    def _resolve_link(link: Path) -> Path | None:
        """Resolve a symlink to its target, or return None."""
        try:
            if link.is_symlink():
                return link.parent / link.readlink()
            return link.resolve()
        except Exception:
            return None


def _serialise_config(config: Any) -> dict[str, Any]:
    """Serialise a config object (dataclass, dict, or primitive) to a plain dict."""
    if hasattr(config, "__dataclass_fields__"):
        result: dict[str, Any] = {}
        for k in config.__dataclass_fields__:
            v = getattr(config, k)
            result[k] = _serialise_config(v)
        return result
    if isinstance(config, dict):
        return {str(k): _serialise_config(v) for k, v in config.items()}
    if isinstance(config, (list, tuple)):
        return [_serialise_config(v) for v in config]
    if hasattr(config, "value"):  # Enum
        return config.value
    return config
