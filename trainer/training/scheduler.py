"""Warmup and composite LR scheduler utilities — Phase 14.9B.

Provides a ``WarmupScheduler`` wrapper that prepends a linear warmup phase
to any existing PyTorch LR scheduler, including ``ReduceLROnPlateau``.

Usage::

    base = CosineAnnealingLR(optimizer, T_max=100)
    scheduler = WarmupScheduler(
        optimizer=optimizer,
        warmup_epochs=5,
        base_scheduler=base,
        start_lr=1e-6,
    )
    # Every epoch:
    scheduler.step()              # non-plateau schedulers
    scheduler.step(metric=0.42)   # plateau-aware step
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.optim as optim
from loguru import logger


class WarmupScheduler:
    """Wraps any LR scheduler with a linear warmup phase.

    During the first ``warmup_epochs`` epochs, the learning rate grows
    linearly from ``start_lr`` to the optimizer's base LR.  After warmup,
    the ``base_scheduler`` takes over.

    Handles ``ReduceLROnPlateau`` transparently — call ``step(metric=...)``
    for plateau-aware stepping, ``step()`` otherwise.

    Args:
        optimizer: The optimizer being wrapped.
        warmup_epochs: Number of epochs for the linear warmup phase.
        base_scheduler: The scheduler to activate after warmup.
        start_lr: Starting LR at epoch 0 of warmup.
            Defaults to ``base_lr * 0.01``.
        is_plateau: True when ``base_scheduler`` is ``ReduceLROnPlateau``.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        base_scheduler: Any,
        start_lr: Optional[float] = None,
        is_plateau: bool = False,
    ) -> None:
        self._optimizer = optimizer
        self._warmup_epochs = warmup_epochs
        self._base = base_scheduler
        self._is_plateau = is_plateau
        self._epoch = 0

        # Base LR is the LR the optimizer was configured with
        self._base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._start_lrs = [
            (start_lr if start_lr is not None else lr * 0.01) for lr in self._base_lrs
        ]

        # Set initial LR to start_lr if warmup active
        if warmup_epochs > 0:
            self._set_lrs(self._start_lrs)
            logger.info(
                "WarmupScheduler: {}-epoch linear warmup active (start_lr={:.2e})",
                warmup_epochs,
                self._start_lrs[0] if self._start_lrs else 0.0,
            )

    # ------------------------------------------------------------------
    # Public step API
    # ------------------------------------------------------------------

    def step(self, metric: Optional[float] = None) -> None:
        """Advance the scheduler by one epoch.

        During warmup, applies linear LR interpolation.
        After warmup, delegates to ``base_scheduler``.

        Args:
            metric: Required only when ``is_plateau=True`` and past warmup.
        """
        self._epoch += 1

        if self._epoch <= self._warmup_epochs:
            self._warmup_step()
        else:
            if self._is_plateau:
                if metric is not None:
                    self._base.step(metric)
            else:
                self._base.step()

    def get_last_lr(self) -> list[float]:
        """Return current learning rates per param group."""
        return [pg["lr"] for pg in self._optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        """Serialize scheduler state for checkpointing."""
        state: dict[str, Any] = {
            "epoch": self._epoch,
            "warmup_epochs": self._warmup_epochs,
            "base_lrs": self._base_lrs,
            "start_lrs": self._start_lrs,
            "is_plateau": self._is_plateau,
        }
        if hasattr(self._base, "state_dict"):
            state["base_state"] = self._base.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore scheduler state from a checkpoint."""
        self._epoch = state["epoch"]
        self._warmup_epochs = state.get("warmup_epochs", self._warmup_epochs)
        self._base_lrs = state.get("base_lrs", self._base_lrs)
        self._start_lrs = state.get("start_lrs", self._start_lrs)
        if "base_state" in state and hasattr(self._base, "load_state_dict"):
            self._base.load_state_dict(state["base_state"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _warmup_step(self) -> None:
        """Linearly interpolate LR from start_lr toward base_lr."""
        progress = self._epoch / max(self._warmup_epochs, 1)  # 0 → 1
        new_lrs = [
            start + progress * (base - start)
            for start, base in zip(self._start_lrs, self._base_lrs)
        ]
        self._set_lrs(new_lrs)

    def _set_lrs(self, lrs: list[float]) -> None:
        for pg, lr in zip(self._optimizer.param_groups, lrs):
            pg["lr"] = lr


def build_warmup_scheduler(
    optimizer: optim.Optimizer,
    base_scheduler: Any,
    warmup_epochs: int,
    is_plateau: bool = False,
) -> WarmupScheduler:
    """Factory — wrap a base scheduler with linear warmup.

    Args:
        optimizer: The optimizer being wrapped.
        base_scheduler: Any PyTorch LR scheduler.
        warmup_epochs: Number of warmup epochs (0 = no warmup).
        is_plateau: True when base_scheduler is ReduceLROnPlateau.

    Returns:
        ``WarmupScheduler`` wrapping ``base_scheduler``.
    """
    return WarmupScheduler(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        base_scheduler=base_scheduler,
        is_plateau=is_plateau,
    )
