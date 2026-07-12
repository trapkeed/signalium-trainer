"""Unit tests for CheckpointManager."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_torch
import torch
import torch.nn as nn

from trainer.checkpoints.manager import CheckpointManager
from trainer.config.schema import TrainerConfig


class _SimpleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@pytest.fixture()
def model() -> _SimpleNet:
    return _SimpleNet()


@pytest.fixture()
def optimizer(model: _SimpleNet) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=1e-3)


@pytest.fixture()
def scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)


def test_saves_latest(
    tmp_path: Path,
    model: _SimpleNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    manager = CheckpointManager(tmp_path, keep_top_k=3)
    manager.save(0, model, optimizer, scheduler, {"val_loss": 0.5}, TrainerConfig())
    assert (tmp_path / "latest.pt").exists()


def test_saves_best_on_improvement(
    tmp_path: Path,
    model: _SimpleNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    manager = CheckpointManager(tmp_path, monitor_metric="val_loss", monitor_mode="min")
    manager.save(0, model, optimizer, scheduler, {"val_loss": 0.5}, TrainerConfig())
    assert (tmp_path / "best.pt").exists()


def test_rotates_epoch_checkpoints(
    tmp_path: Path,
    model: _SimpleNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    manager = CheckpointManager(tmp_path, keep_top_k=2)
    cfg = TrainerConfig()
    for epoch in range(4):
        manager.save(epoch, model, optimizer, scheduler, {"val_loss": 0.5 - epoch * 0.1}, cfg)
    epoch_files = list(tmp_path.glob("epoch_*.pt"))
    assert len(epoch_files) <= 2


def test_load_restores_weights(
    tmp_path: Path,
    model: _SimpleNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    manager = CheckpointManager(tmp_path)
    manager.save(0, model, optimizer, scheduler, {"val_loss": 0.3}, TrainerConfig())

    new_model = _SimpleNet()
    manager.load(tmp_path / "latest.pt", new_model)
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)
