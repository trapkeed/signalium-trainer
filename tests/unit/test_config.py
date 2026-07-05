"""Unit tests for configuration schema."""

from __future__ import annotations

import pytest

from trainer.config.schema import (
    Environment,
    OptimizerType,
    SchedulerType,
    TrainerConfig,
)


def test_trainer_config_defaults() -> None:
    """TrainerConfig instantiates with sensible defaults."""
    cfg = TrainerConfig()
    assert cfg.environment == Environment.DEVELOPMENT
    assert cfg.training.epochs == 100
    assert cfg.training.seed == 42
    assert cfg.optimizer.type == OptimizerType.ADAMW
    assert cfg.scheduler.type == SchedulerType.COSINE
    assert cfg.checkpoint.monitor_mode == "min"


def test_api_config_no_credentials() -> None:
    """APIConfig allows None credentials (resolved at runtime)."""
    from trainer.config.schema import APIConfig
    cfg = APIConfig()
    assert cfg.api_key is None
    assert cfg.token is None


def test_gpu_config_auto_device() -> None:
    """Default GPU device is 'auto'."""
    from trainer.config.schema import GPUConfig
    cfg = GPUConfig()
    assert cfg.device == "auto"


def test_optimizer_type_values() -> None:
    """All OptimizerType members are valid strings."""
    for member in OptimizerType:
        assert isinstance(member.value, str)


def test_scheduler_type_none() -> None:
    """SchedulerType.NONE is a valid member."""
    assert SchedulerType.NONE.value == "none"
