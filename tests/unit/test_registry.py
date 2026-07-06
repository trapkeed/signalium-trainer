"""Unit tests for ModelRegistry."""

from __future__ import annotations

import pytest

from trainer.registry.registry import ModelRegistry


@pytest.fixture(autouse=True)
def clear_registry() -> None:
    """Ensure the registry is empty before and after each test."""
    ModelRegistry.clear()
    yield
    ModelRegistry.clear()


def test_register_and_retrieve() -> None:
    from trainer.models.base import BaseQuantModel
    import torch
    from trainer.config.schema import TrainerConfig

    class _DummyModel(BaseQuantModel):
        @property
        def feature_dim(self) -> int:
            return 8

        @property
        def num_classes(self) -> int:
            return 2

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    ModelRegistry.register("dummy")(_DummyModel)
    retrieved = ModelRegistry.get("dummy")
    assert retrieved is _DummyModel


def test_get_raises_for_unknown() -> None:
    with pytest.raises(KeyError, match="not found"):
        ModelRegistry.get("nonexistent")


def test_duplicate_registration_raises() -> None:
    from trainer.models.base import BaseQuantModel
    import torch

    class _M(BaseQuantModel):
        feature_dim = 4
        num_classes = 2

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    ModelRegistry.register("dup")(_M)
    with pytest.raises(KeyError, match="already registered"):
        ModelRegistry.register("dup")(_M)


def test_list_models() -> None:
    assert ModelRegistry.list_models() == []


def test_clear() -> None:
    from trainer.models.base import BaseQuantModel
    import torch

    class _M2(BaseQuantModel):
        feature_dim = 4
        num_classes = 2

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    ModelRegistry.register("m2")(_M2)
    ModelRegistry.clear()
    assert ModelRegistry.list_models() == []
