"""Central model registry using the registry pattern."""

from __future__ import annotations

from typing import Any, Type

from trainer.models.base import BaseQuantModel


class ModelRegistry:
    """Singleton registry mapping architecture names to model classes.

    Usage::

        @ModelRegistry.register("my_model")
        class MyModel(BaseQuantModel):
            ...

        model_cls = ModelRegistry.get("my_model")
        model = model_cls(config)
    """

    _registry: dict[str, Type[BaseQuantModel]] = {}

    @classmethod
    def register(cls, name: str) -> Any:
        """Class decorator that registers a model under the given name.

        Args:
            name: Registry key (e.g. "lstm_classifier").

        Returns:
            Decorator function.
        """
        def decorator(model_cls: Type[BaseQuantModel]) -> Type[BaseQuantModel]:
            if name in cls._registry:
                raise KeyError(f"Model '{name}' is already registered.")
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseQuantModel]:
        """Retrieve a model class by name.

        Args:
            name: Registry key.

        Returns:
            Model class.

        Raises:
            KeyError: If the name is not found in the registry.
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(f"Model '{name}' not found. Available: {available}")
        return cls._registry[name]

    @classmethod
    def list_models(cls) -> list[str]:
        """Return a sorted list of all registered model names."""
        return sorted(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered models (useful in tests)."""
        cls._registry.clear()
