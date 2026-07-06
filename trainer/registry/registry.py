"""Central model registry with class registration and rich ModelSpec metadata."""

from __future__ import annotations

from typing import Any

from trainer.models.base import BaseQuantModel


class ModelRegistry:
    """Singleton registry mapping architecture names to model classes and specs.

    Supports two registration paths:

    1. Simple decorator (backward compatible)::

        @ModelRegistry.register("my_model")
        class MyModel(BaseQuantModel): ...

    2. Decorator with full ModelSpec::

        from trainer.framework.contracts import ModelSpec, FeatureSchema, OutputSchema

        @ModelRegistry.register(
            "my_model",
            spec=ModelSpec(
                name="my_model",
                version="1.0.0",
                architecture="MyArch",
                input_schema=FeatureSchema(feature_dim=128, seq_len=60),
                output_schema=OutputSchema(num_classes=3),
            ),
        )
        class MyModel(BaseQuantModel): ...

    Retrieval::

        cls = ModelRegistry.get("my_model")
        spec = ModelRegistry.get_spec("my_model")
    """

    _registry: dict[str, type[BaseQuantModel]] = {}
    _specs: dict[str, Any] = {}  # stores ModelSpec objects (typed as Any to avoid circular import)

    @classmethod
    def register(
        cls,
        name: str,
        spec: Any | None = None,
    ) -> Any:
        """Class decorator that registers a model under the given name.

        Args:
            name: Unique registry key (e.g. ``"quant_v1"``).
            spec: Optional ``ModelSpec`` with rich metadata about the model.
                  If omitted the model is registered with no metadata.

        Returns:
            Decorator function.
        """

        def decorator(model_cls: type[BaseQuantModel]) -> type[BaseQuantModel]:
            if name in cls._registry:
                raise KeyError(f"Model '{name}' is already registered.")
            cls._registry[name] = model_cls
            if spec is not None:
                cls._specs[name] = spec
            return model_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseQuantModel]:
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
    def get_spec(cls, name: str) -> Any | None:
        """Retrieve the ModelSpec for a registered model.

        Args:
            name: Registry key.

        Returns:
            ``ModelSpec`` if one was registered, otherwise ``None``.
        """
        return cls._specs.get(name)

    @classmethod
    def list_models(cls) -> list[str]:
        """Return a sorted list of all registered model names."""
        return sorted(cls._registry.keys())

    @classmethod
    def list_specs(cls) -> dict[str, Any]:
        """Return a dict of all models that have registered specs.

        Returns:
            Dict mapping model name → ModelSpec.
        """
        return dict(cls._specs)

    @classmethod
    def has_spec(cls, name: str) -> bool:
        """True if the given model has a registered ModelSpec."""
        return name in cls._specs

    @classmethod
    def summary(cls) -> list[dict[str, Any]]:
        """Return a summary table of all registered models.

        Each entry contains: name, class name, version, architecture.
        Models without a spec show None for version and architecture.

        Returns:
            List of dicts, one per registered model.
        """
        rows = []
        for name in cls.list_models():
            model_cls = cls._registry[name]
            spec = cls._specs.get(name)
            rows.append(
                {
                    "name": name,
                    "class": model_cls.__name__,
                    "version": spec.version if spec else None,
                    "architecture": spec.architecture if spec else None,
                    "has_spec": spec is not None,
                }
            )
        return rows

    @classmethod
    def clear(cls) -> None:
        """Clear all registered models and specs (useful in tests)."""
        cls._registry.clear()
        cls._specs.clear()
