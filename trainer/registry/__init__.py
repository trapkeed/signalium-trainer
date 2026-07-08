"""Model registry — maps architecture names to classes.

`ModelRegistry` (in `trainer.registry.registry`) pulls in `BaseQuantModel`,
which imports torch. `trainer.registry.unified_registry` (the QDE/live
model-lineage registry) has no such dependency and must remain importable
even in environments without a working torch install. `ModelRegistry` is
therefore exposed lazily so importing this package — which Python does
automatically for any `trainer.registry.<submodule>` import — never forces
the torch-dependent chain to load.
"""

from typing import Any

__all__ = ["ModelRegistry"]


def __getattr__(name: str) -> Any:
    if name == "ModelRegistry":
        from trainer.registry.registry import ModelRegistry

        return ModelRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
