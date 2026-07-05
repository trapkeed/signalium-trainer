"""BaseExporter — framework-level abstract model exporter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseFrameworkExporter(ABC):
    """Abstract base for model exporters in the Signalium Model Framework.

    Each exporter is responsible for serialising a trained model into a
    specific production format (TorchScript, ONNX, etc.) and returning
    the paths of all produced artifacts.

    This is the framework-level contract. Concrete implementations live in
    ``trainer/exporters/``.
    """

    @abstractmethod
    def export(self, output_dir: Path, **kwargs: Any) -> list[Path]:
        """Export the model to one or more artifact files.

        Args:
            output_dir: Directory to write exported files to.
            **kwargs: Exporter-specific options.

        Returns:
            Sorted list of paths to the exported artifact files.
        """

    @abstractmethod
    def validate(self, artifact_path: Path) -> bool:
        """Validate a previously exported artifact.

        Args:
            artifact_path: Path to an artifact produced by :meth:`export`.

        Returns:
            True if the artifact is valid; False otherwise.
        """
