"""BaseInferenceAdapter — framework-level abstract inference interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch


class BaseInferenceAdapter(ABC):
    """Abstract base for inference adapters in the Signalium Model Framework.

    An inference adapter wraps a trained model (PyTorch, TorchScript, or ONNX)
    and exposes a unified ``predict`` interface. This decouples downstream
    consumers from the model's internal format.

    Concrete adapters handle:
    - Input pre-processing (normalisation, batching, device placement)
    - Model invocation (PyTorch eager / TorchScript / ONNX Runtime)
    - Output post-processing (softmax, class mapping, confidence gating)
    """

    @abstractmethod
    def load(self, artifact_path: Path) -> None:
        """Load a model artifact from disk.

        Args:
            artifact_path: Path to the serialised model artifact.
        """

    @abstractmethod
    def predict(self, x: torch.Tensor) -> dict[str, Any]:
        """Run inference on a batch of feature windows.

        Args:
            x: Input tensor of shape (batch, seq_len, feature_dim).

        Returns:
            Dict with at minimum:
                - ``probabilities``: class probability tensor (batch, num_classes)
                - ``confidence``:    confidence tensor (batch, 1)
                - ``predicted_class``: integer class index tensor (batch,)
        """

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """True if the model artifact has been successfully loaded."""
