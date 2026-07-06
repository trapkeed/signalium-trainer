"""InferenceAdapter — wraps trained models with a unified predict interface.

Supports PyTorch eager mode, TorchScript, and ONNX Runtime backends
behind a single ``predict()`` call.

Phase 14.5: PyTorchInferenceAdapter is updated to surface all multi-task
outputs from QuantModelV1 when they are available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from trainer.framework.abstractions.inference import BaseInferenceAdapter
from trainer.framework.contracts.model_input import ModelInput
from trainer.framework.contracts.model_output import ModelOutput


class PyTorchInferenceAdapter(BaseInferenceAdapter):
    """Inference adapter for PyTorch eager-mode models (``BaseQuantModel``).

    Phase 14.5 update: when the model returns a ``MultiTaskQuantModelOutput``
    (i.e. it has additional heads enabled), all non-None fields are included
    in the returned dict.  Callers that only use the original fields see no
    change.

    Args:
        model: A loaded ``BaseQuantModel`` instance.
        device: Device for inference. Defaults to CPU.
        confidence_threshold: Optional minimum confidence to gate predictions.
    """

    def __init__(
        self,
        model: Any,
        device: Optional[torch.device] = None,
        confidence_threshold: Optional[float] = None,
    ) -> None:
        self._model = model
        self._device = device or torch.device("cpu")
        self._threshold = confidence_threshold
        self._loaded = True
        self._model.to(self._device)
        self._model.eval()

    def load(self, artifact_path: Path) -> None:
        """Not used for eager-mode adapter — model is passed at construction."""
        raise NotImplementedError(
            "PyTorchInferenceAdapter receives a live model at construction time. "
            "Use TorchScriptInferenceAdapter to load from a .pt file."
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, Any]:
        """Run inference on a batch.

        Returns the full inference contract:
            buy_prob, sell_prob, no_trade_prob, confidence, predicted_class,
            plus (when enabled): expected_move, stop_loss, take_profit,
            holding_time, tp_probability, prediction_timestamp, model_version.

        Args:
            x: Input tensor of shape (batch, seq_len, feature_dim).

        Returns:
            Dict with all available prediction fields.
        """
        x = x.to(self._device)
        out = self._model.predict(x)

        # If this is a MultiTaskQuantModelOutput, use to_dict() directly
        if hasattr(out, "to_dict") and hasattr(out, "expected_move"):
            result_dict = out.to_dict()
            # Apply confidence gate to direction if threshold set
            if self._threshold is not None:
                from trainer.framework.contracts.model_output import (
                    CLASS_NO_TRADE,
                    ModelOutput,
                )

                base = ModelOutput(
                    logits=out.logits,
                    probabilities=out.probabilities,
                    confidence=out.confidence,
                    predicted_class=out.predicted_class,
                )
                gated = base.gate_by_confidence(self._threshold)
                result_dict["predicted_class"] = gated.predicted_class
                result_dict["probabilities"] = gated.probabilities
            return result_dict

        # Legacy path: original QuantModelOutput
        result = ModelOutput(
            logits=out.logits,
            probabilities=out.probabilities,
            confidence=out.confidence,
            predicted_class=out.predicted_class,
        )
        if self._threshold is not None:
            result = result.gate_by_confidence(self._threshold)
        return result.to_dict()


class TorchScriptInferenceAdapter(BaseInferenceAdapter):
    """Inference adapter for TorchScript (.pt) model files.

    Args:
        device: Device for inference. Defaults to CPU.
        confidence_threshold: Optional minimum confidence to gate predictions.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        confidence_threshold: Optional[float] = None,
    ) -> None:
        self._device = device or torch.device("cpu")
        self._threshold = confidence_threshold
        self._model: Optional[Any] = None

    def load(self, artifact_path: Path) -> None:
        """Load a TorchScript model from disk.

        Args:
            artifact_path: Path to a ``.pt`` TorchScript file.
        """
        self._model = torch.jit.load(str(artifact_path), map_location=self._device)
        self._model.eval()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, Any]:
        """Run inference through TorchScript.

        Args:
            x: Input tensor of shape (batch, seq_len, feature_dim).

        Returns:
            Dict with logits, probabilities (softmax), confidence (uniform 1.0),
            and predicted_class.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        x = x.to(self._device)
        logits = self._model(x)
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        confidence = torch.ones(x.shape[0], 1, device=self._device)

        result = ModelOutput(
            logits=logits,
            probabilities=probs,
            confidence=confidence,
            predicted_class=pred,
        )
        if self._threshold is not None:
            result = result.gate_by_confidence(self._threshold)
        return result.to_dict()


class InferenceAdapterFactory:
    """Creates the correct InferenceAdapter from an artifact path or model instance.

    Usage::

        # From a live PyTorch model (with full multi-task outputs)
        adapter = InferenceAdapterFactory.from_model(model)

        # From a TorchScript .pt file
        adapter = InferenceAdapterFactory.from_artifact(Path("exports/model.torchscript.pt"))

    """

    @staticmethod
    def from_model(
        model: Any,
        device: Optional[torch.device] = None,
        confidence_threshold: Optional[float] = None,
    ) -> PyTorchInferenceAdapter:
        """Create a PyTorchInferenceAdapter from a live model.

        Args:
            model: Trained BaseQuantModel instance.
            device: Inference device (default: CPU).
            confidence_threshold: Optional confidence gate.

        Returns:
            Configured PyTorchInferenceAdapter.
        """
        return PyTorchInferenceAdapter(
            model=model,
            device=device,
            confidence_threshold=confidence_threshold,
        )

    @staticmethod
    def from_artifact(
        artifact_path: Path,
        device: Optional[torch.device] = None,
        confidence_threshold: Optional[float] = None,
    ) -> BaseInferenceAdapter:
        """Create and load an adapter from an artifact file.

        Dispatches to the correct adapter based on file extension:
        - ``.pt`` → TorchScriptInferenceAdapter
        - ``.onnx`` → (future) ONNXInferenceAdapter

        Args:
            artifact_path: Path to the serialised model artifact.
            device: Inference device (default: CPU).
            confidence_threshold: Optional confidence gate.

        Returns:
            Loaded BaseInferenceAdapter.
        """
        suffix = artifact_path.suffix.lower()
        if suffix == ".pt":
            adapter: BaseInferenceAdapter = TorchScriptInferenceAdapter(
                device=device, confidence_threshold=confidence_threshold
            )
            adapter.load(artifact_path)
            return adapter
        raise ValueError(
            f"Unsupported artifact format: '{suffix}'. "
            "Supported: .pt (TorchScript). ONNX support coming in a future release."
        )
