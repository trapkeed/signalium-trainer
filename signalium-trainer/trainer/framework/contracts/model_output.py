"""ModelOutput — standardised output contract for all Signalium models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch


# Class index constants — shared across all Signalium classifiers
CLASS_BUY = 0
CLASS_SELL = 1
CLASS_NO_TRADE = 2
CLASS_LABELS = ["BUY", "SELL", "NO_TRADE"]


@dataclass
class ModelOutput:
    """Standardised output container for all Signalium model inference calls.

    Every model registered in the framework must produce a ``ModelOutput``
    (or a subclass) from its ``predict()`` method. This contract guarantees
    that downstream consumers — dashboards, risk systems, evaluation pipelines —
    can consume any model's output without model-specific logic.

    Attributes:
        logits: Raw class logits of shape (batch, num_classes).
                Used for loss computation during training.
        probabilities: Softmax class probabilities of shape (batch, num_classes).
                       Sum to 1.0 along the class dimension.
        confidence: Per-sample confidence score of shape (batch, 1), in [0, 1].
        predicted_class: Integer class index of shape (batch,).
                         0=BUY, 1=SELL, 2=NO_TRADE by convention.
        auxiliary: Optional dict for model-specific extra outputs
                   (e.g. attention weights, per-class named probabilities).
    """

    logits: torch.Tensor
    probabilities: torch.Tensor
    confidence: torch.Tensor
    predicted_class: torch.Tensor
    auxiliary: Optional[dict[str, Any]] = None

    @property
    def buy_prob(self) -> torch.Tensor:
        """BUY class probability for each sample in the batch."""
        return self.probabilities[:, CLASS_BUY]

    @property
    def sell_prob(self) -> torch.Tensor:
        """SELL class probability for each sample in the batch."""
        return self.probabilities[:, CLASS_SELL]

    @property
    def no_trade_prob(self) -> torch.Tensor:
        """NO TRADE class probability for each sample in the batch."""
        return self.probabilities[:, CLASS_NO_TRADE]

    @property
    def predicted_label(self) -> list[str]:
        """Human-readable class labels for each sample in the batch."""
        return [CLASS_LABELS[i] for i in self.predicted_class.tolist()]

    def to_dict(self) -> dict[str, Any]:
        """Return all outputs as a plain dict (e.g. for logging or serialisation).

        Returns:
            Dict with tensor values and the auxiliary sub-dict flattened in.
        """
        result: dict[str, Any] = {
            "logits": self.logits,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
            "predicted_class": self.predicted_class,
            "buy_prob": self.buy_prob,
            "sell_prob": self.sell_prob,
            "no_trade_prob": self.no_trade_prob,
        }
        if self.auxiliary:
            result.update(self.auxiliary)
        return result

    def gate_by_confidence(self, threshold: float) -> "ModelOutput":
        """Return a new ModelOutput containing only samples above the threshold.

        Samples below the threshold are replaced with NO_TRADE predictions.
        Useful for downstream systems that only act on high-confidence signals.

        Args:
            threshold: Minimum confidence value in [0, 1].

        Returns:
            New ModelOutput with low-confidence samples mapped to NO_TRADE.
        """
        mask = self.confidence.squeeze(-1) >= threshold
        new_predicted = torch.where(mask, self.predicted_class,
                                    torch.full_like(self.predicted_class, CLASS_NO_TRADE))
        return ModelOutput(
            logits=self.logits,
            probabilities=self.probabilities,
            confidence=self.confidence,
            predicted_class=new_predicted,
            auxiliary=self.auxiliary,
        )
