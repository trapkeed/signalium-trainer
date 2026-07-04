"""Model output container and final output layer for QuantModelV1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class QuantModelOutput:
    """Structured output from a QuantModelV1 forward pass.

    All tensors have batch dimension as the first axis.

    Attributes:
        logits: Raw class logits, shape (batch, num_classes). Used for loss.
        probabilities: Softmax class probabilities, shape (batch, num_classes).
        confidence: Scalar confidence per sample, shape (batch, 1), in [0, 1].
        predicted_class: Integer class index, shape (batch,).
        buy_prob: BUY class probability, shape (batch,).
        sell_prob: SELL class probability, shape (batch,).
        no_trade_prob: NO TRADE class probability, shape (batch,).
    """

    logits: torch.Tensor
    probabilities: torch.Tensor
    confidence: torch.Tensor
    predicted_class: torch.Tensor
    buy_prob: torch.Tensor
    sell_prob: torch.Tensor
    no_trade_prob: torch.Tensor

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Return all outputs as a plain dict (e.g. for logging or export)."""
        return {
            "logits": self.logits,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
            "predicted_class": self.predicted_class,
            "buy_prob": self.buy_prob,
            "sell_prob": self.sell_prob,
            "no_trade_prob": self.no_trade_prob,
        }


class OutputLayer(nn.Module):
    """Final output layer: applies softmax and assembles QuantModelOutput.

    This layer is stateless — it only transforms logits and confidence
    into the final structured output.

    Args:
        label_names: List of class labels in order [BUY, SELL, NO_TRADE].
    """

    CLASS_BUY = 0
    CLASS_SELL = 1
    CLASS_NO_TRADE = 2

    def __init__(self, label_names: list[str] | None = None) -> None:
        super().__init__()
        self._label_names = label_names or ["BUY", "SELL", "NO_TRADE"]
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        logits: torch.Tensor,
        confidence: torch.Tensor,
    ) -> QuantModelOutput:
        """Assemble the final model output.

        Args:
            logits: Raw class logits of shape (batch, num_classes).
            confidence: Confidence scores of shape (batch, 1).

        Returns:
            QuantModelOutput with all fields populated.
        """
        probs = self.softmax(logits)
        predicted = probs.argmax(dim=-1)

        return QuantModelOutput(
            logits=logits,
            probabilities=probs,
            confidence=confidence,
            predicted_class=predicted,
            buy_prob=probs[:, self.CLASS_BUY],
            sell_prob=probs[:, self.CLASS_SELL],
            no_trade_prob=probs[:, self.CLASS_NO_TRADE],
        )
