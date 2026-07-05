"""Model output containers for QuantModelV1 — Phase 14.5 multi-task extension.

Backward compatibility is fully preserved:
  - ``QuantModelOutput`` retains all original fields.
  - ``MultiTaskQuantModelOutput`` extends it with optional per-head outputs.
    Fields are None when the corresponding head is disabled in config.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Original output (unchanged — backward compatible)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Phase 14.5 — Multi-task output container
# ---------------------------------------------------------------------------


@dataclass
class MultiTaskQuantModelOutput:
    """Extended structured output for the Phase 14.5 multi-task model.

    This is a superset of ``QuantModelOutput``.  Every field from the
    original output is present.  The additional per-head fields are
    ``None`` when the corresponding head is disabled via config.

    Inference contract (all per-sample, batch dimension first):

        Direction:
            logits          (batch, 3)   — raw direction logits
            probabilities   (batch, 3)   — softmax direction probs
            predicted_class (batch,)     — argmax class index
            buy_prob        (batch,)     — P(BUY)
            sell_prob       (batch,)     — P(SELL)
            no_trade_prob   (batch,)     — P(NO TRADE)

        Quantitative heads:
            confidence      (batch, 1)   — model certainty in [0, 1]
            expected_move   (batch, 1)   — predicted price move
            stop_loss       (batch, 1)   — recommended SL distance (≥0)
            take_profit     (batch, 1)   — recommended TP distance (≥0)
            holding_time    (batch, 1)   — expected duration (≥0)
            tp_probability  (batch, 1)   — P(TP reached before SL) in [0,1]

        Metadata:
            prediction_timestamp — UTC ISO-8601 timestamp
            model_version        — semantic version string
    """

    logits: torch.Tensor
    probabilities: torch.Tensor
    confidence: torch.Tensor
    predicted_class: torch.Tensor
    buy_prob: torch.Tensor
    sell_prob: torch.Tensor
    no_trade_prob: torch.Tensor

    expected_move: Optional[torch.Tensor] = None
    stop_loss: Optional[torch.Tensor] = None
    take_profit: Optional[torch.Tensor] = None
    holding_time: Optional[torch.Tensor] = None
    tp_probability: Optional[torch.Tensor] = None

    prediction_timestamp: Optional[str] = None
    model_version: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return all non-None outputs as a plain dict."""
        result: dict[str, Any] = {
            "logits": self.logits,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
            "predicted_class": self.predicted_class,
            "buy_prob": self.buy_prob,
            "sell_prob": self.sell_prob,
            "no_trade_prob": self.no_trade_prob,
        }
        if self.expected_move is not None:
            result["expected_move"] = self.expected_move
        if self.stop_loss is not None:
            result["stop_loss"] = self.stop_loss
        if self.take_profit is not None:
            result["take_profit"] = self.take_profit
        if self.holding_time is not None:
            result["holding_time"] = self.holding_time
        if self.tp_probability is not None:
            result["tp_probability"] = self.tp_probability
        if self.prediction_timestamp is not None:
            result["prediction_timestamp"] = self.prediction_timestamp
        if self.model_version is not None:
            result["model_version"] = self.model_version
        return result

    def to_base_output(self) -> QuantModelOutput:
        """Down-cast to the original QuantModelOutput for backward compatibility."""
        return QuantModelOutput(
            logits=self.logits,
            probabilities=self.probabilities,
            confidence=self.confidence,
            predicted_class=self.predicted_class,
            buy_prob=self.buy_prob,
            sell_prob=self.sell_prob,
            no_trade_prob=self.no_trade_prob,
        )

    @staticmethod
    def _now_utc() -> str:
        return datetime.now(tz=timezone.utc).isoformat()
