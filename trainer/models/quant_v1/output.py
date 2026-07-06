"""Model output containers for QuantModelV1 — Phase 14.6 Institutional Analytics.

Backward compatibility is fully preserved:
  - ``QuantModelOutput`` retains all original fields (unchanged).
  - ``MultiTaskQuantModelOutput`` is extended with 15 new Phase 14.6 fields.
    All new fields are ``None`` when their head is disabled in config.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

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
        """Return all outputs as a plain dict."""
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
    """Applies softmax to logits and assembles QuantModelOutput.

    Stateless — only transforms logits + confidence into a structured output.
    """

    CLASS_BUY = 0
    CLASS_SELL = 1
    CLASS_NO_TRADE = 2

    def __init__(self, label_names: list[str] | None = None) -> None:
        super().__init__()
        self._label_names = label_names or ["BUY", "SELL", "NO_TRADE"]
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, logits: torch.Tensor, confidence: torch.Tensor) -> QuantModelOutput:
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
# Phase 14.5 + 14.6 — Extended multi-task output container
# ---------------------------------------------------------------------------


@dataclass
class MultiTaskQuantModelOutput:
    """Extended output for the Phase 14.6 Institutional Analytics model.

    Superset of ``QuantModelOutput``.  Every original field is present.
    Phase 14.5 and 14.6 fields are ``None`` when the head is disabled.

    ── Direction (always present) ──────────────────────────────────────────
    logits              (batch, 3)  — raw direction logits (for loss)
    probabilities       (batch, 3)  — softmax direction probabilities
    predicted_class     (batch,)    — argmax class index {0=BUY,1=SELL,2=NO_TRADE}
    buy_prob            (batch,)    — P(BUY)
    sell_prob           (batch,)    — P(SELL)
    no_trade_prob       (batch,)    — P(NO TRADE)

    ── Phase 14.5 heads (optional) ─────────────────────────────────────────
    confidence          (batch, 1)  — model certainty ∈ [0, 1]
    expected_move       (batch, 1)  — predicted price move (signed)
    stop_loss           (batch, 1)  — SL distance ≥ 0
    take_profit         (batch, 1)  — TP distance ≥ 0
    holding_time        (batch, 1)  — expected duration ≥ 0
    tp_probability      (batch, 1)  — P(TP before SL) ∈ [0, 1]

    ── Phase 14.6 Institutional heads (optional) ────────────────────────────
    mfe                 (batch, 1)  — max favorable excursion ≥ 0
    mae_excursion       (batch, 1)  — max adverse excursion ≥ 0
    trade_efficiency    (batch, 1)  — % move captured ∈ [0, 100]
    reward_risk         (batch, 1)  — expected R:R ratio ≥ 0
    trade_survival      (batch, K)  — survival probs at K bar horizons ∈ [0,1]
    sl_resilience       (batch, 1)  — P(SL survives noise) ∈ [0, 1]
    tp_achievability    (batch, 1)  — P(TP achieved) ∈ [0, 1]
    entry_quality       (batch, 1)  — entry quality score ∈ [0, 100]
    exit_quality        (batch, 1)  — exit quality score ∈ [0, 100]
    market_stability    (batch, 4)  — stability class probabilities
    trade_decay         (batch, D+1)— [decay_prob, decay curve] ∈ [0, 1]
    profit_distribution (batch, 6)  — [mean, median, var, skew, ci_lo, ci_hi]
    loss_distribution   (batch, 4)  — [mean, median, var, tail_risk]
    position_size_quality (batch,1) — capital allocation quality ∈ [0, 100]
    uncertainty         (batch, 2)  — [epistemic, aleatoric] ≥ 0

    ── Metadata ────────────────────────────────────────────────────────────
    prediction_timestamp — UTC ISO-8601 timestamp
    model_version        — semantic version string
    """

    # ── Always present ───────────────────────────────────────────────────
    logits: torch.Tensor
    probabilities: torch.Tensor
    confidence: torch.Tensor
    predicted_class: torch.Tensor
    buy_prob: torch.Tensor
    sell_prob: torch.Tensor
    no_trade_prob: torch.Tensor

    # ── Phase 14.5 optional ──────────────────────────────────────────────
    expected_move: torch.Tensor | None = None
    stop_loss: torch.Tensor | None = None
    take_profit: torch.Tensor | None = None
    holding_time: torch.Tensor | None = None
    tp_probability: torch.Tensor | None = None

    # ── Phase 14.6 optional ──────────────────────────────────────────────
    mfe: torch.Tensor | None = None
    mae_excursion: torch.Tensor | None = None
    trade_efficiency: torch.Tensor | None = None
    reward_risk: torch.Tensor | None = None
    trade_survival: torch.Tensor | None = None
    sl_resilience: torch.Tensor | None = None
    tp_achievability: torch.Tensor | None = None
    entry_quality: torch.Tensor | None = None
    exit_quality: torch.Tensor | None = None
    market_stability: torch.Tensor | None = None
    trade_decay: torch.Tensor | None = None
    profit_distribution: torch.Tensor | None = None
    loss_distribution: torch.Tensor | None = None
    position_size_quality: torch.Tensor | None = None
    uncertainty: torch.Tensor | None = None

    # ── Metadata ─────────────────────────────────────────────────────────
    prediction_timestamp: str | None = None
    model_version: str | None = None

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
        _opt_fields = [
            "expected_move",
            "stop_loss",
            "take_profit",
            "holding_time",
            "tp_probability",
            "mfe",
            "mae_excursion",
            "trade_efficiency",
            "reward_risk",
            "trade_survival",
            "sl_resilience",
            "tp_achievability",
            "entry_quality",
            "exit_quality",
            "market_stability",
            "trade_decay",
            "profit_distribution",
            "loss_distribution",
            "position_size_quality",
            "uncertainty",
            "prediction_timestamp",
            "model_version",
        ]
        for f in _opt_fields:
            val = getattr(self, f)
            if val is not None:
                result[f] = val
        return result

    def to_base_output(self) -> QuantModelOutput:
        """Down-cast to original QuantModelOutput for backward compatibility."""
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
        return datetime.now(tz=UTC).isoformat()
