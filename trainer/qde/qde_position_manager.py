"""Phase S3 — QDE as a live position manager.

QDE reasons over every open position on a fixed cadence (default 30s),
consuming live telemetry (`PositionTelemetry`) and producing a single
`PositionManagementDecision`. This is the position-management analogue of
`trainer.qde.qde_inference.QDEDecisionModel.decide()` — same "one brain, one
decision" principle, applied to managing an already-open position instead of
opening a new one.

This module does not own a scheduler/loop itself (consistent with the rest
of the codebase — see `trainer/lifecycle/position_monitor.py`, which is also
a pure "evaluate on demand" reasoning function, not a `while True` loop).
The calling service (e.g. a background worker in `apps/trainer-api`) is
responsible for invoking `QDEPositionManager.review()` on a timer; `due_for_review()`
is provided so callers can check cadence without hardcoding "30" everywhere.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from trainer.lifecycle.lifecycle_models import MarketContext, PositionMetrics
from trainer.qde.qde_outputs import OUTPUT_SCHEMA_VERSION, QDEOutputHeads

REVIEW_INTERVAL_SECONDS = 30.0


class PositionManagementAction(str, Enum):
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    ADD = "ADD"
    CLOSE = "CLOSE"
    MOVE_TO_BREAKEVEN = "MOVE_TO_BREAKEVEN"
    TRAIL = "TRAIL"
    LOCK_PROFIT = "LOCK_PROFIT"


@dataclass
class PositionTelemetry:
    """Everything the live position-management loop needs for one review
    cycle. `heads` is the QDE model's latest 10-head read for this
    instrument (produced the same way as for a fresh entry decision — see
    `trainer.qde.qde_inference.QDEDecisionModel`), refreshed each cycle.
    """

    instrument: str
    position: PositionMetrics
    market: MarketContext
    heads: QDEOutputHeads
    free_margin_pct: float = 1.0
    seconds_since_last_review: float = REVIEW_INTERVAL_SECONDS

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "position": self.position.to_dict(),
            "market": self.market.to_dict(),
            "heads": self.heads.to_dict(),
            "free_margin_pct": round(self.free_margin_pct, 4),
            "seconds_since_last_review": round(self.seconds_since_last_review, 1),
        }


@dataclass
class PositionManagementDecision:
    """The single decision produced for one open position on one review
    cycle. Every field here is required by the Phase S3 spec: an action,
    plus confidence / reasoning / management_confidence / risk_signal.
    """

    instrument: str
    action: PositionManagementAction
    confidence: float
    reasoning: str
    management_confidence: float
    risk_signal: float
    model_version_id: str = ""
    schema_version: str = OUTPUT_SCHEMA_VERSION
    decided_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "action": self.action.value,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
            "management_confidence": round(self.management_confidence, 4),
            "risk_signal": round(self.risk_signal, 4),
            "model_version_id": self.model_version_id,
            "schema_version": self.schema_version,
            "decided_at": round(self.decided_at, 3),
        }


class QDEPositionManager:
    """Reasons over open-position telemetry and produces a single management
    decision per review cycle. Reads only from the 10-head signal vector
    (`QDEOutputHeads`) plus position/market telemetry — never from a named
    strategy, and never mutates the safety-system state established in
    Phase S1 (it may only *request* an action; execution and the Phase S1
    safety systems remain the layers that actually move capital).
    """

    def __init__(self, version_id: str = "") -> None:
        self.version_id = version_id

    def due_for_review(self, telemetry: PositionTelemetry) -> bool:
        return telemetry.seconds_since_last_review >= REVIEW_INTERVAL_SECONDS

    def review(self, telemetry: PositionTelemetry) -> PositionManagementDecision:
        heads = telemetry.heads
        pos = telemetry.position
        mkt = telemetry.market

        # ── Hard safety override: never overridden by a signal head ────────
        # Mirrors the Phase S1 principle — a true safety condition (extreme
        # news risk) always wins regardless of what the signal heads say.
        if mkt.news_risk > 0.85:
            return PositionManagementDecision(
                instrument=telemetry.instrument,
                action=PositionManagementAction.CLOSE,
                confidence=1.0,
                reasoning=f"Critical news risk ({mkt.news_risk:.0%}) — closing regardless of signal heads.",
                management_confidence=heads.management_confidence,
                risk_signal=1.0,
                model_version_id=self.version_id,
            )

        action: PositionManagementAction
        reasoning: str

        if heads.reduce_signal >= 0.65:
            action = PositionManagementAction.CLOSE
            reasoning = f"reduce_signal={heads.reduce_signal:.2f} — full exit warranted."
        elif heads.reduce_signal >= 0.35:
            action = PositionManagementAction.REDUCE
            reasoning = f"reduce_signal={heads.reduce_signal:.2f} — partial reduction warranted."
        elif (
            heads.scale_signal >= 0.65
            and telemetry.free_margin_pct > 0.3
            and pos.unrealised_pct >= 0.0
        ):
            action = PositionManagementAction.ADD
            reasoning = f"scale_signal={heads.scale_signal:.2f} with free margin — adding to winner."
        elif pos.unrealised_pct >= 0.02 and heads.reduce_signal < 0.2 and heads.confidence < 0.55:
            # Conviction has faded on a position that is meaningfully in
            # profit — lock the gain rather than risk giving it back.
            action = PositionManagementAction.LOCK_PROFIT
            reasoning = (
                f"unrealised_pct={pos.unrealised_pct:.2%} in profit with fading "
                f"confidence={heads.confidence:.2f} — locking in profit."
            )
        elif pos.unrealised_pct > 0.0 and heads.exit_horizon < 0.4:
            # Short expected remaining horizon on a winning trade — protect
            # the entry rather than let it round-trip back to breakeven.
            action = PositionManagementAction.MOVE_TO_BREAKEVEN
            reasoning = (
                f"unrealised_pct={pos.unrealised_pct:.2%} positive, short exit_horizon="
                f"{heads.exit_horizon:.2f} — moving stop to breakeven."
            )
        elif pos.unrealised_pct > 0.0 and heads.management_confidence >= 0.5:
            action = PositionManagementAction.TRAIL
            reasoning = (
                f"unrealised_pct={pos.unrealised_pct:.2%} positive, management_confidence="
                f"{heads.management_confidence:.2f} — trailing the stop."
            )
        else:
            action = PositionManagementAction.HOLD
            reasoning = (
                f"No head crossed an action threshold "
                f"(reduce={heads.reduce_signal:.2f}, scale={heads.scale_signal:.2f}, "
                f"confidence={heads.confidence:.2f}) — holding."
            )

        return PositionManagementDecision(
            instrument=telemetry.instrument,
            action=action,
            confidence=heads.confidence,
            reasoning=reasoning,
            management_confidence=heads.management_confidence,
            risk_signal=heads.risk_signal,
            model_version_id=self.version_id,
        )
