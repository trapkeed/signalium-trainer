"""Live QDE inference — the single decision-making entry point for execution.

Per `docs/qde/unified_qde_architecture.md`: the QDE is one model producing one
unified decision (entry, exit, scale, size, risk allocation, trade management).
`ExecutionPlanner` must be a pure *executor* of that decision, not a computer
of independent sub-decisions.

This module defines:
  - Five snapshot input types (market, portfolio, position, account, memory),
    reusing existing dataclasses where they already cover the same ground.
  - `QDEInferenceInput` — the aggregate input passed to the model.
  - `QDEDecision` — the single unified output.
  - `QDEDecisionModel` — the abstract interface any active QDE model
    implements (`decide(inputs) -> QDEDecision`).
  - `HeuristicQDEDecisionModel` — an interim baseline used until a trained
    model artifact is registered for a version. It is intentionally the only
    concrete implementation today and is documented as a placeholder, not a
    permanent trading-rule engine (see `docs/qde/hardcoded_behavior_audit.md`).

Nothing in this module is invoked independently by `ExecutionPlanner` sub
engines — it is the reverse: supporting engine output (zones/stop/tp/timing)
is fed *into* the decision as context, and the decision is what gets executed.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from trainer.execution.execution_types import ExecutionForecast, ManagementAction
from trainer.lifecycle.lifecycle_models import MarketContext, PositionMetrics
from trainer.portfolio.portfolio_scorer import PortfolioMetrics

if TYPE_CHECKING:
    from trainer.memory.decision_models import DecisionRecord
    from trainer.qde.qde_models import QDEVersion

# ── Enumerations ─────────────────────────────────────────────────────────────


class EntryAction(str, Enum):
    ENTER = "ENTER"
    WAIT = "WAIT"
    REJECT = "REJECT"


class ExitAction(str, Enum):
    HOLD = "HOLD"
    EXIT_NOW = "EXIT_NOW"
    EXIT_PARTIAL = "EXIT_PARTIAL"


class ScaleAction(str, Enum):
    NONE = "NONE"
    SCALE_IN = "SCALE_IN"
    SCALE_OUT = "SCALE_OUT"


# ── Snapshot inputs ──────────────────────────────────────────────────────────
#
# Market, portfolio, and position snapshots reuse existing dataclasses that
# already model this exact ground (`MarketContext`, `PortfolioMetrics`,
# `PositionMetrics`). Account and memory snapshots are new — no existing type
# covered them.


def default_portfolio_snapshot() -> PortfolioMetrics:
    """A neutral portfolio snapshot for single-instrument / no-portfolio calls."""
    return PortfolioMetrics(
        expected_return=0.0,
        expected_volatility=0.0,
        expected_drawdown=0.0,
        expected_sharpe=0.0,
        expected_sortino=0.0,
        diversification_score=1.0,
        correlation_risk=0.0,
        concentration_score=0.0,
        instrument_count=0,
        long_count=0,
        short_count=0,
        neutral_count=0,
        weighted_confidence=0.0,
    )


@dataclass
class QDEAccountSnapshot:
    """Live account state at decision time — a hard-constraint surface, not a
    trading rule (see architecture doc §6: broker/account-level constraints
    are infrastructure, not strategy)."""

    balance: float = 10_000.0
    equity: float = 10_000.0
    margin_used: float = 0.0
    margin_available: float = 10_000.0
    open_positions_count: int = 0
    daily_pnl_pct: float = 0.0
    weekly_pnl_pct: float = 0.0

    @property
    def free_margin_pct(self) -> float:
        if self.equity < 1e-9:
            return 0.0
        return max(0.0, self.margin_available / self.equity)

    def to_dict(self) -> dict:
        return {
            "balance": round(self.balance, 2),
            "equity": round(self.equity, 2),
            "margin_used": round(self.margin_used, 2),
            "margin_available": round(self.margin_available, 2),
            "open_positions_count": self.open_positions_count,
            "daily_pnl_pct": round(self.daily_pnl_pct, 4),
            "weekly_pnl_pct": round(self.weekly_pnl_pct, 4),
        }


@dataclass
class QDEMemorySnapshot:
    """Compact summary of relevant decision history for this instrument.

    Built from `DecisionMemory.filter(...)` — carries a summary rather than
    the full record list, since the QDE consumes aggregate historical
    context, not a database dump.
    """

    instrument: str = ""
    recent_decision_count: int = 0
    recent_win_rate: float | None = None
    recent_avg_pnl_pct: float = 0.0
    last_decision_type: str = ""
    last_decision_grade: str | None = None

    @classmethod
    def from_records(cls, instrument: str, records: list["DecisionRecord"]) -> "QDEMemorySnapshot":
        if not records:
            return cls(instrument=instrument)

        scored = [r for r in records if r.has_outcome]
        wins = [r for r in scored if r.outcome and r.outcome.actual_direction_correct]
        win_rate = (len(wins) / len(scored)) if scored else None
        avg_pnl = (
            sum(r.outcome.actual_pnl_pct for r in scored) / len(scored) if scored else 0.0
        )
        latest = records[0]  # DecisionMemory.filter returns newest-first

        return cls(
            instrument=instrument,
            recent_decision_count=len(records),
            recent_win_rate=win_rate,
            recent_avg_pnl_pct=avg_pnl,
            last_decision_type=latest.decision_type.value,
            last_decision_grade=latest.grade,
        )

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "recent_decision_count": self.recent_decision_count,
            "recent_win_rate": (
                round(self.recent_win_rate, 4) if self.recent_win_rate is not None else None
            ),
            "recent_avg_pnl_pct": round(self.recent_avg_pnl_pct, 4),
            "last_decision_type": self.last_decision_type,
            "last_decision_grade": self.last_decision_grade,
        }


# ── Aggregate input / output ─────────────────────────────────────────────────


@dataclass
class QDEInferenceInput:
    """The complete set of live inputs handed to the active QDE model.

    `forecast` carries the raw instrument/price/model-signal data (already
    produced upstream by the forecasting model); the five snapshots carry
    market, portfolio, position, account, and memory context. Supporting
    execution-mechanics detail (entry zones, stop/tp candidates, timing) is
    attached so the QDE can factor in concrete price levels, but none of that
    detail is itself a decision — the QDE's output is.
    """

    forecast: ExecutionForecast
    market: MarketContext
    portfolio: PortfolioMetrics
    position: PositionMetrics
    account: QDEAccountSnapshot
    memory: QDEMemorySnapshot

    # Optional execution-mechanics context (price levels only, not decisions)
    entry_price_hint: float = 0.0
    stop_price_hint: float = 0.0
    take_profit_hint: float = 0.0

    def to_dict(self) -> dict:
        return {
            "forecast": self.forecast.to_dict(),
            "market": self.market.to_dict(),
            "portfolio": self.portfolio.to_dict(),
            "position": self.position.to_dict(),
            "account": self.account.to_dict(),
            "memory": self.memory.to_dict(),
            "entry_price_hint": round(self.entry_price_hint, 5),
            "stop_price_hint": round(self.stop_price_hint, 5),
            "take_profit_hint": round(self.take_profit_hint, 5),
        }


@dataclass
class QDEDecision:
    """The single unified decision produced by the active QDE model.

    This is the only object `ExecutionPlanner` executes. Entry, exit, scale,
    size, risk allocation, and trade management are all facets of this one
    output — never independently re-derived downstream.
    """

    instrument: str
    direction: str
    entry_action: EntryAction
    exit_action: ExitAction = ExitAction.HOLD
    scale_action: ScaleAction = ScaleAction.NONE

    position_size_lots: float = 0.0
    risk_allocation_pct: float = 0.0  # fraction of account equity

    management_actions: list[ManagementAction] = field(default_factory=list)

    confidence: float = 0.0
    model_version_id: str = ""
    rationale: str = ""

    decided_at: float = field(default_factory=time.time)

    @property
    def is_entry(self) -> bool:
        return self.entry_action == EntryAction.ENTER

    @property
    def primary_management_action(self) -> ManagementAction:
        return self.management_actions[0] if self.management_actions else ManagementAction.HOLD

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "direction": self.direction,
            "entry_action": self.entry_action.value,
            "exit_action": self.exit_action.value,
            "scale_action": self.scale_action.value,
            "position_size_lots": round(self.position_size_lots, 2),
            "risk_allocation_pct": round(self.risk_allocation_pct, 4),
            "management_actions": [a.value for a in self.management_actions],
            "confidence": round(self.confidence, 4),
            "model_version_id": self.model_version_id,
            "rationale": self.rationale,
            "decided_at": round(self.decided_at, 3),
        }


# ── Decision model interface ─────────────────────────────────────────────────


class QDEDecisionModel(ABC):
    """Abstract interface for an invocable, active QDE model.

    A concrete implementation wraps a specific trained artifact (or, until
    one is deployed, an interim baseline) behind a single `decide()` call.
    `ExecutionPlanner` never inspects or branches on which implementation is
    active — it only calls `decide()`.
    """

    version_id: str = ""

    @abstractmethod
    def decide(self, inputs: QDEInferenceInput) -> QDEDecision:
        """Produce the complete trading decision for one instrument."""


class HeuristicQDEDecisionModel(QDEDecisionModel):
    """Interim baseline decision model.

    Used only when no trained QDE artifact has been registered for the
    active version. This is explicitly a placeholder — it exists so the
    "one model, one decision" execution path has something invocable to call
    end-to-end today, and is intended to be replaced by a real trained model
    behind the same `QDEDecisionModel` interface (registered via
    `QDERegistry.register_decision_model`). It is not a permanent trading
    rule engine and should be treated as scaffolding pending model training,
    consistent with `docs/qde/unified_qde_architecture.md` §4.3.
    """

    def __init__(self, version: "QDEVersion | None" = None) -> None:
        self.version_id = version.version_id if version else ""

    def decide(self, inputs: QDEInferenceInput) -> QDEDecision:
        fc = inputs.forecast

        # Blend forecast confidence with recent memory performance and news
        # risk into a single conviction score — one coherent read, not
        # independent sub-engines.
        conviction = fc.confidence
        if inputs.memory.recent_win_rate is not None:
            conviction = (conviction * 0.7) + (inputs.memory.recent_win_rate * 0.3)
        conviction *= 1.0 - _clamp01(inputs.market.news_risk)

        if inputs.account.free_margin_pct < 0.05:
            entry_action = EntryAction.REJECT
            rationale = "Account free margin below 5% — cannot open new risk."
        elif conviction < 0.45:
            entry_action = EntryAction.REJECT
            rationale = f"Blended conviction {conviction:.2f} below entry threshold."
        elif conviction < 0.55:
            entry_action = EntryAction.WAIT
            rationale = f"Blended conviction {conviction:.2f} marginal — waiting for confirmation."
        else:
            entry_action = EntryAction.ENTER
            rationale = f"Blended conviction {conviction:.2f} supports entry."

        has_position = inputs.position.lots > 1e-9
        exit_action = ExitAction.HOLD
        scale_action = ScaleAction.NONE
        management: list[ManagementAction] = [ManagementAction.HOLD]

        if has_position:
            if inputs.market.news_risk > 0.6 or conviction < 0.35:
                exit_action = ExitAction.EXIT_NOW
                management = [ManagementAction.EXIT_EARLY]
                rationale += " Existing position exited on elevated risk / low conviction."
            elif conviction < 0.5:
                exit_action = ExitAction.EXIT_PARTIAL
                management = [ManagementAction.PARTIAL_CLOSE]
            elif conviction > 0.75 and inputs.account.free_margin_pct > 0.3:
                scale_action = ScaleAction.SCALE_IN
                management = [ManagementAction.SCALE_IN, ManagementAction.TRAIL_STOP]
            else:
                management = [ManagementAction.MOVE_TO_BREAKEVEN, ManagementAction.TRAIL_STOP]

        risk_allocation_pct = 0.0
        position_size_lots = 0.0
        if entry_action == EntryAction.ENTER or scale_action == ScaleAction.SCALE_IN:
            risk_allocation_pct = _clamp01(conviction) * 0.02  # up to 2% of equity
            risk_allocation_pct = min(risk_allocation_pct, inputs.account.free_margin_pct * 0.5)
            entry = inputs.entry_price_hint or fc.current_price
            stop = inputs.stop_price_hint or (
                entry - fc.atr * 1.5 if fc.is_buy else entry + fc.atr * 1.5
            )
            stop_dist = max(abs(entry - stop), 1e-9)
            risk_amount = inputs.account.equity * risk_allocation_pct
            position_size_lots = risk_amount / (100_000 * stop_dist)

        return QDEDecision(
            instrument=fc.instrument,
            direction=fc.direction,
            entry_action=entry_action,
            exit_action=exit_action,
            scale_action=scale_action,
            position_size_lots=max(position_size_lots, 0.0),
            risk_allocation_pct=max(risk_allocation_pct, 0.0),
            management_actions=management,
            confidence=_clamp01(conviction),
            model_version_id=self.version_id,
            rationale=rationale,
        )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
