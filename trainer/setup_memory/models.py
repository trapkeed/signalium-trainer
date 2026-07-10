"""S32 — Setup Memory Engine: state/data model.

Defines the seven-state lifecycle required by the spec and the persisted
per-(symbol, timeframe) record. Kept deliberately separate from
``trainer.qde.qde_observer_types.ObserverState`` — that enum models QDE's
own in-trade management lifecycle (MANAGING/EXITING/etc.); this one models
setup *maturation* prior to and across a single trade, and is designed to
be usable by any consumer, not just QDE.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SetupState(str, Enum):
    """Per-(symbol, timeframe) setup lifecycle state."""

    IDLE = "IDLE"  # no opportunity being tracked
    WATCHING = "WATCHING"  # higher-timeframe bias appeared; watching for structure
    BUILDING_SETUP = "BUILDING_SETUP"  # structure is forming
    ARMED = "ARMED"  # probability exceeded the setup threshold
    ENTRY_WINDOW = "ENTRY_WINDOW"  # trigger conditions appeared; entry imminent
    ACTIVE_POSITION = "ACTIVE_POSITION"  # position executed and open
    COOLDOWN = "COOLDOWN"  # post-closure (or post-invalidation) quiet period


#: Canonical forward lifecycle, exactly as specified.
FORWARD_TRANSITIONS: dict[SetupState, SetupState] = {
    SetupState.IDLE: SetupState.WATCHING,
    SetupState.WATCHING: SetupState.BUILDING_SETUP,
    SetupState.BUILDING_SETUP: SetupState.ARMED,
    SetupState.ARMED: SetupState.ENTRY_WINDOW,
    SetupState.ENTRY_WINDOW: SetupState.ACTIVE_POSITION,
    SetupState.ACTIVE_POSITION: SetupState.COOLDOWN,
    SetupState.COOLDOWN: SetupState.IDLE,
}

#: One step back in maturity — used by weakening-driven demotion. There is
#: deliberately no demotion path out of ACTIVE_POSITION (an open position
#: can only be closed, not "un-armed").
DEMOTION_TRANSITIONS: dict[SetupState, SetupState] = {
    SetupState.WATCHING: SetupState.IDLE,
    SetupState.BUILDING_SETUP: SetupState.WATCHING,
    SetupState.ARMED: SetupState.BUILDING_SETUP,
    SetupState.ENTRY_WINDOW: SetupState.ARMED,
}

#: States from which a setup can be force-invalidated straight to COOLDOWN
#: (anything before a position is actually open).
INVALIDATABLE_STATES = frozenset(
    {
        SetupState.WATCHING,
        SetupState.BUILDING_SETUP,
        SetupState.ARMED,
        SetupState.ENTRY_WINDOW,
    }
)

#: Every state reachable directly from a given state — forward advance,
#: demotion (weakening), or invalidation to COOLDOWN. Used to validate any
#: explicit ``advance_to``/``transition_to`` call fails closed on illegal
#: jumps (e.g. IDLE straight to ARMED).
VALID_TRANSITIONS: dict[SetupState, frozenset[SetupState]] = {
    state: frozenset(
        {state, FORWARD_TRANSITIONS[state]}
        | ({DEMOTION_TRANSITIONS[state]} if state in DEMOTION_TRANSITIONS else set())
        | ({SetupState.COOLDOWN} if state in INVALIDATABLE_STATES else set())
    )
    for state in SetupState
}


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    UNKNOWN = "UNKNOWN"


@dataclass
class TransitionEvent:
    """One row of a setup's audit trail."""

    at: float
    from_state: SetupState
    to_state: SetupState
    reason: str
    confidence_after: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "reason": self.reason,
            "confidence_after": self.confidence_after,
        }


@dataclass
class SetupRecord:
    """Persisted per-(symbol, timeframe) setup memory.

    Every field named in the spec is represented:
    symbol, broker_symbol, setup state, state confidence, state age,
    expected direction, expected move size, expected holding period, time
    spent in state.
    """

    symbol: str
    broker_symbol: str
    timeframe: str
    state: SetupState = SetupState.IDLE
    state_confidence: float = 0.0
    state_entered_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)
    expected_direction: Direction = Direction.UNKNOWN
    expected_move_size: float = 0.0
    expected_holding_period_s: float = 0.0
    consecutive_weakenings: int = 0
    consecutive_strengthenings: int = 0
    invalidation_reason: str = ""

    def state_age_s(self, now: float | None = None) -> float:
        """Time spent continuously in the *current* state."""
        return (now if now is not None else time.time()) - self.state_entered_at

    def total_age_s(self, now: float | None = None) -> float:
        """Time since the record was first created/last updated (mirrors
        ``state_age_s`` unless a caller wants a distinct "record age" —
        kept separate so the two never get silently conflated)."""
        return (now if now is not None else time.time()) - self.last_updated_at

    def key(self) -> str:
        return f"{self.symbol}::{self.timeframe}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "broker_symbol": self.broker_symbol,
            "timeframe": self.timeframe,
            "state": self.state.value,
            "state_confidence": self.state_confidence,
            "state_entered_at": self.state_entered_at,
            "last_updated_at": self.last_updated_at,
            "state_age_s": self.state_age_s(),
            "expected_direction": self.expected_direction.value,
            "expected_move_size": self.expected_move_size,
            "expected_holding_period_s": self.expected_holding_period_s,
            "consecutive_weakenings": self.consecutive_weakenings,
            "consecutive_strengthenings": self.consecutive_strengthenings,
            "invalidation_reason": self.invalidation_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SetupRecord":
        return cls(
            symbol=data["symbol"],
            broker_symbol=data["broker_symbol"],
            timeframe=data["timeframe"],
            state=SetupState(data["state"]),
            state_confidence=float(data["state_confidence"]),
            state_entered_at=float(data["state_entered_at"]),
            last_updated_at=float(data["last_updated_at"]),
            expected_direction=Direction(data.get("expected_direction", Direction.UNKNOWN.value)),
            expected_move_size=float(data.get("expected_move_size", 0.0)),
            expected_holding_period_s=float(data.get("expected_holding_period_s", 0.0)),
            consecutive_weakenings=int(data.get("consecutive_weakenings", 0)),
            consecutive_strengthenings=int(data.get("consecutive_strengthenings", 0)),
            invalidation_reason=data.get("invalidation_reason", ""),
        )
