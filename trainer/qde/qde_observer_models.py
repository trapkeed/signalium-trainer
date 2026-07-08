"""Phase S7 — Market Observer data models.

`SymbolObservationInput` is the single per-symbol snapshot fed into
`QDEMarketObserver.observe()` on every tick of the always-running monitoring
loop. It composes the existing typed snapshots (`MarketContext`,
`PositionMetrics`, `PortfolioMetrics`) rather than re-declaring their
fields, so this module stays a thin aggregation layer over data QDE already
understands.

`ObserverMemory` is the persisted state/context carried between ticks for a
given symbol — this *is* the "memory" the always-running observer
maintains; it is not a rule engine, it is what the model consults as
temporal context (how long has this setup been forming? did we just close
a losing trade? are we in a cooldown?).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from trainer.lifecycle.lifecycle_models import MarketContext, PositionMetrics
from trainer.portfolio.portfolio_scorer import PortfolioMetrics
from trainer.qde.qde_observer_types import ObserverEvent, ObserverState

OBSERVER_SCHEMA_VERSION = "s7_v1"


# ── SymbolObservationInput ───────────────────────────────────────────────────


@dataclass
class SymbolObservationInput:
    """Everything the observer needs for one evaluation tick, for one symbol."""

    instrument: str
    market: MarketContext = field(default_factory=MarketContext)

    # ── Microstructure / volatility / liquidity / session context ──────────
    # Kept as plain scores (0-1 or named states as strings) rather than
    # importing the full `MicrostructureFeatures`/MIE result types, so this
    # module has no hard dependency on those subsystems' internal shapes —
    # callers translate whatever engine they use into these normalised
    # fields (mirrors how `PositionTelemetry.heads` decouples QDE from any
    # one model implementation).
    microstructure_quality: float = 0.5  # 0-1, higher = cleaner/more tradeable
    volatility_state: str = "NORMAL"
    volatility_quality: float = 0.5  # 0-1, 100 = ideal vol for trading
    liquidity_state: str = "NORMAL"
    liquidity_quality: float = 0.5
    session_quality: float = 0.5  # 0-1
    market_regime: str = "RANGE"

    # ── Portfolio / account context ─────────────────────────────────────────
    portfolio: PortfolioMetrics | None = None
    free_margin_pct: float = 1.0

    # ── Position context (populated once a position exists for this symbol) ─
    has_open_position: bool = False
    position: PositionMetrics | None = None

    # ── Trade-lifecycle events since the last tick (caller-reported) ────────
    position_just_opened: bool = False
    position_just_closed: bool = False
    last_close_was_loss: bool = False

    unix_ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "market": self.market.to_dict(),
            "microstructure_quality": round(self.microstructure_quality, 4),
            "volatility_state": self.volatility_state,
            "volatility_quality": round(self.volatility_quality, 4),
            "liquidity_state": self.liquidity_state,
            "liquidity_quality": round(self.liquidity_quality, 4),
            "session_quality": round(self.session_quality, 4),
            "market_regime": self.market_regime,
            "portfolio": self.portfolio.to_dict() if self.portfolio else None,
            "free_margin_pct": round(self.free_margin_pct, 4),
            "has_open_position": self.has_open_position,
            "position": self.position.to_dict() if self.position else None,
            "position_just_opened": self.position_just_opened,
            "position_just_closed": self.position_just_closed,
            "last_close_was_loss": self.last_close_was_loss,
            "unix_ts": round(self.unix_ts, 3),
        }


# ── ObserverMemory ────────────────────────────────────────────────────────────


@dataclass
class ObserverMemory:
    """Persisted per-symbol context carried between observation ticks.

    This is the "memory" of the always-running observer: it is consulted
    (not re-derived from scratch) on every tick, which is what lets QDE
    reason about dwell time, streaks, and cooldowns instead of only ever
    seeing a stateless snapshot.
    """

    instrument: str
    state: ObserverState = ObserverState.OBSERVING
    previous_state: ObserverState = ObserverState.OBSERVING
    entered_state_at: float = field(default_factory=time.time)
    last_event: ObserverEvent = ObserverEvent.NO_CHANGE
    readiness_score: float = 0.0  # rolling 0-1 "how close to a good entry"
    consecutive_losses: int = 0
    cooldown_until: float = 0.0
    entry_window_expires_at: float = 0.0
    last_updated_at: float = field(default_factory=time.time)
    transition_count: int = 0

    @classmethod
    def new(cls, instrument: str) -> ObserverMemory:
        now = time.time()
        return cls(
            instrument=instrument,
            entered_state_at=now,
            last_updated_at=now,
        )

    def dwell_seconds(self, now: float | None = None) -> float:
        now = now if now is not None else time.time()
        return max(0.0, now - self.entered_state_at)

    def cooldown_remaining_s(self, now: float | None = None) -> float:
        now = now if now is not None else time.time()
        return max(0.0, self.cooldown_until - now)

    def to_dict(self) -> dict:
        now = time.time()
        return {
            "instrument": self.instrument,
            "state": self.state.value,
            "previous_state": self.previous_state.value,
            "entered_state_at": round(self.entered_state_at, 3),
            "dwell_seconds": round(self.dwell_seconds(now), 2),
            "last_event": self.last_event.value,
            "readiness_score": round(self.readiness_score, 4),
            "consecutive_losses": self.consecutive_losses,
            "cooldown_until": round(self.cooldown_until, 3),
            "cooldown_remaining_s": round(self.cooldown_remaining_s(now), 2),
            "entry_window_expires_at": round(self.entry_window_expires_at, 3),
            "last_updated_at": round(self.last_updated_at, 3),
            "transition_count": self.transition_count,
        }


# ── ObservationResult ─────────────────────────────────────────────────────────


@dataclass
class ObservationResult:
    """Output of a single `QDEMarketObserver.observe()` call."""

    observation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    instrument: str = ""
    state: ObserverState = ObserverState.OBSERVING
    previous_state: ObserverState = ObserverState.OBSERVING
    transitioned: bool = False
    event: ObserverEvent = ObserverEvent.NO_CHANGE
    readiness_score: float = 0.0
    confidence: float = 0.0
    reasoning: str = ""
    dwell_seconds: float = 0.0
    cooldown_remaining_s: float = 0.0
    guard_tripped: str = ""  # "" | "OVERTRADE" | "REVENGE_TRADE"
    schema_version: str = OBSERVER_SCHEMA_VERSION
    unix_ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "observation_id": self.observation_id,
            "instrument": self.instrument,
            "state": self.state.value,
            "previous_state": self.previous_state.value,
            "transitioned": self.transitioned,
            "event": self.event.value,
            "readiness_score": round(self.readiness_score, 4),
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
            "dwell_seconds": round(self.dwell_seconds, 2),
            "cooldown_remaining_s": round(self.cooldown_remaining_s, 2),
            "guard_tripped": self.guard_tripped,
            "schema_version": self.schema_version,
            "unix_ts": round(self.unix_ts, 3),
        }
