"""S32 — Setup Memory Engine.

Tracks market opportunities over time as a persistent per-(symbol,
timeframe) state machine, so decisions are made from accumulated setup
context rather than a single isolated inference pass. Purpose-built to
prevent two failure modes: late entries (acting only once a move is
already underway, because there was no memory of the setup building) and
overtrading (re-entering the same symbol/timeframe before a prior setup
has genuinely cooled down).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.setup_memory.models import (
    DEMOTION_TRANSITIONS,
    FORWARD_TRANSITIONS,
    INVALIDATABLE_STATES,
    VALID_TRANSITIONS,
    Direction,
    SetupRecord,
    SetupState,
)
from trainer.setup_memory.store import SetupMemoryStore

#: Confidence delta below which a demotion is triggered automatically by
#: ``weaken`` (rather than requiring the caller to call ``invalidate``
#: explicitly). Purely a convenience threshold — callers may also demote
#: proactively for their own reasons via ``advance_to``.
DEFAULT_DEMOTION_CONFIDENCE_FLOOR = 0.15

#: Default maximum time a setup may spend in ENTRY_WINDOW before it is
#: treated as expired and forced back to COOLDOWN — the core anti-late-entry
#: guard: once the trigger window has passed, entering is no longer the same
#: opportunity that was armed.
DEFAULT_ENTRY_WINDOW_TIMEOUT_S = 5 * 60.0

#: Default cooldown duration before COOLDOWN -> IDLE is allowed — the core
#: anti-overtrading guard: a symbol/timeframe cannot be re-watched from
#: scratch immediately after a close or invalidation.
DEFAULT_COOLDOWN_S = 15 * 60.0

#: If a setup sits in WATCHING/BUILDING_SETUP/ARMED for longer than this
#: without strengthening, it is stale and auto-invalidated (setup expiry).
DEFAULT_SETUP_MAX_AGE_S = 4 * 3600.0


class InvalidTransition(Exception):
    """Raised when a caller requests a transition the state machine does
    not allow (e.g. IDLE straight to ARMED)."""


@dataclass
class SetupMemoryConfig:
    demotion_confidence_floor: float = DEFAULT_DEMOTION_CONFIDENCE_FLOOR
    entry_window_timeout_s: float = DEFAULT_ENTRY_WINDOW_TIMEOUT_S
    cooldown_s: float = DEFAULT_COOLDOWN_S
    setup_max_age_s: float = DEFAULT_SETUP_MAX_AGE_S


class SetupMemoryEngine:
    """Public API for the Setup Memory Engine.

    Args:
        store: Injectable :class:`SetupMemoryStore` (tests / custom path).
        config: Tunable thresholds; see :class:`SetupMemoryConfig`.
    """

    def __init__(
        self,
        store: SetupMemoryStore | None = None,
        config: SetupMemoryConfig | None = None,
    ) -> None:
        self._store = store if store is not None else SetupMemoryStore()
        self._config = config if config is not None else SetupMemoryConfig()

    # ── Lookup / creation ────────────────────────────────────────────────

    def get(self, symbol: str, timeframe: str) -> SetupRecord | None:
        """Return the persisted record, if any — does not create one."""
        return self._store.get(symbol, timeframe)

    def get_or_create(self, symbol: str, timeframe: str, broker_symbol: str) -> SetupRecord:
        """Return the existing record for (symbol, timeframe), creating a
        fresh IDLE record if this is the first time it's been seen. State
        persists across restarts because it's read straight from SQLite."""
        record = self._store.get(symbol, timeframe)
        if record is not None:
            return record
        record = SetupRecord(symbol=symbol, broker_symbol=broker_symbol, timeframe=timeframe)
        self._store.upsert(record)
        return record

    def all_records(self) -> list[SetupRecord]:
        return self._store.all_records()

    def history(self, symbol: str, timeframe: str, limit: int = 100) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self._store.history(symbol, timeframe, limit)]

    # ── Core transition primitive ───────────────────────────────────────

    def _transition(
        self,
        record: SetupRecord,
        to_state: SetupState,
        reason: str,
        now: float,
        confidence: float | None = None,
    ) -> SetupRecord:
        if to_state not in VALID_TRANSITIONS[record.state]:
            raise InvalidTransition(
                f"{record.key()}: cannot transition {record.state.value} -> {to_state.value}"
            )
        from_state = record.state
        record.state = to_state
        if from_state != to_state:
            record.state_entered_at = now
        record.last_updated_at = now
        if confidence is not None:
            record.state_confidence = max(0.0, min(1.0, confidence))
        self._store.upsert(record)
        # Always append to the audit trail, even for same-state confidence
        # updates (strengthen/weaken) — this is what lets confidence
        # stability be derived later (e.g. Opportunity Quality Scoring's
        # confidence_stability sub-score) instead of only seeing full
        # state transitions.
        self._store.append_transition(
            record.symbol,
            record.timeframe,
            from_state,
            to_state,
            reason,
            record.state_confidence,
            now,
        )
        if from_state != to_state:
            logger.info(
                "setup_memory: {} {} -> {} ({})",
                record.key(),
                from_state.value,
                to_state.value,
                reason,
            )
        return record

    # ── Forward advancement ──────────────────────────────────────────────

    def advance(
        self,
        symbol: str,
        timeframe: str,
        broker_symbol: str = "",
        reason: str = "",
        confidence: float | None = None,
        expected_direction: Direction | None = None,
        expected_move_size: float | None = None,
        expected_holding_period_s: float | None = None,
        now: float | None = None,
    ) -> SetupRecord:
        """Advance one step along the canonical forward lifecycle
        (IDLE -> WATCHING -> ... -> ACTIVE_POSITION -> COOLDOWN -> IDLE).

        This is the primary entry point callers use as new evidence
        crosses each stage's threshold (e.g. "higher timeframe bias
        appeared" -> ``advance(...)`` moves IDLE to WATCHING).
        """
        now = now if now is not None else time.time()
        record = self.get_or_create(symbol, timeframe, broker_symbol or symbol)
        next_state = FORWARD_TRANSITIONS[record.state]
        if expected_direction is not None:
            record.expected_direction = expected_direction
        if expected_move_size is not None:
            record.expected_move_size = expected_move_size
        if expected_holding_period_s is not None:
            record.expected_holding_period_s = expected_holding_period_s
        record.consecutive_weakenings = 0
        record.consecutive_strengthenings = 0
        return self._transition(
            record, next_state, reason or f"advanced to {next_state.value}", now, confidence
        )

    def advance_to(
        self,
        symbol: str,
        timeframe: str,
        to_state: SetupState,
        reason: str,
        now: float | None = None,
    ) -> SetupRecord:
        """Explicit transition to an arbitrary (but validated) target
        state — used for invalidation/administrative overrides. Prefer
        ``advance``/``strengthen``/``weaken``/``invalidate`` for normal
        lifecycle flow."""
        now = now if now is not None else time.time()
        record = self.get_or_create(symbol, timeframe, symbol)
        return self._transition(record, to_state, reason, now)

    # ── Strength adjustments ─────────────────────────────────────────────

    def strengthen(
        self,
        symbol: str,
        timeframe: str,
        delta: float = 0.1,
        reason: str = "",
        now: float | None = None,
    ) -> SetupRecord:
        """Increase state confidence as corroborating evidence accumulates
        (e.g. another confirming candle, volume expansion). Never changes
        state by itself — promotion only happens via ``advance``, so a
        setup can get *more confident* without prematurely skipping a
        stage."""
        now = now if now is not None else time.time()
        record = self.get_or_create(symbol, timeframe, symbol)
        record.consecutive_strengthenings += 1
        record.consecutive_weakenings = 0
        new_confidence = max(0.0, min(1.0, record.state_confidence + abs(delta)))
        return self._transition(
            record, record.state, reason or "setup strengthened", now, new_confidence
        )

    def weaken(
        self,
        symbol: str,
        timeframe: str,
        delta: float = 0.1,
        reason: str = "",
        now: float | None = None,
    ) -> SetupRecord:
        """Decrease state confidence as evidence fails to confirm. If
        confidence drops below ``demotion_confidence_floor`` and the state
        has a demotion target, the setup is automatically demoted one
        stage (e.g. ARMED -> BUILDING_SETUP) rather than invalidated
        outright — this is what lets a setup "cool off" gracefully instead
        of every wobble forcing a full COOLDOWN cycle."""
        now = now if now is not None else time.time()
        record = self.get_or_create(symbol, timeframe, symbol)
        record.consecutive_weakenings += 1
        record.consecutive_strengthenings = 0
        new_confidence = max(0.0, min(1.0, record.state_confidence - abs(delta)))
        demote_to = DEMOTION_TRANSITIONS.get(record.state)
        if demote_to is not None and new_confidence < self._config.demotion_confidence_floor:
            record.consecutive_weakenings = 0
            return self._transition(
                record, demote_to, reason or "setup weakened below threshold", now, new_confidence
            )
        return self._transition(
            record, record.state, reason or "setup weakened", now, new_confidence
        )

    # ── Invalidation / expiry ────────────────────────────────────────────

    def invalidate(
        self, symbol: str, timeframe: str, reason: str, now: float | None = None
    ) -> SetupRecord:
        """Force a pre-position setup straight into COOLDOWN (e.g.
        structure broke down, higher-timeframe bias flipped). No-op if the
        setup is already IDLE/COOLDOWN; raises if a position is currently
        open — that must be closed through the normal ACTIVE_POSITION ->
        COOLDOWN path, never invalidated out from under an open trade."""
        now = now if now is not None else time.time()
        record = self.get_or_create(symbol, timeframe, symbol)
        if record.state in (SetupState.IDLE, SetupState.COOLDOWN):
            return record
        if record.state not in INVALIDATABLE_STATES:
            raise InvalidTransition(
                f"{record.key()}: cannot invalidate an open position "
                f"(state={record.state.value}); close it first"
            )
        record.invalidation_reason = reason
        return self._transition(record, SetupState.COOLDOWN, f"invalidated: {reason}", now, 0.0)

    def check_expiry(self, symbol: str, timeframe: str, now: float | None = None) -> SetupRecord:
        """Run all time-based expiry checks for one setup and apply any
        resulting transition. Intended to be polled periodically (e.g. once
        per bar) for every tracked (symbol, timeframe) pair:

        - ENTRY_WINDOW open longer than ``entry_window_timeout_s`` without
          execution -> invalidated to COOLDOWN (prevents late entries on a
          trigger that's no longer fresh).
        - WATCHING/BUILDING_SETUP/ARMED open longer than
          ``setup_max_age_s`` without strengthening -> invalidated to
          COOLDOWN (stale setup expiry).
        - COOLDOWN held for at least ``cooldown_s`` -> advances to IDLE
          (release the overtrading guard once enough time has passed).
        """
        now = now if now is not None else time.time()
        record = self.get_or_create(symbol, timeframe, symbol)
        age = record.state_age_s(now)

        if record.state == SetupState.ENTRY_WINDOW and age > self._config.entry_window_timeout_s:
            return self.invalidate(symbol, timeframe, "entry window expired", now)

        if (
            record.state in (SetupState.WATCHING, SetupState.BUILDING_SETUP, SetupState.ARMED)
            and age > self._config.setup_max_age_s
        ):
            return self.invalidate(symbol, timeframe, "setup expired (stale, no progress)", now)

        if record.state == SetupState.COOLDOWN and age > self._config.cooldown_s:
            record.invalidation_reason = ""
            return self._transition(record, SetupState.IDLE, "cooldown elapsed", now, 0.0)

        return record

    def check_all_expiry(self, now: float | None = None) -> list[SetupRecord]:
        """Run ``check_expiry`` across every tracked (symbol, timeframe)."""
        now = now if now is not None else time.time()
        return [
            self.check_expiry(record.symbol, record.timeframe, now)
            for record in self._store.all_records()
        ]

    # ── Model-input feature extraction ──────────────────────────────────

    #: Fixed ordering so the one-hot encoding is stable across calls/model
    #: versions — do not reorder without also retraining/migrating anything
    #: that consumes this vector.
    STATE_ORDER: tuple[SetupState, ...] = (
        SetupState.IDLE,
        SetupState.WATCHING,
        SetupState.BUILDING_SETUP,
        SetupState.ARMED,
        SetupState.ENTRY_WINDOW,
        SetupState.ACTIVE_POSITION,
        SetupState.COOLDOWN,
    )

    def feature_vector(
        self,
        symbol: str,
        timeframe: str,
        now: float | None = None,
        age_normalization_s: float = 3600.0,
    ) -> dict[str, Any]:
        """Compact, model-ready summary of setup memory for (symbol,
        timeframe) — this is the "setup state as an input feature"
        requirement. Consumers (e.g. the Hybrid Encoder's context/market
        memory stream — see ``trainer/models/quant_v1/hybrid/encoder.py``)
        can flatten ``state_onehot`` + the scalar fields directly into a
        feature tensor. Never raises: an unseen (symbol, timeframe) yields
        the neutral IDLE feature vector rather than an error, since "no
        setup memory yet" is itself meaningful signal to the model.
        """
        now = now if now is not None else time.time()
        record = self._store.get(symbol, timeframe)
        if record is None:
            record = SetupRecord(symbol=symbol, broker_symbol=symbol, timeframe=timeframe)
        state_index = self.STATE_ORDER.index(record.state)
        one_hot = [1.0 if i == state_index else 0.0 for i in range(len(self.STATE_ORDER))]
        direction_scalar = {
            Direction.LONG: 1.0,
            Direction.SHORT: -1.0,
            Direction.UNKNOWN: 0.0,
        }[record.expected_direction]
        age_s = record.state_age_s(now)
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "state": record.state.value,
            "state_index": state_index,
            "state_onehot": one_hot,
            "state_confidence": record.state_confidence,
            "state_age_s": age_s,
            "state_age_normalized": (
                min(1.0, age_s / age_normalization_s) if age_normalization_s > 0 else 0.0
            ),
            "expected_direction": record.expected_direction.value,
            "expected_direction_scalar": direction_scalar,
            "expected_move_size": record.expected_move_size,
            "expected_holding_period_s": record.expected_holding_period_s,
        }
