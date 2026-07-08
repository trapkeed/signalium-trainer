"""Phase S7 — QDE as an always-running market observer.

Unlike `qde_inference.py` (a single entry decision) or
`qde_position_manager.py` (a 30s-cadence review of an *open* position),
`QDEMarketObserver` is meant to be ticked continuously for every symbol QDE
watches — including symbols with no open position — so QDE always has a
live internal read of "where is this symbol in its trade lifecycle right
now" (see `ObserverState`). That state is context/memory consulted by the
decision models, not a rule engine that decides trades itself: readiness
scoring here only ever gates *state labelling*, never order placement.

Like the rest of the codebase, this module does not own a scheduler/loop.
The calling service ticks `observe()` per symbol on its own cadence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.qde.qde_observer_models import (
    ObservationResult,
    ObserverMemory,
    SymbolObservationInput,
)
from trainer.qde.qde_observer_types import (
    ObserverEvent,
    ObserverState,
    is_valid_transition,
)

# ── Tunables ──────────────────────────────────────────────────────────────────
# Named constants (not magic numbers) so the anti-overtrading / anti-revenge-
# trading behaviour is auditable and can be tuned without touching logic.

PREPARING_THRESHOLD = 0.40
ARMED_THRESHOLD = 0.65
ENTRY_WINDOW_THRESHOLD = 0.82
SETUP_INVALIDATION_DROP = 0.15  # readiness must fall this far below the
# threshold that earned the current state before we retreat a stage

ENTRY_WINDOW_SECONDS = 300.0  # how long ENTRY_WINDOW stays open w/o a fill
BASE_COOLDOWN_SECONDS = 120.0  # normal post-exit quiet period
LOSS_COOLDOWN_MULTIPLIER = 2.5  # extra cooldown per consecutive loss (revenge-trade guard)
MAX_COOLDOWN_SECONDS = 1800.0
REVENGE_TRADE_LOSS_STREAK = 2  # consecutive losses that trigger the extended guard
MIN_REARM_GAP_SECONDS = 60.0  # overtrading guard: minimum time OBSERVING before re-PREPARING


@dataclass
class _EvalContext:
    now: float
    inp: SymbolObservationInput
    mem: ObserverMemory


class QDEMarketObserver:
    """Per-symbol always-on state tracker. Stateless in itself — all memory
    lives in the `ObserverMemory` objects it holds, keyed by instrument, so
    it can be swapped for a persisted-backed implementation without callers
    changing.
    """

    def __init__(self) -> None:
        self._memory: dict[str, ObserverMemory] = {}

    # ── Public API ────────────────────────────────────────────────────────

    def get_memory(self, instrument: str) -> ObserverMemory:
        if instrument not in self._memory:
            self._memory[instrument] = ObserverMemory.new(instrument)
        return self._memory[instrument]

    def load_memory(self, mem: ObserverMemory) -> None:
        """Restore previously-persisted memory (e.g. after a restart)."""
        self._memory[mem.instrument] = mem

    def all_memory(self) -> dict[str, ObserverMemory]:
        return dict(self._memory)

    def observe(self, inp: SymbolObservationInput) -> ObservationResult:
        """Evaluate one tick for one symbol and advance its lifecycle state.

        Pure w.r.t. its inputs plus the held `ObserverMemory` — no I/O, no
        network calls, no scheduling side effects. Safe to call at any
        cadence the caller chooses.
        """
        now = inp.unix_ts or time.time()
        mem = self.get_memory(inp.instrument)
        ctx = _EvalContext(now=now, inp=inp, mem=mem)

        readiness = self._score_readiness(inp)
        mem.readiness_score = readiness

        new_state, event, guard = self._next_state(ctx, readiness)
        transitioned = new_state != mem.state

        if transitioned:
            if not is_valid_transition(mem.state, new_state):
                logger.warning(
                    f"[Observer] {inp.instrument}: rejected invalid transition "
                    f"{mem.state.value} -> {new_state.value}; holding state"
                )
                new_state = mem.state
                event = ObserverEvent.NO_CHANGE
                transitioned = False

        if transitioned:
            mem.previous_state = mem.state
            mem.state = new_state
            mem.entered_state_at = now
            mem.transition_count += 1
            logger.info(
                f"[Observer] {inp.instrument}: {mem.previous_state.value} -> "
                f"{new_state.value} ({event.value}, readiness={readiness:.2f})"
            )

        mem.last_event = event
        mem.last_updated_at = now

        if inp.position_just_closed:
            if inp.last_close_was_loss:
                mem.consecutive_losses += 1
            else:
                mem.consecutive_losses = 0
            mem.cooldown_until = now + self._cooldown_seconds(mem.consecutive_losses)

        result = ObservationResult(
            instrument=inp.instrument,
            state=mem.state,
            previous_state=mem.previous_state if transitioned else mem.state,
            transitioned=transitioned,
            event=event,
            readiness_score=readiness,
            confidence=self._confidence_for_state(mem.state, readiness),
            reasoning=self._explain(mem.state, event, readiness, guard),
            dwell_seconds=mem.dwell_seconds(now),
            cooldown_remaining_s=mem.cooldown_remaining_s(now),
            guard_tripped=guard,
            unix_ts=now,
        )
        return result

    # ── Readiness scoring ─────────────────────────────────────────────────

    def _score_readiness(self, inp: SymbolObservationInput) -> float:
        """Blend microstructure/volatility/liquidity/session quality into a
        single 0-1 "how close is this to a good opportunity" score. This is
        advisory context, not an entry rule — it only ever moves the
        *label*, and the actual entry/exit call remains QDE's decision
        models' job.
        """
        weights = {
            "microstructure": 0.35,
            "volatility": 0.25,
            "liquidity": 0.20,
            "session": 0.20,
        }
        score = (
            weights["microstructure"] * inp.microstructure_quality
            + weights["volatility"] * inp.volatility_quality
            + weights["liquidity"] * inp.liquidity_quality
            + weights["session"] * inp.session_quality
        )
        return max(0.0, min(1.0, score))

    # ── State machine ─────────────────────────────────────────────────────

    def _next_state(
        self, ctx: _EvalContext, readiness: float
    ) -> tuple[ObserverState, ObserverEvent, str]:
        mem, inp, now = ctx.mem, ctx.inp, ctx.now
        state = mem.state

        # In-trade side of the lifecycle is driven by explicit position
        # events, never by readiness scoring.
        if state == ObserverState.POSITION_ACTIVE:
            if inp.position_just_closed:
                return ObserverState.EXITING, ObserverEvent.EXIT_SIGNALLED, ""
            return ObserverState.MANAGING, ObserverEvent.MANAGEMENT_REVIEW, ""

        if state == ObserverState.MANAGING:
            if inp.position_just_closed:
                return ObserverState.EXITING, ObserverEvent.EXIT_SIGNALLED, ""
            return ObserverState.MANAGING, ObserverEvent.MANAGEMENT_REVIEW, ""

        if state == ObserverState.EXITING:
            if not inp.has_open_position:
                return ObserverState.COOLDOWN, ObserverEvent.POSITION_CLOSED, ""
            return ObserverState.EXITING, ObserverEvent.EXIT_SIGNALLED, ""

        if state == ObserverState.COOLDOWN:
            if mem.cooldown_remaining_s(now) <= 0:
                return ObserverState.OBSERVING, ObserverEvent.COOLDOWN_ELAPSED, ""
            return ObserverState.COOLDOWN, ObserverEvent.NO_CHANGE, ""

        # Pre-trade side: a position opening always wins immediately,
        # regardless of which pre-trade state we were in (a fill can land
        # even if our own state tracking was a beat behind).
        if inp.position_just_opened:
            return ObserverState.POSITION_ACTIVE, ObserverEvent.POSITION_OPENED, ""

        if state == ObserverState.OBSERVING:
            if readiness >= PREPARING_THRESHOLD:
                # Overtrading guard: don't re-arm immediately after leaving
                # cooldown — require a minimum quiet dwell in OBSERVING first.
                if mem.dwell_seconds(now) < MIN_REARM_GAP_SECONDS:
                    return ObserverState.OBSERVING, ObserverEvent.OVERTRADE_GUARD, "OVERTRADE"
                return ObserverState.PREPARING, ObserverEvent.SETUP_FORMING, ""
            return ObserverState.OBSERVING, ObserverEvent.NO_CHANGE, ""

        if state == ObserverState.PREPARING:
            if readiness < PREPARING_THRESHOLD - SETUP_INVALIDATION_DROP:
                return ObserverState.OBSERVING, ObserverEvent.SETUP_INVALIDATED, ""
            if readiness >= ARMED_THRESHOLD:
                return ObserverState.ARMED, ObserverEvent.SETUP_STRENGTHENED, ""
            return ObserverState.PREPARING, ObserverEvent.NO_CHANGE, ""

        if state == ObserverState.ARMED:
            if readiness < ARMED_THRESHOLD - SETUP_INVALIDATION_DROP:
                return ObserverState.OBSERVING, ObserverEvent.SETUP_WEAKENED, ""
            if readiness >= ENTRY_WINDOW_THRESHOLD:
                # Revenge-trade guard: an extended cooldown after a losing
                # streak blocks entry even when the setup looks ready.
                if now < mem.cooldown_until:
                    return ObserverState.ARMED, ObserverEvent.REVENGE_TRADE_GUARD, "REVENGE_TRADE"
                mem.entry_window_expires_at = now + ENTRY_WINDOW_SECONDS
                return ObserverState.ENTRY_WINDOW, ObserverEvent.SETUP_STRENGTHENED, ""
            return ObserverState.ARMED, ObserverEvent.NO_CHANGE, ""

        if state == ObserverState.ENTRY_WINDOW:
            if now >= mem.entry_window_expires_at:
                return ObserverState.OBSERVING, ObserverEvent.ENTRY_WINDOW_EXPIRED, ""
            if readiness < ARMED_THRESHOLD - SETUP_INVALIDATION_DROP:
                return ObserverState.OBSERVING, ObserverEvent.SETUP_INVALIDATED, ""
            return ObserverState.ENTRY_WINDOW, ObserverEvent.NO_CHANGE, ""

        # Should be unreachable; fail safe to OBSERVING rather than raise,
        # since this loop must never crash the always-running monitor.
        logger.warning(f"[Observer] {inp.instrument}: unhandled state {state}, resetting")
        return ObserverState.OBSERVING, ObserverEvent.NO_CHANGE, ""

    # ── Helpers ───────────────────────────────────────────────────────────

    def _cooldown_seconds(self, consecutive_losses: int) -> float:
        if consecutive_losses < REVENGE_TRADE_LOSS_STREAK:
            return BASE_COOLDOWN_SECONDS
        extra = (consecutive_losses - REVENGE_TRADE_LOSS_STREAK + 1) * (
            BASE_COOLDOWN_SECONDS * LOSS_COOLDOWN_MULTIPLIER
        )
        return min(MAX_COOLDOWN_SECONDS, BASE_COOLDOWN_SECONDS + extra)

    def _confidence_for_state(self, state: ObserverState, readiness: float) -> float:
        if state in (ObserverState.ARMED, ObserverState.ENTRY_WINDOW):
            return readiness
        if state in (ObserverState.POSITION_ACTIVE, ObserverState.MANAGING):
            return 1.0
        return round(readiness * 0.5, 4)

    def _explain(
        self, state: ObserverState, event: ObserverEvent, readiness: float, guard: str
    ) -> str:
        if guard:
            return f"{guard} guard active — holding at {state.value} (readiness={readiness:.2f})"
        return f"{state.value} (event={event.value}, readiness={readiness:.2f})"
