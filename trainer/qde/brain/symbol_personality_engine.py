"""QDE Brain — symbol personality engine.

`SymbolPersonalityEngine` is the orchestration facade for continuous
behavioral-personality learning, following the exact composition
pattern already established by `HypothesisEngine` / `SetupQueue`: it is
constructed externally by a caller that already holds a `QDEBrain`, it
owns no duplicate storage of its own (`BrainState.behavioral_profile`
is the single source of truth), and it is not wired into
`QDEBrain.on_market_update` automatically — a caller that observes
behavioral events (a trend run ending, a pullback resolving, a
liquidity sweep, a session closing, a news reaction, a reversal) calls
`observe()` itself, the same way `HypothesisEngine`/`SetupQueue` are
driven externally today.

What this module *adds* that did not exist before:
  * Continuous behavioral-profile learning per symbol, via
    `behavioral_memory.BehavioralMemory` folding raw
    `BehavioralObservation`s into `personality_profiles.PersonalityProfile`.
  * The concrete integration seam for "profiles should influence
    hypothesis confidence, risk appetite, setup requirements, and
    expected holding duration": `adjust_confidence()`,
    `risk_appetite_multiplier()`, `setup_thresholds_for()`, and
    `expected_holding_duration()` each wrap the corresponding pure
    function in `personality_profiles.py`. A caller (e.g.
    `HypothesisEngine` when creating/reinforcing a hypothesis,
    `SetupQueue` when choosing thresholds, a risk-sizing module when
    budgeting) consults these instead of re-deriving personality logic
    itself — this module never mutates a hypothesis, opportunity, or
    risk budget directly, exactly as `HypothesisEngine.executable_hypotheses()`
    and `SetupQueue.ready_opportunities()` are seams rather than
    automatic side effects.

Integration points, each reused rather than re-derived:
  * storage       -> `BrainState.behavioral_profile` (via `QDEBrain.get_state`)
  * learning       -> `behavioral_memory.BehavioralMemory` (unchanged)
  * influence math -> `personality_profiles` pure functions (unchanged)
  * working memory -> `BrainState.working_memory` via `WorkingMemory.add`
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.qde.brain.behavioral_memory import BehavioralMemory, BehavioralObservation
from trainer.qde.brain.personality_profiles import (
    PersonalityProfile,
    confidence_adjustment,
    expected_holding_duration_seconds,
    risk_appetite_multiplier,
    setup_requirement_overrides,
)
from trainer.qde.brain.setup_lifecycle import DEFAULT_SETUP_THRESHOLDS, SetupThresholds
from trainer.qde.brain.working_memory import WorkingMemory

if TYPE_CHECKING:
    from trainer.qde.brain.brain import QDEBrain

# A single dimension moving by at least this much between observations
# is worth surfacing into working memory as a notable personality shift
# (e.g. a symbol's liquidity-sweep frequency climbing sharply) — every
# observation still updates the profile regardless of this threshold.
NOTABLE_SHIFT_DELTA = 0.08

_TRACKED_DIMENSIONS = (
    "trend_persistence",
    "avg_pullback_depth",
    "volatility_clustering",
    "liquidity_sweep_frequency",
    "news_sensitivity",
    "reversal_tendency",
)


class SymbolPersonalityEngine:
    """Learn and query each symbol's continuously-evolving behavioral
    personality profile, and expose the pure influence functions a
    caller can consult when forming confidence, sizing risk, tuning
    setup requirements, or estimating holding duration.
    """

    def __init__(self, brain: "QDEBrain") -> None:
        self._brain = brain
        self._memory = BehavioralMemory()
        self._working_memory = WorkingMemory()

    # ── Continuous learning ──────────────────────────────────────────────

    def observe(
        self, symbol: str, observation: BehavioralObservation, now: float | None = None
    ) -> PersonalityProfile:
        """Fold one behavioral observation into `symbol`'s profile.

        Must be called whenever the caller has fresh behavioral evidence
        (not necessarily every market tick) — a single call only ever
        nudges the profile via EMA, by design: personality is learned
        *over time*, never reset or jumped to from one event.
        """
        now = now if now is not None else time.time()
        state = self._brain.get_state(symbol)
        if not state.behavioral_profile.instrument:
            state.behavioral_profile.instrument = symbol

        before = {dim: getattr(state.behavioral_profile, dim) for dim in _TRACKED_DIMENSIONS}
        profile = self._memory.observe(state.behavioral_profile, observation, now=now)

        for dim in _TRACKED_DIMENSIONS:
            delta = getattr(profile, dim) - before[dim]
            if abs(delta) >= NOTABLE_SHIFT_DELTA:
                self._log_shift(state, dim, before[dim], getattr(profile, dim), now)

        self._persist(state)
        return profile

    def profile_for(self, symbol: str) -> PersonalityProfile:
        return self._brain.get_state(symbol).behavioral_profile

    # ── Influence seam ────────────────────────────────────────────────────

    def adjust_confidence(self, symbol: str, base_confidence: float) -> float:
        """Nudge a hypothesis's confidence using this symbol's learned
        trend-persistence-vs-reversal-tendency skew. See
        `personality_profiles.confidence_adjustment` for the math.
        """
        return confidence_adjustment(self.profile_for(symbol), base_confidence)

    def risk_appetite_multiplier(self, symbol: str) -> float:
        """A multiplier a risk-sizing caller can apply to its own base
        risk budget for this symbol. See
        `personality_profiles.risk_appetite_multiplier` for the math.
        """
        return risk_appetite_multiplier(self.profile_for(symbol))

    def setup_thresholds_for(
        self, symbol: str, base: SetupThresholds = DEFAULT_SETUP_THRESHOLDS
    ) -> SetupThresholds:
        """Tune `setup_lifecycle.SetupThresholds` for this symbol from
        its learned behavior — the concrete "profiles influence setup
        requirements" integration point for `SetupQueue`, which a caller
        may pass in as `SetupQueue(brain, thresholds=...)` per symbol.
        """
        return setup_requirement_overrides(self.profile_for(symbol), base)

    def expected_holding_duration(self, symbol: str, base_seconds: float) -> float:
        """Scale a base expected-holding-duration estimate using this
        symbol's learned trend-persistence-vs-pullback-depth skew. See
        `personality_profiles.expected_holding_duration_seconds`.
        """
        return expected_holding_duration_seconds(self.profile_for(symbol), base_seconds)

    # ── Internals ─────────────────────────────────────────────────────────

    def _log_shift(self, state, dimension: str, before: float, after: float, now: float) -> None:
        self._working_memory.add(
            state.working_memory,
            instrument=state.instrument,
            kind="PERSONALITY_SHIFT",
            content={
                "dimension": dimension,
                "before": round(before, 4),
                "after": round(after, 4),
            },
            salience=min(1.0, 0.4 + abs(after - before)),
            now=now,
        )
        logger.debug(
            "SymbolPersonalityEngine({}): {} shifted {:.2f} -> {:.2f}",
            state.instrument,
            dimension,
            before,
            after,
        )

    def _persist(self, state) -> None:
        db = getattr(self._brain, "_db", None)
        if db is not None:
            db.upsert_state(state.to_dict())


__all__ = ["SymbolPersonalityEngine"]
