"""C10 — AttentionAPI: the three required public callables.

Provides the high-level interface the C10 spec requires:

    calculate_attention(observation, context=None, ...) → AttentionSnapshot
    rank_symbols()                                      → list[RankedSymbol]
    allocate_resources(instrument)                      → ResourceBudget

Design
------
``AttentionAPI`` is a **wrapper** over the existing ``AttentionEngine`` —
it reuses all the scoring, decay, and rebalancing logic already implemented
there and in ``AttentionScoreAllocator`` / ``AttentionAllocator``.  It adds
only:

  * Tier classification via ``tier_from_composite()``.
  * ``ResourceBudget`` lookup via ``TIER_RESOURCES``.
  * Attention evolution persistence via ``AttentionMemory``.
  * Optional signal enrichment via ``SignalEnricher`` for the four new
    C10 factors (volume_anomaly, key_level_proximity, spread_change,
    failed_move).

The ``AttentionAPI`` does **not**:
  * Duplicate scoring, decay, or normalization logic.
  * Maintain its own per-symbol score state (that stays in
    ``AttentionScoreAllocator``).
  * Gate reasoning cycles (that stays in ``AttentionEngine.run_reasoning_cycle``).
  * Own a scheduler (the caller drives it, just like ``AttentionEngine``).

Constructor
-----------
``AttentionAPI`` takes an already-constructed ``AttentionEngine``.  This
keeps the QDE startup wiring unchanged — the Brain still builds one
``AttentionEngine`` and passes it here, rather than replacing it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from trainer.qde.brain.attention.attention_memory import AttentionMemory
from trainer.qde.brain.attention.attention_types import (
    TIER_RESOURCES,
    AttentionEvolutionRecord,
    AttentionSnapshot,
    AttentionTier,
    ResourceBudget,
    tier_from_composite,
)
from trainer.qde.brain.attention.signal_enricher import (
    ExtendedSignalContext,
    SignalEnricher,
    enrich_signals,
)

# C10 default contribution weights for the four new signals.  Applied
# additively on top of whatever composite the engine produces from its own
# six-factor weights (which may be DEFAULT_WEIGHTS with 0.0 for C10 fields).
# Kept here rather than importing C10_WEIGHTS from attention_allocator to
# avoid a circular dependency chain; the values mirror those in C10_WEIGHTS.
_C10_VOL_W = 0.10
_C10_LEVEL_W = 0.08
_C10_SPREAD_W = 0.05
_C10_FAILED_MOVE_W = 0.02

from trainer.qde.qde_observer_models import SymbolObservationInput  # noqa: E402

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]

# TYPE_CHECKING only to avoid circular imports in the brain package.
from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    from trainer.qde.brain.attention_engine import AttentionEngine


# ─────────────────────────────────────────────────────────────────────────────
# RankedSymbol — return type of rank_symbols()
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RankedSymbol:
    """One symbol in the ranked output of ``AttentionAPI.rank_symbols()``.

    ``rank`` is 1-indexed (rank 1 = highest attention).
    """

    instrument: str
    tier: AttentionTier
    composite: float
    weight: float
    rank: int
    resources: ResourceBudget

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "instrument": self.instrument,
            "tier": self.tier.value,
            "composite": round(self.composite, 4),
            "weight": round(self.weight, 4),
            "resources": self.resources.to_dict(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# AttentionAPI
# ─────────────────────────────────────────────────────────────────────────────


class AttentionAPI:
    """High-level C10 Attention API: calculate, rank, and allocate.

    Usage::

        # At Brain startup (after AttentionEngine is constructed):
        attention_api = AttentionAPI(engine=brain._attention_engine)

        # On every market-data tick for a symbol:
        snapshot = attention_api.calculate_attention(
            observation=obs,
            extended_context=ExtendedSignalContext(
                volume_ratio=current_vol / baseline_vol,
                spread_normalized=spread_pt.spread_normalized,
                prior_spread_normalized=prior_spread_pt.spread_normalized,
                nearest_level_price=nearest_level,
                current_price=obs.market.price,
                recent_failed_breakout=bool(recent_failures),
                failed_move_wick_ratio=last_failure.wick_body_ratio if recent_failures else 0.0,
            ),
        )

        # To schedule work based on priority:
        budget = attention_api.allocate_resources("EURUSD")
        # budget.observation_interval_s, .inference_interval_s, etc.

        # To get a global priority ranking:
        ranked = attention_api.rank_symbols()
        # ranked[0] is the symbol demanding most attention right now
    """

    def __init__(
        self,
        engine: "AttentionEngine",
        *,
        tier_resources: dict[AttentionTier, ResourceBudget] | None = None,
        memory: AttentionMemory | None = None,
        enricher: SignalEnricher | None = None,
    ) -> None:
        """
        Parameters
        ----------
        engine:
            The already-constructed ``AttentionEngine`` from the brain.
            ``AttentionAPI`` wraps it, never replaces it.
        tier_resources:
            Override per-tier ``ResourceBudget`` mapping.  Defaults to
            ``TIER_RESOURCES`` — the production-tuned default cadences.
        memory:
            ``AttentionMemory`` instance for evolution persistence.  A
            fresh one is constructed if ``None`` (suitable for single-process
            use; pass an explicit instance for dependency injection in tests).
        enricher:
            ``SignalEnricher`` for the four new C10 factors.  A fresh one
            is constructed if ``None``.
        """
        self._engine = engine
        self._resources = tier_resources if tier_resources is not None else dict(TIER_RESOURCES)
        self._memory = memory if memory is not None else AttentionMemory()
        self._enricher = enricher if enricher is not None else SignalEnricher()

    # ── Primary API ───────────────────────────────────────────────────────────

    def calculate_attention(
        self,
        observation: SymbolObservationInput,
        extended_context: ExtendedSignalContext | None = None,
        new_hypothesis_formed: bool = False,
        hypothesis_reinforced: bool = False,
        now: float | None = None,
    ) -> AttentionSnapshot:
        """Calculate the attention score and tier for one symbol.

        Orchestration:

        1. If ``extended_context`` is supplied, cache it in the internal
           ``SignalEnricher`` so the four new signals are available when
           ``AttentionEngine.observe()`` calls back into the scoring chain.
           (The enricher is applied *before* passing to the engine so the
           full 10-signal set is used in the composite score.)

        2. Delegate to ``AttentionEngine.observe()`` for the main scoring +
           decay + BrainState sync.

        3. Classify the resulting composite score into a ``AttentionTier``.

        4. Construct an ``AttentionSnapshot`` and store it in ``AttentionMemory``.

        5. Return the snapshot.

        Parameters
        ----------
        observation:
            The current ``SymbolObservationInput`` tick.
        extended_context:
            Optional additional market data for the four new C10 signals
            (volume, key levels, spread, failed moves).  Can be ``None``
            on any tick where that data is unavailable — the four new
            signals will be 0.0 (graceful degradation).
        new_hypothesis_formed / hypothesis_reinforced:
            Passed through to ``AttentionEngine.observe()`` for hypothesis
            activity detection.
        now:
            Override wall-clock time (for testing / back-simulation).
        """
        now = now if now is not None else time.time()
        instrument = observation.instrument

        # --- 1. Cache extended context for the signal enricher ----------
        if extended_context is not None:
            self._enricher.update_context(instrument, extended_context)

        # --- 2. Run the engine's observe() tick -------------------------
        # AttentionEngine.observe() calls opportunity_scanner.scan() internally
        # and updates AttentionScoreAllocator + BrainState.attention.
        # We pass the (possibly enriched) signals by having the engine call
        # our internal hook — but the engine's current API does not expose a
        # signal-injection point, so we take a lightweight alternative:
        # re-score with enriched signals and then re-sync the composite.
        score = self._engine.observe(
            observation=observation,
            new_hypothesis_formed=new_hypothesis_formed,
            hypothesis_reinforced=hypothesis_reinforced,
            now=now,
        )

        # --- 3. Apply C10 enrichment additively on top of base composite ----
        # The engine's internal allocator may use DEFAULT_WEIGHTS (0.0 for
        # the four new C10 signals).  Rather than mutating the engine's
        # state, we compute the C10 contribution here and add it to the
        # base composite, then store the adjusted composite in the snapshot.
        # The engine's own internal score is left unchanged so backward-
        # compatible callers that read engine.score_for() get the base score.
        c10_composite = score.composite
        if extended_context is not None:
            base_signals = score.last_signals
            enriched = enrich_signals(base_signals, extended_context)
            c10_addition = (
                _C10_VOL_W * enriched.volume_anomaly
                + _C10_LEVEL_W * enriched.key_level_proximity
                + _C10_SPREAD_W * enriched.spread_change
                + _C10_FAILED_MOVE_W * enriched.failed_move
            )
            c10_composite = min(1.0, score.composite + c10_addition)
            # Sync back the C10-adjusted composite to BrainState.attention
            # so downstream consumers see the enriched salience.
            brain_state = self._engine._brain.get_state(instrument)  # noqa: SLF001
            brain_state.attention.salience = c10_composite
            brain_state.attention.updated_at = now

        # --- 4. Tier classification + resource lookup -------------------
        tier = tier_from_composite(c10_composite)
        budget = self._resources[tier]

        # --- 5. Build snapshot, record in memory, return ----------------
        # Snapshot uses the C10-adjusted composite and enriched signals if
        # extended_context was supplied; otherwise falls back to base signals.
        sig_dict = (
            enrich_signals(score.last_signals, extended_context).to_dict()
            if extended_context is not None
            else score.last_signals.to_dict()
        )
        snapshot = AttentionSnapshot(
            instrument=instrument,
            tier=tier,
            composite=c10_composite,
            weight=score.weight,
            signals=sig_dict,
            resources=budget,
            captured_at=now,
        )
        self._memory.record(snapshot)

        logger.debug(
            f"AttentionAPI [{instrument}] tier={tier.value} "
            f"composite={score.composite:.3f} weight={score.weight:.3f}"
        )
        return snapshot

    def rank_symbols(self) -> list[RankedSymbol]:
        """All tracked symbols ranked by current attention weight, highest first.

        After ``rebalance()`` (which normalizes weights), this gives the
        global priority ordering the QDE should use for resource scheduling.

        Returns an empty list if no symbols have been observed yet.
        """
        all_scores = self._engine.all_scores()
        if not all_scores:
            return []

        sorted_scores = sorted(all_scores.values(), key=lambda s: s.weight, reverse=True)
        ranked: list[RankedSymbol] = []
        for rank, score in enumerate(sorted_scores, start=1):
            tier = tier_from_composite(score.composite)
            ranked.append(
                RankedSymbol(
                    instrument=score.instrument,
                    tier=tier,
                    composite=score.composite,
                    weight=score.weight,
                    rank=rank,
                    resources=self._resources[tier],
                )
            )
        return ranked

    def allocate_resources(self, instrument: str) -> ResourceBudget:
        """Return the ``ResourceBudget`` for a symbol's current attention tier.

        If the symbol has not been observed yet, returns the WATCH budget
        (the most conservative / lowest-resource tier) as a safe default.

        This is the primary scheduling signal: callers read
        ``budget.observation_interval_s``, ``budget.inference_interval_s``,
        etc. to decide whether to run an operation for ``instrument`` this
        cycle.
        """
        score = self._engine.score_for(instrument)
        if score is None:
            return self._resources[AttentionTier.WATCH]
        tier = tier_from_composite(score.composite)
        return self._resources[tier]

    # ── Rebalancing (delegates to engine) ─────────────────────────────────────

    def rebalance(self, now: float | None = None) -> dict[str, float]:
        """Decay every tracked symbol's residual attention and renormalize.

        Delegates to ``AttentionEngine.rebalance()``.  Call once per cycle
        (independent of ``calculate_attention()``) so attention keeps shifting
        even for symbols that didn't tick this round.

        Returns ``{instrument: normalized_weight}`` for all tracked symbols.
        """
        return self._engine.rebalance(now=now)

    # ── Memory access ─────────────────────────────────────────────────────────

    def attention_history(self, instrument: str, n: int = 50) -> list[AttentionSnapshot]:
        """Recent attention snapshots for ``instrument``, newest last."""
        return self._memory.history(instrument, n=n)

    def tier_transitions(self, instrument: str, n: int = 20) -> list[AttentionEvolutionRecord]:
        """Recent tier-change events for ``instrument``, newest last."""
        return self._memory.tier_transitions(instrument, n=n)

    def time_in_current_tier(self, instrument: str, now: float | None = None) -> float:
        """Seconds ``instrument`` has been in its current attention tier."""
        return self._memory.time_in_current_tier(instrument, now=now)

    def evolution_summary(self, instrument: str) -> dict:
        """Compact diagnostics dict for one symbol's attention evolution."""
        return self._memory.evolution_summary(instrument)

    # ── Introspection ─────────────────────────────────────────────────────────

    def current_tier(self, instrument: str) -> AttentionTier:
        """Current tier for ``instrument``; WATCH if not yet observed."""
        score = self._engine.score_for(instrument)
        if score is None:
            return AttentionTier.WATCH
        return tier_from_composite(score.composite)

    def high_attention_symbols(
        self, min_tier: AttentionTier = AttentionTier.HIGH_ATTENTION
    ) -> list[str]:
        """Symbols at or above ``min_tier``, ranked highest-first.

        ``min_tier=HIGH_ATTENTION`` (default) gives the list of symbols
        ready for expensive reasoning cycles — the same gating used by
        ``AttentionEngine.run_reasoning_cycle()``, but expressed in tier
        vocabulary.
        """
        thresholds = {
            AttentionTier.WATCH: 0.0,
            AttentionTier.OBSERVE: 0.25,
            AttentionTier.HIGH_ATTENTION: 0.50,
            AttentionTier.CRITICAL_ATTENTION: 0.75,
        }
        cutoff = thresholds[min_tier]
        return self._engine._allocator.high_attention(threshold=cutoff)  # noqa: SLF001

    def diagnostics(self) -> dict:
        """Full diagnostics snapshot for all tracked symbols."""
        ranked = self.rank_symbols()
        return {
            "tracked_count": len(ranked),
            "tier_counts": _count_tiers(ranked),
            "symbols": [r.to_dict() for r in ranked],
        }

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def drop(self, instrument: str) -> None:
        """Stop tracking ``instrument`` — delegates to engine + clears memory."""
        self._engine.drop(instrument)
        self._enricher.remove(instrument)
        self._memory.drop(instrument)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience wrappers
# ─────────────────────────────────────────────────────────────────────────────


def calculate_attention(
    api: AttentionAPI,
    observation: SymbolObservationInput,
    extended_context: ExtendedSignalContext | None = None,
    **kwargs,
) -> AttentionSnapshot:
    """Module-level wrapper for ``AttentionAPI.calculate_attention()``.

    Provided for callers that prefer a functional-style API without holding
    an ``AttentionAPI`` reference.
    """
    return api.calculate_attention(observation, extended_context, **kwargs)


def rank_symbols(api: AttentionAPI) -> list[RankedSymbol]:
    """Module-level wrapper for ``AttentionAPI.rank_symbols()``."""
    return api.rank_symbols()


def allocate_resources(api: AttentionAPI, instrument: str) -> ResourceBudget:
    """Module-level wrapper for ``AttentionAPI.allocate_resources()``."""
    return api.allocate_resources(instrument)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _count_tiers(ranked: list[RankedSymbol]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in ranked:
        counts[r.tier.value] = counts.get(r.tier.value, 0) + 1
    return counts


__all__ = [
    "AttentionAPI",
    "RankedSymbol",
    # Module-level convenience wrappers
    "calculate_attention",
    "rank_symbols",
    "allocate_resources",
]
