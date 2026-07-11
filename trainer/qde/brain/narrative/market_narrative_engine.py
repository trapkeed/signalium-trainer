"""QDE Brain — market narrative engine.

Top-level orchestrator for the typed-narrative layer.  One
`MarketNarrativeEngine` instance is held by a `QDEBrain`-level consumer
(or alongside one) and called with every `SymbolObservationInput` tick,
exactly as `QDEBrain.on_market_update` is called from the continuous
ingestion driver.

Responsibilities
----------------
1. Maintain a ``dict[NarrativeKind, SymbolNarrative]`` per symbol.
2. On each tick: expire stale narratives → detect new seeds (Builder) →
   evolve existing narratives (Evolution) → open new ones from seeds →
   detect conflicts (ConflictDetector).
3. Expose integration points for the rest of the QDE reasoning stack:

   suggest_hypothesis_seeds(instrument)
       Return `HypothesisSeed` objects from established narratives so the
       `HypothesisEngine` / `HypothesisTracker` can form concrete hypotheses
       that are grounded in a sustained narrative rather than an isolated
       tick.  This is the hypothesis-generation integration point.

   narrative_conviction_boost(instrument, direction)
       0–1 multiplier derived from narratives that align with `direction`,
       discounted by conflict intensity.  Callers in `conviction_engine`
       or `HypothesisEngine.trade_worthy_hypotheses` may add this to
       `ConvictionScore.conviction` as a narrative-alignment bonus.
       This is the conviction-scoring integration point.

   narrative_context(instrument)
       Compact dict for embedding in `QDEInferenceInput.brain_context` so
       the decision model can reason from narrative state.  This is the
       trade-selection integration point.

   opportunity_rank_signal(instrument)
       0–1 attractiveness score for cross-symbol opportunity ranking.
       Higher when narratives are established, coherent, and aligned.
       This is the opportunity-ranking integration point.

What this module does NOT do
-----------------------------
* It never imports `QDEDecision` or places trades.
* It does not re-derive regime / volatility / liquidity classification.
* It does not own a scheduler; the caller controls invocation cadence.
* It does not duplicate the single-instance `MarketNarrative` summary
  already maintained by `NarrativeEngine` in `narratives.py` — the two
  layers are complementary and the existing summary is consumed as input
  context by `NarrativeBuilder`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.qde.brain.brain_models import Hypothesis, MarketNarrative
from trainer.qde.brain.brain_types import BeliefDirection
from trainer.qde.brain.narrative.narrative_builder import NarrativeBuilder, NarrativeSeed
from trainer.qde.brain.narrative.narrative_conflict_detector import (
    NarrativeConflict,
    NarrativeConflictDetector,
)
from trainer.qde.brain.narrative.narrative_evolution import NarrativeEvolution
from trainer.qde.brain.narrative.narrative_store import (
    NarrativeKind,
    NarrativeMaturity,
    NarrativeStore,
    SymbolNarrative,
)
from trainer.qde.qde_observer_models import ObservationResult, SymbolObservationInput

if TYPE_CHECKING:
    pass  # kept for future typed imports


# ── Integration-point output types ───────────────────────────────────────


@dataclass
class HypothesisSeed:
    """A narrative-suggested hypothesis the `HypothesisEngine` should form.

    Callers pass these to `HypothesisEngine.create()` / `HypothesisTracker.form()`
    rather than letting the narrative engine create hypotheses directly — the
    boundary between narrative reasoning and hypothesis storage stays clear.
    """

    instrument: str
    thesis: str
    direction: BeliefDirection
    confidence: float
    source_narrative_kind: NarrativeKind
    evidence: str

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "thesis": self.thesis,
            "direction": self.direction.value,
            "confidence": round(self.confidence, 4),
            "source_narrative_kind": self.source_narrative_kind.value,
            "evidence": self.evidence,
        }


@dataclass
class NarrativeUpdateResult:
    """Summary of what changed on one `on_market_update` call."""

    instrument: str
    narratives_active: int = 0
    narratives_established: int = 0
    seeds_detected: list[NarrativeSeed] = field(default_factory=list)
    conflicts: list[NarrativeConflict] = field(default_factory=list)
    dominant_kind: NarrativeKind | None = None
    conflict_intensity: float = 0.0
    coherence_score: float = 1.0
    opportunity_signal: float = 0.0
    expired_count: int = 0

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "narratives_active": self.narratives_active,
            "narratives_established": self.narratives_established,
            "seeds_detected": [s.to_dict() for s in self.seeds_detected],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "dominant_kind": self.dominant_kind.value if self.dominant_kind else None,
            "conflict_intensity": round(self.conflict_intensity, 4),
            "coherence_score": round(self.coherence_score, 4),
            "opportunity_signal": round(self.opportunity_signal, 4),
            "expired_count": self.expired_count,
        }


# ── Engine ────────────────────────────────────────────────────────────────


# Narrative kinds that imply a bullish setup when active and established.
_BULLISH_KINDS = frozenset(
    {NarrativeKind.TREND_CONTINUATION, NarrativeKind.ACCUMULATION, NarrativeKind.BREAKOUT}
)
# Narrative kinds that imply a bearish setup.
_BEARISH_KINDS = frozenset(
    {NarrativeKind.REVERSAL, NarrativeKind.DISTRIBUTION, NarrativeKind.EXHAUSTION}
)
# Narrative kinds that are regime-agnostic / direction-neutral.
_NEUTRAL_KINDS = frozenset({NarrativeKind.LIQUIDITY_GRAB})

# Direction labels used to build hypothesis thesis strings.
_DIRECTION_LABEL: dict[NarrativeKind, tuple[str, str]] = {
    # (bullish thesis fragment, bearish thesis fragment)
    NarrativeKind.TREND_CONTINUATION: ("upward trend continuation", "downward trend continuation"),
    NarrativeKind.REVERSAL:           ("bullish reversal setup", "bearish reversal setup"),
    NarrativeKind.ACCUMULATION:       ("accumulation before bullish breakout", ""),
    NarrativeKind.DISTRIBUTION:       ("", "distribution before bearish breakdown"),
    NarrativeKind.BREAKOUT:           ("bullish breakout developing", "bearish breakout developing"),
    NarrativeKind.EXHAUSTION:         ("bearish exhaustion of uptrend", "bullish exhaustion of downtrend"),
    NarrativeKind.LIQUIDITY_GRAB:     ("bullish recovery after liquidity grab", "bearish recovery after liquidity grab"),
}


class MarketNarrativeEngine:
    """Orchestrates typed market narratives across all symbols.

    One instance is typically held by whichever component drives the
    continuous-ingestion loop (e.g. alongside `QDEBrain`), and
    `on_market_update` is called in step with `QDEBrain.on_market_update`.
    """

    def __init__(self) -> None:
        # instrument -> {NarrativeKind.value -> SymbolNarrative}
        self._stores: dict[str, dict[str, SymbolNarrative]] = {}
        self._store = NarrativeStore()
        self._builder = NarrativeBuilder()
        self._evolution = NarrativeEvolution()
        self._conflict_detector = NarrativeConflictDetector()

    # ── Primary update entry point ────────────────────────────────────────

    def on_market_update(
        self,
        inp: SymbolObservationInput,
        existing_narrative: MarketNarrative | None,
        active_hypotheses: list[Hypothesis],
        observation_result: ObservationResult | None = None,
        now: float | None = None,
    ) -> NarrativeUpdateResult:
        """Advance all narratives for `inp.instrument` by one tick.

        Steps (in order):
        1. Expire any narratives that have exceeded their max age.
        2. Run `NarrativeBuilder.detect` to find new seeds.
        3. For seeds whose kind already has a living narrative: evolve it.
        4. For seeds whose kind has no living narrative: open a new one.
        5. Evolve *all other* active narratives (kinds with no new seed),
           so they are updated from the current tick even without a new seed.
        6. Detect and return conflicts.
        """
        now = now if now is not None else time.time()
        instrument = inp.instrument
        store = self._get_store(instrument)

        # Extract observer fields with safe defaults (mirrors QDEBrain.on_market_update)
        observer_state = observation_result.state.value if observation_result else "OBSERVING"
        readiness_score = observation_result.readiness_score if observation_result else 0.0
        dwell_seconds = observation_result.dwell_seconds if observation_result else 0.0

        # 1. Expire stale narratives.
        expired_count = self._store.expire_stale(store, now=now)

        # 2. Detect new seeds for this tick.
        seeds = self._builder.detect(
            instrument=instrument,
            inp=inp,
            existing_narrative=existing_narrative,
            active_hypotheses=active_hypotheses,
            observer_state=observer_state,
            readiness_score=readiness_score,
            dwell_seconds=dwell_seconds,
            now=now,
        )

        seeded_kinds: set[str] = set()

        # 3+4. Process each seed: evolve existing or open new.
        for seed in seeds:
            seeded_kinds.add(seed.kind.value)
            existing = self._store.get(store, seed.kind)
            if existing is not None and existing.is_active():
                # Evolve the living narrative rather than replacing it.
                self._evolution.evolve(
                    narrative=existing,
                    inp=inp,
                    existing_narrative=existing_narrative,
                    active_hypotheses=active_hypotheses,
                    observer_state=observer_state,
                    readiness_score=readiness_score,
                    dwell_seconds=dwell_seconds,
                    now=now,
                )
                logger.debug(
                    "MarketNarrativeEngine[{}]: evolved {} → conf={:.0%} {}",
                    instrument, seed.kind.value,
                    existing.confidence, existing.maturity.value,
                )
            else:
                # Open a fresh narrative from the seed.
                new_narrative = seed.to_narrative(instrument, now=now)
                self._store.upsert(store, new_narrative)
                logger.debug(
                    "MarketNarrativeEngine[{}]: opened {} conf={:.0%}",
                    instrument, seed.kind.value, new_narrative.confidence,
                )

        # 5. Evolve any other active narratives for which no seed fired this tick
        #    (they still need a no-signal update so they decay correctly).
        for narrative in list(store.values()):
            if narrative.kind.value in seeded_kinds:
                continue
            if not narrative.is_active():
                continue
            self._evolution.evolve(
                narrative=narrative,
                inp=inp,
                existing_narrative=existing_narrative,
                active_hypotheses=active_hypotheses,
                observer_state=observer_state,
                readiness_score=readiness_score,
                dwell_seconds=dwell_seconds,
                now=now,
            )

        # 6. Detect conflicts among active narratives.
        active = self._store.active(store)
        conflicts = self._conflict_detector.detect(active)
        conflict_intensity = self._conflict_detector.conflict_intensity(conflicts)
        coherence = self._conflict_detector.coherence_score(active, conflicts)
        dominant = self._store.dominant(store)
        opportunity_signal = self._compute_opportunity_signal(store, conflicts)

        result = NarrativeUpdateResult(
            instrument=instrument,
            narratives_active=len(active),
            narratives_established=len(self._store.established(store)),
            seeds_detected=seeds,
            conflicts=conflicts,
            dominant_kind=dominant.kind if dominant else None,
            conflict_intensity=conflict_intensity,
            coherence_score=coherence,
            opportunity_signal=opportunity_signal,
            expired_count=expired_count,
        )

        logger.debug(
            "MarketNarrativeEngine[{}]: {} active narratives, "
            "coherence={:.0%}, opportunity={:.2f}",
            instrument, len(active), coherence, opportunity_signal,
        )
        return result

    # ── Integration points ────────────────────────────────────────────────

    def suggest_hypothesis_seeds(self, instrument: str) -> list[HypothesisSeed]:
        """Return hypothesis seeds from established narratives.

        **Hypothesis-generation integration point.**

        Callers (e.g. the ingestion driver or `HypothesisEngine`) should
        iterate over these and call `HypothesisEngine.create()` for each,
        checking for existing hypotheses with the same thesis before creating
        (the `HypothesisTracker.form` dedup already handles that).
        """
        store = self._get_store(instrument)
        established = self._store.established(store)
        regime = self._dominant_regime_context(store)
        seeds: list[HypothesisSeed] = []

        # Determine the current narrative's directional lean from the conflict detector.
        active = self._store.active(store)
        net_bias = self._conflict_detector.net_bias(active)

        for narrative in established:
            direction, thesis = self._narrative_to_hypothesis_direction(
                narrative, net_bias, regime
            )
            if not thesis:
                continue
            seeds.append(
                HypothesisSeed(
                    instrument=instrument,
                    thesis=thesis,
                    direction=direction,
                    confidence=min(0.85, narrative.confidence),
                    source_narrative_kind=narrative.kind,
                    evidence=(
                        f"narrative:{narrative.kind.value} "
                        f"conf={narrative.confidence:.0%} "
                        f"({narrative.maturity.value})"
                    ),
                )
            )

        return seeds

    def narrative_conviction_boost(
        self,
        instrument: str,
        direction: BeliefDirection,
    ) -> float:
        """0–1 conviction multiplier from narratives aligned with `direction`.

        **Conviction-scoring integration point.**

        Usage example in a caller::

            score = compute_conviction(hypothesis)
            boost = engine.narrative_conviction_boost(instrument, hypothesis.direction)
            # Apply as a capped additive bonus:
            adjusted_conviction = min(1.0, score.conviction + boost * 0.20)

        The boost is discounted when narratives are internally conflicted, so
        a split narrative picture does not inflate conviction.
        """
        store = self._get_store(instrument)
        active = self._store.active(store)
        if not active:
            return 0.0

        # Gather narratives whose kind is directionally aligned.
        aligned_narratives: list[SymbolNarrative] = []
        if direction == BeliefDirection.BULLISH:
            aligned_kinds = _BULLISH_KINDS
        elif direction == BeliefDirection.BEARISH:
            aligned_kinds = _BEARISH_KINDS
        else:
            aligned_kinds = _NEUTRAL_KINDS

        for n in active:
            if n.kind in aligned_kinds and n.is_actionable():
                aligned_narratives.append(n)

        if not aligned_narratives:
            return 0.0

        # Raw boost: confidence-weighted average of aligned narratives, capped at 0.4.
        total_conf = sum(n.confidence for n in aligned_narratives)
        raw_boost = min(0.40, total_conf / max(1, len(active)) * 0.6)

        # Discount by conflict intensity so a conflicted picture does not boost.
        conflicts = self._conflict_detector.detect(active)
        intensity = self._conflict_detector.conflict_intensity(conflicts)
        discounted = raw_boost * max(0.1, 1.0 - intensity)

        return round(discounted, 4)

    def narrative_context(self, instrument: str) -> dict:
        """Compact narrative summary for `QDEInferenceInput.brain_context`.

        **Trade-selection integration point.**

        This dict is designed to be merged into the existing `brain_context`
        dict that `QDEBrain.to_memory_context` already produces, so inference
        can reason from both the single-instance narrative summary (tone,
        trend maturity, session) and the richer typed narrative layer.
        """
        store = self._get_store(instrument)
        active = self._store.active(store)
        dominant = self._store.dominant(store)
        conflicts = self._conflict_detector.detect(active)
        coherence = self._conflict_detector.coherence_score(active, conflicts)
        net_bias = self._conflict_detector.net_bias(active)

        return {
            "typed_narratives": [n.to_dict() for n in active],
            "dominant_narrative": dominant.to_dict() if dominant else None,
            "narrative_coherence": round(coherence, 4),
            "narrative_conflict_intensity": round(
                self._conflict_detector.conflict_intensity(conflicts), 4
            ),
            "narrative_net_bias": round(net_bias, 4),
            "conflicts": [c.to_dict() for c in conflicts],
            "hypothesis_seeds": [
                s.to_dict() for s in self.suggest_hypothesis_seeds(instrument)
            ],
            "opportunity_signal": round(
                self._compute_opportunity_signal(store, conflicts), 4
            ),
        }

    def opportunity_rank_signal(self, instrument: str) -> float:
        """0–1 attractiveness score for cross-symbol opportunity ranking.

        **Opportunity-ranking integration point.**

        Higher when:
        - Multiple established, coherent narratives are active.
        - Conflict intensity is low (the narrative picture is clear).
        - The dominant narrative is high-confidence.
        - There are no liquidity-grab narratives (which imply uncertainty).

        Callers should combine this with the `AttentionAllocator`'s salience
        signal rather than replacing it.
        """
        store = self._get_store(instrument)
        conflicts = self._conflict_detector.detect(self._store.active(store))
        return self._compute_opportunity_signal(store, conflicts)

    # ── State access ──────────────────────────────────────────────────────

    def narratives_for(self, instrument: str) -> list[SymbolNarrative]:
        """All (active + expired) narratives for one symbol."""
        return list(self._get_store(instrument).values())

    def active_narratives_for(self, instrument: str) -> list[SymbolNarrative]:
        return self._store.active(self._get_store(instrument))

    def established_narratives_for(self, instrument: str) -> list[SymbolNarrative]:
        return self._store.established(self._get_store(instrument))

    def narrative_of_kind(
        self, instrument: str, kind: NarrativeKind
    ) -> SymbolNarrative | None:
        return self._store.get(self._get_store(instrument), kind)

    def all_instruments(self) -> list[str]:
        return list(self._stores.keys())

    def to_snapshot(self, instrument: str) -> list[dict]:
        """Serialise all narratives for one symbol (for persistence or telemetry)."""
        return self._store.to_list_dict(self._get_store(instrument))

    def load_snapshot(self, instrument: str, raw: list[dict]) -> None:
        """Restore a previously-serialised snapshot."""
        self._stores[instrument] = NarrativeStore.from_list_dict(raw)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_store(self, instrument: str) -> dict[str, SymbolNarrative]:
        if instrument not in self._stores:
            self._stores[instrument] = {}
        return self._stores[instrument]

    def _compute_opportunity_signal(
        self,
        store: dict[str, SymbolNarrative],
        conflicts: list[NarrativeConflict],
    ) -> float:
        """Aggregate 0-1 opportunity signal from the current narrative state."""
        active = self._store.active(store)
        if not active:
            return 0.0

        established = [n for n in active if n.maturity == NarrativeMaturity.ESTABLISHED]
        dominant = self._store.dominant(store)

        # Base: fraction of active narratives that are established.
        establishment_score = len(established) / max(1, len(active))

        # Bonus: dominant narrative confidence.
        dominant_conf = dominant.confidence if dominant else 0.0

        # Penalty: conflict intensity.
        intensity = self._conflict_detector.conflict_intensity(conflicts)

        # Penalty: liquidity-grab narrative reduces clarity.
        liq_grab = self._store.get(store, NarrativeKind.LIQUIDITY_GRAB)
        liq_penalty = (
            liq_grab.confidence * 0.3
            if liq_grab is not None and liq_grab.is_active()
            else 0.0
        )

        raw = (
            0.35 * establishment_score
            + 0.35 * dominant_conf
            + 0.30 * (1.0 - intensity)
            - liq_penalty
        )
        return round(max(0.0, min(1.0, raw)), 4)

    def _dominant_regime_context(self, store: dict[str, SymbolNarrative]) -> str:
        """Return the regime context from the highest-confidence active narrative."""
        dominant = self._store.dominant(store)
        return dominant.regime_context if dominant else ""

    def _narrative_to_hypothesis_direction(
        self,
        narrative: SymbolNarrative,
        net_bias: float,
        regime_context: str,
    ) -> tuple[BeliefDirection, str]:
        """Translate a narrative kind → (direction, thesis text).

        For direction-neutral kinds (LIQUIDITY_GRAB), net_bias is used to
        pick a side; if bias is truly neutral (|bias| < 0.1) the seed is
        skipped (returns empty thesis) rather than generating a meaningless
        neutral hypothesis.
        """
        regime = regime_context.upper()
        kind = narrative.kind

        if kind in _BULLISH_KINDS:
            # For TREND_CONTINUATION, the actual direction comes from the regime.
            if kind == NarrativeKind.TREND_CONTINUATION:
                if "TREND_UP" in regime:
                    lbl = _DIRECTION_LABEL[kind][0]
                    return BeliefDirection.BULLISH, f"{narrative.instrument} {lbl}"
                elif "TREND_DOWN" in regime:
                    lbl = _DIRECTION_LABEL[kind][1]
                    return BeliefDirection.BEARISH, f"{narrative.instrument} {lbl}"
                elif net_bias >= 0.1:
                    lbl = _DIRECTION_LABEL[kind][0]
                    return BeliefDirection.BULLISH, f"{narrative.instrument} {lbl}"
                elif net_bias <= -0.1:
                    lbl = _DIRECTION_LABEL[kind][1]
                    return BeliefDirection.BEARISH, f"{narrative.instrument} {lbl}"
                else:
                    return BeliefDirection.NEUTRAL, ""
            # ACCUMULATION → bullish; BREAKOUT → net_bias-guided.
            if kind == NarrativeKind.ACCUMULATION:
                lbl = _DIRECTION_LABEL[kind][0]
                return BeliefDirection.BULLISH, f"{narrative.instrument} {lbl}"
            if kind == NarrativeKind.BREAKOUT:
                if net_bias >= 0.0:
                    lbl = _DIRECTION_LABEL[kind][0]
                    return BeliefDirection.BULLISH, f"{narrative.instrument} {lbl}"
                else:
                    lbl = _DIRECTION_LABEL[kind][1]
                    return BeliefDirection.BEARISH, f"{narrative.instrument} {lbl}"

        elif kind in _BEARISH_KINDS:
            if kind == NarrativeKind.REVERSAL:
                # A REVERSAL of an uptrend is bearish, of a downtrend is bullish.
                if "TREND_UP" in regime:
                    lbl = _DIRECTION_LABEL[kind][1]
                    return BeliefDirection.BEARISH, f"{narrative.instrument} {lbl}"
                elif "TREND_DOWN" in regime:
                    lbl = _DIRECTION_LABEL[kind][0]
                    return BeliefDirection.BULLISH, f"{narrative.instrument} {lbl}"
                elif net_bias >= 0.1:
                    lbl = _DIRECTION_LABEL[kind][1]  # reversal of bullish run
                    return BeliefDirection.BEARISH, f"{narrative.instrument} {lbl}"
                elif net_bias <= -0.1:
                    lbl = _DIRECTION_LABEL[kind][0]
                    return BeliefDirection.BULLISH, f"{narrative.instrument} {lbl}"
                else:
                    return BeliefDirection.NEUTRAL, ""
            if kind == NarrativeKind.DISTRIBUTION:
                lbl = _DIRECTION_LABEL[kind][1]
                return BeliefDirection.BEARISH, f"{narrative.instrument} {lbl}"
            if kind == NarrativeKind.EXHAUSTION:
                # Exhaustion of an uptrend → bearish; downtrend → bullish.
                if "TREND_UP" in regime or net_bias >= 0.1:
                    lbl = _DIRECTION_LABEL[kind][0]
                    return BeliefDirection.BEARISH, f"{narrative.instrument} {lbl}"
                else:
                    lbl = _DIRECTION_LABEL[kind][1]
                    return BeliefDirection.BULLISH, f"{narrative.instrument} {lbl}"

        elif kind == NarrativeKind.LIQUIDITY_GRAB:
            if net_bias >= 0.15:
                lbl = _DIRECTION_LABEL[kind][0]
                return BeliefDirection.BULLISH, f"{narrative.instrument} {lbl}"
            elif net_bias <= -0.15:
                lbl = _DIRECTION_LABEL[kind][1]
                return BeliefDirection.BEARISH, f"{narrative.instrument} {lbl}"
            else:
                return BeliefDirection.NEUTRAL, ""

        return BeliefDirection.NEUTRAL, ""


__all__ = [
    "MarketNarrativeEngine",
    "HypothesisSeed",
    "NarrativeUpdateResult",
]
