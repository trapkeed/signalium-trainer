"""QDE Brain — narrative API.

Top-level facade exposing the three spec-required methods for Phase C3:

    ``generate_narrative(instrument, inp, existing_narrative, hypotheses,
                          features, portfolio, session, now)``
    ``store_narrative(instrument, result)``
    ``evaluate_narrative(instrument, kind)``

All three delegate to the components already built in this package:

    generate_narrative()
        1. Recall memory context (memory bridge → episodic / semantic).
        2. Apply memory context to adjust active narrative confidences.
        3. Call ``MarketNarrativeEngine.on_market_update`` to advance all
           narratives for the symbol by one tick.
        4. Apply calibration to established narratives (calibrator).
        5. Compute probability distribution (probability engine).
        6. Return a rich ``NarrativeGenerationResult``.

    store_narrative()
        1. Write each newly-established narrative as an episode to episodic
           memory via the memory bridge.
        2. Record establishments in the outcome tracker (so future calls to
           evaluate_narrative have data).

    evaluate_narrative()
        1. Return quality scores from the outcome tracker for all kinds (or
           a specific kind) for the instrument.
        2. Include current probability distribution for context.
        3. Include calibration factors so the caller can see how the
           calibrator would adjust seeds.

``NarrativeAPI`` holds its own default instances of all sub-components so
it works as a standalone entry point.  Callers that already own a
``MarketNarrativeEngine`` can pass it in to share state.

What this module does NOT do
----------------------------
* It never imports ``QDEDecision`` or places trades.
* It does not own a scheduler or run on a timer.
* It does not re-derive volatility / regime / liquidity — those come from
  the ``SymbolObservationInput`` and ``MarketNarrative`` passed in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.qde.brain.brain_models import Hypothesis, MarketNarrative
from trainer.qde.brain.narrative.market_narrative_engine import (
    MarketNarrativeEngine,
    NarrativeUpdateResult,
)
from trainer.qde.brain.narrative.narrative_calibrator import (
    CalibrationResult,
    NarrativeCalibrator,
)
from trainer.qde.brain.narrative.narrative_conflict_detector import NarrativeConflictDetector
from trainer.qde.brain.narrative.narrative_memory_bridge import (
    NarrativeMemoryBridge,
    NarrativeMemoryContext,
    NarrativePortfolioContext,
)
from trainer.qde.brain.narrative.narrative_outcome_tracker import (
    NarrativeOutcomeTracker,
)
from trainer.qde.brain.narrative.narrative_probability import (
    NarrativeProbabilityDistribution,
    NarrativeProbabilityEngine,
)
from trainer.qde.brain.narrative.narrative_store import (
    NarrativeKind,
    NarrativeMaturity,
    NarrativeStore,
    SymbolNarrative,
)
from trainer.qde.qde_observer_models import ObservationResult, SymbolObservationInput

if TYPE_CHECKING:
    pass


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass
class NarrativeGenerationResult:
    """Rich output of one ``generate_narrative`` call.

    ``update_result``
        The ``NarrativeUpdateResult`` from ``MarketNarrativeEngine`` — includes
        count of active/established narratives, detected seeds, conflicts, and
        the opportunity signal.

    ``probability_distribution``
        Normalized probability distribution across all 7 ``NarrativeKind``
        values.  This is the primary "probabilistic narratives" output.

    ``narratives_active``
        List of all currently active (non-EXPIRED) ``SymbolNarrative`` objects.

    ``memory_context``
        ``NarrativeMemoryContext`` from the memory bridge, or None if the
        bridge was not configured.

    ``portfolio_influence``
        Dict from memory bridge describing how portfolio state modifies
        narrative relevance.

    ``calibration_results``
        Per-kind ``CalibrationResult`` for each established narrative.

    ``dominant_narrative``
        The highest-confidence active narrative, or None.
    """

    instrument: str
    timestamp: float

    update_result: NarrativeUpdateResult
    probability_distribution: NarrativeProbabilityDistribution
    narratives_active: list[SymbolNarrative] = field(default_factory=list)
    memory_context: NarrativeMemoryContext | None = None
    portfolio_influence: dict = field(default_factory=dict)
    calibration_results: list[CalibrationResult] = field(default_factory=list)
    dominant_narrative: SymbolNarrative | None = None

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "timestamp": round(self.timestamp, 3),
            "probability_distribution": self.probability_distribution.to_dict(),
            "update_result": self.update_result.to_dict(),
            "narratives_active": [n.to_dict() for n in self.narratives_active],
            "dominant_narrative": (
                self.dominant_narrative.to_dict() if self.dominant_narrative else None
            ),
            "memory_context": self.memory_context.to_dict() if self.memory_context else None,
            "portfolio_influence": self.portfolio_influence,
            "calibration_results": [c.to_dict() for c in self.calibration_results],
        }


# ── API ───────────────────────────────────────────────────────────────────────


class NarrativeAPI:
    """Phase C3 public API: ``generate_narrative``, ``store_narrative``,
    ``evaluate_narrative``.

    One instance is shared across all instruments (like
    ``MarketNarrativeEngine`` itself).

    Parameters
    ----------
    engine:
        ``MarketNarrativeEngine`` instance.  If None, a fresh one is created.
    outcome_tracker:
        ``NarrativeOutcomeTracker`` instance.  If None, a fresh one is created.
    calibrator:
        ``NarrativeCalibrator`` instance.  If None, a fresh one is created.
    memory_bridge:
        ``NarrativeMemoryBridge`` instance.  If None, a bridge with no C2
        backends is created (all memory recall steps are no-ops).
    """

    def __init__(
        self,
        engine: MarketNarrativeEngine | None = None,
        outcome_tracker: NarrativeOutcomeTracker | None = None,
        calibrator: NarrativeCalibrator | None = None,
        memory_bridge: NarrativeMemoryBridge | None = None,
    ) -> None:
        self._engine = engine or MarketNarrativeEngine()
        self._tracker = outcome_tracker or NarrativeOutcomeTracker()
        self._calibrator = calibrator or NarrativeCalibrator()
        self._bridge = memory_bridge or NarrativeMemoryBridge()
        self._prob_engine = NarrativeProbabilityEngine()
        self._store_ops = NarrativeStore()
        self._conflict = NarrativeConflictDetector()

        # Per-instrument cache of narrative_id → episode_id for outcome linking
        self._episode_ids: dict[str, str] = {}  # narrative_id → episode_id

    # ── Primary API ───────────────────────────────────────────────────────

    def generate_narrative(
        self,
        instrument: str,
        inp: SymbolObservationInput,
        existing_narrative: MarketNarrative | None = None,
        active_hypotheses: list[Hypothesis] | None = None,
        observation_result: ObservationResult | None = None,
        features: dict | None = None,
        portfolio: NarrativePortfolioContext | None = None,
        session: str = "",
        now: float | None = None,
    ) -> NarrativeGenerationResult:
        """Generate narratives for ``instrument`` for the current tick.

        This is the primary entry point.  Call once per market tick per symbol,
        in the same cadence as ``QDEBrain.on_market_update``.

        Parameters
        ----------
        instrument:
            Symbol to generate narratives for.
        inp:
            Current market observation input.
        existing_narrative:
            The ``MarketNarrative`` summary from ``NarrativeEngine`` (the
            single-instance layer) or None.
        active_hypotheses:
            Active hypotheses from ``HypothesisTracker`` or empty list.
        observation_result:
            Observer state + readiness from the observer engine.
        features:
            12-dim feature dict (keys matching ``EpisodeFeatures``) for
            memory recall.  Can be computed from ``inp`` fields or passed
            from the QDE inference input.  None disables memory recall.
        portfolio:
            Current portfolio state for portfolio influence calculation.
        session:
            Current session label for episodic memory recall filtering.
        now:
            Override current time (for replay/testing).
        """
        now = now if now is not None else time.time()
        hyps = active_hypotheses or []

        # ── Step 1: Memory recall ─────────────────────────────────────────
        memory_ctx: NarrativeMemoryContext | None = None
        if features is not None:
            try:
                memory_ctx = self._bridge.recall_context(
                    instrument=instrument,
                    features=features,
                    session=session,
                    portfolio=portfolio,
                    active_narratives=self._engine.active_narratives_for(instrument),
                    now=now,
                )
            except Exception as exc:
                logger.warning(f"[NarrativeAPI] memory recall failed: {exc}")

        # ── Step 2: Apply memory adjustments to active narratives ─────────
        if memory_ctx is not None:
            self._bridge.apply_memory_context(
                self._engine.active_narratives_for(instrument),
                memory_ctx,
            )

        # ── Step 3: Advance narratives (builder + evolution + conflicts) ──
        update_result = self._engine.on_market_update(
            inp=inp,
            existing_narrative=existing_narrative,
            active_hypotheses=hyps,
            observation_result=observation_result,
            now=now,
        )

        # ── Step 4: Calibrate established narratives ──────────────────────
        active = self._engine.active_narratives_for(instrument)
        calib_results: list[CalibrationResult] = []

        for narrative in active:
            if narrative.maturity != NarrativeMaturity.ESTABLISHED:
                continue
            quality = self._tracker.quality_score(instrument, narrative.kind)
            cr = self._calibrator.calibrate_narrative(
                narrative, quality, apply=False  # inspect only, don't mutate yet
            )
            calib_results.append(cr)

        # ── Step 5: Probability distribution ─────────────────────────────
        # Compute boost from memory context if available
        boost_kind = None
        boost_amt = 0.0
        if memory_ctx is not None and memory_ctx.top_concept is not None:
            from trainer.qde.brain.narrative.narrative_memory_bridge import (
                _concept_to_narrative_kind,
            )

            boost_kind = _concept_to_narrative_kind(memory_ctx.top_concept)
            boost_amt = round(memory_ctx.concept_similarity * 0.08, 4)

        dist = self._prob_engine.compute(
            active,
            boost_kind=boost_kind,
            boost_amount=boost_amt,
        )

        # ── Step 6: Dominant narrative ────────────────────────────────────
        dominant = self._store_ops.dominant({n.kind.value: n for n in active})

        return NarrativeGenerationResult(
            instrument=instrument,
            timestamp=now,
            update_result=update_result,
            probability_distribution=dist,
            narratives_active=active,
            memory_context=memory_ctx,
            portfolio_influence=(memory_ctx.portfolio_influence if memory_ctx else {}),
            calibration_results=calib_results,
            dominant_narrative=dominant,
        )

    def store_narrative(
        self,
        instrument: str,
        result: NarrativeGenerationResult,
        features: dict | None = None,
        portfolio: NarrativePortfolioContext | None = None,
        now: float | None = None,
    ) -> list[str]:
        """Persist newly-established narratives to memory.

        Should be called immediately after ``generate_narrative`` to record
        any narratives that just became ESTABLISHED into:
          1. The ``NarrativeOutcomeTracker`` (for quality tracking).
          2. Episodic memory (via ``NarrativeMemoryBridge``) so future calls
             can recall this market state as a historical precedent.

        Returns
        -------
        list[str]
            Episode IDs created in episodic memory (one per stored narrative).
            Empty if episodic memory is not configured.
        """
        now = now if now is not None else time.time()
        episode_ids: list[str] = []

        established = [
            n for n in result.narratives_active if n.maturity == NarrativeMaturity.ESTABLISHED
        ]

        # direction from coherent narrative bias
        net_bias = self._conflict.net_bias(result.narratives_active)
        direction = 1 if net_bias > 0.1 else (-1 if net_bias < -0.1 else 0)

        for narrative in established:
            # Track in outcome tracker (idempotent)
            self._tracker.record_establishment(
                narrative_id=narrative.narrative_id,
                kind=narrative.kind,
                instrument=instrument,
                direction=direction,
                now=now,
            )

            # Store in episodic memory
            if features is not None:
                ep_id = self._bridge.store_narrative_episode(
                    narrative=narrative,
                    features=features,
                    portfolio=portfolio,
                    action="ENTER" if direction != 0 else "NO_TRADE",
                    confidence=narrative.confidence,
                    reasoning=(
                        f"narrative:{narrative.kind.value} established "
                        f"conf={narrative.confidence:.2f} "
                        f"coherence={result.update_result.coherence_score:.2f}"
                    ),
                    now=now,
                )
                if ep_id:
                    self._episode_ids[narrative.narrative_id] = ep_id
                    episode_ids.append(ep_id)

        return episode_ids

    def evaluate_narrative(
        self,
        instrument: str,
        kind: NarrativeKind | None = None,
        now: float | None = None,
    ) -> dict:
        """Return narrative quality evaluation for ``instrument``.

        Parameters
        ----------
        instrument:
            Symbol to evaluate.
        kind:
            If given, returns evaluation for that specific kind only.
            If None, returns evaluation for all 7 kinds.

        Returns
        -------
        dict with keys:
          ``quality_scores``       — per-kind ``NarrativeQualityScore`` dicts.
          ``calibration_factors``  — per-kind scaling factors from calibrator.
          ``probability_distribution`` — current probability distribution
                                     (snapshot of what narratives look like now).
          ``dominant_kind``        — current dominant kind's value or None.
          ``instrument``           — instrument name.
          ``total_resolved``       — total resolved outcome records.
        """
        now = now if now is not None else time.time()

        kinds = [kind] if kind is not None else list(NarrativeKind)

        quality_scores: dict[str, dict] = {}
        calibration_factors: dict[str, float] = {}
        total_resolved = 0

        for k in kinds:
            q = self._tracker.quality_score(instrument, k)
            quality_scores[k.value] = q.to_dict()
            calibration_factors[k.value] = self._calibrator.compute_factor(k, q).factor
            total_resolved += q.sample_count

        # Current probability distribution
        active = self._engine.active_narratives_for(instrument)
        dist = self._prob_engine.compute(active)

        # Global quality for comparison
        global_scores: dict[str, dict] = {}
        for k in kinds:
            global_scores[k.value] = self._tracker.quality_score_global(k).to_dict()

        return {
            "instrument": instrument,
            "quality_scores": quality_scores,
            "global_quality_scores": global_scores,
            "calibration_factors": calibration_factors,
            "probability_distribution": dist.to_dict(),
            "dominant_kind": dist.dominant.value if dist.dominant else None,
            "active_narrative_count": len(active),
            "total_resolved": total_resolved,
            "evaluated_at": round(now, 3),
        }

    def record_outcome(
        self,
        narrative_id: str,
        direction_correct: bool,
        reward_normalized: float,
        resolution_bars: int = 0,
        now: float | None = None,
    ) -> bool:
        """Record the final outcome for a narrative that reached ESTABLISHED.

        Call this when a trade or market event resolves the narrative's
        prediction — e.g. when a position opened from a narrative seed is
        closed.

        Also resolves the linked episodic memory episode if one was created
        in ``store_narrative``.

        Returns True if the outcome was successfully recorded.
        """
        now = now if now is not None else time.time()
        ok = self._tracker.record_outcome(
            narrative_id, direction_correct, reward_normalized, resolution_bars, now
        )
        # If there is a linked episode, resolve it in episodic memory too
        ep_id = self._episode_ids.get(narrative_id)
        if ep_id and ok:
            # Resolve in all instruments (bridge doesn't know which one)
            for instr in self._engine.all_instruments():
                if self._bridge.update_episode_outcome(
                    instr,
                    ep_id,
                    direction_correct=direction_correct,
                    reward_atr=reward_normalized,
                    bars_to_close=resolution_bars,
                    now=now,
                ):
                    break
        return ok

    # ── Convenience pass-throughs ─────────────────────────────────────────

    def narratives_for(self, instrument: str) -> list[SymbolNarrative]:
        """All active narratives for ``instrument``."""
        return self._engine.active_narratives_for(instrument)

    def probability_distribution_for(self, instrument: str) -> NarrativeProbabilityDistribution:
        """Current probability distribution without a full generate call."""
        active = self._engine.active_narratives_for(instrument)
        return self._prob_engine.compute(active)

    def to_snapshot(self, instrument: str) -> dict:
        """Serialize all narrative state for ``instrument``."""
        return {
            "engine_narratives": self._engine.to_snapshot(instrument),
            "outcome_tracker": self._tracker.to_snapshot(),
        }

    def load_snapshot(self, instrument: str, data: dict) -> None:
        """Restore previously-serialized state."""
        if "engine_narratives" in data:
            self._engine.load_snapshot(instrument, data["engine_narratives"])
        if "outcome_tracker" in data:
            self._tracker.load_snapshot(data["outcome_tracker"])

    # ── Access to sub-components (for advanced callers) ───────────────────

    @property
    def engine(self) -> MarketNarrativeEngine:
        return self._engine

    @property
    def outcome_tracker(self) -> NarrativeOutcomeTracker:
        return self._tracker

    @property
    def calibrator(self) -> NarrativeCalibrator:
        return self._calibrator

    @property
    def memory_bridge(self) -> NarrativeMemoryBridge:
        return self._bridge


__all__ = [
    "NarrativeAPI",
    "NarrativeGenerationResult",
]
