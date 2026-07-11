"""QDE Brain — hypothesis API (Phase C4).

Top-level façade exposing the four spec-required methods for Phase C4:

    ``generate_hypotheses(instrument, inp, narrative_seeds, ...)``
    ``update_probabilities(hypothesis_set, new_evidence, ...)``
    ``invalidate_hypothesis(hypothesis_id, reason, ...)``
    ``promote_hypothesis(hypothesis_id, reason, ...)``

All four delegate to the components already built in this package (and in
the existing ``HypothesisEngine`` / ``HypothesisTracker`` machinery):

    generate_hypotheses()
        1. Recall memory context (memory bridge → episodic/semantic).
        2. Enrich narrative seeds with memory-adjusted confidence.
        3. Create or reinforce hypotheses via ``HypothesisEngine.create()``.
        4. Build a ``HypothesisSet`` from the competing hypotheses.
        5. Compute the initial probability distribution.
        6. Register every hypothesis in the lineage tree.

    update_probabilities()
        1. Apply a Bayesian update to the probability distribution.
        2. Reinforce or weaken the underlying ``Hypothesis`` objects via
           ``HypothesisEngine`` according to the evidence strength.
        3. Commit the new distribution to the set and snapshot it.

    invalidate_hypothesis()
        1. Call ``HypothesisEngine.invalidate()`` (delegates — no re-impl).
        2. Record the failure in ``HypothesisFailureTracker``.
        3. Update the lineage node to INVALIDATED.
        4. Renormalize the remaining active scenarios' probabilities.

    promote_hypothesis()
        1. Reinforce the target hypothesis via ``HypothesisEngine.reinforce()``.
        2. Apply a directed probability boost to the target scenario.
        3. Update the lineage to PROMOTED.
        4. Mark the set's ``promoted_scenario_id``.

What this module does NOT do
----------------------------
* Never reimplements ``HypothesisTracker`` form/reinforce/weaken logic.
* Never imports ``QDEDecision`` or places trades.
* Never owns a scheduler.
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

from trainer.qde.brain.brain_models import Hypothesis
from trainer.qde.brain.brain_types import BeliefDirection
from trainer.qde.brain.hypothesis.hypothesis_failure_tracker import (
    FailureMode,
    HypothesisFailureTracker,
)
from trainer.qde.brain.hypothesis.hypothesis_lineage import HypothesisLineage
from trainer.qde.brain.hypothesis.hypothesis_memory_bridge import (
    HypothesisMemoryBridge,
    HypothesisMemoryContext,
)
from trainer.qde.brain.hypothesis.hypothesis_probability import (
    HypothesisProbabilityEngine,
    ProbabilityUpdateResult,
)
from trainer.qde.brain.hypothesis.hypothesis_set import (
    HypothesisSet,
    make_hypothesis_set,
)
from trainer.qde.brain.narrative.market_narrative_engine import HypothesisSeed
from trainer.qde.brain.narrative.narrative_store import NarrativeKind
from trainer.qde.qde_observer_models import SymbolObservationInput

if TYPE_CHECKING:
    from trainer.qde.brain.hypothesis_engine import HypothesisEngine

# How strongly a promotion boosts the promoted scenario's raw_score
_PROMOTE_BOOST = 0.50
# How many HypothesisEngine reinforcements a "promote" triggers
_PROMOTE_REINFORCE_COUNT = 2
# Evidence dampening when update_probabilities internally reinforces/weakens
_INTERNAL_EVIDENCE_BOOST = 0.15  # per reinforce call
_INTERNAL_EVIDENCE_PENALTY = 0.20  # per weaken call


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class HypothesisGenerationResult:
    """Rich output of one ``generate_hypotheses`` call.

    ``hypothesis_set``
        The ``HypothesisSet`` of competing scenarios for this instrument.

    ``seeds_used``
        The ``HypothesisSeed`` objects used to create/reinforce hypotheses.

    ``memory_context``
        Memory recall context (None if no memory backends are configured).

    ``scenarios_created``
        Number of new ``Hypothesis`` objects created in this call.

    ``scenarios_updated``
        Number of existing hypotheses reinforced (de-duplicated by thesis).
    """

    instrument: str
    timestamp: float
    hypothesis_set: HypothesisSet
    seeds_used: list[HypothesisSeed] = field(default_factory=list)
    memory_context: HypothesisMemoryContext | None = None
    scenarios_created: int = 0
    scenarios_updated: int = 0

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "timestamp": round(self.timestamp, 3),
            "hypothesis_set": self.hypothesis_set.to_dict(),
            "seeds_used": [s.to_dict() for s in self.seeds_used],
            "memory_context": self.memory_context.to_dict() if self.memory_context else None,
            "scenarios_created": self.scenarios_created,
            "scenarios_updated": self.scenarios_updated,
        }


# ── API ───────────────────────────────────────────────────────────────────────


class HypothesisAPI:
    """Phase C4 public API: ``generate_hypotheses``, ``update_probabilities``,
    ``invalidate_hypothesis``, ``promote_hypothesis``.

    One instance is shared across all instruments.

    Parameters
    ----------
    hypothesis_engine:
        The existing ``HypothesisEngine`` instance (required — all create /
        reinforce / weaken / invalidate operations delegate to it).
    failure_tracker:
        ``HypothesisFailureTracker`` instance.  None → a fresh one is created.
    lineage:
        ``HypothesisLineage`` instance.  None → a fresh one is created.
    memory_bridge:
        ``HypothesisMemoryBridge`` instance.  None → a bridge with no C2
        backends is created (memory recall steps are no-ops).
    """

    def __init__(
        self,
        hypothesis_engine: "HypothesisEngine",
        failure_tracker: HypothesisFailureTracker | None = None,
        lineage: HypothesisLineage | None = None,
        memory_bridge: HypothesisMemoryBridge | None = None,
    ) -> None:
        self._engine = hypothesis_engine
        self._tracker = failure_tracker or HypothesisFailureTracker()
        self._lineage = lineage or HypothesisLineage()
        self._bridge = memory_bridge or HypothesisMemoryBridge()
        self._prob_eng = HypothesisProbabilityEngine()

        # instrument → HypothesisSet (one active set per instrument at a time)
        self._active_sets: dict[str, HypothesisSet] = {}
        # hypothesis_id → set_id (for invalidate/promote look-up)
        self._hyp_to_set: dict[str, str] = {}

    # ── Primary API ───────────────────────────────────────────────────────

    def generate_hypotheses(
        self,
        instrument: str,
        inp: SymbolObservationInput,
        narrative_seeds: list[HypothesisSeed],
        features: dict | None = None,
        session: str = "",
        market_regime: str = "",
        now: float | None = None,
    ) -> HypothesisGenerationResult:
        """Generate (or refresh) the competing hypothesis set for ``instrument``.

        This is the primary entry point.  Call once per market tick per symbol
        whenever the narrative layer has produced new ``HypothesisSeed`` objects
        (i.e. after ``MarketNarrativeEngine.suggest_hypothesis_seeds``).

        Parameters
        ----------
        instrument:
            Symbol to generate hypotheses for.
        inp:
            Current market observation input (for regime / session context).
        narrative_seeds:
            ``HypothesisSeed`` objects from ``MarketNarrativeEngine`` for this
            instrument.  May be empty (no established narratives).
        features:
            12-dim feature dict for episodic/semantic memory recall.
            None → memory recall is skipped.
        session:
            Current session label (e.g. "LONDON").
        market_regime:
            Current market regime string (e.g. "TREND_UP").
        now:
            Override current time (for replay/testing).

        Returns
        -------
        HypothesisGenerationResult
            Contains the full ``HypothesisSet`` with competing scenarios and
            their initial probability distribution.
        """
        now = now if now is not None else time.time()
        regime_str = market_regime or getattr(inp, "market_regime", "")

        # ── Step 1: Memory recall ─────────────────────────────────────────
        memory_ctx: HypothesisMemoryContext | None = None
        seeds_to_use = list(narrative_seeds)

        if features is not None:
            try:
                memory_ctx = self._bridge.recall_context(
                    instrument=instrument,
                    features=features,
                    session=session,
                    market_regime=regime_str,
                    now=now,
                )
                seeds_to_use = self._bridge.enrich_seeds(seeds_to_use, memory_ctx)
            except Exception as exc:
                logger.warning(f"[HypothesisAPI] memory recall failed: {exc}")

        # ── Step 2: Create / reinforce hypotheses via HypothesisEngine ────
        created = 0
        updated = 0
        created_hyps: list[Hypothesis] = []
        used_seeds: list[HypothesisSeed] = []

        for seed in seeds_to_use:
            if seed.instrument != instrument:
                continue
            try:
                existing_before = list(self._engine._brain.get_state(instrument).hypotheses.keys())
                hyp = self._engine.create(
                    symbol=instrument,
                    thesis=seed.thesis,
                    direction=seed.direction,
                    confidence=seed.confidence,
                    evidence=seed.evidence,
                    now=now,
                )
                if hyp.hypothesis_id in existing_before:
                    updated += 1
                else:
                    created += 1
                    created_hyps.append(hyp)
                used_seeds.append(seed)

                # Register in lineage (idempotent)
                self._lineage.register(
                    hypothesis_id=hyp.hypothesis_id,
                    thesis=hyp.thesis,
                    direction=hyp.direction,
                    instrument=instrument,
                    now=now,
                )
            except Exception as exc:
                logger.warning(f"[HypothesisAPI] create failed for seed '{seed.thesis}': {exc}")

        # ── Step 3: Build HypothesisSet ───────────────────────────────────
        state = self._engine._brain.get_state(instrument)
        active = self._engine._tracker.active(state.hypotheses)

        # Retrieve or create the set for this instrument
        hyp_set = self._active_sets.get(instrument) or make_hypothesis_set(instrument)

        # Add any newly-created hypotheses as new scenarios
        existing_hyp_ids = {s.hypothesis_id for s in hyp_set.scenarios}

        # Map NarrativeKind from the seed that created each hypothesis
        seed_kind_map: dict[str, NarrativeKind] = {}
        for seed in used_seeds:
            # Find the hypothesis that matched this seed's thesis
            for hyp in active:
                if hyp.thesis == seed.thesis:
                    seed_kind_map[hyp.hypothesis_id] = seed.source_narrative_kind
                    break

        for hyp in active:
            if hyp.hypothesis_id not in existing_hyp_ids:
                kind = seed_kind_map.get(hyp.hypothesis_id)
                hyp_set._add_scenario(hyp, narrative_kind=kind)
                self._hyp_to_set[hyp.hypothesis_id] = hyp_set.set_id

            # Update existing scenario's hypothesis reference (confidence may have changed)
            else:
                scenario = hyp_set.scenario_for_hypothesis(hyp.hypothesis_id)
                if scenario is not None:
                    scenario.hypothesis = hyp

        # Remove scenarios whose underlying hypothesis became terminal
        active_ids = {h.hypothesis_id for h in active}
        for scenario in list(hyp_set.scenarios):
            if scenario.hypothesis_id not in active_ids:
                scenario.deactivate()
                hyp_set.invalidated_scenario_ids.add(scenario.scenario_id)

        # ── Step 4: Compute initial probability distribution ──────────────
        probs = self._prob_eng.compute_initial(hyp_set)
        if probs:
            hyp_set._apply_probabilities(probs, trigger="INIT", now=now)

        self._active_sets[instrument] = hyp_set

        logger.debug(
            f"[HypothesisAPI] generate_hypotheses {instrument}: "
            f"{len(hyp_set.active_scenarios)} active scenarios, "
            f"created={created} updated={updated}"
        )

        return HypothesisGenerationResult(
            instrument=instrument,
            timestamp=now,
            hypothesis_set=hyp_set,
            seeds_used=used_seeds,
            memory_context=memory_ctx,
            scenarios_created=created,
            scenarios_updated=updated,
        )

    def update_probabilities(
        self,
        hypothesis_set: HypothesisSet,
        new_evidence: dict[str, float],
        trigger: str = "EVIDENCE",
        dampening: float | None = None,
        now: float | None = None,
    ) -> HypothesisSet:
        """Update the probability distribution over the competing scenarios.

        Applies a Bayesian-style update (see ``HypothesisProbabilityEngine``)
        and also reinforces/weakens the underlying ``Hypothesis`` objects via
        ``HypothesisEngine`` proportionally to the evidence strength, so the
        two layers stay in sync.

        Parameters
        ----------
        hypothesis_set:
            The ``HypothesisSet`` to update.
        new_evidence:
            Dict mapping scenario_id → evidence strength (−1 to +1).
            Positive = supporting evidence for that scenario.
            Negative = contradicting evidence.
        trigger:
            Label for the probability snapshot (e.g. "EVIDENCE", "TICK",
            "NARRATIVE_UPDATE").
        dampening:
            Override the Bayesian dampening coefficient.  Higher = more
            aggressive update.
        now:
            Override current time.

        Returns
        -------
        HypothesisSet
            The updated set (mutated in place, also returned for chaining).
        """
        now = now if now is not None else time.time()

        # ── Bayesian probability update ───────────────────────────────────
        result: ProbabilityUpdateResult = self._prob_eng.bayesian_update(
            hypothesis_set, new_evidence, dampening=dampening
        )
        hypothesis_set._apply_probabilities(result.new_probs, trigger=trigger, now=now)

        # ── Sync underlying Hypothesis objects ────────────────────────────
        # Use a gentle one-step reinforce/weaken per update so even low-
        # confidence hypotheses are not auto-invalidated by repeated calls.
        for scenario_id, strength in new_evidence.items():
            scenario = hypothesis_set.scenario_by_id(scenario_id)
            if scenario is None or not scenario.is_active:
                continue
            hyp_id = scenario.hypothesis_id

            # Skip already-terminal hypotheses (may have been auto-invalidated
            # by the HypothesisEngine itself during a prior reinforce/weaken)
            located = self._engine._locate(hyp_id)
            if located is None:
                continue
            _instr, _state = located
            live_hyp = _state.hypotheses.get(hyp_id)
            if live_hyp is None or live_hyp.is_terminal():
                continue

            if abs(strength) < 0.05:
                continue  # insignificant — skip
            if strength > 0:
                # One reinforce call per update regardless of strength
                self._engine.reinforce(
                    hyp_id,
                    f"probability update (strength={strength:+.2f})",
                )
            else:
                # One weaken call per update; the existing HypothesisEngine
                # weaken() uses a fixed penalty so we don't amplify it.
                self._engine.weaken(
                    hyp_id,
                    f"probability update (strength={strength:+.2f})",
                )

        logger.debug(
            f"[HypothesisAPI] update_probabilities {hypothesis_set.instrument}: "
            f"shift={result.shift:.3f} entropy={result.entropy:.3f} dominant={result.dominant_id}"
        )
        return hypothesis_set

    def invalidate_hypothesis(
        self,
        hypothesis_id: str,
        reason: str,
        hypothesis_set: HypothesisSet | None = None,
        market_direction: BeliefDirection | None = None,
        winning_scenario_thesis: str | None = None,
        failure_mode: FailureMode | None = None,
        now: float | None = None,
    ) -> bool:
        """Invalidate a hypothesis and record its failure.

        Delegates the invalidation itself to ``HypothesisEngine.invalidate()``
        so there is no reimplementation of the invalidation logic.

        Additionally:
        - Records the failure in ``HypothesisFailureTracker``.
        - Updates the lineage node to INVALIDATED.
        - Deactivates the scenario in the ``HypothesisSet`` and renormalizes
          the remaining scenarios' probabilities.

        Parameters
        ----------
        hypothesis_id:
            The hypothesis to invalidate.
        reason:
            Human-readable invalidation reason.
        hypothesis_set:
            The ``HypothesisSet`` this hypothesis belongs to.  If None, the
            API tries to locate it from its internal set registry.
        market_direction:
            Observed market direction after invalidation (for lesson learning).
        winning_scenario_thesis:
            Thesis of the competing scenario that turned out to be correct.
        failure_mode:
            Optional override for the failure mode classification.
        now:
            Override current time.

        Returns
        -------
        bool
            True if the hypothesis was found and invalidated, False otherwise.
        """
        now = now if now is not None else time.time()

        # ── Locate the hypothesis and instrument ──────────────────────────
        located = self._engine._locate(hypothesis_id)
        if located is None:
            logger.debug(f"[HypothesisAPI] invalidate: hypothesis {hypothesis_id} not found")
            return False
        instrument, state = located
        hyp = state.hypotheses.get(hypothesis_id)
        if hyp is None:
            return False

        # Snapshot before invalidation (confidence may already be 0 if auto-invalidated)
        confidence_at_failure = hyp.confidence

        # ── Delegate invalidation to HypothesisEngine (no reimplementation) ─
        # If the hypothesis is already terminal (e.g. auto-invalidated by
        # HypothesisEngine.weaken() crossing the 0.15 floor), skip the
        # engine call but still record the failure and update lineage so
        # the event is not silently lost.
        if not hyp.is_terminal():
            result = self._engine.invalidate(hypothesis_id, reason)
            if result is None:
                return False

        # ── Record failure ────────────────────────────────────────────────
        self._tracker.record_failure(
            hypothesis=hyp,
            set_id=self._hyp_to_set.get(hypothesis_id),
            market_direction=market_direction,
            winning_scenario_thesis=winning_scenario_thesis,
            invalidation_reason=reason,
            failure_mode=failure_mode,
            now=now,
        )

        # ── Update lineage ────────────────────────────────────────────────
        self._lineage.resolve(
            hypothesis_id,
            resolution="INVALIDATED",
            confidence_at_resolution=confidence_at_failure,
            now=now,
        )

        # ── Update the HypothesisSet ──────────────────────────────────────
        hyp_set = hypothesis_set or self._active_sets.get(instrument)
        if hyp_set is not None:
            scenario = hyp_set.scenario_for_hypothesis(hypothesis_id)
            if scenario is not None:
                scenario.deactivate()
                hyp_set.invalidated_scenario_ids.add(scenario.scenario_id)
                # Renormalize remaining active scenarios
                probs = self._prob_eng.compute_initial(hyp_set)
                if probs:
                    hyp_set._apply_probabilities(probs, trigger="INVALIDATION", now=now)

        logger.debug(f"[HypothesisAPI] invalidated {hypothesis_id[:8]} ({reason[:40]})")
        return True

    def promote_hypothesis(
        self,
        hypothesis_id: str,
        reason: str,
        hypothesis_set: HypothesisSet | None = None,
        boost: float = _PROMOTE_BOOST,
        now: float | None = None,
    ) -> bool:
        """Promote a hypothesis to the dominant position in its scenario set.

        "Promote" means:
        - Reinforce the underlying ``Hypothesis`` (multiple times) via
          ``HypothesisEngine.reinforce()`` to build conviction toward EXECUTABLE.
        - Apply a directed probability boost to this scenario in the set so it
          becomes the dominant (rank 1) scenario.
        - Update the lineage to PROMOTED.
        - Set ``hypothesis_set.promoted_scenario_id``.

        This does NOT immediately mark the hypothesis as CONFIRMED — that is
        reserved for when the thesis actually plays out.  Promote is an
        intermediate step meaning "this is now our primary working scenario".

        Parameters
        ----------
        hypothesis_id:
            The hypothesis to promote.
        reason:
            Human-readable promotion reason.
        hypothesis_set:
            The ``HypothesisSet`` this hypothesis belongs to.  If None, the
            API tries to locate it from its internal set registry.
        boost:
            Raw-score boost for the promoted scenario before renormalization.
            Default 0.50 (adds 50pp to the unnormalized weight).
        now:
            Override current time.

        Returns
        -------
        bool
            True if the hypothesis was found and promoted, False otherwise.
        """
        now = now if now is not None else time.time()

        # ── Locate the hypothesis ─────────────────────────────────────────
        located = self._engine._locate(hypothesis_id)
        if located is None:
            logger.debug(f"[HypothesisAPI] promote: hypothesis {hypothesis_id} not found")
            return False
        instrument, state = located
        hyp = state.hypotheses.get(hypothesis_id)
        if hyp is None or hyp.is_terminal():
            return False

        # ── Reinforce via HypothesisEngine (no reimplementation) ─────────
        for i in range(_PROMOTE_REINFORCE_COUNT):
            self._engine.reinforce(
                hypothesis_id,
                f"promoted: {reason} (step {i + 1}/{_PROMOTE_REINFORCE_COUNT})",
            )

        # ── Update lineage ────────────────────────────────────────────────
        self._lineage.resolve(
            hypothesis_id,
            resolution="PROMOTED",
            confidence_at_resolution=hyp.confidence,
            probability_at_resolution=0.0,  # will be updated below
            now=now,
        )

        # ── Update the HypothesisSet ──────────────────────────────────────
        hyp_set = hypothesis_set or self._active_sets.get(instrument)
        if hyp_set is not None:
            scenario = hyp_set.scenario_for_hypothesis(hypothesis_id)
            if scenario is not None:
                hyp_set.promoted_scenario_id = scenario.scenario_id

                # Apply directed boost via evidence dict
                evidence: dict[str, float] = {
                    s.scenario_id: (boost if s.hypothesis_id == hypothesis_id else -boost / 2)
                    for s in hyp_set.active_scenarios
                }
                prob_result = self._prob_eng.bayesian_update(hyp_set, evidence)
                hyp_set._apply_probabilities(prob_result.new_probs, trigger="PROMOTION", now=now)

                # Backfill lineage node's probability_at_resolution
                node = self._lineage.get(hypothesis_id)
                if node and node.resolved_at:
                    node.probability_at_resolution = scenario.probability

        logger.debug(f"[HypothesisAPI] promoted {hypothesis_id[:8]} ({reason[:40]})")
        return True

    # ── Convenience pass-throughs ─────────────────────────────────────────

    def active_set_for(self, instrument: str) -> HypothesisSet | None:
        """Current active ``HypothesisSet`` for ``instrument``."""
        return self._active_sets.get(instrument)

    def all_active_sets(self) -> dict[str, HypothesisSet]:
        """All active sets by instrument."""
        return dict(self._active_sets)

    def failure_summary(self, instrument: str) -> dict:
        """Quick failure summary for ``instrument``."""
        return {
            "lessons": self._tracker.lesson_summary(instrument),
            "common_failure_modes": self._tracker.common_failure_modes(instrument),
            "direction_bias": self._tracker.direction_bias_summary(instrument),
        }

    def lineage_path(self, hypothesis_id: str) -> list[dict]:
        """Full lineage path (root → this node) as a list of dicts."""
        return [n.to_dict() for n in self._lineage.lineage_path(hypothesis_id)]

    # ── Persistence ───────────────────────────────────────────────────────

    def to_snapshot(self, instrument: str) -> dict:
        """Serialize state for ``instrument``."""
        hyp_set = self._active_sets.get(instrument)
        return {
            "active_set": hyp_set.to_dict() if hyp_set else None,
            "failure_tracker": self._tracker.to_snapshot(),
            "lineage": self._lineage.to_snapshot(),
        }

    def load_snapshot(self, instrument: str, data: dict) -> None:
        """Restore previously-serialized state."""
        if "failure_tracker" in data:
            self._tracker.load_snapshot(data["failure_tracker"])
        if "lineage" in data:
            self._lineage.load_snapshot(data["lineage"])
        # Note: active_set contains Hypothesis references that can't be fully
        # restored from dict without the HypothesisEngine; callers should
        # call generate_hypotheses() to rebuild the set on startup.

    # ── Access to sub-components ──────────────────────────────────────────

    @property
    def failure_tracker(self) -> HypothesisFailureTracker:
        return self._tracker

    @property
    def lineage(self) -> HypothesisLineage:
        return self._lineage

    @property
    def memory_bridge(self) -> HypothesisMemoryBridge:
        return self._bridge

    @property
    def probability_engine(self) -> HypothesisProbabilityEngine:
        return self._prob_eng


__all__ = [
    "HypothesisAPI",
    "HypothesisGenerationResult",
]
