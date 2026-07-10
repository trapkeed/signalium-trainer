"""QDE Brain — the top-level facade.

`QDEBrain` is the single entry point that gives QDE a persistent internal
market consciousness. It is deliberately a thin orchestrator over the
other modules in this package (`working_memory`, `long_term_memory`,
`hypotheses`, `narratives`, `attention`, `reflection`, `persistence`) —
each owns one concern, `QDEBrain` wires them together per symbol and
handles cross-session load/save.

Continuous thinking, not just at inference time
------------------------------------------------
`on_market_update()` is meant to be called for **every enabled symbol on
every new market update**, independent of whether a user is connected or
autonomous trading is active — that decoupling is the caller's
responsibility (e.g. a market-data feed dispatcher or the
`/qde/observer/{instrument}/observe` HTTP tick, see `qde_server.py`), the
Brain itself just never refuses to update state based on session/trading
flags. Like `QDEMarketObserver`, it does not own a scheduler/loop.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.qde.brain.attention import AttentionAllocator
from trainer.qde.brain.brain_models import (
    BrainState,
    BrainThought,
    ExperienceRecord,
    MarketBelief,
    ReflectionEntry,
)
from trainer.qde.brain.brain_types import BeliefDirection, NarrativeTone, TrendMaturity
from trainer.qde.brain.hypotheses import HypothesisTracker
from trainer.qde.brain.long_term_memory import LongTermMemory
from trainer.qde.brain.narratives import NarrativeEngine
from trainer.qde.brain.persistence import QDEBrainDatabase
from trainer.qde.brain.reflection import ReflectionEngine
from trainer.qde.brain.working_memory import WorkingMemory
from trainer.qde.qde_observer_models import ObservationResult, SymbolObservationInput

if TYPE_CHECKING:
    from trainer.memory.decision_memory import DecisionMemory
    from trainer.memory.knowledge_base import KnowledgeBase
    from trainer.qde.qde_inference import QDEInferenceInput

REFLECTION_MIN_INTERVAL_SECONDS = 3600.0  # don't re-reflect more than hourly per symbol


class QDEBrain:
    """Persistent, continuously-updated internal market consciousness.

    One `BrainState` is held per instrument in-process; `QDEBrainDatabase`
    persists it so state survives restarts (cross-session continuity).
    """

    def __init__(self, db: QDEBrainDatabase | None = None) -> None:
        self._states: dict[str, BrainState] = {}
        self._db = db
        self._working_memory = WorkingMemory()
        self._long_term_memory = LongTermMemory()
        self._hypotheses = HypothesisTracker()
        self._narratives = NarrativeEngine()
        self._attention = AttentionAllocator()
        self._reflection = ReflectionEngine()
        self._last_reflected_at: dict[str, float] = {}
        if self._db is not None:
            self._restore_from_db()

    def _restore_from_db(self) -> None:
        assert self._db is not None
        for raw in self._db.list_states():
            try:
                state = _state_from_dict(raw)
                self._states[state.instrument] = state
            except (KeyError, ValueError) as exc:
                logger.warning(f"QDEBrain: skipped malformed stored state: {exc}")

    # ── State access ──────────────────────────────────────────────────────

    def get_state(self, instrument: str) -> BrainState:
        if instrument not in self._states:
            self._states[instrument] = BrainState.new(instrument)
        return self._states[instrument]

    def all_states(self) -> dict[str, BrainState]:
        return dict(self._states)

    def load_state(self, state: BrainState) -> None:
        """Restore previously-persisted state (e.g. after a restart)."""
        self._states[state.instrument] = state

    # ── Continuous ingestion ─────────────────────────────────────────────

    def on_market_update(
        self,
        inp: SymbolObservationInput,
        observation_result: ObservationResult | None = None,
    ) -> BrainThought:
        """Advance one symbol's internal state on a new market update.

        Called continuously for every enabled symbol — the caller must not
        gate this on user-connection or autonomous-trading state, since the
        whole point is that QDE keeps thinking even when nobody is watching
        and no trade is imminent.
        """
        now = inp.unix_ts or time.time()
        state = self.get_state(inp.instrument)
        state.tick_count += 1

        observer_state = observation_result.state.value if observation_result else "OBSERVING"
        readiness_score = observation_result.readiness_score if observation_result else 0.0
        dwell_seconds = observation_result.dwell_seconds if observation_result else 0.0

        # 1. Record this tick in working memory (fades unless reinforced).
        self._working_memory.add(
            state.working_memory,
            instrument=inp.instrument,
            kind="OBSERVATION",
            content={
                "market_regime": inp.market_regime,
                "volatility_state": inp.volatility_state,
                "liquidity_state": inp.liquidity_state,
                "observer_state": observer_state,
                "readiness_score": round(readiness_score, 4),
                "summary": f"{inp.market_regime}/{inp.volatility_state} tick, readiness={readiness_score:.2f}",
            },
            salience=min(1.0, 0.3 + readiness_score * 0.5),
            now=now,
        )
        self._working_memory.prune_stale(state.working_memory, now=now)

        # 2. Revise the running narrative incrementally.
        state.narrative = self._narratives.update(
            prior=state.narrative,
            instrument=inp.instrument,
            market_regime=inp.market_regime,
            volatility_state=inp.volatility_state,
            liquidity_state=inp.liquidity_state,
            session_quality=inp.session_quality,
            readiness_score=readiness_score,
            dwell_seconds=dwell_seconds,
            observer_state=observer_state,
            now=now,
        )

        # 3. Form/reinforce/invalidate hypotheses from this tick's signal.
        self._update_hypotheses(state, inp, observer_state, readiness_score, now)

        # 4. Age out stale hypotheses (context aging).
        self._hypotheses.age(state.hypotheses, now=now)

        # 5. Derive a compact directional belief from active hypotheses.
        self._update_belief(state, now)

        # 6. Consolidate strongly-reinforced working memory into long-term.
        self._long_term_memory.consolidate(state.working_memory, state.long_term_memory, now=now)

        # 7. Update this symbol's attention allocation (single-symbol view;
        #    call `reallocate_attention` separately for a cross-symbol pass).
        salience = self._attention.salience(
            readiness_score=readiness_score,
            volatility_quality=inp.volatility_quality,
            active_hypothesis_count=len(self._hypotheses.active(state.hypotheses)),
            max_hypothesis_confidence=max(
                (h.confidence for h in self._hypotheses.active(state.hypotheses)), default=0.0
            ),
        )
        state.attention.salience = salience
        state.attention.updated_at = now

        state.last_updated_at = now

        if self._db is not None:
            self._db.upsert_state(state.to_dict())

        return self._synthesize_thought(state, now)

    def _update_hypotheses(
        self,
        state: BrainState,
        inp: SymbolObservationInput,
        observer_state: str,
        readiness_score: float,
        now: float,
    ) -> None:
        regime = inp.market_regime.upper()

        if regime in ("TREND_UP", "TREND_DOWN") and readiness_score >= 0.4:
            direction = BeliefDirection.BULLISH if regime == "TREND_UP" else BeliefDirection.BEARISH
            thesis = f"{inp.instrument} developing a {regime.lower()} continuation setup"
            hyp = self._hypotheses.form(
                state.hypotheses, inp.instrument, thesis, direction, confidence=0.35, now=now
            )
            if readiness_score >= 0.65:
                self._hypotheses.reinforce(
                    state.hypotheses, hyp.hypothesis_id, f"readiness={readiness_score:.2f}"
                )

        if observer_state in ("ARMED", "ENTRY_WINDOW"):
            for h in self._hypotheses.active(state.hypotheses):
                if h.instrument == inp.instrument:
                    self._hypotheses.reinforce(
                        state.hypotheses, h.hypothesis_id, "entry window opened"
                    )

        if inp.position_just_closed:
            outcome_evidence = "loss" if inp.last_close_was_loss else "win"
            for h in self._hypotheses.active(state.hypotheses):
                if h.instrument != inp.instrument:
                    continue
                if inp.last_close_was_loss:
                    self._hypotheses.weaken(
                        state.hypotheses, h.hypothesis_id, f"trade closed at a {outcome_evidence}"
                    )
                else:
                    self._hypotheses.confirm(state.hypotheses, h.hypothesis_id)

        if regime == "RANGE" and readiness_score < 0.2:
            for h in self._hypotheses.active(state.hypotheses):
                if h.instrument == inp.instrument and h.direction != BeliefDirection.NEUTRAL:
                    self._hypotheses.weaken(
                        state.hypotheses, h.hypothesis_id, "regime flattened back to range"
                    )

    def _update_belief(self, state: BrainState, now: float) -> None:
        direction = self._hypotheses.dominant_direction(state.hypotheses)
        active = self._hypotheses.active(state.hypotheses)
        strength = (
            max((h.confidence for h in active), default=0.0)
            if direction != BeliefDirection.NEUTRAL
            else 0.0
        )

        prior = state.belief
        new_belief = MarketBelief(
            instrument=state.instrument,
            direction=direction,
            strength=strength,
            statement=(
                f"{state.instrument} bias is {direction.value.lower()} "
                f"(confidence {strength:.0%}) based on {len(active)} active hypothes"
                f"{'is' if len(active) == 1 else 'es'}."
            ),
            supporting_hypothesis_ids=[h.hypothesis_id for h in active],
            created_at=prior.created_at if prior else now,
            updated_at=now,
        )
        if prior is not None:
            new_belief.belief_id = prior.belief_id
        state.belief = new_belief

    # ── Cross-symbol attention ────────────────────────────────────────────

    def reallocate_attention(self) -> dict[str, float]:
        """Recompute normalized attention weights across every tracked
        symbol from each symbol's last-known salience.
        """
        salience_by_instrument = {
            instrument: state.attention.salience for instrument, state in self._states.items()
        }
        allocations = self._attention.reallocate(salience_by_instrument)
        for instrument, allocation in allocations.items():
            self._states[instrument].attention = allocation
        return {k: v.weight for k, v in allocations.items()}

    # ── Experience & reflection ───────────────────────────────────────────

    def record_experience(
        self,
        instrument: str,
        kind: str,
        description: str,
        outcome_quality: float,
        regime: str = "",
        tags: list[str] | None = None,
    ) -> ExperienceRecord:
        state = self.get_state(instrument)
        experience = ExperienceRecord(
            instrument=instrument,
            kind=kind,
            description=description,
            outcome_quality=outcome_quality,
            regime=regime,
            tags=tags or [],
        )
        self._long_term_memory.record_experience(state.personality, experience)
        if self._db is not None:
            self._db.log_experience(experience.to_dict())
            self._db.upsert_state(state.to_dict())
        return experience

    def reflect(
        self,
        instrument: str,
        decision_memory: "DecisionMemory | None" = None,
        knowledge_base: "KnowledgeBase | None" = None,
        force: bool = False,
    ) -> ReflectionEntry | None:
        """Run a periodic self-review for a symbol. Rate-limited per symbol
        unless `force=True`, so reflection stays a periodic synthesis step
        rather than re-running on every tick.
        """
        now = time.time()
        last = self._last_reflected_at.get(instrument, 0.0)
        if not force and (now - last) < REFLECTION_MIN_INTERVAL_SECONDS:
            return None

        entry = self._reflection.reflect(
            instrument, decision_memory=decision_memory, knowledge_base=knowledge_base, now=now
        )
        self._last_reflected_at[instrument] = now
        if self._db is not None:
            self._db.log_reflection(entry.to_dict())
        return entry

    # ── Inference integration ─────────────────────────────────────────────

    def to_memory_context(self, instrument: str) -> dict:
        """Compact narrative/hypothesis/belief context suitable for
        `QDEInferenceInput.brain_context`, so inference can draw on
        accumulated internal reasoning instead of only the current tick.
        """
        state = self.get_state(instrument)
        return self._synthesize_thought(state, time.time()).to_dict()

    def enrich_inference_input(
        self, inp: "QDEInferenceInput", instrument: str
    ) -> "QDEInferenceInput":
        """Attach this symbol's current Brain context onto an existing
        `QDEInferenceInput` in place, then return it for convenience.
        """
        inp.brain_context = self.to_memory_context(instrument)
        return inp

    def _synthesize_thought(self, state: BrainState, now: float) -> BrainThought:
        active = self._hypotheses.active(state.hypotheses)
        narrative = state.narrative
        return BrainThought(
            instrument=state.instrument,
            narrative_summary=narrative.summary if narrative else "",
            tone=narrative.tone if narrative else NarrativeTone.NEUTRAL,
            trend_maturity=narrative.trend_maturity if narrative else TrendMaturity.NONE,
            active_hypotheses=[h.to_dict() for h in active],
            dominant_belief=state.belief.to_dict() if state.belief else None,
            attention_weight=state.attention.weight,
            personality_traits=dict(state.personality.traits),
            reasoning=self._explain(state, active),
            tick_count=state.tick_count,
            unix_ts=now,
        )

    def _explain(self, state: BrainState, active_hypotheses: list) -> str:
        if not active_hypotheses:
            return f"{state.instrument}: no active hypotheses, observing."
        strongest = max(active_hypotheses, key=lambda h: h.confidence)
        return (
            f"{state.instrument}: {len(active_hypotheses)} active hypothes"
            f"{'is' if len(active_hypotheses) == 1 else 'es'}; strongest is "
            f"'{strongest.thesis}' ({strongest.status.value}, confidence {strongest.confidence:.0%})."
        )

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, instrument: str) -> None:
        if self._db is None:
            return
        self._db.upsert_state(self.get_state(instrument).to_dict())

    def save_all(self) -> None:
        if self._db is None:
            return
        for state in self._states.values():
            self._db.upsert_state(state.to_dict())


def _state_from_dict(raw: dict) -> BrainState:
    """Reconstruct a `BrainState` from its persisted `to_dict()` form."""
    from trainer.qde.brain.brain_models import (
        AttentionAllocation,
        Hypothesis,
        LongTermMemoryEntry,
        MarketNarrative,
        SymbolPersonality,
        WorkingMemoryItem,
    )
    from trainer.qde.brain.brain_types import HypothesisStatus

    instrument = raw["instrument"]

    working_memory = [
        WorkingMemoryItem(
            item_id=i["item_id"],
            instrument=i["instrument"],
            kind=i["kind"],
            content=i["content"],
            salience=i["salience"],
            created_at=i["created_at"],
            last_reinforced_at=i["last_reinforced_at"],
            reinforcement_count=i["reinforcement_count"],
        )
        for i in raw.get("working_memory", [])
    ]

    long_term_memory = [
        LongTermMemoryEntry(
            entry_id=e["entry_id"],
            instrument=e["instrument"],
            summary=e["summary"],
            tags=e["tags"],
            salience=e["salience"],
            source_item_ids=e["source_item_ids"],
            created_at=e["created_at"],
            last_accessed_at=e["last_accessed_at"],
            access_count=e["access_count"],
        )
        for e in raw.get("long_term_memory", [])
    ]

    hypotheses = {}
    for hid, h in raw.get("hypotheses", {}).items():
        hypotheses[hid] = Hypothesis(
            hypothesis_id=h["hypothesis_id"],
            instrument=h["instrument"],
            thesis=h["thesis"],
            direction=BeliefDirection(h["direction"]),
            status=HypothesisStatus(h["status"]),
            confidence=h["confidence"],
            supporting_evidence=h["supporting_evidence"],
            contradicting_evidence=h["contradicting_evidence"],
            invalidation_reason=h["invalidation_reason"],
            created_at=h["created_at"],
            last_updated_at=h["last_updated_at"],
            reinforcement_count=h["reinforcement_count"],
        )

    narrative = None
    if raw.get("narrative"):
        n = raw["narrative"]
        narrative = MarketNarrative(
            narrative_id=n["narrative_id"],
            instrument=n["instrument"],
            summary=n["summary"],
            tone=NarrativeTone(n["tone"]),
            trend_maturity=TrendMaturity(n["trend_maturity"]),
            session_context=n["session_context"],
            volatility_state=n["volatility_state"],
            liquidity_state=n["liquidity_state"],
            market_regime=n["market_regime"],
            structure_notes=n["structure_notes"],
            created_at=n["created_at"],
            updated_at=n["updated_at"],
            revision_count=n["revision_count"],
        )

    belief = None
    if raw.get("belief"):
        b = raw["belief"]
        belief = MarketBelief(
            belief_id=b["belief_id"],
            instrument=b["instrument"],
            direction=BeliefDirection(b["direction"]),
            strength=b["strength"],
            statement=b["statement"],
            supporting_hypothesis_ids=b["supporting_hypothesis_ids"],
            created_at=b["created_at"],
            updated_at=b["updated_at"],
        )

    p = raw.get("personality", {})
    personality = SymbolPersonality(
        instrument=p.get("instrument", instrument),
        traits=p.get("traits", {}),
        regime_outcome_quality=p.get("regime_outcome_quality", {}),
        sample_count=p.get("sample_count", 0),
        created_at=p.get("created_at", time.time()),
        last_updated_at=p.get("last_updated_at", time.time()),
    )

    a = raw.get("attention", {})
    attention = AttentionAllocation(
        instrument=a.get("instrument", instrument),
        weight=a.get("weight", 0.0),
        salience=a.get("salience", 0.0),
        reason=a.get("reason", ""),
        updated_at=a.get("updated_at", time.time()),
    )

    return BrainState(
        instrument=instrument,
        working_memory=working_memory,
        long_term_memory=long_term_memory,
        hypotheses=hypotheses,
        narrative=narrative,
        belief=belief,
        personality=personality,
        attention=attention,
        tick_count=raw.get("tick_count", 0),
        created_at=raw.get("created_at", time.time()),
        last_updated_at=raw.get("last_updated_at", time.time()),
    )
