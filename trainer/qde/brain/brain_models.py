"""QDE Brain — data models.

These dataclasses model the QDE's internal mental state. They intentionally
reuse existing types where the ground is already covered (e.g. narratives
are built *from* `SymbolObservationInput`/`ObservationResult`, they don't
redefine volatility/liquidity/session fields) and stay plain-old-data so
they serialize cleanly for persistence (`persistence.py`) the same way the
rest of the QDE package does (`to_dict()` convention).
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from trainer.qde.brain.brain_types import (
    BeliefDirection,
    HypothesisStatus,
    NarrativeTone,
    TrendMaturity,
    is_terminal_hypothesis_status,
)
from trainer.qde.brain.personality_profiles import PersonalityProfile

if TYPE_CHECKING:
    from trainer.qde.brain.opportunity_tracker import Opportunity

BRAIN_SCHEMA_VERSION = "brain_v1"

DEFAULT_WORKING_MEMORY_DECAY_PER_SECOND = 1.0 / 3600.0  # ~1/e strength after 1h
DEFAULT_WORKING_MEMORY_CAPACITY = 200
DEFAULT_HYPOTHESIS_MAX_AGE_SECONDS = 24 * 3600.0  # a full trading day


# ── Working memory ────────────────────────────────────────────────────────


@dataclass
class WorkingMemoryItem:
    """One short-lived unit of "what just happened" for a symbol.

    Salience decays exponentially with time since last reinforcement so
    stale context naturally fades out of active consideration (context
    aging) without needing an explicit cleanup pass to be *correct* — the
    cleanup pass (`WorkingMemory.prune_stale`) only exists to bound memory,
    not to decide relevance.
    """

    item_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    instrument: str = ""
    kind: str = "OBSERVATION"  # OBSERVATION | EVENT | HYPOTHESIS_UPDATE | NARRATIVE_UPDATE
    content: dict = field(default_factory=dict)
    salience: float = 0.5  # 0-1, importance at time of creation
    decay_per_second: float = DEFAULT_WORKING_MEMORY_DECAY_PER_SECOND
    created_at: float = field(default_factory=time.time)
    last_reinforced_at: float = field(default_factory=time.time)
    reinforcement_count: int = 0

    def reinforce(self, boost: float = 0.15, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        self.salience = min(1.0, self.current_strength(now) + boost)
        self.last_reinforced_at = now
        self.reinforcement_count += 1

    def current_strength(self, now: float | None = None) -> float:
        now = now if now is not None else time.time()
        elapsed = max(0.0, now - self.last_reinforced_at)
        return self.salience * math.exp(-self.decay_per_second * elapsed)

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "instrument": self.instrument,
            "kind": self.kind,
            "content": self.content,
            "salience": round(self.salience, 4),
            "decay_per_second": self.decay_per_second,
            "created_at": round(self.created_at, 3),
            "last_reinforced_at": round(self.last_reinforced_at, 3),
            "current_strength": round(self.current_strength(), 4),
            "reinforcement_count": self.reinforcement_count,
        }


# ── Long-term memory ──────────────────────────────────────────────────────


@dataclass
class LongTermMemoryEntry:
    """A consolidated, durable memory promoted out of working memory.

    Consolidation (see `long_term_memory.LongTermMemory.consolidate`) is
    what turns "things that happened recently" into "things the Brain has
    learned" — entries here don't decay, they persist across sessions.
    """

    entry_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    instrument: str = ""
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    salience: float = 0.5
    source_item_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    access_count: int = 0

    def touch(self, now: float | None = None) -> None:
        self.last_accessed_at = now if now is not None else time.time()
        self.access_count += 1

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "instrument": self.instrument,
            "summary": self.summary,
            "tags": self.tags,
            "salience": round(self.salience, 4),
            "source_item_ids": self.source_item_ids,
            "created_at": round(self.created_at, 3),
            "last_accessed_at": round(self.last_accessed_at, 3),
            "access_count": self.access_count,
        }


@dataclass
class SymbolPersonality:
    """What the Brain has learned about *how this specific symbol behaves*,
    accumulated over many experiences rather than read off a single tick.

    `traits` are free-form 0-1 scores (e.g. "trend_persistence",
    "mean_reversion_bias", "session_sensitivity", "false_breakout_rate")
    updated via an exponential moving average in `record_experience`, so
    the personality adapts gradually rather than jumping on one data point.
    """

    instrument: str = ""
    traits: dict[str, float] = field(default_factory=dict)
    regime_outcome_quality: dict[str, float] = field(default_factory=dict)  # regime -> avg quality
    sample_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)

    def update_trait(self, name: str, observed_value: float, alpha: float = 0.1) -> None:
        prior = self.traits.get(name, observed_value)
        self.traits[name] = (1 - alpha) * prior + alpha * observed_value
        self.last_updated_at = time.time()

    def update_regime_quality(
        self, regime: str, outcome_quality: float, alpha: float = 0.15
    ) -> None:
        prior = self.regime_outcome_quality.get(regime, outcome_quality)
        self.regime_outcome_quality[regime] = (1 - alpha) * prior + alpha * outcome_quality
        self.last_updated_at = time.time()

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "traits": {k: round(v, 4) for k, v in self.traits.items()},
            "regime_outcome_quality": {
                k: round(v, 4) for k, v in self.regime_outcome_quality.items()
            },
            "sample_count": self.sample_count,
            "created_at": round(self.created_at, 3),
            "last_updated_at": round(self.last_updated_at, 3),
        }


# ── Hypotheses ────────────────────────────────────────────────────────────


@dataclass
class Hypothesis:
    """A standing belief about what a symbol is *setting up to do*, e.g.
    "accumulation before breakout above 1.0950" or "H1 uptrend exhausting
    into session close". Hypotheses are QDE's developing-setup tracking —
    they are consulted as context by inference, never used to place orders
    directly (see `qde_inference.derive_decision_fields` for that boundary).
    """

    hypothesis_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    instrument: str = ""
    thesis: str = ""
    direction: BeliefDirection = BeliefDirection.NEUTRAL
    status: HypothesisStatus = HypothesisStatus.FORMING
    confidence: float = 0.3
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    invalidation_reason: str = ""
    created_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)
    reinforcement_count: int = 0
    max_age_seconds: float = DEFAULT_HYPOTHESIS_MAX_AGE_SECONDS

    def reinforce(self, evidence: str, boost: float = 0.15, now: float | None = None) -> None:
        self.supporting_evidence.append(evidence)
        self.confidence = min(1.0, self.confidence + boost)
        self.reinforcement_count += 1
        self.last_updated_at = now if now is not None else time.time()
        if self.status == HypothesisStatus.FORMING:
            self.status = HypothesisStatus.ACTIVE
        elif self.reinforcement_count >= 3 and self.confidence >= 0.7:
            self.status = HypothesisStatus.REINFORCED

    def weaken(self, evidence: str, penalty: float = 0.2, now: float | None = None) -> None:
        self.contradicting_evidence.append(evidence)
        self.confidence = max(0.0, self.confidence - penalty)
        self.last_updated_at = now if now is not None else time.time()
        if self.confidence <= 0.15:
            self.status = HypothesisStatus.INVALIDATED
            self.invalidation_reason = evidence
        elif self.status in (HypothesisStatus.ACTIVE, HypothesisStatus.REINFORCED):
            self.status = HypothesisStatus.WEAKENING

    def confirm(self, now: float | None = None) -> None:
        self.status = HypothesisStatus.CONFIRMED
        self.confidence = 1.0
        self.last_updated_at = now if now is not None else time.time()

    def invalidate(self, reason: str, now: float | None = None) -> None:
        self.status = HypothesisStatus.INVALIDATED
        self.invalidation_reason = reason
        self.confidence = 0.0
        self.last_updated_at = now if now is not None else time.time()

    def is_stale(self, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        return (now - self.last_updated_at) > self.max_age_seconds

    def is_terminal(self) -> bool:
        return is_terminal_hypothesis_status(self.status)

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "instrument": self.instrument,
            "thesis": self.thesis,
            "direction": self.direction.value,
            "status": self.status.value,
            "confidence": round(self.confidence, 4),
            "supporting_evidence": self.supporting_evidence[-10:],
            "contradicting_evidence": self.contradicting_evidence[-10:],
            "invalidation_reason": self.invalidation_reason,
            "created_at": round(self.created_at, 3),
            "last_updated_at": round(self.last_updated_at, 3),
            "reinforcement_count": self.reinforcement_count,
        }


# ── Narratives & beliefs ──────────────────────────────────────────────────


@dataclass
class MarketNarrative:
    """The Brain's current running "story" for a symbol — a synthesis of
    session/volatility/liquidity/regime context into one coherent
    description, revised on every tick rather than recomputed from scratch,
    so wording/tone changes are incremental (narrative continuity) instead
    of flip-flopping tick to tick.
    """

    narrative_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    instrument: str = ""
    summary: str = ""
    tone: NarrativeTone = NarrativeTone.NEUTRAL
    trend_maturity: TrendMaturity = TrendMaturity.NONE
    session_context: str = ""
    volatility_state: str = "NORMAL"
    liquidity_state: str = "NORMAL"
    market_regime: str = "RANGE"
    structure_notes: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    revision_count: int = 0

    def to_dict(self) -> dict:
        return {
            "narrative_id": self.narrative_id,
            "instrument": self.instrument,
            "summary": self.summary,
            "tone": self.tone.value,
            "trend_maturity": self.trend_maturity.value,
            "session_context": self.session_context,
            "volatility_state": self.volatility_state,
            "liquidity_state": self.liquidity_state,
            "market_regime": self.market_regime,
            "structure_notes": self.structure_notes[-10:],
            "created_at": round(self.created_at, 3),
            "updated_at": round(self.updated_at, 3),
            "revision_count": self.revision_count,
        }


@dataclass
class MarketBelief:
    """A compact directional belief distilled from active hypotheses +
    narrative tone — what the Brain "thinks is true right now", separate
    from any single hypothesis so it can smooth over the noise of
    individual setups forming/invalidating.
    """

    belief_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    instrument: str = ""
    direction: BeliefDirection = BeliefDirection.NEUTRAL
    strength: float = 0.0  # 0-1
    statement: str = ""
    supporting_hypothesis_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "belief_id": self.belief_id,
            "instrument": self.instrument,
            "direction": self.direction.value,
            "strength": round(self.strength, 4),
            "statement": self.statement,
            "supporting_hypothesis_ids": self.supporting_hypothesis_ids,
            "created_at": round(self.created_at, 3),
            "updated_at": round(self.updated_at, 3),
        }


# ── Attention ─────────────────────────────────────────────────────────────


@dataclass
class AttentionAllocation:
    """How much of QDE's limited "attention budget" a symbol currently
    holds. Purely a prioritization/telemetry signal (e.g. for scheduling
    review cadence or dashboarding) — it never gates whether a symbol keeps
    receiving market updates; every enabled symbol is still ingested
    continuously regardless of its attention weight.
    """

    instrument: str = ""
    weight: float = 0.0  # 0-1, normalized across all tracked instruments
    salience: float = 0.0  # raw pre-normalization score
    reason: str = ""
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "weight": round(self.weight, 4),
            "salience": round(self.salience, 4),
            "reason": self.reason,
            "updated_at": round(self.updated_at, 3),
        }


# ── Reflection & experience ───────────────────────────────────────────────


@dataclass
class ReflectionEntry:
    """A periodic self-review: what did QDE observe/decide for this symbol
    over a window, and what did it learn? Reuses `DecisionMemory` /
    `KnowledgeBase` as the source of truth for what actually happened —
    this is a synthesis layer on top, not a parallel decision ledger.
    """

    reflection_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    instrument: str = ""
    period_start: float = 0.0
    period_end: float = 0.0
    insight: str = ""
    lessons: list[str] = field(default_factory=list)
    decisions_reviewed: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "reflection_id": self.reflection_id,
            "instrument": self.instrument,
            "period_start": round(self.period_start, 3),
            "period_end": round(self.period_end, 3),
            "insight": self.insight,
            "lessons": self.lessons,
            "decisions_reviewed": self.decisions_reviewed,
            "created_at": round(self.created_at, 3),
        }


@dataclass
class ExperienceRecord:
    """One unit of "something happened and here's how it went" fed into
    long-term learning (personality traits, regime outcome quality).
    `outcome_quality` is -1..1 (bad..good), left to the caller to derive
    from whatever ground truth is available (e.g. `DecisionOutcome`).
    """

    experience_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    instrument: str = ""
    kind: str = ""  # e.g. "HYPOTHESIS_CONFIRMED", "HYPOTHESIS_INVALIDATED", "DECISION_OUTCOME"
    description: str = ""
    outcome_quality: float = 0.0
    regime: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "experience_id": self.experience_id,
            "instrument": self.instrument,
            "kind": self.kind,
            "description": self.description,
            "outcome_quality": round(self.outcome_quality, 4),
            "regime": self.regime,
            "tags": self.tags,
            "created_at": round(self.created_at, 3),
        }


# ── Aggregate per-symbol state ────────────────────────────────────────────


@dataclass
class BrainState:
    """Everything the Brain knows about one symbol, persisted as a unit.

    This is the "always-on internal consciousness" for a single symbol —
    analogous to `ObserverMemory` for lifecycle state, but for the
    narrative/hypothesis/belief/personality layer instead.
    """

    instrument: str
    working_memory: list[WorkingMemoryItem] = field(default_factory=list)
    long_term_memory: list[LongTermMemoryEntry] = field(default_factory=list)
    hypotheses: dict[str, Hypothesis] = field(default_factory=dict)
    opportunities: dict[str, "Opportunity"] = field(default_factory=dict)
    narrative: MarketNarrative | None = None
    belief: MarketBelief | None = None
    personality: SymbolPersonality = field(default_factory=SymbolPersonality)
    behavioral_profile: PersonalityProfile = field(default_factory=PersonalityProfile)
    attention: AttentionAllocation = field(default_factory=AttentionAllocation)
    tick_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)

    @classmethod
    def new(cls, instrument: str) -> "BrainState":
        return cls(
            instrument=instrument,
            personality=SymbolPersonality(instrument=instrument),
            attention=AttentionAllocation(instrument=instrument),
        )

    def active_hypotheses(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses.values() if not h.is_terminal()]

    def to_dict(self) -> dict:
        return {
            "schema_version": BRAIN_SCHEMA_VERSION,
            "instrument": self.instrument,
            "working_memory": [i.to_dict() for i in self.working_memory],
            "long_term_memory": [e.to_dict() for e in self.long_term_memory],
            "hypotheses": {k: h.to_dict() for k, h in self.hypotheses.items()},
            "opportunities": {k: o.to_dict() for k, o in self.opportunities.items()},
            "narrative": self.narrative.to_dict() if self.narrative else None,
            "belief": self.belief.to_dict() if self.belief else None,
            "personality": self.personality.to_dict(),
            "behavioral_profile": self.behavioral_profile.to_dict(),
            "attention": self.attention.to_dict(),
            "tick_count": self.tick_count,
            "created_at": round(self.created_at, 3),
            "last_updated_at": round(self.last_updated_at, 3),
        }


@dataclass
class BrainThought:
    """The synthesized output of one `QDEBrain.on_market_update()` call —
    what QDE is "thinking" about a symbol right now. This is what gets fed
    into inference as narrative context (see `brain.QDEBrain.to_memory_context`).
    """

    instrument: str
    narrative_summary: str = ""
    tone: NarrativeTone = NarrativeTone.NEUTRAL
    trend_maturity: TrendMaturity = TrendMaturity.NONE
    active_hypotheses: list[dict] = field(default_factory=list)
    dominant_belief: dict | None = None
    attention_weight: float = 0.0
    personality_traits: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    tick_count: int = 0
    unix_ts: float = field(default_factory=time.time)
    uncertainty: dict | None = None
    dominant_conviction: dict | None = None

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "narrative_summary": self.narrative_summary,
            "tone": self.tone.value,
            "trend_maturity": self.trend_maturity.value,
            "active_hypotheses": self.active_hypotheses,
            "dominant_belief": self.dominant_belief,
            "attention_weight": round(self.attention_weight, 4),
            "personality_traits": {k: round(v, 4) for k, v in self.personality_traits.items()},
            "reasoning": self.reasoning,
            "tick_count": self.tick_count,
            "uncertainty": self.uncertainty,
            "dominant_conviction": self.dominant_conviction,
            "unix_ts": round(self.unix_ts, 3),
        }
