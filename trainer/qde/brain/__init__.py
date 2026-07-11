"""QDE Brain — internal market consciousness for the QDE.

This package gives QDE a persistent, continuously-updated internal mental
model of every symbol it watches: working memory, long-term memory, market
narratives, hypotheses, attention allocation, belief tracking, reflection,
experience accumulation, and cross-session persistence.

It deliberately does **not** duplicate existing QDE/memory infrastructure:

  - Lifecycle staging (OBSERVING/PREPARING/ARMED/...) stays owned by
    `qde_observer.QDEMarketObserver`; the Brain *consumes* its
    `ObservationResult`/`SymbolObservationInput` as one input signal among
    several, it never re-implements readiness scoring or state transitions.
  - Action thresholding (ENTER/EXIT/SCALE) stays owned exclusively by
    `qde_inference.derive_decision_fields`; the Brain never derives trade
    actions itself. It only enriches `QDEInferenceInput` with narrative
    context so a decision model's inference is a *consequence* of
    accumulated reasoning rather than an isolated read of the current tick.
  - Durable decision history stays owned by `trainer.memory.DecisionMemory`
    / `trainer.memory.KnowledgeBase`; the Brain's reflection engine reads
    from them rather than keeping a parallel decision ledger.

Like the rest of the QDE codebase, `QDEBrain` does not own a scheduler or
network loop — the caller (a market-data feed dispatcher, or the
`/qde/observer/{instrument}/observe` HTTP tick) is expected to call
`QDEBrain.on_market_update(...)` for every *enabled* symbol continuously,
independent of whether a user is connected or autonomous trading is
enabled, so QDE's internal state advances even when nobody is watching.
"""

from trainer.qde.brain.attention_allocator import (
    AttentionScore,
    AttentionScoreAllocator,
    AttentionWeights,
)
from trainer.qde.brain.attention_engine import AttentionEngine
from trainer.qde.brain.behavioral_memory import BehavioralMemory, BehavioralObservation
from trainer.qde.brain.brain import QDEBrain
from trainer.qde.brain.brain_models import (
    AttentionAllocation,
    BrainState,
    BrainThought,
    CalibrationProfile,
    ExperienceRecord,
    Hypothesis,
    LongTermMemoryEntry,
    MarketBelief,
    MarketNarrative,
    ReflectionEntry,
    SymbolPersonality,
    WorkingMemoryItem,
)
from trainer.qde.brain.brain_state_manager import BrainStateManager, SymbolTickOutcome
from trainer.qde.brain.brain_types import (
    BeliefDirection,
    HypothesisStatus,
    NarrativeTone,
    TrendMaturity,
)
from trainer.qde.brain.conviction_engine import (
    ConvictionComponents,
    ConvictionProfile,
    ConvictionScore,
    classify_profile,
    compute_conviction,
    is_trade_worthy,
)
from trainer.qde.brain.debate_engine import DEFAULT_DOMINANCE_MARGIN, DebateEngine, DebateVerdict
from trainer.qde.brain.forgetting_engine import ForgettingEngine, ForgettingReport
from trainer.qde.brain.hypothesis_engine import HypothesisEngine
from trainer.qde.brain.hypothesis_lifecycle import (
    HypothesisMaturity,
    MaturityThresholds,
    compute_maturity,
    is_terminal_maturity,
)
from trainer.qde.brain.hypothesis_registry import HypothesisRegistry, HypothesisView
from trainer.qde.brain.internal_state_store import InternalStateStore
from trainer.qde.brain.lesson_extractor import LessonApplication, LessonExtractor
from trainer.qde.brain.memory_decay_engine import (
    DEFAULT_CONTEXT_MISMATCH_DECAY_PER_SECOND,
    DEFAULT_RECENCY_HALF_LIFE_SECONDS,
    DEFAULT_REGIME_DECAY_PER_SECOND,
    REGIME_NEUTRAL_QUALITY,
    context_decay,
    recency_weight,
    regime_decay,
    relevance_weight,
)
from trainer.qde.brain.metacognition_engine import MetacognitionEngine, MetacognitiveAssessment
from trainer.qde.brain.opportunity_scanner import OpportunitySignals, scan
from trainer.qde.brain.opportunity_tracker import Opportunity, OpportunityTracker
from trainer.qde.brain.persistence import QDEBrainDatabase
from trainer.qde.brain.personality_profiles import (
    PersonalityProfile,
    confidence_adjustment,
    expected_holding_duration_seconds,
    risk_appetite_multiplier,
    setup_requirement_overrides,
)
from trainer.qde.brain.reflection_engine import TradeReflection, TradeReflectionEngine
from trainer.qde.brain.self_awareness_engine import SelfAwarenessEngine, SelfAwarenessState
from trainer.qde.brain.memory import (
    WorkingMemoryStore,
    TIMEFRAME_CANDLE_LIMITS,
    CandleRecord,
    VolatilityPoint,
    LiquidityEvent,
    TrendState,
    FailedBreakout,
    SessionTransition,
    SpreadPoint,
    CorrelationShift,
)
from trainer.qde.brain.episodic import (
    Episode,
    EpisodeFeatures,
    EpisodeOutcome,
    EpisodePortfolioState,
    EpisodicIndex,
    EpisodicMemory,
    SimilarEpisode,
)
from trainer.qde.brain.semantic import (
    ConceptKind,
    ConceptSeed,
    MarketConcept,
    SemanticMemory,
    SemanticRecallResult,
)
from trainer.qde.brain.hypothesis import (
    HypothesisAPI,
    HypothesisGenerationResult,
    HypothesisSet,
    HypothesisScenario,
    ProbabilitySnapshot,
    make_hypothesis_set,
    HypothesisProbabilityEngine,
    ProbabilityUpdateResult,
    HypothesisFailureTracker,
    HypothesisFailureRecord,
    FailureMode,
    HypothesisLineage,
    LineageNode,
    ResolutionState,
    HypothesisMemoryBridge,
    HypothesisMemoryContext,
)
from trainer.qde.brain.setup_lifecycle import (
    DEFAULT_SETUP_THRESHOLDS,
    SetupEvidence,
    SetupState,
    SetupThresholds,
    is_valid_transition,
    next_state,
)
from trainer.qde.brain.setup_queue import SetupQueue
from trainer.qde.brain.surprise_engine import (
    SurpriseAssessment,
    SurpriseLevel,
    SurpriseSignals,
    compute_surprise,
)
from trainer.qde.brain.symbol_personality_engine import SymbolPersonalityEngine
from trainer.qde.brain.thesis_manager import MAX_EVIDENCE_ITEMS, Thesis, ThesisManager
from trainer.qde.brain.uncertainty_engine import (
    UncertaintyLevel,
    UncertaintyScore,
    UncertaintySignals,
    classify_level,
    compute_uncertainty,
    does_not_understand_market,
    prefers_no_trade,
)

__all__ = [
    "QDEBrain",
    "QDEBrainDatabase",
    "BrainStateManager",
    "SymbolTickOutcome",
    "InternalStateStore",
    "AttentionEngine",
    "AttentionScoreAllocator",
    "AttentionScore",
    "AttentionWeights",
    "OpportunitySignals",
    "scan",
    "Opportunity",
    "OpportunityTracker",
    "SetupQueue",
    "SetupState",
    "SetupEvidence",
    "SetupThresholds",
    "DEFAULT_SETUP_THRESHOLDS",
    "is_valid_transition",
    "next_state",
    "HypothesisEngine",
    "HypothesisRegistry",
    "HypothesisView",
    "HypothesisMaturity",
    "MaturityThresholds",
    "compute_maturity",
    "is_terminal_maturity",
    "BrainState",
    "BrainThought",
    "WorkingMemoryItem",
    "LongTermMemoryEntry",
    "Hypothesis",
    "MarketNarrative",
    "MarketBelief",
    "AttentionAllocation",
    "ReflectionEntry",
    "ExperienceRecord",
    "SymbolPersonality",
    "HypothesisStatus",
    "NarrativeTone",
    "BeliefDirection",
    "TrendMaturity",
    "ConvictionProfile",
    "ConvictionComponents",
    "ConvictionScore",
    "classify_profile",
    "compute_conviction",
    "is_trade_worthy",
    "UncertaintyLevel",
    "UncertaintySignals",
    "UncertaintyScore",
    "classify_level",
    "compute_uncertainty",
    "prefers_no_trade",
    "does_not_understand_market",
    "BehavioralMemory",
    "BehavioralObservation",
    "PersonalityProfile",
    "confidence_adjustment",
    "risk_appetite_multiplier",
    "setup_requirement_overrides",
    "expected_holding_duration_seconds",
    "SymbolPersonalityEngine",
    "Thesis",
    "ThesisManager",
    "MAX_EVIDENCE_ITEMS",
    "DebateEngine",
    "DebateVerdict",
    "DEFAULT_DOMINANCE_MARGIN",
    "ForgettingEngine",
    "ForgettingReport",
    "recency_weight",
    "relevance_weight",
    "context_decay",
    "regime_decay",
    "DEFAULT_RECENCY_HALF_LIFE_SECONDS",
    "DEFAULT_CONTEXT_MISMATCH_DECAY_PER_SECOND",
    "DEFAULT_REGIME_DECAY_PER_SECOND",
    "REGIME_NEUTRAL_QUALITY",
    "CalibrationProfile",
    "TradeReflection",
    "TradeReflectionEngine",
    "SurpriseAssessment",
    "SurpriseLevel",
    "SurpriseSignals",
    "compute_surprise",
    "LessonApplication",
    "LessonExtractor",
    "MetacognitionEngine",
    "MetacognitiveAssessment",
    "SelfAwarenessEngine",
    "SelfAwarenessState",
    # ── Cognitive memory layer (Phase C2) ─────────────────────────────────
    # memory: timeframe-aware structured working memory store
    "WorkingMemoryStore",
    "TIMEFRAME_CANDLE_LIMITS",
    "CandleRecord",
    "VolatilityPoint",
    "LiquidityEvent",
    "TrendState",
    "FailedBreakout",
    "SessionTransition",
    "SpreadPoint",
    "CorrelationShift",
    # episodic: complete market experience records
    "Episode",
    "EpisodeFeatures",
    "EpisodeOutcome",
    "EpisodePortfolioState",
    "EpisodicIndex",
    "EpisodicMemory",
    "SimilarEpisode",
    # semantic: learned market concepts
    "ConceptKind",
    "ConceptSeed",
    "MarketConcept",
    "SemanticMemory",
    "SemanticRecallResult",
]
