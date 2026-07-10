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
from trainer.qde.brain.brain import QDEBrain
from trainer.qde.brain.brain_models import (
    AttentionAllocation,
    BrainState,
    BrainThought,
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
from trainer.qde.brain.hypothesis_engine import HypothesisEngine
from trainer.qde.brain.hypothesis_lifecycle import (
    HypothesisMaturity,
    MaturityThresholds,
    compute_maturity,
    is_terminal_maturity,
)
from trainer.qde.brain.hypothesis_registry import HypothesisRegistry, HypothesisView
from trainer.qde.brain.internal_state_store import InternalStateStore
from trainer.qde.brain.opportunity_scanner import OpportunitySignals, scan
from trainer.qde.brain.opportunity_tracker import Opportunity, OpportunityTracker
from trainer.qde.brain.persistence import QDEBrainDatabase
from trainer.qde.brain.setup_lifecycle import (
    DEFAULT_SETUP_THRESHOLDS,
    SetupEvidence,
    SetupState,
    SetupThresholds,
    is_valid_transition,
    next_state,
)
from trainer.qde.brain.setup_queue import SetupQueue
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
]
