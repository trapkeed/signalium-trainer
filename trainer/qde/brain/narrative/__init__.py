"""QDE Brain — narrative subpackage.

This package maintains evolving typed market narratives per symbol,
operating as a reasoning layer on top of the existing single-instance
`MarketNarrative` summary (``brain.narratives.NarrativeEngine``).

Narrative kinds
---------------
TREND_CONTINUATION  — existing trend is sustaining, continuation likely
REVERSAL            — trend is losing conviction, setup reversing
ACCUMULATION        — quiet range consolidation, smart-money building
DISTRIBUTION        — range after a run, supply emerging, breakdown likely
BREAKOUT            — imminent or active escape from range / continuation zone
EXHAUSTION          — move running out of fuel, momentum fading
LIQUIDITY_GRAB      — stop-hunt / wick event, often precedes reversal

Each narrative carries supporting + conflicting evidence, confidence,
maturity, regime context, session context, and age — and evolves
incrementally on every tick rather than being rebuilt from scratch.

Phase C3 additions
-------------------
  narrative_probability.py      — normalized probability distribution over
                                   all 7 kinds (the "0.64 / 0.23 / 0.13" view)
  narrative_outcome_tracker.py  — records and scores narrative outcomes over time
  narrative_calibrator.py       — adjusts narrative confidence from outcome history
  narrative_memory_bridge.py    — connects narrative layer to C2 episodic/semantic
                                   memory (recall + portfolio influence)
  narrative_api.py              — public façade:
                                   generate_narrative() / store_narrative() /
                                   evaluate_narrative()

Integration with the rest of the QDE
--------------------------------------
Module                  Integration point
----------------------  --------------------------------------------------
hypothesis_engine.py    `MarketNarrativeEngine.suggest_hypothesis_seeds`
conviction_engine.py    `MarketNarrativeEngine.narrative_conviction_boost`
qde_inference.py        `MarketNarrativeEngine.narrative_context`
attention_engine.py     `MarketNarrativeEngine.opportunity_rank_signal`
brain.py                `MarketNarrativeEngine.on_market_update` (alongside
                        `QDEBrain.on_market_update`, same tick cadence)
brain.py (C3)           `NarrativeAPI.generate_narrative` (preferred entry
                        point when memory recall + calibration are needed)
"""

from trainer.qde.brain.narrative.market_narrative_engine import (
    HypothesisSeed,
    MarketNarrativeEngine,
    NarrativeUpdateResult,
)
from trainer.qde.brain.narrative.narrative_api import (
    NarrativeAPI,
    NarrativeGenerationResult,
)
from trainer.qde.brain.narrative.narrative_builder import NarrativeBuilder, NarrativeSeed
from trainer.qde.brain.narrative.narrative_calibrator import (
    CalibrationResult,
    NarrativeCalibrator,
)
from trainer.qde.brain.narrative.narrative_conflict_detector import (
    NarrativeConflict,
    NarrativeConflictDetector,
)
from trainer.qde.brain.narrative.narrative_evolution import NarrativeEvolution
from trainer.qde.brain.narrative.narrative_memory_bridge import (
    NarrativeMemoryBridge,
    NarrativeMemoryContext,
    NarrativePortfolioContext,
)
from trainer.qde.brain.narrative.narrative_outcome_tracker import (
    NarrativeOutcomeRecord,
    NarrativeOutcomeTracker,
    NarrativeQualityScore,
)
from trainer.qde.brain.narrative.narrative_probability import (
    NarrativeProbability,
    NarrativeProbabilityDistribution,
    NarrativeProbabilityEngine,
)
from trainer.qde.brain.narrative.narrative_store import (
    MAX_EVIDENCE_ITEMS,
    NarrativeKind,
    NarrativeMaturity,
    NarrativeStore,
    SymbolNarrative,
    _DEFAULT_MAX_AGE,
)
from trainer.qde.brain.narrative.continuity import (
    ContinuityEngine,
    HORIZON_ORDER,
    HORIZON_WINDOWS,
    LineageEntry,
    LineageEventType,
    NarrativeHorizon,
    NarrativeLineage,
    NarrativeStory,
    SessionSnapshot,
    StoryNode,
    StoryStrength,
    evaluate_story_strength,
    retrieve_story,
    update_narrative,
)

__all__ = [
    # ── Existing (Phase C3-agnostic) ──────────────────────────────────────
    # Engine (primary entry point — full tick update)
    "MarketNarrativeEngine",
    "NarrativeUpdateResult",
    # Integration-point output types
    "HypothesisSeed",
    # Store / model
    "NarrativeKind",
    "NarrativeMaturity",
    "SymbolNarrative",
    "NarrativeStore",
    "MAX_EVIDENCE_ITEMS",
    "_DEFAULT_MAX_AGE",
    # Builder
    "NarrativeSeed",
    "NarrativeBuilder",
    # Evolution
    "NarrativeEvolution",
    # Conflict detection
    "NarrativeConflict",
    "NarrativeConflictDetector",
    # ── Phase C3 additions ────────────────────────────────────────────────
    # Public API façade (generate / store / evaluate)
    "NarrativeAPI",
    "NarrativeGenerationResult",
    # Probability distribution
    "NarrativeProbability",
    "NarrativeProbabilityDistribution",
    "NarrativeProbabilityEngine",
    # Outcome tracking
    "NarrativeOutcomeRecord",
    "NarrativeOutcomeTracker",
    "NarrativeQualityScore",
    # Calibration
    "CalibrationResult",
    "NarrativeCalibrator",
    # Memory bridge (C2 ↔ C3)
    "NarrativeMemoryBridge",
    "NarrativeMemoryContext",
    "NarrativePortfolioContext",
    # ── Phase C14 additions ───────────────────────────────────────────────
    # Horizon enum + constants
    "NarrativeHorizon",
    "HORIZON_WINDOWS",
    "HORIZON_ORDER",
    # Data types
    "StoryNode",
    "LineageEventType",
    "LineageEntry",
    "NarrativeLineage",
    "SessionSnapshot",
    "NarrativeStory",
    "StoryStrength",
    # Public callables (spec)
    "update_narrative",
    "retrieve_story",
    "evaluate_story_strength",
    # Stateful wrapper
    "ContinuityEngine",
]
