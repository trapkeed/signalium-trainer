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

Public surface
--------------
The five modules in this package are intentionally kept separate so each
concern can be imported, tested, and extended independently:

  narrative_store.py            — data model + stateless CRUD
  narrative_builder.py          — pure detection (new narratives)
  narrative_evolution.py        — pure incremental update (existing narratives)
  narrative_conflict_detector.py — conflict detection + coherence scoring
  market_narrative_engine.py    — top-level orchestrator + integration points
"""

from trainer.qde.brain.narrative.market_narrative_engine import (
    HypothesisSeed,
    MarketNarrativeEngine,
    NarrativeUpdateResult,
)
from trainer.qde.brain.narrative.narrative_builder import NarrativeBuilder, NarrativeSeed
from trainer.qde.brain.narrative.narrative_conflict_detector import (
    NarrativeConflict,
    NarrativeConflictDetector,
)
from trainer.qde.brain.narrative.narrative_evolution import NarrativeEvolution
from trainer.qde.brain.narrative.narrative_store import (
    MAX_EVIDENCE_ITEMS,
    NarrativeKind,
    NarrativeMaturity,
    NarrativeStore,
    SymbolNarrative,
    _DEFAULT_MAX_AGE,
)

__all__ = [
    # Engine (primary entry point)
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
]
