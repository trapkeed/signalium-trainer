"""QDE Brain — structural liquidity map package.

Four modules covering the lifecycle of structural liquidity:

  liquidity_memory.py        — data models (OHLCBar, LiquidityLevel,
                               LiquidityEvent, LiquidityMap) and the
                               rolling per-symbol memory store.

  liquidity_event_detector.py — pure, stateless detection of structural
                               events from OHLCV bar sequences (swings,
                               equal levels, session/daily/weekly extremes,
                               sweeps, breakout traps, stop clusters).

  liquidity_tracker.py       — stateless analytics over a LiquidityMap
                               (density, proximity, sweep recency,
                               strength scoring, mutation helpers).

  liquidity_map_engine.py    — top-level orchestrator; the single entry
                               point for callers.  Exposes five integration
                               points for the QDE brain:

                               context_for_inference   → liq_* brain_context
                               conviction_modifier     → ±0.25 float
                               narrative_context       → narrative enrichment
                               opportunity_rank_signal → 0-1 float
                               execution_timing_signal → 0-1 float

Note on scope
-------------
This package covers *structural* liquidity — price levels where orders
cluster, have been taken, or may exist next.  It is distinct from
``mie.liquidity_intelligence``, which classifies *flow* liquidity
(volume/spread quality scores).  Do not merge or conflate the two.
"""

from __future__ import annotations

# ── Memory / data models ──────────────────────────────────────────────────────
from trainer.qde.brain.liquidity.liquidity_memory import (
    LEVEL_TYPE_SIDE,
    LevelSide,
    LevelStatus,
    LiquidityEvent,
    LiquidityEventType,
    LiquidityLevel,
    LiquidityLevelType,
    LiquidityMap,
    LiquidityMemory,
    OHLCBar,
)

# ── Detector ──────────────────────────────────────────────────────────────────
from trainer.qde.brain.liquidity.liquidity_event_detector import (
    LiquidityEventDetector,
)

# ── Tracker ───────────────────────────────────────────────────────────────────
from trainer.qde.brain.liquidity.liquidity_tracker import (
    LiquidityTracker,
)

# ── Engine ────────────────────────────────────────────────────────────────────
from trainer.qde.brain.liquidity.liquidity_map_engine import (
    LiquidityMapEngine,
)

__all__ = [
    # data models
    "OHLCBar",
    "LiquidityLevelType",
    "LevelSide",
    "LevelStatus",
    "LEVEL_TYPE_SIDE",
    "LiquidityLevel",
    "LiquidityEventType",
    "LiquidityEvent",
    "LiquidityMap",
    "LiquidityMemory",
    # core classes
    "LiquidityEventDetector",
    "LiquidityTracker",
    "LiquidityMapEngine",
]
