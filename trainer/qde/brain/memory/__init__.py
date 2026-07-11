"""QDE Brain — memory: timeframe-aware structured working memory.

This package provides ``WorkingMemoryStore`` — a structured, per-symbol,
per-timeframe candle buffer plus event streams for the recent market context
the episodic and semantic layers need.

Distinct from the existing ``trainer.qde.brain.working_memory.WorkingMemory``,
which is a generic salience-decaying item buffer used for
hypothesis/narrative context aging.  ``WorkingMemoryStore`` is a structured,
append-only rolling buffer for raw market observations organized by kind and
timeframe, intended to feed the cognitive memory layer (episodic recall,
semantic clustering, regime transition detection).

What ``WorkingMemoryStore`` tracks
-----------------------------------
* **Candles** — per-timeframe OHLCV deques with configurable depth.
  Defaults: M1→500, M5→300, M15→200, H1→150, H4→100.
* **Volatility evolution** — rolling ATR/spread/vol-state snapshots.
* **Liquidity events** — sweeps, clusters, breakouts, level touches.
* **Trend development** — directional change, maturity, strength readings.
* **Failed breakouts** — level, direction, bar count before reversal.
* **Session transitions** — from/to session with timestamp.
* **Spread evolution** — rolling bid/ask spread readings.
* **Correlation shifts** — cross-asset correlation delta events.

Public API
----------
    store.write_memory(instrument, event_kind, payload, timeframe=None)
    store.recall(instrument, event_kind, n=20, since=None)
    store.candles(instrument, timeframe, n=None)
    store.to_snapshot(instrument) / store.load_snapshot(instrument, data)
"""

from trainer.qde.brain.memory.working_memory_store import (
    TIMEFRAME_CANDLE_LIMITS,
    CandleRecord,
    CorrelationShift,
    FailedBreakout,
    LiquidityEvent,
    SessionTransition,
    SpreadPoint,
    TrendState,
    VolatilityPoint,
    WorkingMemoryStore,
)

__all__ = [
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
]
