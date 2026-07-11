"""QDE Brain — multi-timeframe context subpackage.

Three modules:

- `timeframe_memory`         — `ContextTimeframe` enum (MN/W1/D1/H4/H1/M15/M5),
                               `TimeframeObservation`, `TimeframeMemory`.
- `context_alignment_engine` — `AlignmentPattern`, `CrossTimeframeContext`,
                               `ContextAlignmentEngine`.
- `multi_timeframe_context_engine` — `ContextHypothesisSeed`,
                                     `MultiTimeframeContextEngine`.
"""

from trainer.qde.brain.context.context_alignment_engine import (
    AlignmentPattern,
    ContextAlignmentEngine,
    CrossTimeframeContext,
)
from trainer.qde.brain.context.multi_timeframe_context_engine import (
    ContextHypothesisSeed,
    MultiTimeframeContextEngine,
)
from trainer.qde.brain.context.timeframe_memory import (
    ALL_CONTEXT_TIMEFRAMES,
    CONTEXT_TIMEFRAME_SECONDS,
    TIMEFRAME_TO_CONTEXT,
    ContextTimeframe,
    TimeframeMemory,
    TimeframeObservation,
    coarse_timeframes,
    fine_timeframes,
)

__all__ = [
    # timeframe_memory
    "ALL_CONTEXT_TIMEFRAMES",
    "CONTEXT_TIMEFRAME_SECONDS",
    "TIMEFRAME_TO_CONTEXT",
    "ContextTimeframe",
    "TimeframeMemory",
    "TimeframeObservation",
    "coarse_timeframes",
    "fine_timeframes",
    # context_alignment_engine
    "AlignmentPattern",
    "ContextAlignmentEngine",
    "CrossTimeframeContext",
    # multi_timeframe_context_engine
    "ContextHypothesisSeed",
    "MultiTimeframeContextEngine",
]
