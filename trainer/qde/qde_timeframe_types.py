"""Phase S9 — Hierarchical Timeframes: enums and factual constants only.

`Timeframe` and `TIMEFRAME_SECONDS` describe *what the timeframes are* —
duration, ordering coarsest to finest, which one is the execution
timeframe. They carry no trend/filter/confirmation semantics.

This distinction matters because S9 explicitly forbids hard-coded
cross-timeframe rules ("H1 trend filter", "trade only with higher timeframe
trend", "M15 confirmation"). Naming and ordering six fixed timeframes is
just domain metadata every downstream module needs; it is not a rule about
*how* those timeframes relate to each other. That relationship is left
entirely to the learned attention architecture in
`qde_temporal_attention.py` — nothing in this module or its siblings
branches on "if D1 is bullish" or similar.
"""

from __future__ import annotations

from enum import Enum

QDE_TIMEFRAME_SCHEMA_VERSION = "s9_v1"


class Timeframe(str, Enum):
    """The six timeframes QDE observes simultaneously."""

    D1 = "D1"
    H4 = "H4"
    H1 = "H1"
    M15 = "M15"
    M5 = "M5"
    M1 = "M1"


# Coarsest -> finest. Order reflects nothing more than bar duration; it is
# used for deterministic iteration/serialization, not as an implicit
# priority ranking.
ALL_TIMEFRAMES: tuple[Timeframe, ...] = (
    Timeframe.D1,
    Timeframe.H4,
    Timeframe.H1,
    Timeframe.M15,
    Timeframe.M5,
    Timeframe.M1,
)

EXECUTION_TIMEFRAME = Timeframe.M1

TIMEFRAME_SECONDS: dict[Timeframe, int] = {
    Timeframe.D1: 24 * 60 * 60,
    Timeframe.H4: 4 * 60 * 60,
    Timeframe.H1: 60 * 60,
    Timeframe.M15: 15 * 60,
    Timeframe.M5: 5 * 60,
    Timeframe.M1: 60,
}


def higher_timeframes(tf: Timeframe) -> tuple[Timeframe, ...]:
    """Timeframes strictly coarser than `tf`, coarsest first.

    Purely descriptive helper (duration comparison) — callers still decide
    what, if anything, to do with the result; nothing here asserts that a
    higher timeframe should gate or confirm a lower one.
    """
    idx = ALL_TIMEFRAMES.index(tf)
    return ALL_TIMEFRAMES[:idx]
