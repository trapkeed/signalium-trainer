"""QDE Brain — shared enumerations.

Kept separate (mirrors `qde_observer_types.py`) so `brain_models.py` and
`brain.py` can both import these without a circular dependency.
"""

from __future__ import annotations

from enum import Enum


class HypothesisStatus(str, Enum):
    """Lifecycle of a market hypothesis, from formation to resolution."""

    FORMING = "FORMING"  # just proposed, not enough evidence yet
    ACTIVE = "ACTIVE"  # has at least one piece of supporting evidence
    REINFORCED = "REINFORCED"  # repeatedly confirmed by new evidence
    WEAKENING = "WEAKENING"  # contradicting evidence outweighing support
    INVALIDATED = "INVALIDATED"  # explicitly disproven
    CONFIRMED = "CONFIRMED"  # played out as expected (terminal, successful)
    EXPIRED = "EXPIRED"  # aged out without resolving either way


_TERMINAL_HYPOTHESIS_STATUSES = frozenset(
    {HypothesisStatus.INVALIDATED, HypothesisStatus.CONFIRMED, HypothesisStatus.EXPIRED}
)


def is_terminal_hypothesis_status(status: HypothesisStatus) -> bool:
    return status in _TERMINAL_HYPOTHESIS_STATUSES


class NarrativeTone(str, Enum):
    """The dominant "story" the Brain currently tells about a symbol."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    CHOPPY = "CHOPPY"
    BREAKOUT_PENDING = "BREAKOUT_PENDING"
    EXHAUSTION = "EXHAUSTION"


class BeliefDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class TrendMaturity(str, Enum):
    """Descriptive stage of a trend's life-cycle, purely observational —
    never used to gate order placement (see `qde_inference.py` for the only
    place execution decisions are made).
    """

    NONE = "NONE"
    EARLY = "EARLY"
    ESTABLISHED = "ESTABLISHED"
    LATE = "LATE"
    EXHAUSTED = "EXHAUSTED"


class MemoryTier(str, Enum):
    WORKING = "WORKING"
    LONG_TERM = "LONG_TERM"
