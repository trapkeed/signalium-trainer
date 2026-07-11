"""QDE Brain — multi-timeframe context: memory layer.

Defines the extended timeframe vocabulary (`ContextTimeframe`) that adds
Monthly and Weekly to the five context-relevant timeframes already in
`qde_timeframe_types.Timeframe` (D1/H4/H1/M15/M5 — M1 is the execution
timeframe and is intentionally excluded here).

`TimeframeObservation` is the descriptive, derived state for one timeframe
at one point in time.  Every field is a neutral measurement (momentum
magnitude, range position, candle character) — never a label like
"BULLISH" or "PULLBACK"; those interpretations belong to the alignment
engine one layer up.

`TimeframeMemory` is the stateful rolling history store: one long-lived
instance per process, keyed by instrument+timeframe, retaining a
configurable depth of past observations for history-aware pattern
detection (momentum slope, zero-crossing detection).
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.qde.qde_timeframe_types import Timeframe

# ── ContextTimeframe ──────────────────────────────────────────────────────────


class ContextTimeframe(str, Enum):
    """Extended timeframe vocabulary for the context layer.

    Adds MN (Monthly) and W1 (Weekly) to the six timeframes that
    `qde_timeframe_types.Timeframe` already covers.  Enum values are
    intentionally identical to the existing `Timeframe` values for the five
    shared members (D1–M5) so that serialised forms are interchangeable
    where the ranges overlap.

    M1 (execution timeframe) is deliberately absent — the context layer
    concerns itself with structural timeframes, not the bar-by-bar
    execution stream.
    """

    MN = "MN"    # Monthly  (~30 days)
    W1 = "W1"    # Weekly   (7 days)
    D1 = "D1"    # Daily
    H4 = "H4"    # 4-hour
    H1 = "H1"    # 1-hour
    M15 = "M15"  # 15-minute
    M5 = "M5"    # 5-minute


# Ordered coarsest → finest.  This ordering reflects only bar duration; it
# carries no priority or filter semantics.
ALL_CONTEXT_TIMEFRAMES: tuple[ContextTimeframe, ...] = (
    ContextTimeframe.MN,
    ContextTimeframe.W1,
    ContextTimeframe.D1,
    ContextTimeframe.H4,
    ContextTimeframe.H1,
    ContextTimeframe.M15,
    ContextTimeframe.M5,
)

CONTEXT_TIMEFRAME_SECONDS: dict[ContextTimeframe, int] = {
    ContextTimeframe.MN:  30 * 24 * 3600,
    ContextTimeframe.W1:   7 * 24 * 3600,
    ContextTimeframe.D1:       24 * 3600,
    ContextTimeframe.H4:    4 * 3600,
    ContextTimeframe.H1:        3600,
    ContextTimeframe.M15:      15 * 60,
    ContextTimeframe.M5:        5 * 60,
}

# Mapping from the existing qde_timeframe_types.Timeframe to ContextTimeframe
# for the five shared members.  M1 is absent; callers use it only for
# execution, not for context ingestion.
TIMEFRAME_TO_CONTEXT: dict[Timeframe, ContextTimeframe] = {
    Timeframe.D1:  ContextTimeframe.D1,
    Timeframe.H4:  ContextTimeframe.H4,
    Timeframe.H1:  ContextTimeframe.H1,
    Timeframe.M15: ContextTimeframe.M15,
    Timeframe.M5:  ContextTimeframe.M5,
}

# The seven timeframes are split 4/3 into a "coarse" structural group and a
# "fine" execution-context group.  The split is a factual property of the
# timeframe ladder (the four longest vs. the three shortest context TFs);
# it is not a trading rule about which timeframe should "confirm" another.
_COARSE_COUNT = 4


def coarse_timeframes() -> tuple[ContextTimeframe, ...]:
    """The four structurally longer context timeframes: MN/W1/D1/H4."""
    return ALL_CONTEXT_TIMEFRAMES[:_COARSE_COUNT]


def fine_timeframes() -> tuple[ContextTimeframe, ...]:
    """The three structurally shorter context timeframes: H1/M15/M5."""
    return ALL_CONTEXT_TIMEFRAMES[_COARSE_COUNT:]


# ── TimeframeObservation ──────────────────────────────────────────────────────


@dataclass
class TimeframeObservation:
    """Descriptive state for one timeframe at one point in time.

    Every field is a neutral observation of price/momentum behaviour on
    this timeframe alone — no "BULLISH"/"BEARISH" label, no structural
    assertion like "this is a pullback".  The alignment engine reads these
    raw measurements and derives patterns; the patterns are never baked in
    here.

    `staleness_ratio` is the bar age divided by the timeframe duration.
    A ratio of 1.0 means one full timeframe period has elapsed since the
    last bar closed.  Callers can use `is_fresh()` to gate on a sensible
    default threshold.
    """

    timeframe: ContextTimeframe
    as_of: float = 0.0                # close_time of the bar this describes

    # ── Momentum ─────────────────────────────────────────────────────────
    momentum: float = 0.0             # TimeframeFeatureVector.momentum_n (normalised)
    momentum_sign: int = 0            # +1 / 0 / -1

    # ── Price position ────────────────────────────────────────────────────
    range_position: float = 0.5       # 0 = at period low, 1 = at period high

    # ── Candle character ─────────────────────────────────────────────────
    body_ratio: float = 0.0           # |close−open| / (high−low), 0–1
    close_direction: int = 0          # +1 close > open; −1 close < open; 0 doji

    # ── Volatility ────────────────────────────────────────────────────────
    realized_vol: float = 0.0

    # ── Freshness ─────────────────────────────────────────────────────────
    bars_available: int = 0
    staleness_ratio: float = 0.0      # bar_age / timeframe_seconds

    def is_fresh(self, max_staleness: float = 2.5) -> bool:
        """True if bar data exists and is not excessively stale.

        `max_staleness=2.5` means "bar is no older than 2.5 full timeframe
        periods" — generous enough for coarser timeframes where bars close
        infrequently, conservative enough to exclude cold-start placeholders.
        """
        return self.bars_available > 0 and self.staleness_ratio < max_staleness

    def to_dict(self) -> dict:
        return {
            "timeframe": self.timeframe.value,
            "as_of": round(self.as_of, 3),
            "momentum": round(self.momentum, 6),
            "momentum_sign": self.momentum_sign,
            "range_position": round(self.range_position, 4),
            "body_ratio": round(self.body_ratio, 4),
            "close_direction": self.close_direction,
            "realized_vol": round(self.realized_vol, 6),
            "bars_available": self.bars_available,
            "staleness_ratio": round(self.staleness_ratio, 3),
        }

    @classmethod
    def from_dict(cls, data: dict) -> TimeframeObservation:
        return cls(
            timeframe=ContextTimeframe(data["timeframe"]),
            as_of=data.get("as_of", 0.0),
            momentum=data.get("momentum", 0.0),
            momentum_sign=data.get("momentum_sign", 0),
            range_position=data.get("range_position", 0.5),
            body_ratio=data.get("body_ratio", 0.0),
            close_direction=data.get("close_direction", 0),
            realized_vol=data.get("realized_vol", 0.0),
            bars_available=data.get("bars_available", 0),
            staleness_ratio=data.get("staleness_ratio", 0.0),
        )


# ── Feature vector → observation derivation ──────────────────────────────────

# Momentum below this absolute threshold is treated as "flat" for sign purposes.
_MOMENTUM_FLAT = 1e-9


def _derive_observation(
    tf: ContextTimeframe,
    feature_vector,       # TimeframeFeatureVector | None
    now: float,
) -> TimeframeObservation:
    """Derive a `TimeframeObservation` from a `TimeframeFeatureVector`.

    Handles ``None`` gracefully — returns a cold-start placeholder with
    ``bars_available=0`` so downstream code knows the observation is
    unreliable and `is_fresh()` returns False.
    """
    if feature_vector is None:
        return TimeframeObservation(timeframe=tf, as_of=now)

    fv = feature_vector
    mom = float(fv.momentum_n)
    mom_sign = 0 if abs(mom) < _MOMENTUM_FLAT else (1 if mom > 0 else -1)

    lr = float(fv.log_return_1)
    close_dir = 0 if abs(lr) < _MOMENTUM_FLAT else (1 if lr > 0 else -1)

    tf_secs = CONTEXT_TIMEFRAME_SECONDS.get(tf, 3600)
    bar_age = max(0.0, now - float(fv.as_of)) if float(fv.as_of) > 0 else 0.0
    staleness = bar_age / tf_secs if tf_secs > 0 else 0.0

    return TimeframeObservation(
        timeframe=tf,
        as_of=float(fv.as_of),
        momentum=mom,
        momentum_sign=mom_sign,
        range_position=float(fv.range_position),
        body_ratio=float(fv.body_ratio),
        close_direction=close_dir,
        realized_vol=float(fv.realized_vol),
        bars_available=int(fv.bars_available),
        staleness_ratio=staleness,
    )


# ── TimeframeMemory ───────────────────────────────────────────────────────────

_DEFAULT_HISTORY_DEPTH = 12  # observations retained per (instrument, timeframe)


class TimeframeMemory:
    """Per-symbol rolling history of `TimeframeObservation` records.

    One long-lived instance per process (or per test) for all symbols and
    timeframes.  Stateful — holds the rolling deques — but all read methods
    are pure (they do not mutate stored state).

    Internal keys use `ContextTimeframe.value` strings so that serialised
    snapshots and reloaded instances are structurally identical.
    """

    def __init__(self, history_depth: int = _DEFAULT_HISTORY_DEPTH) -> None:
        self._depth = history_depth
        # instrument → tf_value → deque[TimeframeObservation]
        self._history: dict[str, dict[str, deque[TimeframeObservation]]] = {}

    # ── Ingestion ─────────────────────────────────────────────────────────

    def record(
        self,
        instrument: str,
        tf: ContextTimeframe,
        feature_vector,       # TimeframeFeatureVector | None
        now: float | None = None,
    ) -> TimeframeObservation:
        """Derive and store an observation from a feature vector.

        Returns the stored observation so callers can chain without a
        separate read.
        """
        ts = now if now is not None else time.time()
        obs = _derive_observation(tf, feature_vector, ts)
        self._push(instrument, obs)
        return obs

    def record_raw(self, instrument: str, obs: TimeframeObservation) -> None:
        """Store a pre-built observation (used by `load_snapshot`)."""
        self._push(instrument, obs)

    def _push(self, instrument: str, obs: TimeframeObservation) -> None:
        sym = self._history.setdefault(instrument, {})
        q = sym.setdefault(obs.timeframe.value, deque(maxlen=self._depth))
        q.append(obs)

    # ── Current observations ─────────────────────────────────────────────

    def current_observations(
        self,
        instrument: str,
        now: float | None = None,
    ) -> dict[ContextTimeframe, TimeframeObservation]:
        """Most recent observation per timeframe for `instrument`.

        Only timeframes that have at least one stored observation are
        included.  Each returned observation has a freshly computed
        `staleness_ratio` relative to `now` (the stored copy is not
        mutated).
        """
        ts = now if now is not None else time.time()
        result: dict[ContextTimeframe, TimeframeObservation] = {}
        for tf_val, q in self._history.get(instrument, {}).items():
            if not q:
                continue
            obs = q[-1]
            tf = ContextTimeframe(tf_val)
            tf_secs = CONTEXT_TIMEFRAME_SECONDS.get(tf, 3600)
            bar_age = max(0.0, ts - obs.as_of) if obs.as_of > 0 else 0.0
            staleness = bar_age / tf_secs if tf_secs > 0 else 0.0
            result[tf] = dataclasses.replace(obs, staleness_ratio=staleness)
        return result

    # ── History queries ───────────────────────────────────────────────────

    def observation_history(
        self,
        instrument: str,
        tf: ContextTimeframe,
    ) -> list[TimeframeObservation]:
        """All stored observations for `tf`, oldest → newest."""
        q = self._history.get(instrument, {}).get(tf.value)
        return list(q) if q else []

    def momentum_trend(
        self,
        instrument: str,
        tf: ContextTimeframe,
    ) -> float:
        """Signed linear slope of momentum over the stored observation window.

        Positive → momentum has been strengthening (in whichever direction
        it is pointing); negative → momentum has been weakening (potential
        exhaustion signal).  Returns 0.0 when fewer than two valid
        observations are available.

        "Strengthening" here means the *magnitude* has been growing in the
        current direction — the slope is computed on the raw signed value
        so a move from +0.1 → +0.5 is +slope and a move from +0.5 → +0.1
        is −slope, which is the exhaustion pattern.
        """
        history = [
            o for o in self.observation_history(instrument, tf) if o.bars_available > 0
        ]
        momenta = [o.momentum for o in history]
        if len(momenta) < 2:
            return 0.0
        n = len(momenta)
        xs = list(range(n))
        x_mean = sum(xs) / n
        y_mean = sum(momenta) / n
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, momenta))
        den = sum((x - x_mean) ** 2 for x in xs)
        return num / den if den > 1e-12 else 0.0

    def recently_crossed_zero(
        self,
        instrument: str,
        tf: ContextTimeframe,
        lookback: int = 3,
    ) -> bool:
        """True if momentum sign changed within the last `lookback` observations.

        Ignores flat (sign=0) observations at either end — a sign change is
        only recorded when both the first and last non-zero signs in the
        window differ.  This prevents noisy near-zero crossings from
        triggering a TRANSITION classification.
        """
        history = self.observation_history(instrument, tf)
        valid = [o for o in history[-lookback:] if o.bars_available > 0 and o.momentum_sign != 0]
        if len(valid) < 2:
            return False
        return valid[0].momentum_sign != valid[-1].momentum_sign

    # ── Persistence ───────────────────────────────────────────────────────

    def to_snapshot(self, instrument: str) -> list[dict]:
        """Serialise all stored observations for `instrument` as a list of dicts."""
        rows: list[dict] = []
        for q in self._history.get(instrument, {}).values():
            rows.extend(obs.to_dict() for obs in q)
        return rows

    def load_snapshot(self, instrument: str, rows: list[dict]) -> None:
        """Restore history from a previously serialised snapshot."""
        self._history.pop(instrument, None)
        for row in rows:
            obs = TimeframeObservation.from_dict(row)
            self._push(instrument, obs)
