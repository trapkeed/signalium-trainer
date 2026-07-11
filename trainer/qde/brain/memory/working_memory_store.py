"""QDE Brain — memory: timeframe-aware structured working memory store.

``WorkingMemoryStore`` is a per-symbol structured rolling buffer for raw
market observations.  It is the "immediate context" layer that feeds the
episodic and semantic memory systems with recent history.

Distinct from ``trainer.qde.brain.working_memory.WorkingMemory``
-----------------------------------------------------------------
The existing ``WorkingMemory`` is a salience-decaying item buffer used
for hypothesis/narrative context aging — general-purpose, item-centric,
priority-sorted.  ``WorkingMemoryStore`` is a *structured, append-only*
rolling buffer organized by observation kind and timeframe, with no
salience decay.  Both coexist on ``BrainState``; they answer different
questions:

  * ``WorkingMemory``      — "what items are still salient right now?"
  * ``WorkingMemoryStore`` — "what did the market look like recently,
                              organized by kind?"

What is tracked
---------------
Candles (per timeframe)
    M1→500, M5→300, M15→200, H1→150, H4→100 (configurable).

Volatility evolution
    ATR, spread, vol-state, volatility_quality — one reading per tick
    that carries these values.  Bounded to ``_DEFAULT_EVOL_DEPTH``.

Liquidity events
    Sweeps, clusters, breakout traps, level touches.  Each is a compact
    descriptor: kind, price level, direction, bars_ago at time of
    recording.  Bounded to ``_DEFAULT_EVENTS_DEPTH``.

Trend development
    Direction, maturity, strength, atr-normalized momentum readings
    sampled at each tick.  Bounded to ``_DEFAULT_EVOL_DEPTH``.

Failed breakouts
    Level that was tested, direction attempted, how many bars before
    reversal, and the bar-range ratio of the wick vs. body.  Bounded
    to ``_DEFAULT_EVENTS_DEPTH``.

Session transitions
    From-session → to-session with the timestamp the transition was
    detected.  Bounded to ``_DEFAULT_EVENTS_DEPTH``.

Spread evolution
    Per-tick bid/ask spread in pips plus a normalized 0-1 quality
    reading relative to the symbol's typical spread.  Bounded to
    ``_DEFAULT_EVOL_DEPTH``.

Correlation shifts
    Cross-asset correlation delta events: which asset pair shifted,
    by how much, and in which direction.  Bounded to
    ``_DEFAULT_EVENTS_DEPTH``.

Public API
----------
    store.write_memory(instrument, event_kind, payload, timeframe=None)
        Append a new observation.  ``event_kind`` must be one of the
        ``_KINDS`` constants.  ``timeframe`` is required for
        ``event_kind="candle"``.

    store.recall(instrument, event_kind, n=20, since=None)
        Return up to ``n`` most recent records of the given kind for
        ``instrument``, optionally filtered to those newer than
        ``since`` (Unix timestamp).  Returns a list of dicts.

    store.candles(instrument, timeframe, n=None)
        Return up to ``n`` most recent ``CandleRecord`` objects for
        ``instrument`` on ``timeframe``.  ``n=None`` returns all.

    store.to_snapshot(instrument) → dict
    store.load_snapshot(instrument, data) → None
        Serialize / restore a single instrument's memory.
        Used by the cognitive persistence layer.

    store.instruments() → list[str]
        All instruments with at least one recorded observation.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

# ── Timeframe candle limits ───────────────────────────────────────────────────

TIMEFRAME_CANDLE_LIMITS: dict[str, int] = {
    "M1": 500,
    "M5": 300,
    "M15": 200,
    "H1": 150,
    "H4": 100,
    # Aliases
    "1m": 500,
    "5m": 300,
    "15m": 200,
    "1h": 150,
    "4h": 100,
    "1M": 500,
    "5M": 300,
    "15M": 200,
    "1H": 150,
    "4H": 100,
}

_DEFAULT_CANDLE_LIMIT = 200  # for unrecognised timeframes
_DEFAULT_EVOL_DEPTH = 500  # volatility / trend / spread evolution
_DEFAULT_EVENTS_DEPTH = 200  # discrete events (sweeps, breakouts, …)


# ── Observation record types ──────────────────────────────────────────────────


@dataclass
class CandleRecord:
    """One OHLCV bar for a specific timeframe.

    ``timeframe`` is the label used as the key in ``TIMEFRAME_CANDLE_LIMITS``
    (e.g. ``"M1"``, ``"H1"``).  ``timestamp`` is the bar-open Unix time.
    """

    timeframe: str = "M1"
    timestamp: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0

    @property
    def is_up(self) -> bool:
        return self.close >= self.open

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def full_range(self) -> float:
        return self.high - self.low

    def to_dict(self) -> dict:
        return {
            "timeframe": self.timeframe,
            "timestamp": round(self.timestamp, 3),
            "open": round(self.open, 6),
            "high": round(self.high, 6),
            "low": round(self.low, 6),
            "close": round(self.close, 6),
            "volume": round(self.volume, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> CandleRecord:
        return cls(
            timeframe=d.get("timeframe", "M1"),
            timestamp=d.get("timestamp", 0.0),
            open=d.get("open", 0.0),
            high=d.get("high", 0.0),
            low=d.get("low", 0.0),
            close=d.get("close", 0.0),
            volume=d.get("volume", 0.0),
        )


@dataclass
class VolatilityPoint:
    """One volatility reading for the evolution stream.

    ``atr``                  — absolute ATR value.
    ``spread``               — bid/ask spread (same price units as ATR).
    ``volatility``           — normalized 0-1 reading from the MIE.
    ``vol_state``            — string label, e.g. "NORMAL", "ELEVATED".
    ``volatility_quality``   — 0-1 quality / confidence in the reading.
    """

    timestamp: float = 0.0
    atr: float = 0.0
    spread: float = 0.0
    volatility: float = 0.0
    vol_state: str = "NORMAL"
    volatility_quality: float = 0.5

    def to_dict(self) -> dict:
        return {
            "timestamp": round(self.timestamp, 3),
            "atr": round(self.atr, 6),
            "spread": round(self.spread, 6),
            "volatility": round(self.volatility, 4),
            "vol_state": self.vol_state,
            "volatility_quality": round(self.volatility_quality, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> VolatilityPoint:
        return cls(
            timestamp=d.get("timestamp", 0.0),
            atr=d.get("atr", 0.0),
            spread=d.get("spread", 0.0),
            volatility=d.get("volatility", 0.0),
            vol_state=d.get("vol_state", "NORMAL"),
            volatility_quality=d.get("volatility_quality", 0.5),
        )


@dataclass
class LiquidityEvent:
    """One discrete liquidity-layer event.

    ``kind``      — e.g. "SWEEP", "CLUSTER", "BREAKOUT_TRAP", "LEVEL_TOUCH",
                    "EQUAL_HIGHS", "EQUAL_LOWS".
    ``level``     — price level at which the event occurred.
    ``direction`` — +1 = bullish (sweep of sell stops), −1 = bearish.
    ``bars_ago``  — how many bars before this tick the event was detected.
                    0 = current bar.
    ``magnitude`` — 0-1 relative significance (e.g. wick/body ratio,
                    cluster size).
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    timestamp: float = 0.0
    kind: str = "SWEEP"
    level: float = 0.0
    direction: int = 0
    bars_ago: int = 0
    magnitude: float = 0.5

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": round(self.timestamp, 3),
            "kind": self.kind,
            "level": round(self.level, 6),
            "direction": self.direction,
            "bars_ago": self.bars_ago,
            "magnitude": round(self.magnitude, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> LiquidityEvent:
        return cls(
            event_id=d.get("event_id", str(uuid.uuid4())[:10]),
            timestamp=d.get("timestamp", 0.0),
            kind=d.get("kind", "SWEEP"),
            level=d.get("level", 0.0),
            direction=d.get("direction", 0),
            bars_ago=d.get("bars_ago", 0),
            magnitude=d.get("magnitude", 0.5),
        )


@dataclass
class TrendState:
    """One trend-development reading.

    ``direction``  — +1 bullish, −1 bearish, 0 neutral.
    ``maturity``   — string label from ``TrendMaturity`` enum, e.g.
                     "NONE", "EARLY", "ESTABLISHED", "LATE", "EXHAUSTED".
    ``strength``   — 0-1 directional strength.
    ``momentum``   — signed −1..+1 normalized momentum.
    ``trend_valid`` — whether the upstream regime classifier considers the
                      trend structurally valid.
    """

    timestamp: float = 0.0
    direction: int = 0
    maturity: str = "NONE"
    strength: float = 0.0
    momentum: float = 0.0
    trend_valid: bool = False

    def to_dict(self) -> dict:
        return {
            "timestamp": round(self.timestamp, 3),
            "direction": self.direction,
            "maturity": self.maturity,
            "strength": round(self.strength, 4),
            "momentum": round(self.momentum, 4),
            "trend_valid": self.trend_valid,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TrendState:
        return cls(
            timestamp=d.get("timestamp", 0.0),
            direction=d.get("direction", 0),
            maturity=d.get("maturity", "NONE"),
            strength=d.get("strength", 0.0),
            momentum=d.get("momentum", 0.0),
            trend_valid=d.get("trend_valid", False),
        )


@dataclass
class FailedBreakout:
    """One failed breakout event.

    A failed breakout is a move beyond a structural level that reversed
    within ``bars_until_reversal`` bars, leaving a trap.

    ``level``                — price of the breached structural level.
    ``direction``            — +1 = upside breakout failed, −1 = downside.
    ``bars_until_reversal``  — how many bars before the close re-entered
                               the prior range.
    ``wick_body_ratio``      — the wick that exceeded the level divided
                               by the bar body; higher = more aggressive
                               trap.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    timestamp: float = 0.0
    level: float = 0.0
    direction: int = 1
    bars_until_reversal: int = 1
    wick_body_ratio: float = 1.0

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": round(self.timestamp, 3),
            "level": round(self.level, 6),
            "direction": self.direction,
            "bars_until_reversal": self.bars_until_reversal,
            "wick_body_ratio": round(self.wick_body_ratio, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> FailedBreakout:
        return cls(
            event_id=d.get("event_id", str(uuid.uuid4())[:10]),
            timestamp=d.get("timestamp", 0.0),
            level=d.get("level", 0.0),
            direction=d.get("direction", 1),
            bars_until_reversal=d.get("bars_until_reversal", 1),
            wick_body_ratio=d.get("wick_body_ratio", 1.0),
        )


@dataclass
class SessionTransition:
    """A trading session transition event.

    ``from_session``  — e.g. "ASIAN", "LONDON", "NY", "OFF_HOURS".
    ``to_session``    — the new session label.
    """

    timestamp: float = 0.0
    from_session: str = ""
    to_session: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": round(self.timestamp, 3),
            "from_session": self.from_session,
            "to_session": self.to_session,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SessionTransition:
        return cls(
            timestamp=d.get("timestamp", 0.0),
            from_session=d.get("from_session", ""),
            to_session=d.get("to_session", ""),
        )


@dataclass
class SpreadPoint:
    """One spread reading in the evolution stream.

    ``spread_pips``       — raw spread in price-unit pips (ask − bid).
    ``spread_normalized`` — 0-1 normalized relative to the symbol's
                            historical spread range (0=tight, 1=wide).
    ``microstructure_quality`` — 0-1 from the MIE fill-quality signal.
    """

    timestamp: float = 0.0
    spread_pips: float = 0.0
    spread_normalized: float = 0.5
    microstructure_quality: float = 0.5

    def to_dict(self) -> dict:
        return {
            "timestamp": round(self.timestamp, 3),
            "spread_pips": round(self.spread_pips, 6),
            "spread_normalized": round(self.spread_normalized, 4),
            "microstructure_quality": round(self.microstructure_quality, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> SpreadPoint:
        return cls(
            timestamp=d.get("timestamp", 0.0),
            spread_pips=d.get("spread_pips", 0.0),
            spread_normalized=d.get("spread_normalized", 0.5),
            microstructure_quality=d.get("microstructure_quality", 0.5),
        )


@dataclass
class CorrelationShift:
    """A cross-asset correlation change event.

    ``asset_pair``          — e.g. "EURUSD/DXY", "GBPUSD/EURUSD".
    ``correlation_before``  — correlation before the shift (−1..+1).
    ``correlation_after``   — correlation after (−1..+1).
    ``delta``               — ``correlation_after − correlation_before``.
    ``significance``        — 0-1 estimated significance of the shift.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    timestamp: float = 0.0
    asset_pair: str = ""
    correlation_before: float = 0.0
    correlation_after: float = 0.0
    delta: float = 0.0
    significance: float = 0.0

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": round(self.timestamp, 3),
            "asset_pair": self.asset_pair,
            "correlation_before": round(self.correlation_before, 4),
            "correlation_after": round(self.correlation_after, 4),
            "delta": round(self.delta, 4),
            "significance": round(self.significance, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> CorrelationShift:
        return cls(
            event_id=d.get("event_id", str(uuid.uuid4())[:10]),
            timestamp=d.get("timestamp", 0.0),
            asset_pair=d.get("asset_pair", ""),
            correlation_before=d.get("correlation_before", 0.0),
            correlation_after=d.get("correlation_after", 0.0),
            delta=d.get("delta", 0.0),
            significance=d.get("significance", 0.0),
        )


# ── Event kind constants ──────────────────────────────────────────────────────

KIND_CANDLE = "candle"
KIND_VOLATILITY = "volatility"
KIND_LIQUIDITY = "liquidity"
KIND_TREND = "trend"
KIND_BREAKOUT = "failed_breakout"
KIND_SESSION = "session_transition"
KIND_SPREAD = "spread"
KIND_CORRELATION = "correlation"

_ALL_KINDS = frozenset(
    {
        KIND_CANDLE,
        KIND_VOLATILITY,
        KIND_LIQUIDITY,
        KIND_TREND,
        KIND_BREAKOUT,
        KIND_SESSION,
        KIND_SPREAD,
        KIND_CORRELATION,
    }
)

# Per-symbol state
_SymbolCandles = dict[str, Deque[CandleRecord]]  # timeframe → deque
_SymbolVolatility = Deque[VolatilityPoint]
_SymbolLiquidity = Deque[LiquidityEvent]
_SymbolTrend = Deque[TrendState]
_SymbolBreakouts = Deque[FailedBreakout]
_SymbolSessions = Deque[SessionTransition]
_SymbolSpread = Deque[SpreadPoint]
_SymbolCorrelations = Deque[CorrelationShift]


# ── WorkingMemoryStore ────────────────────────────────────────────────────────


class WorkingMemoryStore:
    """Timeframe-aware structured rolling memory for market observations.

    One instance is shared across all instruments.  Per-instrument state
    is stored in dicts keyed by instrument name.

    Usage pattern
    -------------
    On every market tick, callers push the relevant observations::

        store = WorkingMemoryStore()

        # Push a new candle (required every tick that has a closed bar):
        store.write_memory("EURUSD", "candle", candle_dict, timeframe="M5")

        # Push the volatility reading from the tick:
        store.write_memory("EURUSD", "volatility", vol_dict)

        # Push a liquidity event (only when detected):
        store.write_memory("EURUSD", "liquidity", liq_event_dict)

    Later, retrieve for an episode snapshot::

        recent_candles = store.candles("EURUSD", "M5", n=50)
        vol_history    = store.recall("EURUSD", "volatility", n=100)
    """

    def __init__(
        self,
        candle_limits: dict[str, int] | None = None,
        evol_depth: int = _DEFAULT_EVOL_DEPTH,
        events_depth: int = _DEFAULT_EVENTS_DEPTH,
        default_tf_limit: int = _DEFAULT_CANDLE_LIMIT,
    ) -> None:
        self._candle_limits = {**TIMEFRAME_CANDLE_LIMITS, **(candle_limits or {})}
        self._evol_depth = max(1, evol_depth)
        self._events_depth = max(1, events_depth)
        self._default_tf_limit = max(1, default_tf_limit)

        # per-instrument state
        self._candles: dict[str, dict[str, Deque[CandleRecord]]] = {}
        self._volatility: dict[str, Deque[VolatilityPoint]] = {}
        self._liquidity: dict[str, Deque[LiquidityEvent]] = {}
        self._trend: dict[str, Deque[TrendState]] = {}
        self._breakouts: dict[str, Deque[FailedBreakout]] = {}
        self._sessions: dict[str, Deque[SessionTransition]] = {}
        self._spread: dict[str, Deque[SpreadPoint]] = {}
        self._correlations: dict[str, Deque[CorrelationShift]] = {}

    # ── Write ─────────────────────────────────────────────────────────────

    def write_memory(
        self,
        instrument: str,
        event_kind: str,
        payload: dict,
        timeframe: str | None = None,
        now: float | None = None,
    ) -> None:
        """Append one observation to the appropriate stream.

        Parameters
        ----------
        instrument:
            Symbol name, e.g. ``"EURUSD"``.
        event_kind:
            One of the ``KIND_*`` constants.  Use the string constants
            directly or pass ``"candle"``, ``"volatility"``, etc.
        payload:
            A dict matching the ``to_dict()`` format of the appropriate
            record type.  Missing keys fall back to defaults.
        timeframe:
            Required when ``event_kind == "candle"``.  E.g. ``"M5"``.
        now:
            Override current time.  Used for replay / testing.
        """
        now = now if now is not None else time.time()
        kind = event_kind.lower()

        if kind == KIND_CANDLE:
            tf = (timeframe or payload.get("timeframe", "M1")).upper()
            # Normalize common aliases
            tf = _normalize_timeframe(tf)
            limit = self._candle_limits.get(tf, self._default_tf_limit)
            sym_candles = self._candles.setdefault(instrument, {})
            q = sym_candles.setdefault(tf, deque(maxlen=limit))
            rec = CandleRecord.from_dict({**payload, "timeframe": tf})
            if rec.timestamp == 0.0:
                rec.timestamp = now
            q.append(rec)

        elif kind == KIND_VOLATILITY:
            q = self._volatility.setdefault(instrument, deque(maxlen=self._evol_depth))
            rec = VolatilityPoint.from_dict(payload)  # type: ignore[assignment]
            if rec.timestamp == 0.0:
                rec.timestamp = now
            q.append(rec)

        elif kind == KIND_LIQUIDITY:
            q = self._liquidity.setdefault(instrument, deque(maxlen=self._events_depth))
            rec = LiquidityEvent.from_dict(payload)  # type: ignore[assignment]
            if rec.timestamp == 0.0:
                rec.timestamp = now
            q.append(rec)

        elif kind == KIND_TREND:
            q = self._trend.setdefault(instrument, deque(maxlen=self._evol_depth))
            rec = TrendState.from_dict(payload)  # type: ignore[assignment]
            if rec.timestamp == 0.0:
                rec.timestamp = now
            q.append(rec)

        elif kind == KIND_BREAKOUT:
            q = self._breakouts.setdefault(instrument, deque(maxlen=self._events_depth))
            rec = FailedBreakout.from_dict(payload)  # type: ignore[assignment]
            if rec.timestamp == 0.0:
                rec.timestamp = now
            q.append(rec)

        elif kind == KIND_SESSION:
            q = self._sessions.setdefault(instrument, deque(maxlen=self._events_depth))
            rec = SessionTransition.from_dict(payload)  # type: ignore[assignment]
            if rec.timestamp == 0.0:
                rec.timestamp = now
            q.append(rec)

        elif kind == KIND_SPREAD:
            q = self._spread.setdefault(instrument, deque(maxlen=self._evol_depth))
            rec = SpreadPoint.from_dict(payload)  # type: ignore[assignment]
            if rec.timestamp == 0.0:
                rec.timestamp = now
            q.append(rec)

        elif kind == KIND_CORRELATION:
            q = self._correlations.setdefault(instrument, deque(maxlen=self._events_depth))
            rec = CorrelationShift.from_dict(payload)  # type: ignore[assignment]
            if rec.timestamp == 0.0:
                rec.timestamp = now
            q.append(rec)

    # ── Read ──────────────────────────────────────────────────────────────

    def candles(
        self,
        instrument: str,
        timeframe: str,
        n: int | None = None,
    ) -> list[CandleRecord]:
        """Return up to ``n`` most recent candles (oldest first).

        ``n=None`` returns the full buffer.
        """
        tf = _normalize_timeframe(timeframe)
        store = self._candles.get(instrument, {})
        q = store.get(tf, deque())
        bars = list(q)
        if n is not None:
            bars = bars[-n:]
        return bars

    def recall(
        self,
        instrument: str,
        event_kind: str,
        n: int = 20,
        since: float | None = None,
    ) -> list[dict]:
        """Return up to ``n`` most recent observations of ``event_kind``.

        Results are sorted oldest → newest.  ``since`` filters to
        records whose timestamp is ≥ ``since``.

        Returns dicts (not typed objects) so callers don't need to know
        the internal record type.
        """
        kind = event_kind.lower()
        q: Deque | None = None

        if kind == KIND_VOLATILITY:
            q = self._volatility.get(instrument)
        elif kind == KIND_LIQUIDITY:
            q = self._liquidity.get(instrument)
        elif kind == KIND_TREND:
            q = self._trend.get(instrument)
        elif kind == KIND_BREAKOUT:
            q = self._breakouts.get(instrument)
        elif kind == KIND_SESSION:
            q = self._sessions.get(instrument)
        elif kind == KIND_SPREAD:
            q = self._spread.get(instrument)
        elif kind == KIND_CORRELATION:
            q = self._correlations.get(instrument)
        elif kind == KIND_CANDLE:
            # For candles, merge all timeframes sorted by timestamp
            all_bars = []
            for tf_bars in self._candles.get(instrument, {}).values():
                all_bars.extend(b.to_dict() for b in tf_bars)
            all_bars.sort(key=lambda d: d["timestamp"])
            if since is not None:
                all_bars = [d for d in all_bars if d["timestamp"] >= since]
            return all_bars[-n:]

        if q is None:
            return []

        records = list(q)
        if since is not None:
            records = [r for r in records if r.timestamp >= since]

        out = [r.to_dict() for r in records[-n:]]
        return out

    def latest(self, instrument: str, event_kind: str) -> dict | None:
        """Return the single most recent record of ``event_kind``, or None."""
        recs = self.recall(instrument, event_kind, n=1)
        return recs[-1] if recs else None

    def instruments(self) -> list[str]:
        """All instruments with at least one observation."""
        keys: set[str] = set()
        keys.update(self._candles)
        keys.update(self._volatility)
        keys.update(self._liquidity)
        keys.update(self._trend)
        keys.update(self._breakouts)
        keys.update(self._sessions)
        keys.update(self._spread)
        keys.update(self._correlations)
        return sorted(keys)

    def timeframes(self, instrument: str) -> list[str]:
        """All timeframes for which candle data exists for ``instrument``."""
        return sorted(self._candles.get(instrument, {}).keys())

    def candle_count(self, instrument: str, timeframe: str) -> int:
        tf = _normalize_timeframe(timeframe)
        store = self._candles.get(instrument, {})
        return len(store.get(tf, deque()))

    def event_count(self, instrument: str, event_kind: str) -> int:
        kind = event_kind.lower()
        if kind == KIND_VOLATILITY:
            return len(self._volatility.get(instrument, deque()))
        if kind == KIND_LIQUIDITY:
            return len(self._liquidity.get(instrument, deque()))
        if kind == KIND_TREND:
            return len(self._trend.get(instrument, deque()))
        if kind == KIND_BREAKOUT:
            return len(self._breakouts.get(instrument, deque()))
        if kind == KIND_SESSION:
            return len(self._sessions.get(instrument, deque()))
        if kind == KIND_SPREAD:
            return len(self._spread.get(instrument, deque()))
        if kind == KIND_CORRELATION:
            return len(self._correlations.get(instrument, deque()))
        return 0

    # ── Persistence ───────────────────────────────────────────────────────

    def to_snapshot(self, instrument: str) -> dict:
        """Serialize one instrument's full memory to a JSON-safe dict."""
        tf_candles: dict[str, list[dict]] = {}
        for tf, q in self._candles.get(instrument, {}).items():
            tf_candles[tf] = [c.to_dict() for c in q]

        return {
            "candles": tf_candles,
            "volatility": [r.to_dict() for r in self._volatility.get(instrument, [])],
            "liquidity": [r.to_dict() for r in self._liquidity.get(instrument, [])],
            "trend": [r.to_dict() for r in self._trend.get(instrument, [])],
            "breakouts": [r.to_dict() for r in self._breakouts.get(instrument, [])],
            "sessions": [r.to_dict() for r in self._sessions.get(instrument, [])],
            "spread": [r.to_dict() for r in self._spread.get(instrument, [])],
            "correlations": [r.to_dict() for r in self._correlations.get(instrument, [])],
        }

    def load_snapshot(self, instrument: str, data: dict) -> None:
        """Restore one instrument's memory from a previously-saved snapshot."""
        # Candles
        sym_candles: dict[str, Deque[CandleRecord]] = {}
        for tf, bars in data.get("candles", {}).items():
            tf = _normalize_timeframe(tf)
            limit = self._candle_limits.get(tf, self._default_tf_limit)
            q: Deque[CandleRecord] = deque(maxlen=limit)
            for b in bars:
                try:
                    q.append(CandleRecord.from_dict(b))
                except Exception:
                    pass
            sym_candles[tf] = q
        self._candles[instrument] = sym_candles

        def _load(store: dict, key: str, cls, depth: int) -> None:
            q2: Deque = deque(maxlen=depth)
            for d in data.get(key, []):
                try:
                    q2.append(cls.from_dict(d))
                except Exception:
                    pass
            store[instrument] = q2

        _load(self._volatility, "volatility", VolatilityPoint, self._evol_depth)
        _load(self._liquidity, "liquidity", LiquidityEvent, self._events_depth)
        _load(self._trend, "trend", TrendState, self._evol_depth)
        _load(self._breakouts, "breakouts", FailedBreakout, self._events_depth)
        _load(self._sessions, "sessions", SessionTransition, self._events_depth)
        _load(self._spread, "spread", SpreadPoint, self._evol_depth)
        _load(self._correlations, "correlations", CorrelationShift, self._events_depth)


# ── Helper ────────────────────────────────────────────────────────────────────

_TF_NORMALIZER: dict[str, str] = {
    "1M": "M1",
    "1m": "M1",
    "5M": "M5",
    "5m": "M5",
    "15M": "M15",
    "15m": "M15",
    "1H": "H1",
    "1h": "H1",
    "4H": "H4",
    "4h": "H4",
    "1D": "D1",
    "1d": "D1",
    "D": "D1",
}


def _normalize_timeframe(tf: str) -> str:
    """Normalize timeframe aliases to canonical form (M1, M5, H1, ...)."""
    return _TF_NORMALIZER.get(tf, tf.upper())
