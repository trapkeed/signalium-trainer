"""QDE Brain — structural liquidity: data models and memory store.

This module covers *structural* liquidity — price levels where orders
accumulate, where they have been taken, and where they may exist next.
It is categorically separate from `mie.liquidity_intelligence`, which
classifies *flow* liquidity (is the market liquid enough to trade right
now via volume/spread scores?).

`OHLCBar` — minimal OHLCV record used by the detection layer.
`LiquidityLevel` — one price level with structural metadata.
`LiquidityEvent` — a discrete structural event (sweep, cluster found …).
`LiquidityMap` — per-symbol container of levels + event log.
`LiquidityMemory` — long-lived store of bar history + maps per symbol.

Design rule: all types are plain observational records. None carry
directional trading assertions (no "equal highs = sell", no "swept low
= reversal"). The model is free to learn relationships from these
features; this layer only records what happened, not what it means.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

# ── LiquidityLevelType ────────────────────────────────────────────────────────


class LiquidityLevelType(str, Enum):
    """Structural origin of a liquidity level.

    Types describe *how* the level formed — which price structure created
    it — not what trading action should follow.  The QDE learns those
    relationships from outcome data; they are never encoded here.
    """

    SWING_HIGH = "SWING_HIGH"  # local high in a lookback window
    SWING_LOW = "SWING_LOW"  # local low in a lookback window
    EQUAL_HIGH = "EQUAL_HIGH"  # ≥2 swing highs within tolerance
    EQUAL_LOW = "EQUAL_LOW"  # ≥2 swing lows within tolerance
    SESSION_HIGH = "SESSION_HIGH"  # highest price in a trading session block
    SESSION_LOW = "SESSION_LOW"
    DAILY_HIGH = "DAILY_HIGH"  # highest price in a UTC calendar day
    DAILY_LOW = "DAILY_LOW"
    WEEKLY_HIGH = "WEEKLY_HIGH"  # highest price in a UTC calendar week
    WEEKLY_LOW = "WEEKLY_LOW"
    STOP_CLUSTER = "STOP_CLUSTER"  # dense cluster of ≥3 nearby swing levels
    BREAKOUT_TRAP = "BREAKOUT_TRAP"  # level where price appeared to break out then reversed


class LevelSide(str, Enum):
    """Whether the level marks a high or low in price space.

    STOP_CLUSTER has both high and low members; using NONE avoids
    asserting which side the cluster belongs to.
    """

    HIGH = "HIGH"
    LOW = "LOW"
    NONE = "NONE"


# Lookup: what side does each level type belong to?
LEVEL_TYPE_SIDE: dict[LiquidityLevelType, LevelSide] = {
    LiquidityLevelType.SWING_HIGH: LevelSide.HIGH,
    LiquidityLevelType.SWING_LOW: LevelSide.LOW,
    LiquidityLevelType.EQUAL_HIGH: LevelSide.HIGH,
    LiquidityLevelType.EQUAL_LOW: LevelSide.LOW,
    LiquidityLevelType.SESSION_HIGH: LevelSide.HIGH,
    LiquidityLevelType.SESSION_LOW: LevelSide.LOW,
    LiquidityLevelType.DAILY_HIGH: LevelSide.HIGH,
    LiquidityLevelType.DAILY_LOW: LevelSide.LOW,
    LiquidityLevelType.WEEKLY_HIGH: LevelSide.HIGH,
    LiquidityLevelType.WEEKLY_LOW: LevelSide.LOW,
    LiquidityLevelType.STOP_CLUSTER: LevelSide.NONE,
    LiquidityLevelType.BREAKOUT_TRAP: LevelSide.NONE,
}


class LevelStatus(str, Enum):
    """Current lifecycle status of a structural level."""

    INTACT = "INTACT"  # formed, never touched
    TESTED = "TESTED"  # price approached, did not break through
    SWEPT = "SWEPT"  # price wicked through and closed back (stop-hunt)
    BROKEN = "BROKEN"  # price closed decisively through, no reversal
    RETESTED = "RETESTED"  # price returned to a previously broken/swept level


# ── OHLCBar ───────────────────────────────────────────────────────────────────


@dataclass
class OHLCBar:
    """Minimal OHLCV record for structural analysis.

    Kept deliberately thin — only the fields the detector needs.
    Bridge from `qde_timeframe_models.TimeframeBar` via
    `OHLCBar.from_timeframe_bar()`.
    """

    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    timestamp: float = 0.0  # epoch seconds (bar open or close time)

    @classmethod
    def from_timeframe_bar(cls, bar: object) -> OHLCBar:
        """Bridge from `qde_timeframe_models.TimeframeBar`."""
        return cls(
            open=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=float(getattr(bar, "volume", 0.0)),
            timestamp=float(getattr(bar, "open_time", 0.0) or getattr(bar, "close_time", 0.0)),
        )

    @classmethod
    def from_dict(cls, d: dict) -> OHLCBar:
        return cls(
            open=d.get("open", 0.0),
            high=d.get("high", 0.0),
            low=d.get("low", 0.0),
            close=d.get("close", 0.0),
            volume=d.get("volume", 0.0),
            timestamp=d.get("timestamp", 0.0),
        )

    def to_dict(self) -> dict:
        return {
            "open": round(self.open, 5),
            "high": round(self.high, 5),
            "low": round(self.low, 5),
            "close": round(self.close, 5),
            "volume": round(self.volume, 2),
            "timestamp": round(self.timestamp, 3),
        }


# ── LiquidityLevel ────────────────────────────────────────────────────────────


@dataclass
class LiquidityLevel:
    """One structural price level in the liquidity map.

    `strength` is a 0-1 score updated by `LiquidityTracker` as the level
    is tested, swept, or found to be part of a cluster.  It is a relative
    measure of structural significance — not a probability, not a
    directional signal.

    `test_count` counts how many times price approached within touch
    tolerance without sweeping; `sweep_count` counts confirmed sweeps.
    """

    level_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    instrument: str = ""
    level_type: LiquidityLevelType = LiquidityLevelType.SWING_HIGH
    side: LevelSide = LevelSide.HIGH
    price: float = 0.0
    status: LevelStatus = LevelStatus.INTACT
    strength: float = 0.40  # 0–1
    formed_at: float = field(default_factory=time.time)
    last_tested_at: float = 0.0
    last_status_change_at: float = field(default_factory=time.time)
    test_count: int = 0
    sweep_count: int = 0
    source_bar_timestamp: float = 0.0  # timestamp of the bar that formed this level
    cluster_member_count: int = 1  # >1 for equal-level / cluster entries
    notes: str = ""  # free-form (never a trading rule)

    def age_seconds(self, now: float | None = None) -> float:
        return max(0.0, (now or time.time()) - self.formed_at)

    def is_active(self) -> bool:
        """True unless the level has been decisively broken."""
        return self.status not in (LevelStatus.BROKEN,)

    def to_dict(self) -> dict:
        return {
            "level_id": self.level_id,
            "instrument": self.instrument,
            "level_type": self.level_type.value,
            "side": self.side.value,
            "price": round(self.price, 5),
            "status": self.status.value,
            "strength": round(self.strength, 4),
            "formed_at": round(self.formed_at, 3),
            "last_tested_at": round(self.last_tested_at, 3),
            "last_status_change_at": round(self.last_status_change_at, 3),
            "test_count": self.test_count,
            "sweep_count": self.sweep_count,
            "source_bar_timestamp": round(self.source_bar_timestamp, 3),
            "cluster_member_count": self.cluster_member_count,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LiquidityLevel:
        return cls(
            level_id=d["level_id"],
            instrument=d.get("instrument", ""),
            level_type=LiquidityLevelType(d["level_type"]),
            side=LevelSide(d["side"]),
            price=d["price"],
            status=LevelStatus(d["status"]),
            strength=d.get("strength", 0.4),
            formed_at=d.get("formed_at", 0.0),
            last_tested_at=d.get("last_tested_at", 0.0),
            last_status_change_at=d.get("last_status_change_at", 0.0),
            test_count=d.get("test_count", 0),
            sweep_count=d.get("sweep_count", 0),
            source_bar_timestamp=d.get("source_bar_timestamp", 0.0),
            cluster_member_count=d.get("cluster_member_count", 1),
            notes=d.get("notes", ""),
        )


# ── LiquidityEvent ────────────────────────────────────────────────────────────


class LiquidityEventType(str, Enum):
    """Category of structural event recorded on the liquidity map."""

    LEVEL_FORMED = "LEVEL_FORMED"
    LEVEL_TESTED = "LEVEL_TESTED"
    LEVEL_SWEPT = "LEVEL_SWEPT"  # wick through + close back
    LEVEL_BROKEN = "LEVEL_BROKEN"  # decisive close through
    LEVEL_RETESTED = "LEVEL_RETESTED"  # return to broken/swept level
    CLUSTER_FORMED = "CLUSTER_FORMED"  # equal-level group identified
    BREAKOUT_TRAP = "BREAKOUT_TRAP"  # false breakout detected
    STOP_CLUSTER_FOUND = "STOP_CLUSTER_FOUND"  # dense level cluster identified


@dataclass
class LiquidityEvent:
    """A discrete structural event recorded on the liquidity map.

    `context` is a free-form description of the price relationship that
    triggered the event — never a trading instruction.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    instrument: str = ""
    event_type: LiquidityEventType = LiquidityEventType.LEVEL_FORMED
    level_id: str = ""
    level_type: LiquidityLevelType = LiquidityLevelType.SWING_HIGH
    level_price: float = 0.0
    bar_timestamp: float = 0.0  # bar on which the event was detected
    detected_at: float = field(default_factory=time.time)
    context: str = ""  # e.g. "high=1.09510 > level=1.09490, close=1.09480"

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "instrument": self.instrument,
            "event_type": self.event_type.value,
            "level_id": self.level_id,
            "level_type": self.level_type.value,
            "level_price": round(self.level_price, 5),
            "bar_timestamp": round(self.bar_timestamp, 3),
            "detected_at": round(self.detected_at, 3),
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LiquidityEvent:
        return cls(
            event_id=d["event_id"],
            instrument=d.get("instrument", ""),
            event_type=LiquidityEventType(d["event_type"]),
            level_id=d.get("level_id", ""),
            level_type=LiquidityLevelType(d["level_type"]),
            level_price=d.get("level_price", 0.0),
            bar_timestamp=d.get("bar_timestamp", 0.0),
            detected_at=d.get("detected_at", 0.0),
            context=d.get("context", ""),
        )


# ── LiquidityMap ──────────────────────────────────────────────────────────────


@dataclass
class LiquidityMap:
    """Per-symbol structural liquidity map.

    `levels` is keyed by `level_id` for O(1) lookup.
    `events` is a bounded FIFO (maxlen=500) so the map never grows
    unboundedly from event history alone.
    """

    instrument: str = ""
    levels: dict[str, LiquidityLevel] = field(default_factory=dict)
    events: deque[LiquidityEvent] = field(default_factory=lambda: deque(maxlen=500))
    last_bar: OHLCBar | None = None
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "levels": [lv.to_dict() for lv in self.levels.values()],
            "events": [ev.to_dict() for ev in self.events],
            "last_bar": self.last_bar.to_dict() if self.last_bar else None,
            "updated_at": round(self.updated_at, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> LiquidityMap:
        mp = cls(instrument=d.get("instrument", ""))
        for lv_d in d.get("levels", []):
            lv = LiquidityLevel.from_dict(lv_d)
            mp.levels[lv.level_id] = lv
        for ev_d in d.get("events", []):
            mp.events.append(LiquidityEvent.from_dict(ev_d))
        if d.get("last_bar"):
            mp.last_bar = OHLCBar.from_dict(d["last_bar"])
        mp.updated_at = d.get("updated_at", 0.0)
        return mp


# ── LiquidityMemory ───────────────────────────────────────────────────────────

_DEFAULT_BAR_HISTORY = 500  # bars retained per symbol (enough for weekly extremes)
_DEFAULT_MAX_LEVELS = 300  # max total levels; broken levels pruned first


class LiquidityMemory:
    """Per-symbol rolling OHLCV history and structural liquidity map.

    One long-lived instance per process.  The bar history is the raw input
    the detector reads; the map is the detector's output that the engine
    incrementally updates.

    Stateful — holds the rolling deques and maps — but exposes only pure
    read accessors; mutations happen through explicit push/update calls.
    """

    def __init__(
        self,
        bar_history_depth: int = _DEFAULT_BAR_HISTORY,
        max_levels: int = _DEFAULT_MAX_LEVELS,
    ) -> None:
        self._depth = bar_history_depth
        self._max_levels = max_levels
        # instrument → deque[OHLCBar]  (oldest → newest)
        self._bars: dict[str, deque[OHLCBar]] = {}
        # instrument → LiquidityMap
        self._maps: dict[str, LiquidityMap] = {}

    # ── Bar ingestion ─────────────────────────────────────────────────────

    def push_bar(self, instrument: str, bar: OHLCBar) -> None:
        """Append a new bar; updates the map's last_bar reference."""
        q = self._bars.setdefault(instrument, deque(maxlen=self._depth))
        q.append(bar)
        mp = self._map(instrument)
        mp.last_bar = bar
        mp.updated_at = bar.timestamp or time.time()

    def push_bars(self, instrument: str, bars: list[OHLCBar]) -> None:
        for b in bars:
            self.push_bar(instrument, b)

    def bars(self, instrument: str) -> list[OHLCBar]:
        """Stored bars for `instrument`, oldest → newest."""
        return list(self._bars.get(instrument, []))

    # ── Map access ────────────────────────────────────────────────────────

    def _map(self, instrument: str) -> LiquidityMap:
        if instrument not in self._maps:
            self._maps[instrument] = LiquidityMap(instrument=instrument)
        return self._maps[instrument]

    def map_for(self, instrument: str) -> LiquidityMap:
        """Return the live LiquidityMap for `instrument`."""
        return self._map(instrument)

    def add_level(self, instrument: str, level: LiquidityLevel) -> None:
        self._map(instrument).levels[level.level_id] = level

    def add_event(self, instrument: str, event: LiquidityEvent) -> None:
        self._map(instrument).events.append(event)

    def prune_broken_levels(self, instrument: str) -> int:
        """Remove BROKEN levels when the map exceeds `_max_levels`, oldest first."""
        mp = self._map(instrument)
        if len(mp.levels) <= self._max_levels:
            return 0
        broken = sorted(
            [lid for lid, lv in mp.levels.items() if lv.status == LevelStatus.BROKEN],
            key=lambda lid: mp.levels[lid].formed_at,
        )
        to_remove = len(mp.levels) - self._max_levels
        removed = 0
        for lid in broken[:to_remove]:
            del mp.levels[lid]
            removed += 1
        return removed

    # ── Persistence ───────────────────────────────────────────────────────

    def to_snapshot(self, instrument: str) -> dict:
        return {
            "map": self._map(instrument).to_dict(),
            "bars": [b.to_dict() for b in self.bars(instrument)],
        }

    def load_snapshot(self, instrument: str, data: dict) -> None:
        if "map" in data:
            self._maps[instrument] = LiquidityMap.from_dict(data["map"])
        if "bars" in data:
            q = deque(maxlen=self._depth)
            for bd in data["bars"]:
                q.append(OHLCBar.from_dict(bd))
            self._bars[instrument] = q
