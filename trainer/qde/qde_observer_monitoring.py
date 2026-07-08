"""Phase S7 — Market Observer telemetry / monitoring.

Rolling-window telemetry for `QDEMarketObserver`, mirroring
`trainer/mlops/monitoring.py`'s `ProductionMonitor` pattern. Tracks the
health signals that matter for the S7 goals directly:

  - Overtrading:      OVERTRADE_GUARD trip rate, re-arm frequency.
  - Low-quality entry: ENTRY_WINDOW readiness distribution.
  - Revenge trading:   REVENGE_TRADE_GUARD trip rate, cooldown durations.
  - Late entries:      ENTRY_WINDOW_EXPIRED rate (opportunities missed).
  - Opportunity capture: fraction of ARMED episodes that reach POSITION_ACTIVE.

Usage::

    monitor = ObserverMonitor(window=1000)
    monitor.record(result.to_dict())
    snapshot = monitor.get_snapshot("EURUSD")
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

_TERMINAL_PRE_TRADE_EVENTS = {
    "ENTRY_WINDOW_EXPIRED",
    "SETUP_INVALIDATED",
    "SETUP_WEAKENED",
}


@dataclass
class ObserverEventRecord:
    """One `ObservationResult` recorded for telemetry."""

    instrument: str
    state: str
    previous_state: str
    event: str
    readiness_score: float
    guard_tripped: str = ""
    transitioned: bool = False
    unix_ts: float = field(default_factory=time.time)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class ObserverSnapshot:
    """Aggregated telemetry for one instrument over the rolling window."""

    instrument: str
    window: int = 1000
    n_records: int = 0
    state_distribution: dict[str, float] = field(default_factory=dict)
    transitions_count: int = 0
    overtrade_guard_trips: int = 0
    revenge_trade_guard_trips: int = 0
    entry_window_expired_count: int = 0
    setups_armed_count: int = 0
    positions_opened_count: int = 0
    opportunity_capture_rate: float = 0.0  # positions_opened / setups_armed
    avg_readiness_at_entry_window: float = 0.0
    overtrade_rate: float = 0.0  # guard trips / transitions
    revenge_trade_rate: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "window": self.window,
            "n_records": self.n_records,
            "state_distribution": self.state_distribution,
            "transitions_count": self.transitions_count,
            "overtrade_guard_trips": self.overtrade_guard_trips,
            "revenge_trade_guard_trips": self.revenge_trade_guard_trips,
            "entry_window_expired_count": self.entry_window_expired_count,
            "setups_armed_count": self.setups_armed_count,
            "positions_opened_count": self.positions_opened_count,
            "opportunity_capture_rate": round(self.opportunity_capture_rate, 4),
            "avg_readiness_at_entry_window": round(self.avg_readiness_at_entry_window, 4),
            "overtrade_rate": round(self.overtrade_rate, 4),
            "revenge_trade_rate": round(self.revenge_trade_rate, 4),
            "timestamp": self.timestamp,
        }


class ObserverMonitor:
    """Rolling-window telemetry recorder for all observed instruments."""

    def __init__(self, window: int = 1000) -> None:
        self.window = window
        self._buffers: dict[str, deque[ObserverEventRecord]] = {}
        self._total_records: dict[str, int] = {}

    def record(self, observation: dict[str, Any]) -> ObserverEventRecord:
        rec = ObserverEventRecord(
            instrument=observation["instrument"],
            state=observation["state"],
            previous_state=observation["previous_state"],
            event=observation["event"],
            readiness_score=observation.get("readiness_score", 0.0),
            guard_tripped=observation.get("guard_tripped", ""),
            transitioned=observation.get("transitioned", False),
            unix_ts=observation.get("unix_ts", time.time()),
        )
        buf = self._buffers.setdefault(rec.instrument, deque(maxlen=self.window))
        buf.append(rec)
        self._total_records[rec.instrument] = self._total_records.get(rec.instrument, 0) + 1
        return rec

    def get_snapshot(self, instrument: str) -> ObserverSnapshot:
        buf = list(self._buffers.get(instrument, []))
        n = len(buf)
        if n == 0:
            return ObserverSnapshot(instrument=instrument, window=self.window)

        state_counts: dict[str, int] = {}
        for r in buf:
            state_counts[r.state] = state_counts.get(r.state, 0) + 1
        distribution = {k: round(v / n, 4) for k, v in state_counts.items()}

        transitions = [r for r in buf if r.transitioned]
        overtrade_trips = sum(1 for r in buf if r.guard_tripped == "OVERTRADE")
        revenge_trips = sum(1 for r in buf if r.guard_tripped == "REVENGE_TRADE")
        expired = sum(1 for r in buf if r.event == "ENTRY_WINDOW_EXPIRED")
        armed_events = [r for r in transitions if r.state == "ARMED"]
        entry_window_events = [r for r in transitions if r.state == "ENTRY_WINDOW"]
        opened_events = [r for r in transitions if r.event == "POSITION_OPENED"]

        n_armed = len(armed_events)
        n_opened = len(opened_events)
        capture_rate = (n_opened / n_armed) if n_armed > 0 else 0.0

        avg_readiness_entry = (
            sum(r.readiness_score for r in entry_window_events) / len(entry_window_events)
            if entry_window_events
            else 0.0
        )

        n_transitions = len(transitions) or 1

        return ObserverSnapshot(
            instrument=instrument,
            window=self.window,
            n_records=n,
            state_distribution=distribution,
            transitions_count=len(transitions),
            overtrade_guard_trips=overtrade_trips,
            revenge_trade_guard_trips=revenge_trips,
            entry_window_expired_count=expired,
            setups_armed_count=n_armed,
            positions_opened_count=n_opened,
            opportunity_capture_rate=capture_rate,
            avg_readiness_at_entry_window=avg_readiness_entry,
            overtrade_rate=overtrade_trips / n_transitions,
            revenge_trade_rate=revenge_trips / n_transitions,
        )

    def all_snapshots(self) -> dict[str, ObserverSnapshot]:
        return {instrument: self.get_snapshot(instrument) for instrument in self._buffers}

    def n_records(self, instrument: str) -> int:
        return self._total_records.get(instrument, 0)
