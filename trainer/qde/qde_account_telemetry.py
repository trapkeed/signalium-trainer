"""Phase S8 — Account telemetry schema.

Mirrors `qde_observer_monitoring.py`'s `ObserverMonitor` pattern: rolling
window of raw events feeding an aggregated snapshot, no owned loop, caller
records after each `assess_risk()` call.

Tracks the health signals that matter for account awareness:
  - Posture distribution: how often is the account AGGRESSIVE vs DEFENSIVE
    vs NO_TRADE, over the window.
  - Capital protection activity: NO_TRADE rate, forced-stop rate.
  - Risk-signal stability: mean/volatility of risk_signal (a fund manager
    whose dial swings wildly tick-to-tick is as concerning as one who never
    moves it).
  - Time-in-defensive-mode: fraction of the window spent DEFENSIVE or
    REDUCE_RISK, an early-warning indicator distinct from outright stops.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

_DEFENSIVE_ACTIONS = {"DEFENSIVE", "REDUCE_RISK"}


@dataclass
class AccountTelemetryEvent:
    """One `AccountRiskAssessment` recorded for telemetry."""

    account_id: str
    risk_signal: float
    action: str
    forces_no_trade: bool = False
    reason_codes: list[str] = field(default_factory=list)
    unix_ts: float = field(default_factory=time.time)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class AccountTelemetrySnapshot:
    """Aggregated telemetry for one account over the rolling window."""

    account_id: str
    window: int = 1000
    n_records: int = 0
    action_distribution: dict[str, float] = field(default_factory=dict)
    avg_risk_signal: float = 0.5
    risk_signal_volatility: float = 0.0  # stdev over the window
    no_trade_rate: float = 0.0
    forced_stop_count: int = 0
    defensive_time_pct: float = 0.0  # fraction of window in DEFENSIVE/REDUCE_RISK
    top_reason_codes: dict[str, int] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "window": self.window,
            "n_records": self.n_records,
            "action_distribution": self.action_distribution,
            "avg_risk_signal": round(self.avg_risk_signal, 4),
            "risk_signal_volatility": round(self.risk_signal_volatility, 4),
            "no_trade_rate": round(self.no_trade_rate, 4),
            "forced_stop_count": self.forced_stop_count,
            "defensive_time_pct": round(self.defensive_time_pct, 4),
            "top_reason_codes": self.top_reason_codes,
            "timestamp": self.timestamp,
        }


class AccountTelemetryMonitor:
    """Rolling-window telemetry recorder for all tracked accounts."""

    def __init__(self, window: int = 1000) -> None:
        self.window = window
        self._buffers: dict[str, deque[AccountTelemetryEvent]] = {}
        self._total_records: dict[str, int] = {}

    def record(self, assessment: dict[str, Any]) -> AccountTelemetryEvent:
        evt = AccountTelemetryEvent(
            account_id=assessment["account_id"],
            risk_signal=assessment.get("risk_signal", 0.5),
            action=assessment.get("action", "NORMAL"),
            forces_no_trade=bool(assessment.get("forces_no_trade", False)),
            reason_codes=list(assessment.get("reason_codes", [])),
            unix_ts=assessment.get("unix_ts", time.time()),
        )
        buf = self._buffers.setdefault(evt.account_id, deque(maxlen=self.window))
        buf.append(evt)
        self._total_records[evt.account_id] = self._total_records.get(evt.account_id, 0) + 1
        return evt

    def get_snapshot(self, account_id: str) -> AccountTelemetrySnapshot:
        buf = list(self._buffers.get(account_id, []))
        n = len(buf)
        if n == 0:
            return AccountTelemetrySnapshot(account_id=account_id, window=self.window)

        action_counts: dict[str, int] = {}
        reason_counts: dict[str, int] = {}
        for e in buf:
            action_counts[e.action] = action_counts.get(e.action, 0) + 1
            for code in e.reason_codes:
                reason_counts[code] = reason_counts.get(code, 0) + 1
        distribution = {k: round(v / n, 4) for k, v in action_counts.items()}

        signals = [e.risk_signal for e in buf]
        avg_signal = sum(signals) / n
        variance = sum((s - avg_signal) ** 2 for s in signals) / n
        volatility = variance**0.5

        no_trade_count = sum(1 for e in buf if e.action == "NO_TRADE")
        forced_stops = sum(1 for e in buf if e.forces_no_trade)
        defensive_count = sum(1 for e in buf if e.action in _DEFENSIVE_ACTIONS)

        top_reasons = dict(sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True)[:10])

        return AccountTelemetrySnapshot(
            account_id=account_id,
            window=self.window,
            n_records=n,
            action_distribution=distribution,
            avg_risk_signal=avg_signal,
            risk_signal_volatility=volatility,
            no_trade_rate=no_trade_count / n,
            forced_stop_count=forced_stops,
            defensive_time_pct=defensive_count / n,
            top_reason_codes=top_reasons,
        )

    def all_snapshots(self) -> dict[str, AccountTelemetrySnapshot]:
        return {account_id: self.get_snapshot(account_id) for account_id in self._buffers}

    def n_records(self, account_id: str) -> int:
        return self._total_records.get(account_id, 0)
