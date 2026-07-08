"""Phase S7 — Market Observer Dashboard.

4 dashboard views built from `QDEObserverDatabase` transition history and
`ObserverMonitor` snapshots, for the observability requirement of Phase S7:

1. State Timeline       — state distribution over time, bucketed.
2. Dwell Histogram       — how long symbols spend in each state.
3. Transition Funnel     — OBSERVING → ... → POSITION_ACTIVE conversion.
4. Guard Trips           — overtrading / revenge-trading guard activity.
"""

from __future__ import annotations

import time
from typing import Any


def _bucket(ts: float, granularity: str = "hour") -> str:
    t = time.gmtime(ts)
    if granularity == "day":
        return f"{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d}"
    if granularity == "minute":
        return f"{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d} {t.tm_hour:02d}:{t.tm_min:02d}"
    return f"{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d} {t.tm_hour:02d}:00"


# Lifecycle order used for the funnel view (pre-trade + entry side only —
# the funnel measures opportunity capture, not post-entry management).
_FUNNEL_STATES = ["OBSERVING", "PREPARING", "ARMED", "ENTRY_WINDOW", "POSITION_ACTIVE"]


class ObserverDashboard:
    """Generates all 4 market-observer dashboard views."""

    # ── 1. State Timeline ─────────────────────────────────────────────────

    def state_timeline(
        self, transitions: list[dict[str, Any]], granularity: str = "hour"
    ) -> dict[str, Any]:
        if not transitions:
            return {"view": "OBSERVER_STATE_TIMELINE", "periods": [], "data": {}}

        buckets: dict[str, dict[str, int]] = {}
        for t in transitions:
            b = _bucket(t.get("unix_ts", t.get("created_at", time.time())), granularity)
            buckets.setdefault(b, {})
            state = t.get("state", "UNKNOWN")
            buckets[b][state] = buckets[b].get(state, 0) + 1

        return {
            "view": "OBSERVER_STATE_TIMELINE",
            "granularity": granularity,
            "periods": sorted(buckets.keys()),
            "data": buckets,
            "total": len(transitions),
        }

    # ── 2. Dwell Histogram ────────────────────────────────────────────────

    def dwell_histogram(self, states: list[dict[str, Any]]) -> dict[str, Any]:
        """`states` is a list of current `ObserverMemory.to_dict()` snapshots
        (e.g. from `QDEObserverAPI.list_states`)."""
        buckets: dict[str, list[float]] = {}
        for s in states:
            buckets.setdefault(s.get("state", "UNKNOWN"), []).append(s.get("dwell_seconds", 0.0))

        data = {
            state: {
                "count": len(vals),
                "avg_dwell_seconds": round(sum(vals) / len(vals), 2) if vals else 0.0,
                "max_dwell_seconds": round(max(vals), 2) if vals else 0.0,
            }
            for state, vals in buckets.items()
        }
        return {"view": "OBSERVER_DWELL_HISTOGRAM", "data": data, "total": len(states)}

    # ── 3. Transition Funnel ──────────────────────────────────────────────

    def transition_funnel(self, transitions: list[dict[str, Any]]) -> dict[str, Any]:
        """Counts distinct instruments that reached each pre-trade stage at
        least once within the given transition window, approximating
        opportunity-capture conversion (OBSERVING -> ... -> POSITION_ACTIVE).
        """
        reached: dict[str, set[str]] = {state: set() for state in _FUNNEL_STATES}
        for t in transitions:
            state = t.get("state")
            instrument = t.get("instrument", "")
            if state in reached and instrument:
                reached[state].add(instrument)

        funnel = [{"state": s, "instrument_count": len(reached[s])} for s in _FUNNEL_STATES]
        base = funnel[0]["instrument_count"] or 1
        for row in funnel:
            row["conversion_from_observing"] = round(row["instrument_count"] / base, 4)

        return {"view": "OBSERVER_TRANSITION_FUNNEL", "funnel": funnel}

    # ── 4. Guard Trips ────────────────────────────────────────────────────

    def guard_trips(
        self, transitions: list[dict[str, Any]], granularity: str = "day"
    ) -> dict[str, Any]:
        buckets: dict[str, dict[str, int]] = {}
        totals = {"OVERTRADE": 0, "REVENGE_TRADE": 0}
        for t in transitions:
            guard = t.get("guard_tripped", "")
            if not guard:
                continue
            b = _bucket(t.get("unix_ts", t.get("created_at", time.time())), granularity)
            buckets.setdefault(b, {"OVERTRADE": 0, "REVENGE_TRADE": 0})
            buckets[b][guard] = buckets[b].get(guard, 0) + 1
            totals[guard] = totals.get(guard, 0) + 1

        return {
            "view": "OBSERVER_GUARD_TRIPS",
            "granularity": granularity,
            "periods": sorted(buckets.keys()),
            "data": buckets,
            "totals": totals,
        }
