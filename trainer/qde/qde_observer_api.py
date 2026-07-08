"""Phase S7 — Market Observer API.

Public interface wrapping `QDEMarketObserver` + `QDEObserverDatabase` +
`ObserverMonitor` behind the same `ApiResponse` envelope convention used by
`qde_api.py`, so `qde_server.py` can expose it over HTTP the same way.

Endpoints:
 1. observe            — Feed one tick of market/portfolio/position data for
                          a symbol, advance its lifecycle state.
 2. get_state           — Current `ObserverMemory` snapshot for a symbol.
 3. list_states         — All tracked symbols' current state (optionally
                           filtered by `ObserverState`).
 4. history             — Transition log for a symbol.
 5. summary             — Aggregate counts across all tracked symbols.
 6. telemetry           — `ObserverSnapshot` telemetry for a symbol.
 7. telemetry_all       — Telemetry snapshots for every symbol with activity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.qde.qde_observer import QDEMarketObserver
from trainer.qde.qde_observer_dashboard import ObserverDashboard
from trainer.qde.qde_observer_database import QDEObserverDatabase
from trainer.qde.qde_observer_models import ObserverMemory, SymbolObservationInput
from trainer.qde.qde_observer_monitoring import ObserverMonitor
from trainer.qde.qde_observer_types import ObserverState


@dataclass
class ApiResponse:
    success: bool
    endpoint: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"success": self.success, "endpoint": self.endpoint, "data": self.data}
        if self.error:
            d["error"] = self.error
        return d


def _ok(endpoint: str, data: dict[str, Any]) -> ApiResponse:
    return ApiResponse(True, endpoint, data)


def _err(endpoint: str, error: str) -> ApiResponse:
    return ApiResponse(False, endpoint, error=error)


class QDEObserverAPI:
    """Public API for Phase S7's always-running market observer."""

    def __init__(self, db_path: str | None = None, monitor_window: int = 1000) -> None:
        self._engine = QDEMarketObserver()
        self._db = QDEObserverDatabase(db_path or "data/qde_observer.db")
        self._monitor = ObserverMonitor(window=monitor_window)
        self._dashboard = ObserverDashboard()
        self._restore_from_db()

    def _restore_from_db(self) -> None:
        """Rehydrate in-memory state from persistence on startup, so a
        process restart doesn't silently reset every symbol back to
        OBSERVING (which would defeat the overtrade/revenge-trade guards).
        """
        for snapshot in self._db.list_states():
            try:
                mem = ObserverMemory(
                    instrument=snapshot["instrument"],
                    state=ObserverState(snapshot["state"]),
                    previous_state=ObserverState(snapshot["previous_state"]),
                    entered_state_at=snapshot["entered_state_at"],
                    last_updated_at=snapshot["last_updated_at"],
                    readiness_score=snapshot.get("readiness_score", 0.0),
                    consecutive_losses=snapshot.get("consecutive_losses", 0),
                    cooldown_until=snapshot.get("cooldown_until", 0.0),
                    entry_window_expires_at=snapshot.get("entry_window_expires_at", 0.0),
                    transition_count=snapshot.get("transition_count", 0),
                )
                self._engine.load_memory(mem)
            except (KeyError, ValueError) as exc:
                logger.warning(f"QDEObserverAPI: skipped malformed stored state: {exc}")

    # ── Observation ───────────────────────────────────────────────────────

    def observe(self, inp: SymbolObservationInput) -> ApiResponse:
        try:
            result = self._engine.observe(inp)
        except Exception as exc:  # noqa: BLE001 - never crash the monitor loop
            logger.error(f"QDEObserverAPI.observe({inp.instrument}) failed: {exc}")
            return _err("observe", str(exc))

        mem = self._engine.get_memory(inp.instrument)
        self._db.upsert_state(mem.to_dict())
        if result.transitioned:
            self._db.log_transition(result.to_dict())
        self._monitor.record(result.to_dict())
        return _ok("observe", result.to_dict())

    # ── State inspection ──────────────────────────────────────────────────

    def get_state(self, instrument: str) -> ApiResponse:
        mem = self._engine.get_memory(instrument)
        return _ok("get_state", mem.to_dict())

    def list_states(self, state: str | None = None) -> ApiResponse:
        all_mem = self._engine.all_memory()
        rows = [m.to_dict() for m in all_mem.values() if not state or m.state.value == state]
        return _ok("list_states", {"count": len(rows), "states": rows})

    def history(self, instrument: str, limit: int = 200) -> ApiResponse:
        rows = self._db.list_transitions(instrument=instrument, limit=limit)
        return _ok("history", {"instrument": instrument, "count": len(rows), "transitions": rows})

    def summary(self) -> ApiResponse:
        all_mem = self._engine.all_memory()
        by_state: dict[str, int] = {}
        for m in all_mem.values():
            by_state[m.state.value] = by_state.get(m.state.value, 0) + 1
        return _ok(
            "summary",
            {
                "tracked_instruments": len(all_mem),
                "by_state": by_state,
            },
        )

    # ── Telemetry ─────────────────────────────────────────────────────────

    def telemetry(self, instrument: str) -> ApiResponse:
        snap = self._monitor.get_snapshot(instrument)
        return _ok("telemetry", snap.to_dict())

    def telemetry_all(self) -> ApiResponse:
        snaps = self._monitor.all_snapshots()
        return _ok(
            "telemetry_all",
            {"count": len(snaps), "instruments": {k: v.to_dict() for k, v in snaps.items()}},
        )

    # ── Dashboards ────────────────────────────────────────────────────────

    def dashboard_state_timeline(self, granularity: str = "hour", limit: int = 2000) -> ApiResponse:
        transitions = self._db.list_transitions(limit=limit)
        return _ok(
            "dashboard_state_timeline", self._dashboard.state_timeline(transitions, granularity)
        )

    def dashboard_dwell_histogram(self) -> ApiResponse:
        states = [m.to_dict() for m in self._engine.all_memory().values()]
        return _ok("dashboard_dwell_histogram", self._dashboard.dwell_histogram(states))

    def dashboard_transition_funnel(self, limit: int = 2000) -> ApiResponse:
        transitions = self._db.list_transitions(limit=limit)
        return _ok("dashboard_transition_funnel", self._dashboard.transition_funnel(transitions))

    def dashboard_guard_trips(self, granularity: str = "day", limit: int = 2000) -> ApiResponse:
        transitions = self._db.list_transitions(limit=limit)
        return _ok("dashboard_guard_trips", self._dashboard.guard_trips(transitions, granularity))
