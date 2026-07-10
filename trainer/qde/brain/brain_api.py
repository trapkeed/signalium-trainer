"""QDE Brain — public API.

Wraps `QDEBrain` + `QDEBrainDatabase` behind the same `ApiResponse` envelope
convention used by `qde_observer_api.py`/`qde_api.py`, so `qde_server.py`
can expose it over HTTP identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.qde.brain.brain import QDEBrain
from trainer.qde.brain.persistence import QDEBrainDatabase
from trainer.qde.qde_observer_models import ObservationResult, SymbolObservationInput


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


class QDEBrainAPI:
    """Public API for QDE's internal market consciousness subsystem."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db = QDEBrainDatabase(db_path or "data/qde_brain.db")
        self._brain = QDEBrain(db=self._db)

    @property
    def brain(self) -> QDEBrain:
        """Direct access for in-process callers (e.g. enriching a live
        `QDEInferenceInput` without a network round-trip)."""
        return self._brain

    # ── Continuous ingestion ─────────────────────────────────────────────

    def on_market_update(
        self, inp: SymbolObservationInput, observation_result: ObservationResult | None = None
    ) -> ApiResponse:
        try:
            thought = self._brain.on_market_update(inp, observation_result=observation_result)
        except Exception as exc:  # noqa: BLE001 - never crash the ingestion loop
            logger.error(f"QDEBrainAPI.on_market_update({inp.instrument}) failed: {exc}")
            return _err("on_market_update", str(exc))
        return _ok("on_market_update", thought.to_dict())

    # ── Inspection ────────────────────────────────────────────────────────

    def get_thought(self, instrument: str) -> ApiResponse:
        return _ok("get_thought", self._brain.to_memory_context(instrument))

    def get_state(self, instrument: str) -> ApiResponse:
        return _ok("get_state", self._brain.get_state(instrument).to_dict())

    def list_states(self) -> ApiResponse:
        states = self._brain.all_states()
        return _ok(
            "list_states",
            {"count": len(states), "instruments": {k: v.to_dict() for k, v in states.items()}},
        )

    def reallocate_attention(self) -> ApiResponse:
        weights = self._brain.reallocate_attention()
        return _ok(
            "reallocate_attention", {"weights": {k: round(v, 4) for k, v in weights.items()}}
        )

    # ── Experience & reflection ───────────────────────────────────────────

    def record_experience(
        self,
        instrument: str,
        kind: str,
        description: str,
        outcome_quality: float,
        regime: str = "",
        tags: list[str] | None = None,
    ) -> ApiResponse:
        experience = self._brain.record_experience(
            instrument, kind, description, outcome_quality, regime=regime, tags=tags
        )
        return _ok("record_experience", experience.to_dict())

    def reflect(self, instrument: str, force: bool = False) -> ApiResponse:
        entry = self._brain.reflect(instrument, force=force)
        if entry is None:
            return _ok(
                "reflect",
                {"skipped": True, "reason": "reflected too recently; use force=true to override"},
            )
        return _ok("reflect", entry.to_dict())

    def reflection_history(self, instrument: str, limit: int = 200) -> ApiResponse:
        rows = self._db.list_reflections(instrument=instrument, limit=limit)
        return _ok(
            "reflection_history",
            {"instrument": instrument, "count": len(rows), "reflections": rows},
        )

    def experience_history(self, instrument: str, limit: int = 200) -> ApiResponse:
        rows = self._db.list_experiences(instrument=instrument, limit=limit)
        return _ok(
            "experience_history",
            {"instrument": instrument, "count": len(rows), "experiences": rows},
        )
