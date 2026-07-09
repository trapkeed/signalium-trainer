"""S20 — Telemetry Ingestion Router.

Admin-only FastAPI router for all eight telemetry streams. Mounted into the
main Research Lab server (research_lab_server.py) under /telemetry/*.

Every endpoint:
  - Requires the same admin bearer token as all Research Lab endpoints.
  - Validates the payload via Pydantic.
  - Writes immediately to the TelemetryWarehouse (append-only, no soft-deletes).
  - Returns the ingested event dict so callers can confirm what was stored.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from trainer.research_lab.telemetry_warehouse import TelemetryWarehouse

# ── Request Models ─────────────────────────────────────────────────────────────
# All requests share model_version, symbol, timestamp as mandatory fields.


class IngestDecisionRequest(BaseModel):
    model_version: str
    symbol: str
    timestamp: float
    direction: str
    confidence: float
    reasoning_summary: str = ""
    entry_price: float | None = None
    stop_distance: float | None = None
    take_profit_distance: float | None = None
    position_size: float | None = None
    session: str = ""
    market_conditions: dict[str, Any] = Field(default_factory=dict)
    regime: str = ""
    decision_id: str = ""
    event_id: str = ""  # Idempotency key: provide to deduplicate re-deliveries


class IngestTradeRequest(BaseModel):
    model_version: str
    symbol: str
    timestamp: float
    direction: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    hold_duration_s: float
    exit_reason: str = ""
    confidence_at_entry: float = 0.0
    market_conditions: dict[str, Any] = Field(default_factory=dict)
    session: str = ""
    regime: str = ""
    trade_id: str = ""
    event_id: str = ""  # Idempotency key


class IngestPositionRequest(BaseModel):
    model_version: str
    symbol: str
    timestamp: float
    action: str
    size: float
    price: float
    confidence: float = 0.0
    cumulative_pnl: float = 0.0
    market_conditions: dict[str, Any] = Field(default_factory=dict)
    position_actions: list[dict[str, Any]] = Field(default_factory=list)
    position_id: str = ""
    event_id: str = ""  # Idempotency key


class IngestFailureRequest(BaseModel):
    model_version: str
    symbol: str
    timestamp: float
    failure_type: str
    severity: str = "MEDIUM"
    details: str = ""
    affected_order_id: str = ""
    event_id: str = ""  # Idempotency key


class IngestReasoningRequest(BaseModel):
    model_version: str
    symbol: str
    timestamp: float
    decision_id: str
    output_heads: dict[str, Any] = Field(default_factory=dict)
    feature_values: dict[str, float] = Field(default_factory=dict)
    confidence_breakdown: dict[str, float] = Field(default_factory=dict)
    event_id: str = ""  # Idempotency key


class IngestExecutionRequest(BaseModel):
    model_version: str
    symbol: str
    timestamp: float
    order_type: str
    requested_price: float
    executed_price: float
    size: float
    execution_time_ms: float = 0.0
    venue: str = ""
    status: str = "FILLED"
    order_id: str = ""
    event_id: str = ""  # Idempotency key


class IngestCounterfactualRequest(BaseModel):
    model_version: str
    symbol: str
    timestamp: float
    trade_id: str
    what_if_description: str
    actual_pnl: float
    counterfactual_pnl: float
    actual_outcome: dict[str, Any] = Field(default_factory=dict)
    counterfactual_outcome: dict[str, Any] = Field(default_factory=dict)
    event_id: str = ""  # Idempotency key


class IngestLifecycleRequest(BaseModel):
    model_version: str
    symbol: str
    timestamp: float
    kind: str
    previous_state: str = ""
    new_state: str = ""
    trigger: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    event_id: str = ""  # Idempotency key


class IngestInferenceRequest(BaseModel):
    model_version: str
    symbol: str = ""
    timestamp: float
    inference_count: int
    latency_ms_p50: float = 0.0
    latency_ms_p95: float = 0.0
    latency_ms_max: float = 0.0
    queue_length: int = 0
    throughput_per_s: float = 0.0
    cpu_pct: float = 0.0
    memory_pct: float = 0.0
    gpu_pct: float | None = None
    window_s: float = 60.0
    event_id: str = ""  # Idempotency key


# ── Router factory ─────────────────────────────────────────────────────────────


def make_telemetry_router(
    warehouse: TelemetryWarehouse,
    auth_dep: Any,
) -> APIRouter:
    """Create and return the telemetry ingestion router.

    Args:
        warehouse: Shared TelemetryWarehouse instance.
        auth_dep: FastAPI Depends dependency that enforces admin auth.
    """
    router = APIRouter(prefix="/telemetry", tags=["telemetry"])

    def _wrap(event_dict: dict[str, Any]) -> dict[str, Any]:
        """Wrap an ingested event in a standard response envelope."""
        return {"success": True, "event": event_dict}

    def _err(detail: str) -> None:
        raise HTTPException(status_code=400, detail=detail)

    # ── POST /telemetry/decisions ─────────────────────────────────────────────

    @router.post("/decisions")
    def ingest_decision(req: IngestDecisionRequest, _: None = Depends(auth_dep)) -> dict:
        """Ingest a QDE decision event from Signalium."""
        try:
            event = warehouse.ingest_decision(**req.model_dump())
            return _wrap(event.to_dict())
        except Exception as exc:
            _err(str(exc))

    # ── POST /telemetry/trades ────────────────────────────────────────────────

    @router.post("/trades")
    def ingest_trade(req: IngestTradeRequest, _: None = Depends(auth_dep)) -> dict:
        """Ingest a completed trade outcome from Signalium."""
        try:
            event = warehouse.ingest_trade(**req.model_dump())
            return _wrap(event.to_dict())
        except Exception as exc:
            _err(str(exc))

    # ── POST /telemetry/positions ─────────────────────────────────────────────

    @router.post("/positions")
    def ingest_position(req: IngestPositionRequest, _: None = Depends(auth_dep)) -> dict:
        """Ingest a position lifecycle event from Signalium."""
        try:
            event = warehouse.ingest_position(**req.model_dump())
            return _wrap(event.to_dict())
        except Exception as exc:
            _err(str(exc))

    # ── POST /telemetry/failures ──────────────────────────────────────────────

    @router.post("/failures")
    def ingest_failure(req: IngestFailureRequest, _: None = Depends(auth_dep)) -> dict:
        """Ingest an execution-level failure event from Signalium."""
        try:
            event = warehouse.ingest_failure(**req.model_dump())
            return _wrap(event.to_dict())
        except Exception as exc:
            _err(str(exc))

    # ── POST /telemetry/reasoning ─────────────────────────────────────────────

    @router.post("/reasoning")
    def ingest_reasoning(req: IngestReasoningRequest, _: None = Depends(auth_dep)) -> dict:
        """Ingest raw model output heads and feature values for a decision."""
        try:
            event = warehouse.ingest_reasoning(**req.model_dump())
            return _wrap(event.to_dict())
        except Exception as exc:
            _err(str(exc))

    # ── POST /telemetry/execution ─────────────────────────────────────────────

    @router.post("/execution")
    def ingest_execution(req: IngestExecutionRequest, _: None = Depends(auth_dep)) -> dict:
        """Ingest an order execution record from Signalium."""
        try:
            event = warehouse.ingest_execution(**req.model_dump())
            return _wrap(event.to_dict())
        except Exception as exc:
            _err(str(exc))

    # ── POST /telemetry/counterfactuals ───────────────────────────────────────

    @router.post("/counterfactuals")
    def ingest_counterfactual(
        req: IngestCounterfactualRequest, _: None = Depends(auth_dep)
    ) -> dict:
        """Ingest a counterfactual retrospective event from Signalium."""
        try:
            event = warehouse.ingest_counterfactual(**req.model_dump())
            return _wrap(event.to_dict())
        except Exception as exc:
            _err(str(exc))

    # ── POST /telemetry/lifecycle ─────────────────────────────────────────────

    @router.post("/lifecycle")
    def ingest_lifecycle(req: IngestLifecycleRequest, _: None = Depends(auth_dep)) -> dict:
        """Ingest a QDE lifecycle state transition from Signalium."""
        try:
            event = warehouse.ingest_lifecycle(**req.model_dump())
            return _wrap(event.to_dict())
        except Exception as exc:
            _err(str(exc))

    # ── POST /telemetry/inference ─────────────────────────────────────────────

    @router.post("/inference")
    def ingest_inference(req: IngestInferenceRequest, _: None = Depends(auth_dep)) -> dict:
        """Ingest a periodic inference-serving telemetry sample (S25) from
        Signalium's live inference host: prediction throughput/latency and
        system resource usage (CPU/memory/GPU/queue length)."""
        try:
            event = warehouse.ingest_inference(**req.model_dump())
            return _wrap(event.to_dict())
        except Exception as exc:
            _err(str(exc))

    # ── GET /telemetry/status ─────────────────────────────────────────────────

    @router.get("/status")
    def telemetry_status(_: None = Depends(auth_dep)) -> dict:
        """Return row counts for all nine telemetry streams."""
        counts = warehouse.stream_counts()
        total = sum(counts.values())
        return {"success": True, "stream_counts": counts, "total_events": total}

    return router
