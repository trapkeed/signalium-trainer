"""Phase 27.6 — QDE HTTP Server.

Lightweight FastAPI server that exposes the QDELifecycleAPI over HTTP.
This is the upstream service that the Express router proxies to.

Start with:
    uvicorn trainer.qde.qde_server:app --port 8001 --reload

Environment variables:
    QDE_REGISTRY_PATH   Path to the JSON registry file (default: /tmp/qde_registry.json)
    QDE_AUTO_PROMOTE    Set to "true" to enable automatic promotion (default: false)
    QDE_SERVER_PORT     Port to listen on (default: 8001)
"""

from __future__ import annotations

import os
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError as exc:
    raise ImportError(
        "FastAPI and Pydantic are required for the QDE server. "
        "Install with: pip install fastapi uvicorn pydantic"
    ) from exc

from trainer.lifecycle.lifecycle_models import MarketContext, PositionMetrics
from trainer.portfolio.portfolio_scorer import PortfolioMetrics
from trainer.qde.qde_api import QDELifecycleAPI
from trainer.qde.qde_observer_api import QDEObserverAPI
from trainer.qde.qde_observer_models import SymbolObservationInput
from trainer.qde.qde_outputs import Direction, QDEOutputHeads
from trainer.qde.qde_position_manager import (
    REVIEW_INTERVAL_SECONDS,
    PositionTelemetry,
    QDEPositionManager,
)

# ── Application Setup ─────────────────────────────────────────────────────────

app = FastAPI(
    title="QDE Lifecycle API",
    description="Quant Decision Engine lifecycle management",
    version="27.6.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_api = QDELifecycleAPI(
    storage_path=os.environ.get("QDE_REGISTRY_PATH"),
    auto_promote=os.environ.get("QDE_AUTO_PROMOTE", "").lower() == "true",
)

_position_manager = QDEPositionManager()

_observer_api = QDEObserverAPI(db_path=os.environ.get("QDE_OBSERVER_DB_PATH"))


# ── Request / Response models ─────────────────────────────────────────────────


class RegisterVersionRequest(BaseModel):
    label: str = ""
    parent_version_id: str | None = None
    tags: list[str] = []
    description: str = ""
    artifact_uri: str = ""
    version_number: int | None = None


class AdvancePipelineRequest(BaseModel):
    metrics: dict[str, float] = {}
    logs: list[str] = []


class FailPipelineRequest(BaseModel):
    error: str
    logs: list[str] = []


class SkipPipelineRequest(BaseModel):
    reason: str = ""


class EvaluatePromotionRequest(BaseModel):
    notes: str = ""


class ApprovePromotionRequest(BaseModel):
    approver: str = ""
    notes: str = ""


class RollbackRequest(BaseModel):
    reason: str
    initiated_by: str = "system"
    notes: str = ""


class RollbackPreviousRequest(BaseModel):
    reason: str
    initiated_by: str = "system"
    notes: str = ""


class ArchiveVersionRequest(BaseModel):
    archiver: str = ""
    reason: str = ""


class RetireVersionRequest(BaseModel):
    retirer: str = ""
    reason: str = ""


class ShadowTestRequest(BaseModel):
    challenger_version_id: str
    control_version_id: str | None = None


class SideBySideRequest(BaseModel):
    challenger_version_id: str
    control_version_id: str | None = None


class ABResultsRequest(BaseModel):
    control_metrics: dict[str, float]
    challenger_metrics: dict[str, float]


class ConcludeABRequest(BaseModel):
    winner: str
    conclusion: str


class PositionTelemetryRequest(BaseModel):
    instrument: str
    position: dict[str, float] = {}
    market: dict[str, Any] = {}
    heads: dict[str, Any] = {}
    free_margin_pct: float = 1.0
    seconds_since_last_review: float = 30.0


class ObservationRequest(BaseModel):
    instrument: str
    market: dict[str, Any] = {}
    microstructure_quality: float = 0.5
    volatility_state: str = "NORMAL"
    volatility_quality: float = 0.5
    liquidity_state: str = "NORMAL"
    liquidity_quality: float = 0.5
    session_quality: float = 0.5
    market_regime: str = "RANGE"
    portfolio: dict[str, Any] | None = None
    free_margin_pct: float = 1.0
    has_open_position: bool = False
    position: dict[str, float] | None = None
    position_just_opened: bool = False
    position_just_closed: bool = False
    last_close_was_loss: bool = False


# ── Helper ────────────────────────────────────────────────────────────────────


def _respond(response: Any) -> dict:
    d = response.to_dict()
    if not d.get("success"):
        raise HTTPException(status_code=400, detail=d.get("error", "Unknown error"))
    return d


# ── Registry Endpoints ────────────────────────────────────────────────────────


@app.post("/qde/versions")
def register_version(req: RegisterVersionRequest) -> dict:
    return _respond(
        _api.register_version(
            label=req.label,
            parent_version_id=req.parent_version_id,
            tags=req.tags,
            description=req.description,
            artifact_uri=req.artifact_uri,
            version_number=req.version_number,
        )
    )


@app.get("/qde/versions")
def list_versions(stage: str | None = None) -> dict:
    return _respond(_api.list_versions(stage=stage))


@app.get("/qde/versions/{version_id}")
def get_version(version_id: str) -> dict:
    return _respond(_api.get_version(version_id))


@app.get("/qde/versions/{version_id}/lineage")
def version_lineage(version_id: str) -> dict:
    return _respond(_api.version_lineage(version_id))


@app.get("/qde/active")
def active_version() -> dict:
    return _respond(_api.active_version())


@app.get("/qde/summary")
def registry_summary() -> dict:
    return _respond(_api.registry_summary())


# ── Pipeline Endpoints ────────────────────────────────────────────────────────


@app.post("/qde/versions/{version_id}/pipeline/start")
def start_pipeline(version_id: str) -> dict:
    return _respond(_api.start_pipeline(version_id))


@app.post("/qde/pipeline/{run_id}/advance")
def advance_pipeline(run_id: str, req: AdvancePipelineRequest) -> dict:
    return _respond(_api.advance_pipeline(run_id, metrics=req.metrics, logs=req.logs))


@app.post("/qde/pipeline/{run_id}/fail")
def fail_pipeline(run_id: str, req: FailPipelineRequest) -> dict:
    return _respond(_api.fail_pipeline(run_id, error=req.error, logs=req.logs))


@app.post("/qde/pipeline/{run_id}/skip")
def skip_pipeline_stage(run_id: str, req: SkipPipelineRequest) -> dict:
    return _respond(_api.skip_pipeline_stage(run_id, reason=req.reason))


@app.get("/qde/pipeline/{run_id}")
def get_pipeline_run(run_id: str) -> dict:
    return _respond(_api.get_pipeline_run(run_id))


@app.get("/qde/pipeline")
def list_pipeline_runs() -> dict:
    return _respond(_api.list_pipeline_runs())


# ── Promotion Endpoints ───────────────────────────────────────────────────────


@app.post("/qde/versions/{version_id}/promotion/evaluate")
def evaluate_promotion(version_id: str, req: EvaluatePromotionRequest) -> dict:
    return _respond(_api.evaluate_promotion(version_id, notes=req.notes))


@app.post("/qde/versions/{version_id}/promotion/approve")
def approve_promotion(version_id: str, req: ApprovePromotionRequest) -> dict:
    return _respond(_api.approve_promotion(version_id, approver=req.approver, notes=req.notes))


@app.post("/qde/versions/{version_id}/promotion/auto")
def auto_promote(version_id: str, req: EvaluatePromotionRequest) -> dict:
    return _respond(_api.auto_promote(version_id, notes=req.notes))


@app.get("/qde/promotion/history")
def promotion_history() -> dict:
    return _respond(_api.promotion_history())


@app.get("/qde/promotion/criteria")
def promotion_criteria() -> dict:
    return _respond(_api.promotion_criteria())


# ── Rollback Endpoints ────────────────────────────────────────────────────────


@app.post("/qde/rollback/{target_version_id}")
def rollback_to(target_version_id: str, req: RollbackRequest) -> dict:
    return _respond(
        _api.rollback_to(
            target_version_id=target_version_id,
            reason=req.reason,
            initiated_by=req.initiated_by,
            notes=req.notes,
        )
    )


@app.post("/qde/rollback")
def rollback_to_previous(req: RollbackPreviousRequest) -> dict:
    return _respond(
        _api.rollback_to_previous(
            reason=req.reason,
            initiated_by=req.initiated_by,
            notes=req.notes,
        )
    )


@app.get("/qde/rollback/history")
def rollback_history() -> dict:
    return _respond(_api.rollback_history())


@app.get("/qde/rollback/targets")
def eligible_rollback_targets() -> dict:
    return _respond(_api.eligible_rollback_targets())


# ── Archive / Retire Endpoints ────────────────────────────────────────────────


@app.post("/qde/versions/{version_id}/archive")
def archive_version(version_id: str, req: ArchiveVersionRequest) -> dict:
    return _respond(
        _api.archive_version(version_id, archiver=req.archiver, reason=req.reason)
    )


@app.post("/qde/versions/{version_id}/retire")
def retire_version(version_id: str, req: RetireVersionRequest) -> dict:
    return _respond(
        _api.retire_version(version_id, retirer=req.retirer, reason=req.reason)
    )


# ── A/B Testing Endpoints ─────────────────────────────────────────────────────


@app.post("/qde/ab/shadow")
def create_shadow_test(req: ShadowTestRequest) -> dict:
    return _respond(
        _api.create_shadow_test(
            challenger_version_id=req.challenger_version_id,
            control_version_id=req.control_version_id,
        )
    )


@app.post("/qde/ab/side-by-side")
def create_side_by_side_test(req: SideBySideRequest) -> dict:
    return _respond(
        _api.create_side_by_side_test(
            challenger_version_id=req.challenger_version_id,
            control_version_id=req.control_version_id,
        )
    )


@app.post("/qde/ab/{test_id}/results")
def record_ab_results(test_id: str, req: ABResultsRequest) -> dict:
    return _respond(
        _api.record_ab_results(
            test_id=test_id,
            control_metrics=req.control_metrics,
            challenger_metrics=req.challenger_metrics,
        )
    )


@app.post("/qde/ab/{test_id}/conclude")
def conclude_ab_test(test_id: str, req: ConcludeABRequest) -> dict:
    return _respond(
        _api.conclude_ab_test(
            test_id=test_id,
            winner=req.winner,
            conclusion=req.conclusion,
        )
    )


@app.get("/qde/ab/{test_id}/compare")
def compare_ab_test(test_id: str) -> dict:
    return _respond(_api.compare_ab_test(test_id))


@app.get("/qde/ab")
def list_ab_tests(
    status: str | None = None,
    test_type: str | None = None,
) -> dict:
    return _respond(_api.list_ab_tests(status=status, test_type=test_type))


# ── Position Management (Phase S3) ───────────────────────────────────────────
# Stateless: no registry/DB involved. Callers post a fresh telemetry snapshot
# every review cycle (default 30s — see trainer/qde/qde_position_manager.py)
# and get back a single PositionManagementDecision. See
# docs/qde/phase_s3_api_contract.md for the full contract.


@app.post("/qde/position-management/review")
def review_position_management(req: PositionTelemetryRequest) -> dict:
    heads_kwargs = dict(req.heads)
    heads_kwargs.pop("instrument", None)
    raw_direction = heads_kwargs.pop("direction", "FLAT")
    try:
        direction = Direction(raw_direction)
        heads = QDEOutputHeads(instrument=req.instrument, direction=direction, **heads_kwargs)
        telemetry = PositionTelemetry(
            instrument=req.instrument,
            position=PositionMetrics(**req.position),
            market=MarketContext(**req.market),
            heads=heads,
            free_margin_pct=req.free_margin_pct,
            seconds_since_last_review=req.seconds_since_last_review,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    decision = _position_manager.review(telemetry)
    return {"success": True, "decision": decision.to_dict()}


@app.get("/qde/position-management/interval")
def position_management_interval() -> dict:
    return {"success": True, "review_interval_seconds": REVIEW_INTERVAL_SECONDS}


# ── Market Observer (Phase S7) ───────────────────────────────────────────────
# Always-running per-symbol monitoring: callers tick `/qde/observer/{instrument}
# /observe` on their own cadence (including for symbols with no open
# position) so QDE maintains a live lifecycle state per symbol even between
# trades. State/telemetry persist across restarts via QDEObserverDatabase.


@app.post("/qde/observer/{instrument}/observe")
def observe_symbol(instrument: str, req: ObservationRequest) -> dict:
    try:
        inp = SymbolObservationInput(
            instrument=instrument,
            market=MarketContext(**req.market),
            microstructure_quality=req.microstructure_quality,
            volatility_state=req.volatility_state,
            volatility_quality=req.volatility_quality,
            liquidity_state=req.liquidity_state,
            liquidity_quality=req.liquidity_quality,
            session_quality=req.session_quality,
            market_regime=req.market_regime,
            portfolio=PortfolioMetrics(**req.portfolio) if req.portfolio else None,
            free_margin_pct=req.free_margin_pct,
            has_open_position=req.has_open_position,
            position=PositionMetrics(**req.position) if req.position else None,
            position_just_opened=req.position_just_opened,
            position_just_closed=req.position_just_closed,
            last_close_was_loss=req.last_close_was_loss,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _respond(_observer_api.observe(inp))


@app.get("/qde/observer/{instrument}/state")
def observer_state(instrument: str) -> dict:
    return _respond(_observer_api.get_state(instrument))


@app.get("/qde/observer/states")
def observer_states(state: str | None = None) -> dict:
    return _respond(_observer_api.list_states(state=state))


@app.get("/qde/observer/{instrument}/history")
def observer_history(instrument: str, limit: int = 200) -> dict:
    return _respond(_observer_api.history(instrument, limit=limit))


@app.get("/qde/observer/summary")
def observer_summary() -> dict:
    return _respond(_observer_api.summary())


@app.get("/qde/observer/{instrument}/telemetry")
def observer_telemetry(instrument: str) -> dict:
    return _respond(_observer_api.telemetry(instrument))


@app.get("/qde/observer/telemetry")
def observer_telemetry_all() -> dict:
    return _respond(_observer_api.telemetry_all())


@app.get("/qde/observer/dashboard/state-timeline")
def observer_dashboard_state_timeline(granularity: str = "hour", limit: int = 2000) -> dict:
    return _respond(_observer_api.dashboard_state_timeline(granularity=granularity, limit=limit))


@app.get("/qde/observer/dashboard/dwell-histogram")
def observer_dashboard_dwell_histogram() -> dict:
    return _respond(_observer_api.dashboard_dwell_histogram())


@app.get("/qde/observer/dashboard/transition-funnel")
def observer_dashboard_transition_funnel(limit: int = 2000) -> dict:
    return _respond(_observer_api.dashboard_transition_funnel(limit=limit))


@app.get("/qde/observer/dashboard/guard-trips")
def observer_dashboard_guard_trips(granularity: str = "day", limit: int = 2000) -> dict:
    return _respond(_observer_api.dashboard_guard_trips(granularity=granularity, limit=limit))


# ── Health ────────────────────────────────────────────────────────────────────


@app.get("/healthz")
def health() -> dict:
    return {"status": "ok", "service": "qde-lifecycle"}
