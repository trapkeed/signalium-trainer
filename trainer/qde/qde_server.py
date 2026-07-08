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

from trainer.qde.qde_api import QDELifecycleAPI

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


class ShadowTestRequest(BaseModel):
    challenger_version_id: str
    control_version_id: str | None = None


class CanaryTestRequest(BaseModel):
    challenger_version_id: str
    traffic_fraction: float = 0.05
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


# ── A/B Testing Endpoints ─────────────────────────────────────────────────────


@app.post("/qde/ab/shadow")
def create_shadow_test(req: ShadowTestRequest) -> dict:
    return _respond(
        _api.create_shadow_test(
            challenger_version_id=req.challenger_version_id,
            control_version_id=req.control_version_id,
        )
    )


@app.post("/qde/ab/canary")
def create_canary_test(req: CanaryTestRequest) -> dict:
    return _respond(
        _api.create_canary_test(
            challenger_version_id=req.challenger_version_id,
            traffic_fraction=req.traffic_fraction,
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


# ── Health ────────────────────────────────────────────────────────────────────


@app.get("/healthz")
def health() -> dict:
    return {"status": "ok", "service": "qde-lifecycle"}
