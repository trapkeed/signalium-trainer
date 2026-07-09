"""Phase 28.5 — Research Lab HTTP Server.

Admin-only FastAPI server. Every endpoint is protected by a static bearer token.
The server has no public documentation UI, no API browser, and no frontend.
It is consumed exclusively by the Signalium Admin Portal via the Express proxy.

Start with:
    RESEARCH_LAB_ADMIN_TOKEN=<secret> uvicorn trainer.research_lab.research_lab_server:app --port 8002

Environment variables:
    RESEARCH_LAB_ADMIN_TOKEN  Required. Shared secret for all admin requests.
    RESEARCH_LAB_DB_PATH      Path to the SQLite database file (default: data/research_lab.db).
    RESEARCH_LAB_PORT         Port (default 8002).

ACCESS MODEL
------------
- This server runs on a private port (8002) inside the trainer container.
- It is NEVER exposed directly to the public internet.
- The Express API (port 8080) acts as the sole entry point for external callers.
- The Express API enforces admin authentication before proxying to this server.
- This server also enforces its own admin token as a defence-in-depth measure.
- No retail user may reach any endpoint in this server.
- No public documentation UI is served (docs_url=None, redoc_url=None).
- No OpenAPI schema endpoint is served (openapi_url=None).

RESPONSIBILITIES (8 core, confirmed by administrator specification)
-------------------------------------------------------------------
 1. Experiment tracking
 2. Backtesting
 3. Validation (deployment gates)
 4. Dataset generation
 5. Failure analysis
 6. Version comparisons
 7. Promotion approval
 8. Rollback approval
"""

from __future__ import annotations

import os
from typing import Any

try:
    from fastapi import Depends, FastAPI, HTTPException, Security
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel
except ImportError as exc:
    raise ImportError(
        "FastAPI and Pydantic are required. Install with: pip install fastapi uvicorn pydantic"
    ) from exc

from trainer.research_lab.approval_gate import ApprovalGate
from trainer.research_lab.approval_models import ApprovalAlreadyResolved, ApprovalGateType, DEFAULT_TTL_SECONDS
from trainer.research_lab.artifact_models import ARTIFACT_TYPES, ArtifactRecord, ArtifactType, TrainingRunRecord
from trainer.research_lab.artifact_registry import ArtifactRegistry, ArtifactHashMismatchError, DuplicateArtifactError
from trainer.research_lab.health_monitor import AlertSeverity, DriftDimension, HealthMonitorConfig, ResearchLabHealthMonitor
from trainer.research_lab.pipeline_controller import ResearchLabPipelineController
from trainer.research_lab.pipeline_models import PipelineJobSpec
from trainer.research_lab.discovery_models import DiscoveryQuery, DiscoverySummary
from trainer.research_lab.experiment_creator import ExperimentCreator
from trainer.research_lab.failure_investigation_engine import FailureInvestigationEngine
from trainer.research_lab.feature_discovery_pipeline import FeatureDiscoveryPipeline
from trainer.research_lab.investigation_models import InvestigationConfig, InvestigationQuery
from trainer.research_lab.label_discovery_engine import LabelDiscoveryEngine
from trainer.research_lab.recommendation_models import RecommendationConfig
from trainer.research_lab.research_lab_api import ResearchLabAPI
from trainer.research_lab.retraining_recommendation_engine import (
    RetrainingRecommendationEngine,
)
from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.telemetry_server import make_telemetry_router
from trainer.research_lab.telemetry_warehouse import TelemetryWarehouse

# ── Setup ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Research Lab API",
    description=(
        "Administrator-only QDE research and improvement subsystem. "
        "No public access. No frontend. Internal use only."
    ),
    version="28.5.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# No CORS middleware — internal server. All traffic arrives via the Express proxy.

# ── Telemetry router (S20) ────────────────────────────────────────────────────
# Mounted after the app and auth dep are both defined.

_db_path = os.environ.get("RESEARCH_LAB_DB_PATH", "data/research_lab.db")
_db = ResearchLabDatabase(_db_path)
_api = ResearchLabAPI(db=_db)
_warehouse = TelemetryWarehouse(_db)
_engine = FailureInvestigationEngine(_warehouse, _db)
_creator = ExperimentCreator(_db)
_feature_pipeline = FeatureDiscoveryPipeline(_warehouse, _db)
_label_engine = LabelDiscoveryEngine(_warehouse, _db)
_rec_engine = RetrainingRecommendationEngine(_warehouse, _db, RecommendationConfig())
_approval_gate = ApprovalGate(_db)
_pipeline_ctrl = ResearchLabPipelineController(_db, _approval_gate)
_artifact_registry = ArtifactRegistry(_db)
_health_monitor = ResearchLabHealthMonitor(_db, _warehouse, rec_engine=_rec_engine)
_bearer = HTTPBearer(auto_error=False)

# ── Auth ──────────────────────────────────────────────────────────────────────

_CONFIGURED_TOKEN = os.environ.get("RESEARCH_LAB_ADMIN_TOKEN", "")


def _require_admin(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    """Defence-in-depth admin auth. The Express proxy enforces auth first.

    Even if a caller bypasses the Express proxy and contacts port 8002 directly,
    they cannot access any endpoint without a valid admin token.
    """
    if not _CONFIGURED_TOKEN:
        raise HTTPException(
            503,
            detail="Research Lab not configured (RESEARCH_LAB_ADMIN_TOKEN not set)",
        )
    token = credentials.credentials if credentials else None
    if not token or token != _CONFIGURED_TOKEN:
        raise HTTPException(403, detail="Forbidden. Invalid or missing admin token.")


# Mount the telemetry router — _require_admin is now defined above.
_telemetry_router = make_telemetry_router(_warehouse, _require_admin)
app.include_router(_telemetry_router)

# ── Request Models ─────────────────────────────────────────────────────────────


class CreateDatasetRequest(BaseModel):
    name: str
    version: str
    source: str
    instruments: list[str]
    start_date: str
    end_date: str
    row_count: int = 0
    feature_count: int = 0
    tags: list[str] = []
    notes: str = ""


class CreateGenerationJobRequest(BaseModel):
    name: str
    instruments: list[str]
    start_date: str
    end_date: str
    feature_schema: dict[str, Any] = {}
    label_schema: dict[str, Any] = {}
    requested_by: str = ""
    notes: str = ""


class ValidatingGenerationJobRequest(BaseModel):
    row_count: int = 0
    feature_count: int = 0


class CompleteGenerationJobRequest(BaseModel):
    quality_metrics: dict[str, float] = {}


class FailGenerationJobRequest(BaseModel):
    error: str


class RecordFeatureRequest(BaseModel):
    name: str
    source: str
    importance_score: float
    method: str = ""
    notes: str = ""


class FeatureStatusRequest(BaseModel):
    notes: str = ""


class CreateExperimentRequest(BaseModel):
    name: str
    qde_version: str
    training_dataset: str
    researcher: str = ""
    tags: list[str] = []
    hyperparameters: dict[str, Any] = {}
    notes: str = ""


class CompleteExperimentRequest(BaseModel):
    metrics: dict[str, float]
    notes: str = ""


class FailExperimentRequest(BaseModel):
    error: str


class RunBenchmarkRequest(BaseModel):
    name: str
    qde_version: str
    benchmark_type: str
    parameters: dict[str, Any] = {}
    results: dict[str, Any] = {}


class RegisterBacktestRequest(BaseModel):
    experiment_id: str
    start_date: str
    end_date: str
    instruments: list[str]
    metrics: dict[str, float]
    regime_breakdown: dict[str, Any] = {}
    notes: str = ""


class CreateSearchRequest(BaseModel):
    name: str
    qde_version: str
    parameter_space: dict[str, Any]
    strategy: str = "BAYESIAN"
    max_trials: int = 50
    notes: str = ""


class RecordTrialRequest(BaseModel):
    parameters: dict[str, float]
    metrics: dict[str, float]


class EvaluateGatesRequest(BaseModel):
    baseline_version: str
    min_sharpe: float = 0.5
    max_drawdown_pct: float = 15.0
    min_win_rate: float = 0.45
    min_profit_factor: float = 1.2


class ValidateRetrainingRequest(BaseModel):
    baseline_version: str


class ApproveDeploymentRequest(BaseModel):
    approver: str
    notes: str = ""


class RejectDeploymentRequest(BaseModel):
    rejector: str
    reason: str


class CreateRollbackRequestModel(BaseModel):
    current_version_id: str
    target_version_id: str
    reason: str
    requested_by: str


class ApproveRollbackRequest(BaseModel):
    approver: str
    notes: str = ""


class RejectRollbackRequest(BaseModel):
    rejector: str
    notes: str = ""


class WithdrawRollbackRequest(BaseModel):
    notes: str = ""


class AuthorizeLifecycleActionRequest(BaseModel):
    action: str
    version_id: str
    approver: str
    reason: str = ""


class ConsumeLifecycleAuthorizationRequest(BaseModel):
    version_id: str
    action: str


class RecordFailureRequest(BaseModel):
    qde_version: str
    category: str
    severity: str
    description: str
    affected_instruments: list[str] = []
    loss_pct: float | None = None
    notes: str = ""


class InvestigateFailureRequest(BaseModel):
    root_cause: str
    notes: str = ""


class ResolveFailureRequest(BaseModel):
    resolution: str
    resolver: str = ""


class AnalyzeRegimeRequest(BaseModel):
    qde_version: str
    regime_metrics: dict[str, Any]
    notes: str = ""


class CompareModelsRequest(BaseModel):
    versions: list[str]
    metrics: list[str] = []
    regime_filter: str | None = None
    notes: str = ""


class CreateShadowTestRequest(BaseModel):
    name: str
    challenger_version: str
    champion_version: str
    instruments: list[str] = []
    min_decisions: int = 100
    notes: str = ""


class RecordShadowDecisionRequest(BaseModel):
    instrument: str
    champion_decision: str
    challenger_decision: str
    champion_confidence: float
    challenger_confidence: float
    regime: str | None = None


class ConcludeShadowTestRequest(BaseModel):
    notes: str = ""


class CreateNoteRequest(BaseModel):
    content: str
    author: str = ""
    linked_entity_type: str = ""
    linked_entity_id: str = ""
    tags: list[str] = []


# Artifact registry request models (S20) — must be defined before route declarations
class RegisterArtifactRequest(BaseModel):
    artifact_type: str
    name: str
    path: str
    artifact_hash: str
    training_run_id: str = ""
    experiment_id: str = ""
    model_version: str = ""
    dataset_version: str = ""
    feature_version: str = ""
    code_commit: str = ""
    config_hash: str = ""
    pipeline_job_id: str = ""
    mlflow_run_id: str = ""
    size_bytes: int = 0
    registered_by: str = "admin"
    tags: list[str] = []
    notes: str = ""
    extra: dict[str, Any] = {}
    verify_file: bool = False


class RegisterTrainingRunRequest(BaseModel):
    experiment_id: str
    dataset_version: str
    feature_version: str
    code_commit: str
    config_hash: str
    artifact_hash: str
    requested_by: str = "admin"
    pipeline_job_id: str = ""
    mlflow_run_id: str = ""
    model_version: str = ""
    label_version: str = ""
    notes: str = ""
    metadata: dict[str, Any] = {}


# Pipeline request model (S20) — must be defined before route declarations
class RunPipelineRequest(BaseModel):
    experiment_id: str | None = None
    recommendation_id: str | None = None
    requested_by: str = "admin"
    stages_to_run: list[str] = []
    stage_configs: dict[str, Any] = {}
    notes: str = ""


# Human Approval Gate request models (S20) — must be defined before route declarations
class CreateApprovalRequest(BaseModel):
    gate_type: str
    resource_id: str
    resource_type: str = ""
    requested_by: str = "admin"
    action_summary: str = ""
    requestor_context: dict[str, Any] = {}
    ttl_seconds: float = DEFAULT_TTL_SECONDS


class ApproveApprovalRequest(BaseModel):
    approved_by: str = "admin"
    reason: str = ""


class RejectApprovalRequest(BaseModel):
    rejected_by: str = "admin"
    reason: str


# ── Helper ────────────────────────────────────────────────────────────────────


def _respond(response: Any) -> dict:
    d = response.to_dict()
    if not d.get("success"):
        raise HTTPException(status_code=400, detail=d.get("error", "Unknown error"))
    return d


# ── Health ────────────────────────────────────────────────────────────────────


@app.get("/research/healthz")
def health(_: None = Depends(_require_admin)) -> dict:
    """Admin health check. Requires valid admin token."""
    stats = _db.stats()
    return {"status": "ok", "service": "research-lab", "access": "admin-only", "db": stats}


@app.get("/research/summary")
def system_summary(_: None = Depends(_require_admin)) -> dict:
    return _respond(_api.system_summary())


# ── 1. Dataset Generation ─────────────────────────────────────────────────────


@app.post("/research/datasets")
def create_dataset(req: CreateDatasetRequest, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.create_dataset(**req.model_dump()))


@app.get("/research/datasets")
def list_datasets(_: None = Depends(_require_admin)) -> dict:
    return _respond(_api.list_datasets())


@app.get("/research/datasets/{dataset_id}")
def get_dataset(dataset_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.get_dataset(dataset_id))


@app.post("/research/datasets/{dataset_id}/generate")
def create_generation_job(
    dataset_id: str, req: CreateGenerationJobRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(
        _api.create_generation_job(
            dataset_id=dataset_id,
            **req.model_dump(),
        )
    )


@app.get("/research/datasets/{dataset_id}/generation-jobs")
def list_generation_jobs_for_dataset(dataset_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.list_generation_jobs(dataset_id=dataset_id))


@app.get("/research/generation-jobs")
def list_all_generation_jobs(status: str | None = None, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.list_generation_jobs(status=status))


@app.get("/research/generation-jobs/{job_id}")
def get_generation_job(job_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.get_generation_job(job_id))


@app.post("/research/generation-jobs/{job_id}/start")
def start_generation_job(job_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.start_generation_job(job_id))


@app.post("/research/generation-jobs/{job_id}/validating")
def validating_generation_job(
    job_id: str,
    req: ValidatingGenerationJobRequest,
    _: None = Depends(_require_admin),
) -> dict:
    return _respond(
        _api.validating_generation_job(
            job_id,
            row_count=req.row_count,
            feature_count=req.feature_count,
        )
    )


@app.post("/research/generation-jobs/{job_id}/complete")
def complete_generation_job(
    job_id: str,
    req: CompleteGenerationJobRequest,
    _: None = Depends(_require_admin),
) -> dict:
    return _respond(
        _api.complete_generation_job(
            job_id,
            quality_metrics=req.quality_metrics or None,
        )
    )


@app.post("/research/generation-jobs/{job_id}/fail")
def fail_generation_job(
    job_id: str, req: FailGenerationJobRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(_api.fail_generation_job(job_id, error=req.error))


# ── 2. Feature Discovery ───────────────────────────────────────────────────────


@app.post("/research/features")
def record_feature(req: RecordFeatureRequest, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.record_feature(**req.model_dump()))


@app.post("/research/features/{feature_id}/accept")
def accept_feature(
    feature_id: str, req: FeatureStatusRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(_api.accept_feature(feature_id, notes=req.notes))


@app.post("/research/features/{feature_id}/reject")
def reject_feature(
    feature_id: str, req: FeatureStatusRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(_api.reject_feature(feature_id, notes=req.notes))


@app.get("/research/features")
def list_features(status: str | None = None, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.list_features(status=status))


# ── 3. Experiment Tracking ─────────────────────────────────────────────────────


@app.post("/research/experiments")
def create_experiment(req: CreateExperimentRequest, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.create_experiment(**req.model_dump()))


@app.post("/research/experiments/{experiment_id}/start")
def start_experiment(experiment_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.start_experiment(experiment_id))


@app.post("/research/experiments/{experiment_id}/complete")
def complete_experiment(
    experiment_id: str, req: CompleteExperimentRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(_api.complete_experiment(experiment_id, metrics=req.metrics, notes=req.notes))


@app.post("/research/experiments/{experiment_id}/fail")
def fail_experiment(
    experiment_id: str, req: FailExperimentRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(_api.fail_experiment(experiment_id, error=req.error))


@app.get("/research/experiments/summary")
def experiment_summary(_: None = Depends(_require_admin)) -> dict:
    return _respond(_api.experiment_summary())


@app.get("/research/experiments/{experiment_id}")
def get_experiment(experiment_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.get_experiment(experiment_id))


@app.get("/research/experiments")
def list_experiments(
    qde_version: str | None = None,
    status: str | None = None,
    researcher: str | None = None,
    _: None = Depends(_require_admin),
) -> dict:
    return _respond(
        _api.list_experiments(qde_version=qde_version, status=status, researcher=researcher)
    )


# ── 4. Backtesting ────────────────────────────────────────────────────────────


@app.post("/research/benchmarks")
def run_benchmark(req: RunBenchmarkRequest, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.run_benchmark(**req.model_dump()))


@app.get("/research/benchmarks/{benchmark_id}")
def get_benchmark(benchmark_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.get_benchmark(benchmark_id))


@app.get("/research/benchmarks")
def list_benchmarks(_: None = Depends(_require_admin)) -> dict:
    return _respond(_api.list_benchmarks())


@app.post("/research/backtests")
def register_backtest(req: RegisterBacktestRequest, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.register_backtest(**req.model_dump()))


@app.get("/research/backtests")
def list_backtests(experiment_id: str | None = None, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.list_backtests(experiment_id=experiment_id))


# ── 5. Validation (Deployment Gates) ──────────────────────────────────────────


@app.post("/research/experiments/{experiment_id}/gates")
def evaluate_gates(
    experiment_id: str, req: EvaluateGatesRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(
        _api.evaluate_deployment_gates(
            experiment_id=experiment_id,
            baseline_version=req.baseline_version,
            min_sharpe=req.min_sharpe,
            max_drawdown_pct=req.max_drawdown_pct,
            min_win_rate=req.min_win_rate,
            min_profit_factor=req.min_profit_factor,
        )
    )


@app.post("/research/experiments/{experiment_id}/validate-retraining")
def validate_retraining(
    experiment_id: str, req: ValidateRetrainingRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(
        _api.validate_retraining(
            experiment_id=experiment_id,
            baseline_version=req.baseline_version,
        )
    )


# ── 6. Promotion Approval ─────────────────────────────────────────────────────


@app.post("/research/experiments/{experiment_id}/approve")
def approve_deployment(
    experiment_id: str, req: ApproveDeploymentRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(
        _api.approve_deployment(
            experiment_id=experiment_id,
            approver=req.approver,
            notes=req.notes,
        )
    )


@app.post("/research/experiments/{experiment_id}/reject")
def reject_deployment(
    experiment_id: str, req: RejectDeploymentRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(
        _api.reject_deployment(
            experiment_id=experiment_id,
            rejector=req.rejector,
            reason=req.reason,
        )
    )


@app.get("/research/deployments")
def deployment_history(_: None = Depends(_require_admin)) -> dict:
    return _respond(_api.deployment_history())


# ── 7. Rollback Approval ──────────────────────────────────────────────────────


@app.post("/research/rollbacks")
def create_rollback_request(
    req: CreateRollbackRequestModel, _: None = Depends(_require_admin)
) -> dict:
    return _respond(_api.create_rollback_request(**req.model_dump()))


@app.get("/research/rollbacks")
def list_rollback_requests(status: str | None = None, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.list_rollback_requests(status=status))


@app.get("/research/rollbacks/{request_id}")
def get_rollback_request(request_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.get_rollback_request(request_id))


@app.post("/research/rollbacks/{request_id}/approve")
def approve_rollback(
    request_id: str, req: ApproveRollbackRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(_api.approve_rollback(request_id, approver=req.approver, notes=req.notes))


@app.post("/research/rollbacks/{request_id}/reject")
def reject_rollback(
    request_id: str, req: RejectRollbackRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(_api.reject_rollback(request_id, rejector=req.rejector, notes=req.notes))


@app.post("/research/rollbacks/{request_id}/withdraw")
def withdraw_rollback(
    request_id: str, req: WithdrawRollbackRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(_api.withdraw_rollback(request_id, notes=req.notes))


# ── 7b. Lifecycle Authorizations ───────────────────────────────────────────────
#
# Research Lab is the sole authority for PROMOTE / ROLLBACK / ARCHIVE / RETIRE.
# PROMOTE and ROLLBACK authorizations are issued as a side effect of
# approve_deployment / approve_rollback above. ARCHIVE and RETIRE are
# authorized directly via /research/lifecycle/authorize. The gateway
# (apps/trainer-api) consumes an authorization immediately before executing
# the corresponding registry mutation via /research/lifecycle/consume.


@app.post("/research/lifecycle/authorize")
def authorize_lifecycle_action(
    req: AuthorizeLifecycleActionRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(
        _api.authorize_lifecycle_action(
            action=req.action,
            version_id=req.version_id,
            approver=req.approver,
            reason=req.reason,
        )
    )


@app.post("/research/lifecycle/consume")
def consume_lifecycle_authorization(
    req: ConsumeLifecycleAuthorizationRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(
        _api.consume_lifecycle_authorization(version_id=req.version_id, action=req.action)
    )


@app.get("/research/lifecycle/authorizations")
def list_lifecycle_authorizations(
    version_id: str | None = None,
    action: str | None = None,
    status: str | None = None,
    _: None = Depends(_require_admin),
) -> dict:
    return _respond(
        _api.list_lifecycle_authorizations(version_id=version_id, action=action, status=status)
    )


@app.get("/research/lifecycle/authorizations/{authorization_id}")
def get_lifecycle_authorization(authorization_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.get_lifecycle_authorization(authorization_id))


# ── 8. Failure Analysis ───────────────────────────────────────────────────────


@app.post("/research/failures")
def record_failure(req: RecordFailureRequest, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.record_failure(**req.model_dump()))


@app.post("/research/failures/{failure_id}/investigate")
def investigate_failure(
    failure_id: str, req: InvestigateFailureRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(
        _api.investigate_failure(failure_id, root_cause=req.root_cause, notes=req.notes)
    )


@app.post("/research/failures/{failure_id}/resolve")
def resolve_failure(
    failure_id: str, req: ResolveFailureRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(
        _api.resolve_failure(failure_id, resolution=req.resolution, resolver=req.resolver)
    )


@app.get("/research/failures/summary")
def failure_summary(_: None = Depends(_require_admin)) -> dict:
    return _respond(_api.failure_summary())


@app.get("/research/failures")
def list_failures(
    category: str | None = None,
    severity: str | None = None,
    qde_version: str | None = None,
    status: str | None = None,
    _: None = Depends(_require_admin),
) -> dict:
    return _respond(
        _api.list_failures(
            category=category, severity=severity, qde_version=qde_version, status=status
        )
    )


# ── Market Regime Analysis ────────────────────────────────────────────────────


@app.post("/research/regime")
def analyze_regime(req: AnalyzeRegimeRequest, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.analyze_regime(**req.model_dump()))


@app.get("/research/regime/{report_id}")
def get_regime_report(report_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.get_regime_report(report_id))


@app.get("/research/regime")
def list_regime_reports(qde_version: str | None = None, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.list_regime_reports(qde_version=qde_version))


# ── Version Comparisons ───────────────────────────────────────────────────────


@app.post("/research/comparisons")
def compare_models(req: CompareModelsRequest, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.compare_models(**req.model_dump()))


@app.get("/research/comparisons/{comparison_id}")
def get_comparison(comparison_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.get_comparison(comparison_id))


@app.get("/research/comparisons")
def list_comparisons(_: None = Depends(_require_admin)) -> dict:
    return _respond(_api.list_comparisons())


# ── Parameter Search ──────────────────────────────────────────────────────────


@app.post("/research/searches")
def create_search(req: CreateSearchRequest, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.create_search(**req.model_dump()))


@app.post("/research/searches/{search_id}/trials")
def record_trial(
    search_id: str, req: RecordTrialRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(_api.record_trial(search_id, parameters=req.parameters, metrics=req.metrics))


@app.post("/research/searches/{search_id}/complete")
def complete_search(search_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.complete_search(search_id))


@app.get("/research/searches/{search_id}")
def get_search(search_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.get_search(search_id))


@app.get("/research/searches")
def list_searches(qde_version: str | None = None, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.list_searches(qde_version=qde_version))


# ── Shadow Testing ────────────────────────────────────────────────────────────


@app.post("/research/shadow-tests")
def create_shadow_test(req: CreateShadowTestRequest, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.create_shadow_test(**req.model_dump()))


@app.post("/research/shadow-tests/{test_id}/decisions")
def record_shadow_decision(
    test_id: str, req: RecordShadowDecisionRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(_api.record_shadow_decision(test_id, **req.model_dump()))


@app.post("/research/shadow-tests/{test_id}/conclude")
def conclude_shadow_test(
    test_id: str, req: ConcludeShadowTestRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(_api.conclude_shadow_test(test_id, notes=req.notes))


@app.get("/research/shadow-tests/{test_id}")
def get_shadow_test(test_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.get_shadow_test(test_id))


@app.get("/research/shadow-tests")
def list_shadow_tests(
    challenger_version: str | None = None,
    status: str | None = None,
    _: None = Depends(_require_admin),
) -> dict:
    return _respond(_api.list_shadow_tests(challenger_version=challenger_version, status=status))


# ── Research Notes ────────────────────────────────────────────────────────────


@app.post("/research/notes")
def create_note(req: CreateNoteRequest, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.create_note(**req.model_dump()))


@app.get("/research/notes")
def list_notes(
    author: str | None = None,
    linked_entity_type: str | None = None,
    linked_entity_id: str | None = None,
    _: None = Depends(_require_admin),
) -> dict:
    return _respond(
        _api.list_notes(
            author=author,
            linked_entity_type=linked_entity_type,
            linked_entity_id=linked_entity_id,
        )
    )


@app.get("/research/notes/{note_id}")
def get_note(note_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.get_note(note_id))


# ── Failure Investigation Engine (S20) ────────────────────────────────────────


class InvestigateRequest(BaseModel):
    model_version: str | None = None
    symbol: str | None = None
    since: float | None = None
    until: float | None = None
    failure_kinds: list[str] | None = None
    limit_per_stream: int = 2000


@app.post("/research/investigate")
def run_investigation(req: InvestigateRequest, _: None = Depends(_require_admin)) -> dict:
    """Run the Failure Investigation Engine against the telemetry warehouse.

    Returns all FailureReport objects detected, sorted by severity_score descending.
    Reports are also persisted to the database for later retrieval.
    """
    from trainer.research_lab.investigation_models import InvestigationFailureKind

    try:
        kinds = None
        if req.failure_kinds:
            kinds = [InvestigationFailureKind(k) for k in req.failure_kinds]
        query = InvestigationQuery(
            model_version=req.model_version,
            symbol=req.symbol,
            since=req.since,
            until=req.until,
            failure_kinds=kinds,
            limit_per_stream=req.limit_per_stream,
        )
        reports = _engine.investigate(query)
        return {
            "success": True,
            "report_count": len(reports),
            "reports": [r.to_dict() for r in reports],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/research/reports")
def list_failure_reports(
    failure_kind: str | None = None,
    model_version: str | None = None,
    min_severity: float = 0.0,
    since: float | None = None,
    until: float | None = None,
    limit: int = 100,
    _: None = Depends(_require_admin),
) -> dict:
    """Read persisted FailureReport records. Sorted by severity_score descending."""
    rows = _engine.get_reports(
        failure_kind=failure_kind,
        model_version=model_version,
        min_severity=min_severity,
        since=since,
        until=until,
        limit=limit,
    )
    return {"success": True, "report_count": len(rows), "reports": rows}


@app.get("/research/reports/{report_id}")
def get_failure_report(report_id: str, _: None = Depends(_require_admin)) -> dict:
    """Fetch a single FailureReport by ID."""
    row = _db.get_failure_report(report_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Report {report_id!r} not found")
    return {"success": True, "report": row}


# ── Auto Discovery (S20) ──────────────────────────────────────────────────────


class DiscoverRequest(BaseModel):
    model_version: str | None = None
    since: float | None = None
    until: float | None = None
    symbol: str | None = None
    min_failure_severity: float = 0.3
    max_proposed_per_run: int = 5
    min_feature_samples: int = 5
    min_label_samples: int = 3
    limit_per_stream: int = 5000


@app.post("/research/discover")
def run_discovery(req: DiscoverRequest, _: None = Depends(_require_admin)) -> dict:
    """Trigger all three S20 discovery pipelines in sequence:
      1. ExperimentCreator — proposes experiments from failure reports
      2. FeatureDiscoveryPipeline — discovers candidate feature signals
      3. LabelDiscoveryEngine — generates hindsight label datasets

    All results are persisted. Returns a DiscoverySummary.
    """
    try:
        dq = DiscoveryQuery(
            model_version=req.model_version,
            since=req.since,
            until=req.until,
            symbol=req.symbol,
            min_failure_severity=req.min_failure_severity,
            max_proposed_per_run=req.max_proposed_per_run,
            min_feature_samples=req.min_feature_samples,
            min_label_samples=req.min_label_samples,
            limit_per_stream=req.limit_per_stream,
        )
        summary = DiscoverySummary()
        _creator.propose(dq, summary)
        _feature_pipeline.discover(dq, summary)
        _label_engine.discover(dq, summary)
        return {"success": True, "summary": summary.to_dict()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/research/proposed-experiments")
def list_proposed_experiments(_: None = Depends(_require_admin)) -> dict:
    """Return all auto-proposed experiments awaiting admin approval."""
    rows = _creator.list_proposed()
    return {"success": True, "total": len(rows), "experiments": rows}


@app.post("/research/proposed-experiments/{experiment_id}/approve")
def approve_proposed_experiment(
    experiment_id: str, _: None = Depends(_require_admin)
) -> dict:
    """Advance a PROPOSED experiment to PENDING (admin approval)."""
    try:
        _creator.approve(experiment_id)
        return {"success": True, "experiment_id": experiment_id, "new_status": "PENDING"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/research/proposed-experiments/{experiment_id}/reject")
def reject_proposed_experiment(
    experiment_id: str, _: None = Depends(_require_admin)
) -> dict:
    """Reject a PROPOSED experiment (admin decision)."""
    try:
        _creator.reject(experiment_id)
        return {"success": True, "experiment_id": experiment_id, "new_status": "ARCHIVED"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/research/label-datasets")
def list_label_datasets(
    label_type: str | None = None,
    model_version: str | None = None,
    limit: int = 50,
    _: None = Depends(_require_admin),
) -> dict:
    """Return persisted LabelDataset records sorted newest first."""
    rows = _label_engine.list_datasets(
        label_type=label_type, model_version=model_version, limit=limit
    )
    return {"success": True, "total": len(rows), "datasets": rows}


# ── Artifact Registry (S20) ───────────────────────────────────────────────────


@app.post("/artifacts/register")
def register_artifact(
    req: RegisterArtifactRequest, _: None = Depends(_require_admin)
) -> dict:
    """Manually register an artifact produced by a training pipeline run.

    The registry is **metadata-only**: artifacts are stored by the training
    infrastructure; this endpoint records provenance so every artifact can be
    traced back to the dataset, feature pipeline, code commit, and config that
    produced it.

    Raises 400 if:
    - ``artifact_type`` is not one of the six canonical types
    - ``artifact_hash`` is not a 64-character SHA-256 hex string
    - A duplicate ``(artifact_type, artifact_hash)`` is already registered
    - ``verify_file=true`` and the file's computed hash does not match
    """
    try:
        record = ArtifactRecord.new(
            artifact_type=req.artifact_type,
            name=req.name,
            path=req.path,
            artifact_hash=req.artifact_hash,
            training_run_id=req.training_run_id,
            experiment_id=req.experiment_id,
            model_version=req.model_version,
            dataset_version=req.dataset_version,
            feature_version=req.feature_version,
            code_commit=req.code_commit,
            config_hash=req.config_hash,
            pipeline_job_id=req.pipeline_job_id,
            mlflow_run_id=req.mlflow_run_id,
            size_bytes=req.size_bytes,
            registered_by=req.registered_by,
            tags=req.tags,
            notes=req.notes,
            extra=req.extra,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        _artifact_registry.register_artifact(record, verify_file=req.verify_file)
    except (DuplicateArtifactError, ArtifactHashMismatchError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {"success": True, "artifact": record.to_dict()}


@app.get("/artifacts")
def list_artifacts(
    artifact_type: str | None = None,
    model_version: str | None = None,
    training_run_id: str | None = None,
    experiment_id: str | None = None,
    limit: int = 50,
    _: None = Depends(_require_admin),
) -> dict:
    """List registered artifacts with optional filtering.

    Filter by ``artifact_type`` (WEIGHTS | CONFIG | CALIBRATION | ONNX |
    TORCHSCRIPT | DEPLOYMENT_MANIFEST), ``model_version``, ``training_run_id``,
    or ``experiment_id``.  Results are ordered newest first.
    """
    rows = _artifact_registry.list_artifacts(
        artifact_type=artifact_type,
        model_version=model_version,
        training_run_id=training_run_id,
        experiment_id=experiment_id,
        limit=limit,
    )
    return {"success": True, "total": len(rows), "artifacts": rows}


@app.get("/artifacts/types")
def list_artifact_types(_: None = Depends(_require_admin)) -> dict:
    """Return the list of valid artifact types."""
    return {"success": True, "artifact_types": ARTIFACT_TYPES}


@app.post("/artifacts/runs")
def register_training_run(
    req: RegisterTrainingRunRequest, _: None = Depends(_require_admin)
) -> dict:
    """Register a training run provenance record.

    A training run record ties together every ingredient of a training job —
    dataset version, feature version, git commit, config hash, and primary
    weights hash — making it fully reproducible.
    """
    try:
        record = TrainingRunRecord.new(
            experiment_id=req.experiment_id,
            dataset_version=req.dataset_version,
            feature_version=req.feature_version,
            code_commit=req.code_commit,
            config_hash=req.config_hash,
            artifact_hash=req.artifact_hash,
            requested_by=req.requested_by,
            pipeline_job_id=req.pipeline_job_id,
            mlflow_run_id=req.mlflow_run_id,
            model_version=req.model_version,
            label_version=req.label_version,
            notes=req.notes,
            metadata=req.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    _artifact_registry.register_training_run(record)
    return {"success": True, "training_run": record.to_dict()}


@app.get("/artifacts/runs")
def list_training_runs(
    experiment_id: str | None = None,
    pipeline_job_id: str | None = None,
    limit: int = 50,
    _: None = Depends(_require_admin),
) -> dict:
    """List training run provenance records, newest first."""
    rows = _artifact_registry.list_training_runs(
        experiment_id=experiment_id,
        pipeline_job_id=pipeline_job_id,
        limit=limit,
    )
    return {"success": True, "total": len(rows), "training_runs": rows}


@app.get("/artifacts/runs/{run_id}")
def get_training_run(run_id: str, _: None = Depends(_require_admin)) -> dict:
    """Return a specific training run by ID."""
    run = _artifact_registry.get_training_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Training run {run_id!r} not found")
    # Include all artifacts produced by this run
    arts = _artifact_registry.artifacts_for_run(run_id)
    return {"success": True, "training_run": run.to_dict(), "artifacts": arts}


@app.get("/artifacts/{artifact_id}")
def get_artifact(artifact_id: str, _: None = Depends(_require_admin)) -> dict:
    """Return full provenance for a specific artifact, including its training run."""
    prov = _artifact_registry.provenance(artifact_id)
    if prov is None:
        raise HTTPException(status_code=404, detail=f"Artifact {artifact_id!r} not found")
    return {"success": True, **prov}


# ── Model Health Monitoring (S20) ─────────────────────────────────────────────


@app.post("/health/check")
def run_health_check(
    model_version: str | None = None,
    since: float | None = None,
    _: None = Depends(_require_admin),
) -> dict:
    """Trigger an on-demand health check across all five drift dimensions.

    Evaluates:
    - **Confidence drift** — mean output confidence shift baseline vs recent
    - **Feature drift** — output probability distribution (TV distance)
    - **Market drift** — regime distribution shift (TV distance)
    - **Data quality** — telemetry staleness and event-rate drop
    - **Regime drift** — per-regime win-rate degradation

    CRITICAL alerts automatically create a ``RetrainingRecommendation``
    (deduplicated against open recommendations for the same dimension).

    Query params:
        model_version: Restrict to a specific model version. ``None`` = all data.
        since: Unix timestamp; restrict telemetry to events after this time.
    """
    result = _health_monitor.run_check(model_version=model_version, since=since)
    return {"success": True, "health_check": result.to_dict()}


@app.get("/health/status")
def get_health_status(
    model_version: str | None = None,
    _: None = Depends(_require_admin),
) -> dict:
    """Return a current health summary without re-running checks.

    Aggregates persisted alerts by dimension and severity.
    """
    status = _health_monitor.get_status(model_version=model_version)
    return {"success": True, "status": status}


@app.get("/health/alerts")
def list_health_alerts(
    model_version: str | None = None,
    severity: str | None = None,
    dimension: str | None = None,
    limit: int = 50,
    _: None = Depends(_require_admin),
) -> dict:
    """List persisted drift alerts, newest first.

    Filter by ``model_version``, ``severity`` (WARNING | CRITICAL), or
    ``dimension`` (FEATURE | CONFIDENCE | MARKET | DATA_QUALITY | REGIME).
    """
    alerts = _health_monitor.get_alerts(
        model_version=model_version,
        severity=severity,
        dimension=dimension,
        limit=limit,
    )
    return {"success": True, "total": len(alerts), "alerts": alerts}


# ── Training Pipeline (S20) ───────────────────────────────────────────────────


@app.post("/pipeline/run")
def run_pipeline(
    req: RunPipelineRequest, _: None = Depends(_require_admin)
) -> dict:
    """Enqueue and run a training pipeline job.

    Accepts an experiment_id, recommendation_id, or both.  Stages run in the
    canonical order: DATASET_GENERATION → FEATURE_GENERATION → TRAINING →
    VALIDATION → BACKTEST → CERTIFICATION → SHADOW_TEST.  Pass
    ``stages_to_run`` to restrict execution to a subset (in canonical order).

    The TRAINING stage requires an approved TRAINING_START gate for the job_id
    via ``POST /approvals``.  Without it the job is returned with status FAILED
    and the TRAINING stage result shows status BLOCKED.
    """
    try:
        # Empty stages_to_run is valid: the controller interprets it as "all stages".
        # PipelineJobSpec validates the presence of experiment_id/recommendation_id.
        spec = PipelineJobSpec(
            requested_by=req.requested_by,
            experiment_id=req.experiment_id,
            recommendation_id=req.recommendation_id,
            stages_to_run=req.stages_to_run,  # [] → all stages (controller default)
            stage_configs=req.stage_configs,
            notes=req.notes,
        )
        # Validate stage names early so the caller gets a 400 before the job is
        # created, rather than a silent no-op or a partially-run job.
        from trainer.research_lab.pipeline_controller import ResearchLabPipelineController as _PCtrl
        _PCtrl._resolve_stages(req.stages_to_run)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    job = _pipeline_ctrl.run_pipeline(spec)
    return {"success": True, "job": job.to_dict()}


@app.get("/pipeline/jobs")
def list_pipeline_jobs(
    status: str | None = None,
    experiment_id: str | None = None,
    recommendation_id: str | None = None,
    limit: int = 50,
    _: None = Depends(_require_admin),
) -> dict:
    """List pipeline jobs with optional filtering.

    Filter by ``status`` (QUEUED | RUNNING | COMPLETED | FAILED | CANCELLED),
    ``experiment_id``, or ``recommendation_id``.  Results are ordered by
    creation time descending.
    """
    rows = _pipeline_ctrl.list_jobs(
        status=status,
        experiment_id=experiment_id,
        recommendation_id=recommendation_id,
        limit=limit,
    )
    return {"success": True, "total": len(rows), "jobs": rows}


@app.get("/pipeline/jobs/{job_id}")
def get_pipeline_job(
    job_id: str, _: None = Depends(_require_admin)
) -> dict:
    """Fetch a specific pipeline job by ID, including full stage results and events."""
    job = _pipeline_ctrl.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Pipeline job {job_id!r} not found")
    return {"success": True, "job": job.to_dict()}


# ── Human Approval Gates (S20) ────────────────────────────────────────────────


@app.post("/approvals")
def create_approval_request(
    req: CreateApprovalRequest, _: None = Depends(_require_admin)
) -> dict:
    """Create an approval request for a gated Research Lab action.

    Idempotent: a second call for the same gate_type + resource_id while a
    PENDING request is still open returns the existing request.
    """
    try:
        gate_type = ApprovalGateType(req.gate_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown gate type: {req.gate_type!r}")
    approval = _approval_gate.create_request(
        gate_type=gate_type,
        resource_id=req.resource_id,
        resource_type=req.resource_type,
        requested_by=req.requested_by,
        action_summary=req.action_summary,
        requestor_context=req.requestor_context,
        ttl_seconds=req.ttl_seconds,
    )
    return {"success": True, "approval_request": approval.to_dict()}


@app.get("/approvals")
def list_approval_requests(
    gate_type: str | None = None,
    resource_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
    _: None = Depends(_require_admin),
) -> dict:
    """List approval requests. Stale PENDING requests are auto-expired before returning.

    Filter by gate_type (EXPERIMENT_LAUNCH | TRAINING_START | PROMOTION | DEPLOYMENT | ROLLBACK),
    resource_id, and/or status (PENDING | APPROVED | REJECTED | EXPIRED | CONSUMED).
    """
    rows = _approval_gate.list_requests(
        gate_type=gate_type, resource_id=resource_id, status=status, limit=limit
    )
    return {"success": True, "total": len(rows), "approval_requests": rows}


@app.get("/approvals/{request_id}")
def get_approval_request(
    request_id: str, _: None = Depends(_require_admin)
) -> dict:
    """Fetch a specific approval request by ID."""
    req = _approval_gate.get_request(request_id)
    if req is None:
        raise HTTPException(status_code=404, detail=f"Approval request {request_id!r} not found")
    return {"success": True, "approval_request": req.to_dict()}


@app.post("/approvals/{request_id}/approve")
def approve_approval_request(
    request_id: str,
    req: ApproveApprovalRequest,
    _: None = Depends(_require_admin),
) -> dict:
    """Admin approves a PENDING approval request (PENDING → APPROVED).

    The approved request is available for atomic consumption by the gated action.
    """
    try:
        approved = _approval_gate.approve(
            request_id, approved_by=req.approved_by, reason=req.reason
        )
        return {"success": True, "approval_request": approved.to_dict()}
    except (ValueError, ApprovalAlreadyResolved) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/approvals/{request_id}/reject")
def reject_approval_request(
    request_id: str,
    req: RejectApprovalRequest,
    _: None = Depends(_require_admin),
) -> dict:
    """Admin rejects a PENDING approval request (PENDING → REJECTED).

    The gated action will be blocked when it next calls require_approval().
    """
    try:
        rejected = _approval_gate.reject(
            request_id, rejected_by=req.rejected_by, reason=req.reason
        )
        return {"success": True, "approval_request": rejected.to_dict()}
    except (ValueError, ApprovalAlreadyResolved) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ── Retraining Recommendation Engine (S20) ────────────────────────────────────


class EvaluateRequest(BaseModel):
    model_version: str | None = None
    since: float | None = None
    until: float | None = None


class ResolveRequest(BaseModel):
    resolved_by: str = "admin"
    notes: str = ""


@app.post("/research/evaluate")
def run_evaluation(req: EvaluateRequest, _: None = Depends(_require_admin)) -> dict:
    """Run all five evaluation dimensions against the current telemetry.

    For any dimension that crosses its threshold, a RetrainingRecommendation is
    persisted in PROPOSED status and returned in the response.
    Idempotent: subsequent calls deduplicate open recommendations.
    """
    try:
        results = _rec_engine.evaluate(
            model_version=req.model_version,
            since=req.since,
            until=req.until,
        )
        return {
            "success": True,
            "n_evaluated": len(results),
            "n_triggered": sum(1 for r in results if r.triggered),
            "results": [r.to_dict() for r in results],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/research/recommendations")
def list_recommendations(
    status: str | None = None,
    model_version: str | None = None,
    limit: int = 50,
    _: None = Depends(_require_admin),
) -> dict:
    """List retraining recommendations. Optionally filter by status
    (PROPOSED | APPROVED | DISMISSED) and model_version."""
    rows = _rec_engine.list_recommendations(
        status=status, model_version=model_version, limit=limit
    )
    return {"success": True, "total": len(rows), "recommendations": rows}


@app.get("/research/recommendations/{recommendation_id}")
def get_recommendation(
    recommendation_id: str, _: None = Depends(_require_admin)
) -> dict:
    """Fetch a single RetrainingRecommendation by ID."""
    rec = _rec_engine.get_recommendation(recommendation_id)
    if rec is None:
        raise HTTPException(
            status_code=404, detail=f"Recommendation {recommendation_id!r} not found"
        )
    return {"success": True, "recommendation": rec}


@app.post("/research/recommendations/{recommendation_id}/approve")
def approve_recommendation(
    recommendation_id: str,
    req: ResolveRequest,
    _: None = Depends(_require_admin),
) -> dict:
    """Approve a PROPOSED recommendation (admin action).

    Advances status to APPROVED. This does not start retraining automatically —
    that is handled by the Training Pipeline Integration.
    """
    try:
        rec = _rec_engine.approve(recommendation_id, resolved_by=req.resolved_by)
        return {"success": True, "recommendation": rec}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/research/recommendations/{recommendation_id}/dismiss")
def dismiss_recommendation(
    recommendation_id: str,
    req: ResolveRequest,
    _: None = Depends(_require_admin),
) -> dict:
    """Dismiss a PROPOSED recommendation (admin decision).

    Advances status to DISMISSED. Records the resolver and optional notes.
    """
    try:
        rec = _rec_engine.dismiss(
            recommendation_id, resolved_by=req.resolved_by, notes=req.notes
        )
        return {"success": True, "recommendation": rec}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
