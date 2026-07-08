"""Phase 28.4 — Research Lab HTTP Server.

Admin-only FastAPI server. Every endpoint is protected by a static bearer token.
The server has no public documentation UI, no API browser, and no frontend.
It is consumed exclusively by the Signalium Admin Portal via the Express proxy.

Start with:
    RESEARCH_LAB_ADMIN_TOKEN=<secret> uvicorn trainer.research_lab.research_lab_server:app --port 8002

Environment variables:
    RESEARCH_LAB_ADMIN_TOKEN  Required. Shared secret for all admin requests.
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

from trainer.research_lab.research_lab_api import ResearchLabAPI

# ── Setup ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Research Lab API",
    description=(
        "Administrator-only QDE research and improvement subsystem. "
        "No public access. No frontend. Internal use only."
    ),
    version="28.4.0",
    docs_url=None,  # No public API browser — admin subsystem has no UI
    redoc_url=None,  # No public documentation UI
    openapi_url=None,  # No public schema endpoint
)

# No CORS middleware — this server is internal. Browsers do not contact it
# directly. All external traffic arrives via the Express proxy on port 8080.

_api = ResearchLabAPI()
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


class RecordFeatureRequest(BaseModel):
    name: str
    category: str
    importance_score: float
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
    version_ids: list[str]
    metrics_per_version: dict[str, dict[str, float]]
    notes: str = ""


class RegisterBacktestRequest(BaseModel):
    experiment_id: str
    dataset_id: str
    metrics: dict[str, float]
    notes: str = ""


class CreateSearchRequest(BaseModel):
    name: str
    strategy: str
    search_space: dict[str, Any]
    objective: str
    maximize: bool = True
    n_trials: int = 20
    qde_version: str = ""
    notes: str = ""


class RecordTrialRequest(BaseModel):
    params: dict[str, Any]
    score: float
    metadata: dict[str, Any] = {}


class EvaluateGatesRequest(BaseModel):
    baseline_experiment_id: str | None = None
    max_drawdown_tolerance: float = 0.02


class ApproveDeploymentRequest(BaseModel):
    gate_id: str
    approver: str
    notes: str = ""


class RejectDeploymentRequest(BaseModel):
    rejector: str
    reason: str


class RecordFailureRequest(BaseModel):
    category: str
    severity: str
    qde_version: str
    description: str
    instrument: str = ""
    regime: str = ""
    loss_amount: float | None = None
    drawdown_magnitude: float | None = None
    tags: list[str] = []
    metadata: dict[str, Any] = {}


class InvestigateFailureRequest(BaseModel):
    root_cause: str
    notes: str = ""


class ResolveFailureRequest(BaseModel):
    resolution: str
    resolver: str = ""


class AnalyzeRegimeRequest(BaseModel):
    qde_version: str
    regime_results: dict[str, dict[str, float]]
    notes: str = ""


class CompareModelsRequest(BaseModel):
    version_ids: list[str]
    metrics_per_version: dict[str, dict[str, float]]
    notes: str = ""


# ── Shadow Testing Request Models ──────────────────────────────────────────────


class CreateShadowTestRequest(BaseModel):
    name: str
    challenger_version: str
    champion_version: str
    instruments: list[str]
    min_decisions: int = 500
    notes: str = ""


class RecordShadowDecisionRequest(BaseModel):
    instrument: str
    champion_decision_type: str
    challenger_decision_type: str
    champion_confidence: float
    challenger_confidence: float
    regime: str = ""
    market_context: dict[str, Any] = {}


class ConcludeShadowTestRequest(BaseModel):
    notes: str = ""


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
    return {"status": "ok", "service": "research-lab", "access": "admin-only"}


@app.get("/research/summary")
def system_summary(_: None = Depends(_require_admin)) -> dict:
    return _respond(_api.system_summary())


# ── Datasets ──────────────────────────────────────────────────────────────────


@app.post("/research/datasets")
def create_dataset(req: CreateDatasetRequest, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.create_dataset(**req.model_dump()))


@app.get("/research/datasets")
def list_datasets(_: None = Depends(_require_admin)) -> dict:
    return _respond(_api.list_datasets())


@app.get("/research/datasets/{dataset_id}")
def get_dataset(dataset_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.get_dataset(dataset_id))


# ── Features ──────────────────────────────────────────────────────────────────


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


# ── Experiments ───────────────────────────────────────────────────────────────


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


@app.get("/research/experiments/summary")
def experiment_summary(_: None = Depends(_require_admin)) -> dict:
    return _respond(_api.experiment_summary())


# ── Benchmarking ──────────────────────────────────────────────────────────────


@app.post("/research/benchmarks")
def run_benchmark(req: RunBenchmarkRequest, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.run_benchmark(**req.model_dump()))


@app.get("/research/benchmarks/{benchmark_id}")
def get_benchmark(benchmark_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.get_benchmark(benchmark_id))


@app.get("/research/benchmarks")
def list_benchmarks(_: None = Depends(_require_admin)) -> dict:
    return _respond(_api.list_benchmarks())


# ── Backtests ─────────────────────────────────────────────────────────────────


@app.post("/research/backtests")
def register_backtest(req: RegisterBacktestRequest, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.register_backtest(**req.model_dump()))


@app.get("/research/backtests")
def list_backtests(experiment_id: str | None = None, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.list_backtests(experiment_id=experiment_id))


# ── Parameter Search ──────────────────────────────────────────────────────────


@app.post("/research/searches")
def create_search(req: CreateSearchRequest, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.create_search(**req.model_dump()))


@app.post("/research/searches/{search_id}/trials")
def record_trial(
    search_id: str, req: RecordTrialRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(
        _api.record_trial(search_id, params=req.params, score=req.score, metadata=req.metadata)
    )


@app.post("/research/searches/{search_id}/complete")
def complete_search(search_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.complete_search(search_id))


@app.get("/research/searches/{search_id}")
def get_search(search_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.get_search(search_id))


@app.get("/research/searches")
def list_searches(qde_version: str | None = None, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.list_searches(qde_version=qde_version))


# ── Deployment Gates / Validation ─────────────────────────────────────────────


@app.post("/research/experiments/{experiment_id}/gates")
def evaluate_gates(
    experiment_id: str, req: EvaluateGatesRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(
        _api.evaluate_deployment_gates(
            experiment_id=experiment_id,
            baseline_experiment_id=req.baseline_experiment_id,
            max_drawdown_tolerance=req.max_drawdown_tolerance,
        )
    )


@app.post("/research/experiments/{experiment_id}/validate-retraining")
def validate_retraining(
    experiment_id: str, req: EvaluateGatesRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(
        _api.validate_retraining(
            experiment_id=experiment_id,
            incumbent_experiment_id=req.baseline_experiment_id or "",
        )
    )


# ── Deployment Approval ───────────────────────────────────────────────────────


@app.post("/research/experiments/{experiment_id}/approve")
def approve_deployment(
    experiment_id: str, req: ApproveDeploymentRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(
        _api.approve_deployment(
            experiment_id=experiment_id,
            gate_id=req.gate_id,
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


# ── Failure Analysis ──────────────────────────────────────────────────────────


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


@app.get("/research/failures/summary")
def failure_summary(_: None = Depends(_require_admin)) -> dict:
    return _respond(_api.failure_summary())


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


# ── Model Comparison ──────────────────────────────────────────────────────────


@app.post("/research/comparisons")
def compare_models(req: CompareModelsRequest, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.compare_models(**req.model_dump()))


@app.get("/research/comparisons/{comparison_id}")
def get_comparison(comparison_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.get_comparison(comparison_id))


@app.get("/research/comparisons")
def list_comparisons(_: None = Depends(_require_admin)) -> dict:
    return _respond(_api.list_comparisons())


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
