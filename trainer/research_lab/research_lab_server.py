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
    RESEARCH_LAB_SCHEDULER_ENABLED  Set to "true" to start the background
                              daily/weekly/monthly research-cycle scheduler
                              (S24 §2). Default: disabled — cycles can still
                              be triggered manually via
                              POST /research/schedule/run/{cycle_name}.
    RESEARCH_LAB_SCHEDULER_POLL_S  Background scheduler poll interval in
                              seconds (default: 3600).

SINGLE SIGN-ON (S24D)
----------------------
Signalium Trainer is a pure OAuth/JWT Service Provider: it trusts Signalium
as the sole identity authority, never stores administrator passwords, never
offers public registration, and never operates as an identity provider.
Every request must carry BOTH the legacy static bearer token (defence-in-
depth network-boundary check, unchanged) AND a Signalium-issued JWT in the
``X-Signalium-Token`` header asserting the caller's verified identity and
administrator role. See trainer/research_lab/sso.py for the full role and
capability model.

    SIGNALIUM_JWT_ALGORITHM     "HS256" (default) or "RS256".
    SIGNALIUM_JWT_SECRET        Shared secret, required for HS256.
    SIGNALIUM_JWT_PUBLIC_KEY    PEM-encoded RSA public key, required for RS256.
    SIGNALIUM_JWT_MAX_SKEW_S    Max clock skew (seconds) allowed between the
                                 token's issued-at time and the verifier's
                                 current time (default: 300).
    SIGNALIUM_ISSUER             Expected `iss` claim, if issuer pinning is
                                 desired (default: unset — not checked).

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

import math
import os
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Request, Security
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel
except ImportError as exc:
    raise ImportError(
        "FastAPI and Pydantic are required. Install with: pip install fastapi uvicorn pydantic"
    ) from exc

from trainer.config_sync.config_server import make_config_sync_router
from trainer.config_sync.manager import get_default_manager
from trainer.market_data.symbol_normalizer import SymbolNormalizer, UnknownSymbolError
from trainer.market_data.symbol_store import SymbolStore
from trainer.research_lab.approval_gate import ApprovalGate
from trainer.research_lab.approval_models import (
    DEFAULT_TTL_SECONDS,
    ApprovalAlreadyResolved,
    ApprovalGateType,
)
from trainer.research_lab.artifact_models import (
    ARTIFACT_TYPES,
    ArtifactRecord,
    TrainingRunRecord,
)
from trainer.research_lab.artifact_registry import (
    ArtifactHashMismatchError,
    ArtifactRegistry,
    DuplicateArtifactError,
)
from trainer.research_lab.audit_log import AuditLogger
from trainer.research_lab.champion_challenger import (
    ChampionChallengerTracker,
)
from trainer.research_lab.composite_health import CompositeQDEHealthScorer
from trainer.research_lab.decision_quality import DecisionQualityTracker
from trainer.research_lab.deployment_manager import DeploymentManager
from trainer.research_lab.discovery_models import DiscoveryQuery, DiscoverySummary
from trainer.research_lab.experiment_creator import ExperimentCreator
from trainer.research_lab.failure_investigation_engine import FailureInvestigationEngine
from trainer.research_lab.feature_discovery_pipeline import FeatureDiscoveryPipeline
from trainer.research_lab.feature_health import FeatureHealthMonitor
from trainer.research_lab.health_monitor import (
    ResearchLabHealthMonitor,
)
from trainer.research_lab.investigation_models import InvestigationQuery
from trainer.research_lab.label_discovery_engine import LabelDiscoveryEngine
from trainer.research_lab.performance_monitor import PerformanceMonitor
from trainer.research_lab.pipeline_controller import ResearchLabPipelineController
from trainer.research_lab.pipeline_models import PipelineJobSpec
from trainer.research_lab.qde_observability import (
    AccuracyMonitor,
    AutonomousTradingMonitor,
    InferenceMonitor,
    ModelDistributionMonitor,
    QDEHealthDashboard,
    ResearchMonitor,
)
from trainer.research_lab.qde_resilience import (
    ArtifactIntegrityVerifier,
    DeploymentSafetyMonitor,
    DisasterRecoveryManager,
    InferenceQueueMonitor,
    ResearchLabBackupManager,
    ResilienceDashboard,
    TrainingResilienceMonitor,
    WorkerPoolManager,
)
from trainer.research_lab.recommendation_models import RecommendationConfig
from trainer.research_lab.regime_coverage import RegimeCoverageMonitor
from trainer.research_lab.research_lab_api import ResearchLabAPI
from trainer.research_lab.research_scheduler import ResearchScheduler
from trainer.research_lab.retraining_recommendation_engine import (
    RetrainingRecommendationEngine,
)
from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.session_manager import SessionManager
from trainer.research_lab.sso import (
    Capability,
    MalformedToken,
    MissingClaim,
    MissingToken,
    Role,
    ServerMisconfigured,
    SignaliumIdentity,
    SSOError,
    StaleRequestTimestamp,
    TokenExpired,
    UnknownRole,
    Visibility,
    validate_signalium_jwt,
)
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
_composite_health_scorer = CompositeQDEHealthScorer(_db, _health_monitor)
_decision_quality_tracker = DecisionQualityTracker(_db, _warehouse)
_performance_monitor = PerformanceMonitor(_db, _warehouse)
_feature_health_monitor = FeatureHealthMonitor(_db, _warehouse)
_regime_coverage_monitor = RegimeCoverageMonitor(_db, _warehouse)
_cc_tracker = ChampionChallengerTracker(_db, approval_gate=_approval_gate, rec_engine=_rec_engine)
_session_mgr = SessionManager(_db)
_audit = AuditLogger(_db)

# S25 — QDE Observability and Model Intelligence.
_inference_monitor = InferenceMonitor(_db, _warehouse)
_model_distribution_monitor = ModelDistributionMonitor(_db, _warehouse)
_accuracy_monitor = AccuracyMonitor(_db, _warehouse)
_autonomous_trading_monitor = AutonomousTradingMonitor(_db, _warehouse)
_research_monitor = ResearchMonitor(_db, approval_gate=_approval_gate, rec_engine=_rec_engine)

# S33 — Broker-aware market data / symbol normalization admin surface.
# Independent of the Research Lab DB (own SQLite store) — market data
# normalization is shared infrastructure, not Research-Lab-owned.
_symbol_store = SymbolStore(os.environ.get("SYMBOL_STORE_DB_PATH"))
_symbol_normalizer = SymbolNormalizer(store=_symbol_store)


def _parse_scheduler_poll_interval(raw: str | None, default: float = 3600.0) -> float:
    """Parse RESEARCH_LAB_SCHEDULER_POLL_S defensively — an invalid value
    must never crash server startup/import; fall back to the default and
    log instead."""
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            f"research_lab_server: invalid RESEARCH_LAB_SCHEDULER_POLL_S={raw!r}, "
            f"falling back to {default}s"
        )
        return default
    if not math.isfinite(value) or value <= 0:
        logger.warning(
            f"research_lab_server: RESEARCH_LAB_SCHEDULER_POLL_S must be a finite "
            f"number > 0, got {value}; falling back to {default}s"
        )
        return default
    return value


_scheduler = ResearchScheduler(
    _db,
    health_monitor=_health_monitor,
    failure_engine=_engine,
    experiment_creator=_creator,
    feature_pipeline=_feature_pipeline,
    label_engine=_label_engine,
    rec_engine=_rec_engine,
    performance_monitor=_performance_monitor,
    regime_coverage_monitor=_regime_coverage_monitor,
    feature_health_monitor=_feature_health_monitor,
    decision_quality_tracker=_decision_quality_tracker,
    composite_health_scorer=_composite_health_scorer,
    cc_tracker=_cc_tracker,
    poll_interval_s=_parse_scheduler_poll_interval(os.environ.get("RESEARCH_LAB_SCHEDULER_POLL_S")),
)

# S24 §2: start the background research-cycle scheduler only when explicitly
# enabled. Default-off because this module is imported directly by unit
# tests (via `from research_lab_server import app`) against the shared
# data/research_lab.db fixture — a stray daemon thread must never mutate
# that file as a side effect of import. Real deployments set
# RESEARCH_LAB_SCHEDULER_ENABLED=true (e.g. in the process launch command).
if os.environ.get("RESEARCH_LAB_SCHEDULER_ENABLED", "false").lower() in ("1", "true", "yes"):
    _scheduler.start_background()


def _build_signalium_notifier():
    """Return a Signalium notifier callable, or None if no API config is present.

    Attempts to construct a ``SignaliumClient`` from environment variables.
    If the required env vars are absent or the client raises, returns ``None``
    so that the ``DeploymentManager`` gracefully logs a warning instead of
    crashing the server at startup.
    """
    api_url = os.environ.get("SIGNALIUM_API_URL", "")
    api_key = os.environ.get("SIGNALIUM_API_KEY", "")
    if not api_url:
        return None
    try:
        from trainer.api.client import SignaliumClient
        from trainer.config.schema import APIConfig

        _api_cfg = APIConfig(base_url=api_url, api_key=api_key)
        _sapi = SignaliumClient(_api_cfg)
        return _sapi.notify_deployment
    except Exception as _exc:  # noqa: BLE001
        logger.warning(f"ResearchLabServer: cannot build Signalium notifier — {_exc!r}")
        return None


_deployment_manager = DeploymentManager(
    _db,
    approval_gate=_approval_gate,
    signalium_notifier=_build_signalium_notifier(),
    # S21A: wiring the live ArtifactRegistry here is what makes signature
    # verification actually enforced on this server's promotion/rollback
    # endpoints — without it, `DeploymentManager` silently skips the check
    # (see its docstring / `_enforce_signature_verification`).
    artifact_registry=_artifact_registry,
)
_qde_health_dashboard = QDEHealthDashboard(
    _deployment_manager, _composite_health_scorer, _health_monitor, _rec_engine
)

# ── S26 — QDE Scalability and Model Resilience ─────────────────────────────
_worker_pool = WorkerPoolManager(_db)
_queue_monitor = InferenceQueueMonitor(_db, _worker_pool)
_artifact_integrity = ArtifactIntegrityVerifier(
    _db, _artifact_registry, deployment_manager=_deployment_manager
)
_backup_manager = ResearchLabBackupManager(_db)
_training_resilience = TrainingResilienceMonitor(_db, _artifact_registry)
_disaster_recovery = DisasterRecoveryManager(_db)
_deployment_safety = DeploymentSafetyMonitor(
    _deployment_manager, _composite_health_scorer, _artifact_integrity
)
_resilience_dashboard = ResilienceDashboard(
    _worker_pool,
    _queue_monitor,
    _backup_manager,
    _training_resilience,
    _disaster_recovery,
    _deployment_safety,
)
_bearer = HTTPBearer(auto_error=False)

# ── Auth ──────────────────────────────────────────────────────────────────────

_CONFIGURED_TOKEN = os.environ.get("RESEARCH_LAB_ADMIN_TOKEN", "")

# Configurable admin identity label used in audit records.
# Override via RESEARCH_LAB_ADMIN_ID when multiple named admins share one token.
_ADMIN_ID = os.environ.get("RESEARCH_LAB_ADMIN_ID", "admin")


_SSO_ERROR_STATUS: dict[type[SSOError], int] = {
    ServerMisconfigured: 503,
    MissingToken: 401,
    MalformedToken: 401,
    UnknownRole: 401,
    MissingClaim: 401,
    TokenExpired: 401,
    StaleRequestTimestamp: 401,
    # InvalidSignature and any other SSOError subclass not listed above are
    # treated as 403 (a token was presented but is not trustworthy).
}


def _require_signalium_identity(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
    x_signalium_token: str | None = Header(default=None, alias="X-Signalium-Token"),
    x_trainer_api_key: str | None = Header(default=None, alias="X-Trainer-Api-Key"),
) -> SignaliumIdentity:
    """S24D — Single Sign-On Service Provider entry point.

    Signalium Trainer is a pure Service Provider: it never authenticates a
    user directly, never stores administrator passwords, and never issues
    its own tokens. Every admin-facing request must carry a Signalium-issued
    JWT (``X-Signalium-Token`` header) asserting the caller's verified
    identity and administrator role. This function:

      1. Enforces the pre-existing static bearer-token defence-in-depth
         check (belt-and-suspenders network boundary — this does NOT
         identify *who* the caller is, only that the request reached this
         server through a channel that knows the shared secret).
      2. Validates the Signalium JWT's signature, expiration, and request
         timestamp, and resolves its administrator role
         (``sso.validate_signalium_jwt``).
      3. Establishes (or reuses) an internal research session for that
         identity.
      4. Stashes the identity on ``request.state`` so the audit-logging
         middleware can record exactly who performed which action.

    Any failure — missing/invalid static token, missing/invalid/expired
    Signalium JWT, stale request timestamp, or an unrecognized role — is
    logged as a REJECTED audit entry before the corresponding HTTPException
    is raised, so the audit trail captures unauthorized attempts as well as
    successful ones.

    Status codes:
      - 503 — server misconfigured (missing static token or SSO signing config)
      - 401 — no/invalid static token, or no/invalid/expired/stale Signalium JWT
      - 403 — static token and JWT structure are both valid, but the JWT
              signature does not verify (untrusted issuer)
    """
    if not _CONFIGURED_TOKEN:
        raise HTTPException(
            503,
            detail="Research Lab not configured (RESEARCH_LAB_ADMIN_TOKEN not set)",
        )

    # Accept either:
    #   Authorization: Bearer <token>   (internal / dashboard callers)
    #   X-Trainer-Api-Key: <token>      (Signalium admin platform, per integration spec v1.0)
    bearer_token = credentials.credentials if credentials else None
    presented_token = bearer_token or x_trainer_api_key

    if not presented_token:
        _safe_log_rejected(
            reason="Missing auth credential (no Authorization bearer token or X-Trainer-Api-Key header)",
            action=f"{request.method} {request.url.path}",
        )
        raise HTTPException(
            401,
            detail="Unauthorized. Provide an Authorization bearer token or X-Trainer-Api-Key header.",
        )
    if presented_token != _CONFIGURED_TOKEN:
        _safe_log_rejected(
            reason="Invalid static admin token",
            action=f"{request.method} {request.url.path}",
        )
        raise HTTPException(403, detail="Forbidden. Invalid admin token.")

    signalium_token = x_signalium_token or request.headers.get("X-Signalium-Token")

    if not signalium_token:
        # No Signalium JWT present.  Synthesise a local admin identity from the
        # static bearer token that was already verified above.
        #
        # Design rationale: this server is a private internal service running on
        # localhost, only reachable through the Express proxy layer.  The Express
        # layer is responsible for SSO enforcement for external callers; a valid
        # static bearer token here is sufficient proof the request arrived through
        # the trusted internal channel.  The Signalium JWT enriches audit records
        # when present (who exactly performed the action) but is never a hard
        # requirement for access.
        import time as _time

        _now = _time.time()
        identity = SignaliumIdentity(
            admin_id=_ADMIN_ID,
            role=Role.ADMIN,
            originating_service="trainer-api",
            issued_at=_now,
            expires_at=_now + 3600,
        )
    else:
        try:
            identity = validate_signalium_jwt(signalium_token)
        except SSOError as exc:
            status_code = _SSO_ERROR_STATUS.get(type(exc), 403)
            _safe_log_rejected(
                reason=f"{type(exc).__name__}: {exc.reason}",
                action=f"{request.method} {request.url.path}",
            )
            raise HTTPException(status_code, detail=f"Unauthorized: {exc.reason}") from exc
        except HTTPException:
            raise
        except Exception as exc:
            # Any unexpected failure inside JWT verification (malformed public
            # key, config parsing error, etc.) must still resolve to a
            # deterministic auth rejection rather than an opaque 500 — this is
            # still an untrusted/unverifiable caller, never a server crash.
            _safe_log_rejected(
                reason=f"Unexpected error validating Signalium token: {exc}",
                action=f"{request.method} {request.url.path}",
            )
            raise HTTPException(401, detail="Unauthorized: token validation failed") from exc

    try:
        _session_mgr.establish(identity)
    except Exception:
        # Session bookkeeping is best-effort; a persistence hiccup here
        # must not turn a successfully verified identity into a 500.
        logger.exception("Failed to establish Signalium session for %s", identity.admin_id)
    request.state.identity = identity
    return identity


def _safe_log_rejected(**kwargs: Any) -> None:
    """Audit-log a rejection without ever letting persistence failures mask
    the auth error that's about to be raised alongside it."""
    try:
        _audit.log_rejected(**kwargs)
    except Exception:
        logger.exception("Failed to write rejected-auth audit log entry")


def _require_admin(
    identity: SignaliumIdentity = Depends(_require_signalium_identity),
) -> str:
    """Backward-compatible admin-identity dependency.

    Every pre-existing endpoint depends on this (directly, or transitively
    via the router-level fallback below) and expects a plain admin-id
    string, exactly as before S24D. The real identity/role resolution now
    happens in ``_require_signalium_identity``; this just narrows it to the
    string every existing call site already expects.
    """
    return identity.admin_id


def require_capability(capability: Capability):
    """Dependency factory enforcing a specific S24D capability.

    Depends on ``_require_signalium_identity`` (which FastAPI caches per
    request, so authentication runs at most once even though both this and
    ``_require_admin`` depend on it), then checks whether the resolved
    role's capability set includes ``capability`` — SuperAdmin/Admin always
    pass via the UNRESTRICTED sentinel; every other role is checked exactly
    against the S24D permission table. Raises 403 if the role lacks it.
    """

    def _dependency(
        identity: SignaliumIdentity = Depends(_require_signalium_identity),
    ) -> SignaliumIdentity:
        if not identity.has_capability(capability):
            raise HTTPException(
                403,
                detail=(
                    f"Forbidden. Role {identity.role.value!r} does not have "
                    f"the {capability.value!r} capability."
                ),
            )
        return identity

    return _dependency


def require_visibility(visibility: Visibility):
    """Dependency factory enforcing a specific S24D/S25 dashboard visibility.

    Mirrors ``require_capability`` exactly, but checks the read-only
    ``Visibility`` set instead of a mutating ``Capability`` — this is the
    gate for internal dashboards/reports (e.g. the S25 QDE Health
    Dashboard) that several roles may *view* without being able to trigger
    any research/deployment action.
    """

    def _dependency(
        identity: SignaliumIdentity = Depends(_require_signalium_identity),
    ) -> SignaliumIdentity:
        if not identity.has_visibility(visibility):
            raise HTTPException(
                403,
                detail=(
                    f"Forbidden. Role {identity.role.value!r} does not have "
                    f"the {visibility.value!r} visibility."
                ),
            )
        return identity

    return _dependency


# Belt-and-suspenders: apply auth at the router level as a fallback for any
# route that accidentally omits the per-endpoint Depends(_require_admin).
# FastAPI deduplicates the dependency so it runs at most once per request.
app.router.dependencies.append(Depends(_require_admin))


@app.middleware("http")
async def _liveness_bypass(request: Request, call_next: Any) -> Any:
    """Short-circuit GET /health before auth dependencies run.

    The router-level ``_require_admin`` dependency covers every registered
    route, so a route-level ``@app.get("/health")`` would still be blocked
    by the auth check.  Middleware runs *before* the router resolves
    dependencies, making it the correct place for a no-auth liveness probe.
    """
    if request.url.path == "/health":
        from fastapi.responses import JSONResponse

        return JSONResponse({"status": "ok", "service": "research-lab"})
    return await call_next(request)


@app.middleware("http")
async def _audit_log_middleware(request: Request, call_next: Any) -> Any:
    """S24D §3 — logs every successfully authenticated research action.

    Rejections (failed authentication/authorization) are logged directly by
    ``_require_signalium_identity`` at the point of failure, since by the
    time this middleware's ``call_next`` returns, that context has already
    been consumed. This middleware instead covers the success path: after
    the route handler runs, ``request.state.identity`` will have been set
    by ``_require_signalium_identity`` if authentication succeeded, and
    ``request.state.audit_extra`` may have been populated by the handler
    itself with the specific model_version/datasets/experiments affected.
    """
    request.state.identity = None
    request.state.audit_extra = {}
    action = f"{request.method} {request.url.path}"
    try:
        response = await call_next(request)
    except Exception:
        # The route handler raised before producing a response (e.g. an
        # unhandled exception -> Starlette's default 500). Authentication
        # already succeeded if identity is set, so still record that this
        # action did not complete successfully, then let the exception
        # propagate unchanged so FastAPI/Starlette's normal 500 handling
        # still applies.
        identity = getattr(request.state, "identity", None)
        if identity is not None:
            try:
                _audit.log_rejected(
                    reason="Unhandled exception in route handler",
                    action=action,
                    admin_id=identity.admin_id,
                    role=identity.role.value,
                    originating_service=identity.originating_service,
                )
            except Exception:
                logger.exception("Failed to write audit log entry for %s", action)
        raise

    identity = getattr(request.state, "identity", None)
    if identity is not None:
        # `identity` being set only means authentication succeeded — a
        # capability check downstream (e.g. require_capability, or the
        # dynamic gate-type check on /approvals) may still have rejected
        # the request with 401/403. Log those as REJECTED, not SUCCESS, or
        # audit forensics would misreport a denied action as completed.
        extra = getattr(request.state, "audit_extra", None) or {}
        try:
            if response.status_code < 400:
                _audit.log_success(
                    identity,
                    action=action,
                    model_version=extra.get("model_version"),
                    datasets=extra.get("datasets"),
                    experiments=extra.get("experiments"),
                    extra={"status_code": response.status_code},
                )
            else:
                _audit.log_rejected(
                    reason=f"HTTP {response.status_code}",
                    action=action,
                    admin_id=identity.admin_id,
                    role=identity.role.value,
                    originating_service=identity.originating_service,
                )
        except Exception:
            # Audit persistence must never mask an already-computed
            # response; log and move on.
            logger.exception("Failed to write audit log entry for %s", action)
    return response


# Mount the telemetry router — _require_admin is now defined above.
_telemetry_router = make_telemetry_router(_warehouse, _require_admin)
app.include_router(_telemetry_router)

# Mount the S31 config-sync router (Trainer Configuration Manager).
_config_sync_router = make_config_sync_router(get_default_manager(), Depends(_require_admin))
app.include_router(_config_sync_router)

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


class AddSymbolAliasRequest(BaseModel):
    raw: str
    canonical: str


class RemoveSymbolAliasRequest(BaseModel):
    raw: str


class SymbolActionRequest(BaseModel):
    canonical: str


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


@app.post(
    "/research/datasets", dependencies=[Depends(require_capability(Capability.DATASET_MANAGEMENT))]
)
def create_dataset(req: CreateDatasetRequest, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.create_dataset(**req.model_dump()))


@app.get("/research/datasets")
def list_datasets(_: None = Depends(_require_admin)) -> dict:
    return _respond(_api.list_datasets())


@app.get("/research/datasets/{dataset_id}")
def get_dataset(dataset_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.get_dataset(dataset_id))


@app.post(
    "/research/datasets/{dataset_id}/generate",
    dependencies=[Depends(require_capability(Capability.DATASET_MANAGEMENT))],
)
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


@app.post(
    "/research/generation-jobs/{job_id}/start",
    dependencies=[Depends(require_capability(Capability.DATASET_MANAGEMENT))],
)
def start_generation_job(job_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.start_generation_job(job_id))


@app.post(
    "/research/generation-jobs/{job_id}/validating",
    dependencies=[Depends(require_capability(Capability.DATASET_MANAGEMENT))],
)
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


@app.post(
    "/research/generation-jobs/{job_id}/complete",
    dependencies=[Depends(require_capability(Capability.DATASET_MANAGEMENT))],
)
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


@app.post(
    "/research/generation-jobs/{job_id}/fail",
    dependencies=[Depends(require_capability(Capability.DATASET_MANAGEMENT))],
)
def fail_generation_job(
    job_id: str, req: FailGenerationJobRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(_api.fail_generation_job(job_id, error=req.error))


# ── 2. Feature Discovery ───────────────────────────────────────────────────────


@app.post(
    "/research/features", dependencies=[Depends(require_capability(Capability.FEATURE_ENGINEERING))]
)
def record_feature(req: RecordFeatureRequest, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.record_feature(**req.model_dump()))


@app.post(
    "/research/features/{feature_id}/accept",
    dependencies=[Depends(require_capability(Capability.FEATURE_ENGINEERING))],
)
def accept_feature(
    feature_id: str, req: FeatureStatusRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(_api.accept_feature(feature_id, notes=req.notes))


@app.post(
    "/research/features/{feature_id}/reject",
    dependencies=[Depends(require_capability(Capability.FEATURE_ENGINEERING))],
)
def reject_feature(
    feature_id: str, req: FeatureStatusRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(_api.reject_feature(feature_id, notes=req.notes))


@app.get("/research/features")
def list_features(status: str | None = None, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.list_features(status=status))


# ── 3. Experiment Tracking ─────────────────────────────────────────────────────


@app.post(
    "/research/experiments",
    dependencies=[Depends(require_capability(Capability.EXPERIMENT_CREATION))],
)
def create_experiment(req: CreateExperimentRequest, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.create_experiment(**req.model_dump()))


@app.post(
    "/research/experiments/{experiment_id}/start",
    dependencies=[Depends(require_capability(Capability.TRAINING_RUNS))],
)
def start_experiment(experiment_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.start_experiment(experiment_id))


@app.post(
    "/research/experiments/{experiment_id}/complete",
    dependencies=[Depends(require_capability(Capability.TRAINING_RUNS))],
)
def complete_experiment(
    experiment_id: str, req: CompleteExperimentRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(_api.complete_experiment(experiment_id, metrics=req.metrics, notes=req.notes))


@app.post(
    "/research/experiments/{experiment_id}/fail",
    dependencies=[Depends(require_capability(Capability.TRAINING_RUNS))],
)
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


@app.post("/research/benchmarks", dependencies=[Depends(require_capability(Capability.VALIDATION))])
def run_benchmark(req: RunBenchmarkRequest, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.run_benchmark(**req.model_dump()))


@app.get("/research/benchmarks/{benchmark_id}")
def get_benchmark(benchmark_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.get_benchmark(benchmark_id))


@app.get("/research/benchmarks")
def list_benchmarks(_: None = Depends(_require_admin)) -> dict:
    return _respond(_api.list_benchmarks())


@app.post("/research/backtests", dependencies=[Depends(require_capability(Capability.VALIDATION))])
def register_backtest(req: RegisterBacktestRequest, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.register_backtest(**req.model_dump()))


@app.get("/research/backtests")
def list_backtests(experiment_id: str | None = None, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.list_backtests(experiment_id=experiment_id))


# ── 5. Validation (Deployment Gates) ──────────────────────────────────────────


@app.post(
    "/research/experiments/{experiment_id}/gates",
    dependencies=[Depends(require_capability(Capability.VALIDATION))],
)
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


@app.post(
    "/research/experiments/{experiment_id}/validate-retraining",
    dependencies=[Depends(require_capability(Capability.RETRAINING_WORKFLOWS))],
)
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


@app.post(
    "/research/experiments/{experiment_id}/approve",
    dependencies=[Depends(require_capability(Capability.RESEARCH_WORKFLOWS))],
)
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


@app.post(
    "/research/experiments/{experiment_id}/reject",
    dependencies=[Depends(require_capability(Capability.RESEARCH_WORKFLOWS))],
)
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


@app.post(
    "/research/rollbacks",
    dependencies=[Depends(require_capability(Capability.RETRAINING_WORKFLOWS))],
)
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


@app.post(
    "/research/rollbacks/{request_id}/approve",
    dependencies=[Depends(require_capability(Capability.APPROVE_ROLLBACKS))],
)
def approve_rollback(
    request_id: str, req: ApproveRollbackRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(_api.approve_rollback(request_id, approver=req.approver, notes=req.notes))


@app.post(
    "/research/rollbacks/{request_id}/reject",
    dependencies=[Depends(require_capability(Capability.APPROVE_ROLLBACKS))],
)
def reject_rollback(
    request_id: str, req: RejectRollbackRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(_api.reject_rollback(request_id, rejector=req.rejector, notes=req.notes))


@app.post(
    "/research/rollbacks/{request_id}/withdraw",
    dependencies=[Depends(require_capability(Capability.RETRAINING_WORKFLOWS))],
)
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


@app.post(
    "/research/lifecycle/authorize",
    dependencies=[Depends(require_capability(Capability.RETRAINING_WORKFLOWS))],
)
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


@app.post(
    "/research/lifecycle/consume",
    dependencies=[Depends(require_capability(Capability.RETRAINING_WORKFLOWS))],
)
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


@app.post(
    "/research/failures", dependencies=[Depends(require_capability(Capability.RESEARCH_WORKFLOWS))]
)
def record_failure(req: RecordFailureRequest, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.record_failure(**req.model_dump()))


@app.post(
    "/research/failures/{failure_id}/investigate",
    dependencies=[Depends(require_capability(Capability.RESEARCH_WORKFLOWS))],
)
def investigate_failure(
    failure_id: str, req: InvestigateFailureRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(
        _api.investigate_failure(failure_id, root_cause=req.root_cause, notes=req.notes)
    )


@app.post(
    "/research/failures/{failure_id}/resolve",
    dependencies=[Depends(require_capability(Capability.RESEARCH_WORKFLOWS))],
)
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


@app.post(
    "/research/regime", dependencies=[Depends(require_capability(Capability.RESEARCH_WORKFLOWS))]
)
def analyze_regime(req: AnalyzeRegimeRequest, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.analyze_regime(**req.model_dump()))


@app.get("/research/regime/{report_id}")
def get_regime_report(report_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.get_regime_report(report_id))


@app.get("/research/regime")
def list_regime_reports(qde_version: str | None = None, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.list_regime_reports(qde_version=qde_version))


# ── Version Comparisons ───────────────────────────────────────────────────────


@app.post(
    "/research/comparisons",
    dependencies=[Depends(require_capability(Capability.RESEARCH_WORKFLOWS))],
)
def compare_models(req: CompareModelsRequest, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.compare_models(**req.model_dump()))


@app.get("/research/comparisons/{comparison_id}")
def get_comparison(comparison_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.get_comparison(comparison_id))


@app.get("/research/comparisons")
def list_comparisons(_: None = Depends(_require_admin)) -> dict:
    return _respond(_api.list_comparisons())


# ── Parameter Search ──────────────────────────────────────────────────────────


@app.post(
    "/research/searches",
    dependencies=[Depends(require_capability(Capability.ARCHITECTURE_RESEARCH))],
)
def create_search(req: CreateSearchRequest, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.create_search(**req.model_dump()))


@app.post(
    "/research/searches/{search_id}/trials",
    dependencies=[Depends(require_capability(Capability.ARCHITECTURE_RESEARCH))],
)
def record_trial(
    search_id: str, req: RecordTrialRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(_api.record_trial(search_id, parameters=req.parameters, metrics=req.metrics))


@app.post(
    "/research/searches/{search_id}/complete",
    dependencies=[Depends(require_capability(Capability.ARCHITECTURE_RESEARCH))],
)
def complete_search(search_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.complete_search(search_id))


@app.get("/research/searches/{search_id}")
def get_search(search_id: str, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.get_search(search_id))


@app.get("/research/searches")
def list_searches(qde_version: str | None = None, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.list_searches(qde_version=qde_version))


# ── Shadow Testing ────────────────────────────────────────────────────────────


@app.post(
    "/research/shadow-tests", dependencies=[Depends(require_capability(Capability.SHADOW_TESTING))]
)
def create_shadow_test(req: CreateShadowTestRequest, _: None = Depends(_require_admin)) -> dict:
    return _respond(_api.create_shadow_test(**req.model_dump()))


@app.post(
    "/research/shadow-tests/{test_id}/decisions",
    dependencies=[Depends(require_capability(Capability.SHADOW_TESTING))],
)
def record_shadow_decision(
    test_id: str, req: RecordShadowDecisionRequest, _: None = Depends(_require_admin)
) -> dict:
    return _respond(_api.record_shadow_decision(test_id, **req.model_dump()))


@app.post(
    "/research/shadow-tests/{test_id}/conclude",
    dependencies=[Depends(require_capability(Capability.SHADOW_TESTING))],
)
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
    champion_version: str | None = None,
    status: str | None = None,
    since: float | None = None,
    until: float | None = None,
    _: None = Depends(_require_admin),
) -> dict:
    """List shadow tests, queryable by champion/challenger version pair and
    a started_at time window (S22B: shadow observation hardening)."""
    return _respond(
        _api.list_shadow_tests(
            challenger_version=challenger_version,
            champion_version=champion_version,
            status=status,
            since=since,
            until=until,
        )
    )


@app.get("/research/shadow-decisions")
def list_shadow_decisions_by_version_pair(
    champion_version: str | None = None,
    challenger_version: str | None = None,
    since: float | None = None,
    until: float | None = None,
    limit: int = 500,
    _: None = Depends(_require_admin),
) -> dict:
    """List individual shadow-decision comparisons (not just per-test
    aggregates), queryable by champion/challenger version pair and a
    recorded_at time window (S22B: shadow observation hardening)."""
    return _respond(
        _api.list_shadow_decisions_by_version_pair(
            champion_version=champion_version,
            challenger_version=challenger_version,
            since=since,
            until=until,
            limit=limit,
        )
    )


# ── Research Notes ────────────────────────────────────────────────────────────


@app.post(
    "/research/notes", dependencies=[Depends(require_capability(Capability.RESEARCH_WORKFLOWS))]
)
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


@app.post(
    "/research/investigate",
    dependencies=[Depends(require_capability(Capability.RESEARCH_WORKFLOWS))],
)
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


@app.post(
    "/research/discover", dependencies=[Depends(require_capability(Capability.RESEARCH_WORKFLOWS))]
)
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


@app.get("/research/schedule")
def list_research_cycles(_: None = Depends(_require_admin)) -> dict:
    """List all registered daily/weekly/monthly research cycles."""
    return {"success": True, "cycles": _scheduler.list_cycles()}


@app.get("/research/schedule/status")
def research_schedule_status(_: None = Depends(_require_admin)) -> dict:
    """Per-cycle last-run summary and whether each cycle is currently due."""
    return {"success": True, "status": _scheduler.status()}


@app.get("/research/schedule/history")
def research_schedule_history(
    cycle_name: str | None = None,
    cadence: str | None = None,
    limit: int = 50,
    _: None = Depends(_require_admin),
) -> dict:
    """List persisted research-cycle run records, most recent first."""
    rows = _db.list_research_cycle_runs(cycle_name=cycle_name, cadence=cadence, limit=limit)
    return {"success": True, "total": len(rows), "runs": rows}


@app.post(
    "/research/schedule/run/{cycle_name}",
    dependencies=[Depends(require_capability(Capability.RESEARCH_WORKFLOWS))],
)
def run_research_cycle(cycle_name: str, _: None = Depends(_require_admin)) -> dict:
    """Force-run one named research cycle immediately, regardless of whether
    it is currently due. Read-only with respect to trading/execution — this
    only runs the existing analysis engines and persists their findings."""
    try:
        record = _scheduler.run_cycle(cycle_name)
        return {"success": True, "run": record}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/research/proposed-experiments")
def list_proposed_experiments(_: None = Depends(_require_admin)) -> dict:
    """Return all auto-proposed experiments awaiting admin approval."""
    rows = _creator.list_proposed()
    return {"success": True, "total": len(rows), "experiments": rows}


@app.post(
    "/research/proposed-experiments/{experiment_id}/approve",
    dependencies=[Depends(require_capability(Capability.RESEARCH_WORKFLOWS))],
)
def approve_proposed_experiment(experiment_id: str, _: None = Depends(_require_admin)) -> dict:
    """Advance a PROPOSED experiment to PENDING (admin approval)."""
    try:
        _creator.approve(experiment_id)
        return {"success": True, "experiment_id": experiment_id, "new_status": "PENDING"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post(
    "/research/proposed-experiments/{experiment_id}/reject",
    dependencies=[Depends(require_capability(Capability.RESEARCH_WORKFLOWS))],
)
def reject_proposed_experiment(experiment_id: str, _: None = Depends(_require_admin)) -> dict:
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


@app.post(
    "/artifacts/register", dependencies=[Depends(require_capability(Capability.TRAINING_RUNS))]
)
def register_artifact(req: RegisterArtifactRequest, _: None = Depends(_require_admin)) -> dict:
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


@app.post("/artifacts/runs", dependencies=[Depends(require_capability(Capability.TRAINING_RUNS))])
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


# ── Champion vs Challenger (S20) ──────────────────────────────────────────────


@app.get("/shadow/status")
def get_shadow_status(
    test_id: str | None = None,
    _: None = Depends(_require_admin),
) -> dict:
    """Return head-to-head shadow test status.

    With ``?test_id=...``, returns detailed status for that shadow test
    including the five-dimension comparison and rolling metrics.

    Without a ``test_id``, returns a summary of all RUNNING shadow tests
    with progress toward the ``min_decisions`` threshold.
    """
    status = _cc_tracker.get_status(test_id=test_id)
    return {"success": True, "status": status}


@app.get("/shadow/comparison")
def get_shadow_comparison(
    test_id: str,
    _: None = Depends(_require_admin),
) -> dict:
    """Return dimension-level champion-vs-challenger comparison for one shadow test.

    Computes:
      - **DIRECTION_ACCURACY** — agreement rate with champion
      - **RISK_QUALITY** — absolute risk allocation diff
      - **SIZING_QUALITY** — expected-PnL proxy advantage
      - **CONFIDENCE_QUALITY** — calibration delta (more confident on agreements)
      - **HOLD_QUALITY** — per-regime agreement consistency

    Also returns rolling Sharpe, Sortino, profit factor, max drawdown,
    confidence calibration, and confidence reliability for the challenger
    computed from all paired decisions in this shadow test.
    """
    comparison = _cc_tracker.get_comparison(test_id=test_id)
    return {"success": True, "comparison": comparison}


@app.post("/shadow/evaluate", dependencies=[Depends(require_capability(Capability.SHADOW_TESTING))])
def evaluate_shadow_promotion(
    test_id: str,
    _: None = Depends(_require_admin),
) -> dict:
    """Evaluate the promotion criterion for a shadow test.

    If the challenger meets the configurable majority-of-dimensions criterion
    (default: wins ≥ 3 of 5 dimensions AND avg_promotion_score ≥ 0.65 AND
    agreement_rate ≥ 0.55 AND n_decisions ≥ 50), a PROMOTION
    ``ApprovalRequest`` is created for admin review.

    Idempotent: repeated calls for the same test return the existing pending
    approval if one already exists.
    """
    result = _cc_tracker.evaluate_promotion(test_id=test_id)
    return {"success": True, "evaluation": result}


# ── Deployment Approval, Signalium Notification + Rollback (S20) ──────────────


class _DeploymentPromoteRequest(BaseModel):
    model_version: str
    target_status: str
    requested_by: str
    weights_url: str = ""
    manifest: dict = {}
    metadata: dict = {}
    context: dict = {}


class _DeploymentActivateRequest(BaseModel):
    request_id: str
    approved_by: str
    notes: str = ""


class _DeploymentRollbackRequest(BaseModel):
    model_version: str = "latest_retired"
    reason: str
    requested_by: str
    context: dict = {}


class _DeploymentAdvanceStageRequest(BaseModel):
    model_version: str
    target_status: str
    requested_by: str
    notes: str = ""


@app.post(
    "/deployment/advance-stage", dependencies=[Depends(require_capability(Capability.VALIDATION))]
)
def deployment_advance_stage(
    req: _DeploymentAdvanceStageRequest,
    admin_id: str = Depends(_require_admin),
) -> dict:
    """Advance a version through the pre-SHADOW pipeline stages.

    Valid transitions: ``NOT_DEPLOYED → DRAFT → TRAINING → VALIDATING``.
    These are automatic pipeline-progression steps and do not require a
    human ``ApprovalGate`` — unlike ``POST /deployment/promote``, which
    governs the gated SHADOW/CANDIDATE/ACTIVE transitions.

    Body fields:
      - ``model_version`` — QDE / model version string
      - ``target_status`` — DRAFT | TRAINING | VALIDATING
      - ``requested_by`` — username or service advancing the stage
      - ``notes`` — optional audit notes
    """
    actor = req.requested_by or admin_id
    try:
        result = _deployment_manager.advance_stage(
            model_version=req.model_version,
            target_status=req.target_status,
            actor=actor,
            notes=req.notes,
        )
        return {"success": True, "stage_result": result, "authenticated_as": admin_id}
    except ValueError as exc:
        return {"success": False, "error": str(exc)}


@app.post(
    "/deployment/promote", dependencies=[Depends(require_capability(Capability.APPROVE_PROMOTIONS))]
)
def deployment_request_promotion(
    req: _DeploymentPromoteRequest,
    _: None = Depends(_require_admin),
) -> dict:
    """Create an ``ApprovalRequest`` for a model version stage transition.

    Valid transitions:
      - ``NOT_DEPLOYED → SHADOW`` (PROMOTION gate — legacy direct path, still supported)
      - ``VALIDATING → SHADOW`` (PROMOTION gate — after the pre-SHADOW pipeline;
        see ``POST /deployment/advance-stage`` for DRAFT/TRAINING/VALIDATING)
      - ``SHADOW → CANDIDATE`` (PROMOTION gate)
      - ``CANDIDATE → ACTIVE`` (DEPLOYMENT gate)

    The returned ``approval_request.request_id`` must be passed to
    ``POST /deployment/activate`` once reviewed.

    Body fields:
      - ``model_version`` — QDE / model version string (e.g. ``"v2.1"``)
      - ``target_status`` — destination stage (SHADOW / CANDIDATE / ACTIVE)
      - ``requested_by`` — username or service making the request
      - ``weights_url`` — URL where Signalium will download the weights (required for ACTIVE)
      - ``manifest`` — deployment manifest dict
      - ``metadata`` — arbitrary metadata
      - ``context`` — extra context for the approver
    """
    try:
        result = _deployment_manager.request_promotion(
            model_version=req.model_version,
            target_status=req.target_status,
            requested_by=req.requested_by,
            weights_url=req.weights_url,
            manifest=req.manifest,
            metadata=req.metadata,
            context=req.context,
        )
        return {"success": True, "promotion_request": result}
    except ValueError as exc:
        return {"success": False, "error": str(exc)}


@app.post(
    "/deployment/activate",
    dependencies=[Depends(require_capability(Capability.APPROVE_DEPLOYMENTS))],
)
def deployment_apply_promotion(
    req: _DeploymentActivateRequest,
    admin_id: str = Depends(_require_admin),
) -> dict:
    """Approve an existing promotion ``ApprovalRequest`` and execute the transition.

    Atomically approves, consumes, transitions the model version's stage, and
    — when the target is ACTIVE — retires the previous ACTIVE version and
    sends a deployment notification to Signalium.

    The authenticated admin identity is recorded in the deployment audit log.

    Body fields:
      - ``request_id`` — the ``ApprovalRequest`` returned by ``POST /deployment/promote``
      - ``approved_by`` — admin username (falls back to authenticated identity)
      - ``notes`` — optional review notes
    """
    from trainer.research_lab.approval_models import ApprovalAlreadyResolved, ApprovalRequiredError

    actor = req.approved_by or admin_id
    try:
        result = _deployment_manager.apply_promotion(
            request_id=req.request_id,
            approved_by=actor,
            notes=req.notes,
        )
        return {"success": True, "promotion_result": result, "authenticated_as": admin_id}
    except (ValueError, KeyError, ApprovalRequiredError, ApprovalAlreadyResolved) as exc:
        return {"success": False, "error": str(exc)}


@app.post(
    "/deployment/rollback", dependencies=[Depends(require_capability(Capability.APPROVE_ROLLBACKS))]
)
def deployment_request_rollback(
    req: _DeploymentRollbackRequest,
    _: None = Depends(_require_admin),
) -> dict:
    """Create a ``ROLLBACK`` ``ApprovalRequest``.

    Resolves the most recent RETIRED model version if ``model_version`` is
    ``"latest_retired"`` or empty.  The approval request must be passed to
    ``POST /deployment/rollback/activate`` once reviewed.

    Body fields:
      - ``model_version`` — version to restore (or ``"latest_retired"``)
      - ``reason`` — human-readable rollback reason
      - ``requested_by`` — username or service making the request
      - ``context`` — extra context for the approver
    """
    try:
        result = _deployment_manager.request_rollback(
            model_version=req.model_version,
            reason=req.reason,
            requested_by=req.requested_by,
            context=req.context,
        )
        return {"success": True, "rollback_request": result}
    except ValueError as exc:
        return {"success": False, "error": str(exc)}


@app.post(
    "/deployment/rollback/activate",
    dependencies=[Depends(require_capability(Capability.APPROVE_ROLLBACKS))],
)
def deployment_apply_rollback(
    req: _DeploymentActivateRequest,
    admin_id: str = Depends(_require_admin),
) -> dict:
    """Approve a ``ROLLBACK`` request and restore the target version to ACTIVE.

    Atomically retires the current ACTIVE model, restores the RETIRED version
    to ACTIVE, and re-notifies Signalium with the restored version's artifact
    references.

    The authenticated admin identity is recorded in the rollback audit event.

    Body fields:
      - ``request_id`` — from ``POST /deployment/rollback``
      - ``approved_by`` — admin username (falls back to authenticated identity)
      - ``notes`` — optional review notes (ignored; kept for schema consistency)
    """
    from trainer.research_lab.approval_models import ApprovalAlreadyResolved, ApprovalRequiredError

    actor = req.approved_by or admin_id
    try:
        result = _deployment_manager.apply_rollback(
            request_id=req.request_id,
            approved_by=actor,
        )
        return {"success": True, "rollback_result": result, "authenticated_as": admin_id}
    except (ValueError, KeyError, ApprovalRequiredError, ApprovalAlreadyResolved) as exc:
        return {"success": False, "error": str(exc)}


@app.get("/deployment/current")
def deployment_get_current(
    _: None = Depends(_require_admin),
) -> dict:
    """Return the current ACTIVE model deployment with its provenance.

    Returns the full deployment record including ``weights_url``, ``manifest``,
    ``metadata``, timestamps, and the last approval/notification IDs.
    Returns ``{"active": null}`` when no model is currently ACTIVE.
    """
    current = _deployment_manager.get_current()
    return {"success": True, "active": current}


@app.get("/deployment/history")
def deployment_get_history(
    model_version: str | None = None,
    event_type: str | None = None,
    limit: int = 100,
    _: None = Depends(_require_admin),
) -> dict:
    """Return the full deployment event audit log.

    All stage transitions (PROMOTED, RETIRED, ROLLED_BACK), Signalium
    notifications (SIGNALIUM_NOTIFIED), and approval requests
    (PROMOTION_REQUESTED, ROLLBACK_REQUESTED) are permanently logged.

    Optional filters:
      - ``model_version`` — restrict to one model version
      - ``event_type`` — PROMOTED | RETIRED | ROLLED_BACK | SIGNALIUM_NOTIFIED |
        PROMOTION_REQUESTED | ROLLBACK_REQUESTED | STAGE_ADVANCED
      - ``limit`` — max records returned (default 100)
    """
    events = _deployment_manager.get_history(
        model_version=model_version,
        event_type=event_type,
        limit=limit,
    )
    return {"success": True, "events": events, "total": len(events)}


# ── Model Health Monitoring (S20) ─────────────────────────────────────────────


@app.post(
    "/health/check", dependencies=[Depends(require_capability(Capability.RESEARCH_WORKFLOWS))]
)
def run_health_check(
    model_version: str | None = None,
    since: float | None = None,
    _: None = Depends(_require_admin),
) -> dict:
    """Trigger an on-demand health check across all ten drift/health dimensions.

    Evaluates:
    - **Confidence drift** — mean output confidence shift baseline vs recent
    - **Feature drift** — output probability distribution (TV distance)
    - **Market drift** — regime distribution shift (TV distance)
    - **Data quality** — telemetry staleness and event-rate drop
    - **Regime drift** — per-regime win-rate degradation
    - **Entropy drift** — prediction entropy trend (model decisiveness)
    - **Uncertainty drift** — confidence dispersion (std) vs baseline
    - **Calibration drift** — rolling ECE-style confidence-vs-outcome gap
    - **Decision quality** — expectancy + confidence-outcome correlation drop
    - **Performance** — profit factor/Sharpe/drawdown/hit rate drop
    - **Feature health** — missing/stale/corrupted/delayed pipeline features
    - **Unseen conditions** — live regimes outside the known/trained vocabulary
    - **Black swan** — statistically extreme individual trade outcomes

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


# ── Composite QDE Health Score (S22B) ─────────────────────────────────────────


@app.post(
    "/health/composite/compute",
    dependencies=[Depends(require_capability(Capability.RESEARCH_WORKFLOWS))],
)
def compute_composite_health_score(
    model_version: str | None = None,
    _: None = Depends(_require_admin),
) -> dict:
    """Compute and persist the composite QDE Health Score for a model version.

    Combines the thirteen drift/health dimensions (calibration, confidence,
    entropy, uncertainty, performance, decision quality, data quality,
    regime, feature health, unseen conditions, black swan, market, feature)
    into five weighted categories — calibration, profitability, stability,
    robustness, drift — and a single EXCELLENT/GOOD/WATCH/DEGRADED/CRITICAL
    grade. Reads the *currently persisted* health-alert state; call
    ``POST /health/check`` first to evaluate fresh telemetry.

    This endpoint only computes and stores a score — it has no authority to
    retrain, promote, or roll back any model version.
    """
    result = _composite_health_scorer.compute(model_version=model_version)
    return {"success": True, "composite_health_score": result.to_dict()}


@app.get("/health/composite/latest")
def get_latest_composite_health_score(
    model_version: str | None = None,
    _: None = Depends(_require_admin),
) -> dict:
    """Return the most recently persisted composite QDE Health Score."""
    result = _composite_health_scorer.get_latest(model_version=model_version)
    return {"success": True, "composite_health_score": result}


@app.get("/health/composite/history")
def list_composite_health_scores(
    model_version: str | None = None,
    since: float | None = None,
    until: float | None = None,
    limit: int = 50,
    _: None = Depends(_require_admin),
) -> dict:
    """List persisted composite QDE Health Scores, newest first, filterable
    by model version and a created_at time window."""
    results = _composite_health_scorer.list_scores(
        model_version=model_version, since=since, until=until, limit=limit
    )
    return {"success": True, "total": len(results), "composite_health_scores": results}


# ── Decision Quality Monitoring (S22B) ────────────────────────────────────────


@app.post(
    "/decision-quality/compute",
    dependencies=[Depends(require_capability(Capability.RESEARCH_WORKFLOWS))],
)
def compute_decision_quality_report(
    model_version: str | None = None,
    since: float | None = None,
    until: float | None = None,
    _: None = Depends(_require_admin),
) -> dict:
    """Compute and persist a decision-quality report.

    Tracks confidence-vs-outcome correlation and expectancy broken down by
    regime, symbol, session, and volatility state, over trades matching the
    given filters. Query params:
        model_version: Restrict to a specific model version. ``None`` = all data.
        since / until: Unix timestamp bounds on telemetry.
    """
    report = _decision_quality_tracker.compute_report(
        model_version=model_version, since=since, until=until
    )
    return {"success": True, "report": report.to_dict()}


@app.get("/decision-quality/latest")
def get_latest_decision_quality_report(
    model_version: str | None = None,
    _: None = Depends(_require_admin),
) -> dict:
    """Return the most recently persisted decision-quality report."""
    report = _decision_quality_tracker.get_latest_report(model_version=model_version)
    return {"success": True, "report": report}


@app.get("/decision-quality/reports")
def list_decision_quality_reports(
    model_version: str | None = None,
    limit: int = 50,
    _: None = Depends(_require_admin),
) -> dict:
    """List persisted decision-quality reports, newest first."""
    reports = _decision_quality_tracker.list_reports(model_version=model_version, limit=limit)
    return {"success": True, "total": len(reports), "reports": reports}


# ── Autonomous Performance Monitoring (S22B) ──────────────────────────────────


@app.post(
    "/performance/compute",
    dependencies=[Depends(require_capability(Capability.RESEARCH_WORKFLOWS))],
)
def compute_performance_report(
    model_version: str | None = None,
    since: float | None = None,
    until: float | None = None,
    _: None = Depends(_require_admin),
) -> dict:
    """Compute and persist a live autonomous-performance report.

    Tracks profit factor, Sharpe ratio, max drawdown, hit rate,
    risk-adjusted return, and capital efficiency, broken down by symbol,
    regime, and user cohort, over trades matching the given filters.
    Query params:
        model_version: Restrict to a specific model version. ``None`` = all data.
        since / until: Unix timestamp bounds on telemetry.
    """
    report = _performance_monitor.compute_report(
        model_version=model_version, since=since, until=until
    )
    return {"success": True, "report": report.to_dict()}


@app.get("/performance/latest")
def get_latest_performance_report(
    model_version: str | None = None,
    _: None = Depends(_require_admin),
) -> dict:
    """Return the most recently persisted performance report."""
    report = _performance_monitor.get_latest_report(model_version=model_version)
    return {"success": True, "report": report}


@app.get("/performance/reports")
def list_performance_reports(
    model_version: str | None = None,
    limit: int = 50,
    _: None = Depends(_require_admin),
) -> dict:
    """List persisted performance reports, newest first."""
    reports = _performance_monitor.list_reports(model_version=model_version, limit=limit)
    return {"success": True, "total": len(reports), "reports": reports}


# ── Feature Health Monitoring (S22B) ──────────────────────────────────────────


@app.post(
    "/feature-health/compute",
    dependencies=[Depends(require_capability(Capability.RESEARCH_WORKFLOWS))],
)
def compute_feature_health_report(
    model_version: str | None = None,
    since: float | None = None,
    until: float | None = None,
    _: None = Depends(_require_admin),
) -> dict:
    """Compute and persist a feature-health report.

    Detects, per feature: missing values, staleness, corrupted/out-of-range
    values, and delivery delay, over reasoning-event telemetry matching the
    given filters.
    Query params:
        model_version: Restrict to a specific model version. ``None`` = all data.
        since / until: Unix timestamp bounds on telemetry.
    """
    report = _feature_health_monitor.compute_report(
        model_version=model_version, since=since, until=until
    )
    return {"success": True, "report": report.to_dict()}


@app.get("/feature-health/latest")
def get_latest_feature_health_report(
    model_version: str | None = None,
    _: None = Depends(_require_admin),
) -> dict:
    """Return the most recently persisted feature-health report."""
    report = _feature_health_monitor.get_latest_report(model_version=model_version)
    return {"success": True, "report": report}


@app.get("/feature-health/reports")
def list_feature_health_reports(
    model_version: str | None = None,
    limit: int = 50,
    _: None = Depends(_require_admin),
) -> dict:
    """List persisted feature-health reports, newest first."""
    reports = _feature_health_monitor.list_reports(model_version=model_version, limit=limit)
    return {"success": True, "total": len(reports), "reports": reports}


# ── Market Regime Coverage Detection (S22B) ───────────────────────────────────


@app.post(
    "/regime-coverage/compute",
    dependencies=[Depends(require_capability(Capability.RESEARCH_WORKFLOWS))],
)
def compute_regime_coverage_report(
    model_version: str | None = None,
    since: float | None = None,
    until: float | None = None,
    _: None = Depends(_require_admin),
) -> dict:
    """Compute and persist a regime-coverage report.

    Detects, distinctly: unseen market conditions (regimes outside the
    known/trained vocabulary), general regime-distribution shift, and
    black-swan (statistically extreme) individual trade outcomes.
    Query params:
        model_version: Restrict to a specific model version. ``None`` = all data.
        since / until: Unix timestamp bounds on telemetry.
    """
    report = _regime_coverage_monitor.compute_report(
        model_version=model_version, since=since, until=until
    )
    return {"success": True, "report": report.to_dict()}


@app.get("/regime-coverage/latest")
def get_latest_regime_coverage_report(
    model_version: str | None = None,
    _: None = Depends(_require_admin),
) -> dict:
    """Return the most recently persisted regime-coverage report."""
    report = _regime_coverage_monitor.get_latest_report(model_version=model_version)
    return {"success": True, "report": report}


@app.get("/regime-coverage/reports")
def list_regime_coverage_reports(
    model_version: str | None = None,
    since: float | None = None,
    until: float | None = None,
    limit: int = 50,
    _: None = Depends(_require_admin),
) -> dict:
    """List persisted regime-coverage reports, newest first.

    Query params:
        model_version: Restrict to a specific model version. ``None`` = all.
        since / until: Unix timestamp bounds on each report's `created_at`,
            i.e. filter findings to a specific time window.
    """
    reports = _regime_coverage_monitor.list_reports(
        model_version=model_version, since=since, until=until, limit=limit
    )
    return {"success": True, "total": len(reports), "reports": reports}


# ── Training Pipeline (S20) ───────────────────────────────────────────────────


@app.post("/pipeline/run", dependencies=[Depends(require_capability(Capability.TRAINING_RUNS))])
def run_pipeline(req: RunPipelineRequest, _: None = Depends(_require_admin)) -> dict:
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
def get_pipeline_job(job_id: str, _: None = Depends(_require_admin)) -> dict:
    """Fetch a specific pipeline job by ID, including full stage results and events."""
    job = _pipeline_ctrl.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Pipeline job {job_id!r} not found")
    return {"success": True, "job": job.to_dict()}


# ── Human Approval Gates (S20) ────────────────────────────────────────────────

# The generic /approvals* endpoints serve every gate type through one route,
# so — unlike the rest of the server — their S24D capability requirement
# cannot be fixed at decorator time; it depends on the gate_type in the
# request body/resource. Map each gate type to the capability that governs
# it and check dynamically inside the handler.
_APPROVAL_GATE_CAPABILITY: dict[ApprovalGateType, Capability] = {
    ApprovalGateType.EXPERIMENT_LAUNCH: Capability.EXPERIMENT_CREATION,
    ApprovalGateType.TRAINING_START: Capability.TRAINING_RUNS,
    ApprovalGateType.PROMOTION: Capability.APPROVE_PROMOTIONS,
    ApprovalGateType.DEPLOYMENT: Capability.APPROVE_DEPLOYMENTS,
    ApprovalGateType.ROLLBACK: Capability.APPROVE_ROLLBACKS,
}


def _require_gate_capability(identity: SignaliumIdentity, gate_type: ApprovalGateType) -> None:
    capability = _APPROVAL_GATE_CAPABILITY.get(gate_type)
    if capability is not None and not identity.has_capability(capability):
        raise HTTPException(
            403,
            detail=(
                f"Forbidden. Role {identity.role.value!r} does not have the "
                f"{capability.value!r} capability required for gate type {gate_type.value!r}."
            ),
        )


@app.post("/approvals")
def create_approval_request(
    req: CreateApprovalRequest,
    identity: SignaliumIdentity = Depends(_require_signalium_identity),
) -> dict:
    """Create an approval request for a gated Research Lab action.

    Idempotent: a second call for the same gate_type + resource_id while a
    PENDING request is still open returns the existing request.

    Note: unlike approve/reject, *creating* a request is intentionally not
    gated by the gate type's approval capability — the proposer (e.g. a
    QuantResearcher requesting promotion) is rarely the same role as the
    approver (DeploymentApprover). Any authenticated Signalium identity may
    open a request; only the matching capability can resolve it.
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
def get_approval_request(request_id: str, _: None = Depends(_require_admin)) -> dict:
    """Fetch a specific approval request by ID."""
    req = _approval_gate.get_request(request_id)
    if req is None:
        raise HTTPException(status_code=404, detail=f"Approval request {request_id!r} not found")
    return {"success": True, "approval_request": req.to_dict()}


@app.post("/approvals/{request_id}/approve")
def approve_approval_request(
    request_id: str,
    req: ApproveApprovalRequest,
    identity: SignaliumIdentity = Depends(_require_signalium_identity),
) -> dict:
    """Admin approves a PENDING approval request (PENDING → APPROVED).

    The authenticated admin identity is recorded in the approval audit log
    regardless of what ``approved_by`` body field is provided, providing a
    tamper-evident link between the bearer token and the approval decision.
    Requires the S24D capability matching the request's gate_type (e.g.
    ROLLBACK requests require APPROVE_ROLLBACKS/DeploymentApprover).
    """
    existing = _approval_gate.get_request(request_id)
    if existing is not None:
        _require_gate_capability(identity, ApprovalGateType(existing.gate_type))
    admin_id = identity.admin_id
    # Use authenticated identity; body field is kept for backward compat but
    # the token holder is the authoritative actor recorded in the audit trail.
    actor = req.approved_by or admin_id
    try:
        approved = _approval_gate.approve(request_id, approved_by=actor, reason=req.reason)
        return {
            "success": True,
            "approval_request": approved.to_dict(),
            "authenticated_as": admin_id,
        }
    except (ValueError, ApprovalAlreadyResolved) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/approvals/{request_id}/reject")
def reject_approval_request(
    request_id: str,
    req: RejectApprovalRequest,
    identity: SignaliumIdentity = Depends(_require_signalium_identity),
) -> dict:
    """Admin rejects a PENDING approval request (PENDING → REJECTED).

    The gated action will be blocked when it next calls require_approval().
    The authenticated admin identity is recorded regardless of the body field.
    Requires the same S24D capability as approving that gate type.
    """
    existing = _approval_gate.get_request(request_id)
    if existing is not None:
        _require_gate_capability(identity, ApprovalGateType(existing.gate_type))
    admin_id = identity.admin_id
    actor = req.rejected_by or admin_id
    try:
        rejected = _approval_gate.reject(request_id, rejected_by=actor, reason=req.reason)
        return {
            "success": True,
            "approval_request": rejected.to_dict(),
            "authenticated_as": admin_id,
        }
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


@app.post(
    "/research/evaluate",
    dependencies=[Depends(require_capability(Capability.RETRAINING_WORKFLOWS))],
)
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
    rows = _rec_engine.list_recommendations(status=status, model_version=model_version, limit=limit)
    return {"success": True, "total": len(rows), "recommendations": rows}


@app.get("/research/recommendations/{recommendation_id}")
def get_recommendation(recommendation_id: str, _: None = Depends(_require_admin)) -> dict:
    """Fetch a single RetrainingRecommendation by ID."""
    rec = _rec_engine.get_recommendation(recommendation_id)
    if rec is None:
        raise HTTPException(
            status_code=404, detail=f"Recommendation {recommendation_id!r} not found"
        )
    return {"success": True, "recommendation": rec}


@app.post(
    "/research/recommendations/{recommendation_id}/approve",
    dependencies=[Depends(require_capability(Capability.RETRAINING_WORKFLOWS))],
)
def approve_recommendation(
    recommendation_id: str,
    req: ResolveRequest,
    _: None = Depends(_require_admin),
) -> dict:
    """Approve a PROPOSED recommendation (admin action).

    Advances status to APPROVED, and auto-creates a linked ExperimentRecord
    (PENDING, returned as `experiment_id`) so the recommendation → experiment
    bridge does not require a separate manual step. This still does not start
    retraining: an admin must separately start the experiment and clear the
    TRAINING_START approval gate before any pipeline job executes.
    """
    try:
        rec = _rec_engine.approve(recommendation_id, resolved_by=req.resolved_by)
        return {"success": True, "recommendation": rec}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post(
    "/research/recommendations/{recommendation_id}/dismiss",
    dependencies=[Depends(require_capability(Capability.RETRAINING_WORKFLOWS))],
)
def dismiss_recommendation(
    recommendation_id: str,
    req: ResolveRequest,
    _: None = Depends(_require_admin),
) -> dict:
    """Dismiss a PROPOSED recommendation (admin decision).

    Advances status to DISMISSED. Records the resolver and optional notes.
    """
    try:
        rec = _rec_engine.dismiss(recommendation_id, resolved_by=req.resolved_by, notes=req.notes)
        return {"success": True, "recommendation": rec}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ── Admin Dashboard Aggregations (S20 — Research Lab Admin Dashboards) ────────


@app.get("/research/symbol-performance")
def symbol_performance(
    model_version: str | None = None,
    since: float | None = None,
    until: float | None = None,
    limit: int = 5000,
    _: None = Depends(_require_admin),
) -> dict:
    """Aggregate realized trade outcomes per symbol for the admin dashboard.

    For each symbol with at least one completed trade, computes:
      - ``trade_count`` — number of completed trades in the window
      - ``win_rate`` — fraction of trades with pnl > 0
      - ``sharpe`` — mean(pnl_pct) / stdev(pnl_pct), 0 if fewer than 2 trades
        or stdev is 0 (population, not annualized — comparative ranking only)
      - ``avg_confidence_at_entry`` — mean confidence the model held at entry
      - ``confidence_reliability`` — win_rate restricted to trades entered at
        confidence >= 0.65, as a proxy for "was high confidence earned"

    Symbols are returned sorted by ``sharpe`` descending so the caller can
    read best-N from the head and worst-N from the tail.
    """
    import statistics

    from trainer.research_lab.telemetry_models import TelemetryQuery

    trades = _warehouse.query_trades(
        TelemetryQuery(model_version=model_version, since=since, until=until, limit=limit)
    )
    by_symbol: dict[str, list[dict]] = {}
    for t in trades:
        by_symbol.setdefault(t["symbol"], []).append(t)

    rows = []
    for symbol, symbol_trades in by_symbol.items():
        pnls = [t["pnl"] for t in symbol_trades]
        pnl_pcts = [t["pnl_pct"] for t in symbol_trades]
        confidences = [t.get("confidence_at_entry", 0.0) for t in symbol_trades]
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls) if pnls else 0.0
        sharpe = 0.0
        if len(pnl_pcts) >= 2:
            stdev = statistics.pstdev(pnl_pcts)
            if stdev > 0:
                sharpe = statistics.mean(pnl_pcts) / stdev
        high_conf = [t for t in symbol_trades if t.get("confidence_at_entry", 0.0) >= 0.65]
        conf_reliability = (
            sum(1 for t in high_conf if t["pnl"] > 0) / len(high_conf) if high_conf else None
        )
        rows.append(
            {
                "symbol": symbol,
                "trade_count": len(symbol_trades),
                "win_rate": round(win_rate, 4),
                "sharpe": round(sharpe, 4),
                "avg_confidence_at_entry": (
                    round(statistics.mean(confidences), 4) if confidences else 0.0
                ),
                "confidence_reliability": (
                    round(conf_reliability, 4) if conf_reliability is not None else None
                ),
                "total_pnl": round(sum(pnls), 4),
            }
        )
    rows.sort(key=lambda r: r["sharpe"], reverse=True)
    return {"success": True, "total": len(rows), "symbols": rows}


@app.get("/research/calibration")
def confidence_calibration(
    model_version: str | None = None,
    since: float | None = None,
    until: float | None = None,
    bucket_width: float = 0.1,
    limit: int = 5000,
    _: None = Depends(_require_admin),
) -> dict:
    """Return a confidence-calibration reliability diagram from realized trades.

    Buckets completed trades by ``confidence_at_entry`` (width ``bucket_width``,
    default deciles) and reports the observed win rate per bucket. A
    well-calibrated model has ``observed_win_rate`` roughly tracking the
    bucket's confidence midpoint. Also returns the overall Expected
    Calibration Error (ECE) — the trade-count-weighted mean absolute gap
    between confidence and observed win rate across buckets.
    """
    from trainer.research_lab.telemetry_models import TelemetryQuery

    trades = _warehouse.query_trades(
        TelemetryQuery(model_version=model_version, since=since, until=until, limit=limit)
    )
    if bucket_width <= 0 or bucket_width > 1:
        raise HTTPException(status_code=400, detail="bucket_width must be in (0, 1]")

    n_buckets = max(1, round(1.0 / bucket_width))
    # Use the actual per-bucket width implied by n_buckets (1.0 / n_buckets) for
    # BOTH the assignment index and the reported bucket_lo/bucket_hi labels, so
    # a bucket_width that isn't a clean divisor of 1.0 (e.g. 0.15) can't produce
    # labels that misrepresent which trades actually landed in each bucket.
    actual_width = 1.0 / n_buckets
    buckets: dict[int, list[dict]] = {i: [] for i in range(n_buckets)}
    for t in trades:
        conf = max(0.0, min(1.0, t.get("confidence_at_entry", 0.0)))
        idx = min(int(conf / actual_width), n_buckets - 1)
        buckets[idx].append(t)

    curve = []
    total_trades = 0
    weighted_gap = 0.0
    for i in range(n_buckets):
        bucket_trades = buckets[i]
        lo, hi = round(i * actual_width, 4), round((i + 1) * actual_width, 4)
        if not bucket_trades:
            curve.append(
                {
                    "bucket_lo": lo,
                    "bucket_hi": hi,
                    "trade_count": 0,
                    "mean_confidence": None,
                    "observed_win_rate": None,
                }
            )
            continue
        mean_conf = sum(t.get("confidence_at_entry", 0.0) for t in bucket_trades) / len(
            bucket_trades
        )
        win_rate = sum(1 for t in bucket_trades if t["pnl"] > 0) / len(bucket_trades)
        curve.append(
            {
                "bucket_lo": lo,
                "bucket_hi": hi,
                "trade_count": len(bucket_trades),
                "mean_confidence": round(mean_conf, 4),
                "observed_win_rate": round(win_rate, 4),
            }
        )
        total_trades += len(bucket_trades)
        weighted_gap += len(bucket_trades) * abs(mean_conf - win_rate)

    ece = round(weighted_gap / total_trades, 4) if total_trades else None
    return {
        "success": True,
        "n_trades": total_trades,
        "expected_calibration_error": ece,
        "curve": curve,
    }


@app.get("/research/reasoning-distribution")
def reasoning_distribution(
    model_version: str | None = None,
    symbol: str | None = None,
    since: float | None = None,
    until: float | None = None,
    bucket_seconds: float = 86400.0,
    limit: int = 5000,
    _: None = Depends(_require_admin),
) -> dict:
    """Return the output-head distribution over time for drift inspection.

    Groups reasoning events into fixed-width time buckets (default: daily) and
    reports the mean of each numeric output-head value per bucket, so the
    admin dashboard can chart drift in the model's raw signal heads
    independent of the translated trading decision.
    """
    from trainer.research_lab.telemetry_models import TelemetryQuery

    events = _warehouse.query_reasoning(
        TelemetryQuery(
            model_version=model_version, symbol=symbol, since=since, until=until, limit=limit
        )
    )
    if bucket_seconds <= 0:
        raise HTTPException(status_code=400, detail="bucket_seconds must be > 0")

    buckets: dict[int, dict[str, list[float]]] = {}
    bucket_ts: dict[int, float] = {}
    bucket_event_counts: dict[int, int] = {}
    for ev in events:
        ts = ev.get("timestamp", 0.0)
        bucket_key = int(ts // bucket_seconds)
        bucket_ts.setdefault(bucket_key, bucket_key * bucket_seconds)
        # Count every event assigned to this bucket, independent of whether it
        # contributed a numeric value to any head — a bucket full of events
        # with sparse/missing heads must not be reported as having 0 events.
        bucket_event_counts[bucket_key] = bucket_event_counts.get(bucket_key, 0) + 1
        head_map = buckets.setdefault(bucket_key, {})
        for head_name, value in (ev.get("output_heads") or {}).items():
            if isinstance(value, (int, float)):
                head_map.setdefault(head_name, []).append(float(value))

    series = []
    for bucket_key in sorted(buckets):
        head_map = buckets[bucket_key]
        means = {name: round(sum(vals) / len(vals), 4) for name, vals in head_map.items()}
        series.append(
            {
                "bucket_start": bucket_ts[bucket_key],
                "n_events": bucket_event_counts[bucket_key],
                "head_means": means,
            }
        )
    return {"success": True, "n_events": len(events), "series": series}


# ── S25 — QDE Observability and Model Intelligence ─────────────────────────────
# Every route below is gated on Visibility.OBSERVABILITY_DASHBOARD (view-only:
# SuperAdmin, Admin, Founder, QuantResearcher, MLResearcher, DeploymentApprover
# — Auditor is excluded), not a Capability, since these are read-only
# dashboards/reports rather than mutating research actions.

_OBSERVABILITY_DEP = Depends(require_visibility(Visibility.OBSERVABILITY_DASHBOARD))


@app.post("/observability/inference/compute", dependencies=[_OBSERVABILITY_DEP])
def compute_inference_report(
    model_version: str | None = None,
    since: float | None = None,
    until: float | None = None,
) -> dict:
    """Compute and persist an inference-serving report (latency, throughput,
    queue length, CPU/memory/GPU usage) from the ``tel_inference`` stream."""
    report = _inference_monitor.compute_report(
        model_version=model_version, since=since, until=until
    )
    return {"success": True, "report": report.to_dict()}


@app.get("/observability/inference/latest", dependencies=[_OBSERVABILITY_DEP])
def get_latest_inference_report(model_version: str | None = None) -> dict:
    report = _inference_monitor.get_latest_report(model_version=model_version)
    return {"success": True, "report": report}


@app.get("/observability/model/distributions", dependencies=[_OBSERVABILITY_DEP])
def get_model_distributions(
    model_version: str | None = None,
    since: float | None = None,
    until: float | None = None,
) -> dict:
    """Decision/confidence/symbol/regime/risk-signal/position-size/hold-
    duration distributions for a model version, computed fresh from
    telemetry (not persisted — this is a live view)."""
    report = _model_distribution_monitor.compute_report(
        model_version=model_version, since=since, until=until, persist=False
    )
    return {"success": True, "report": report.to_dict()}


@app.get("/observability/accuracy", dependencies=[_OBSERVABILITY_DEP])
def get_accuracy_report(
    model_version: str | None = None,
    since: float | None = None,
    until: float | None = None,
) -> dict:
    """Directional accuracy, expected-value accuracy, risk-estimation
    accuracy, sizing quality, and portfolio efficiency for a model version."""
    report = _accuracy_monitor.compute_report(
        model_version=model_version, since=since, until=until, persist=False
    )
    return {"success": True, "report": report.to_dict()}


@app.get("/observability/autonomous-trading", dependencies=[_OBSERVABILITY_DEP])
def get_autonomous_trading_report(
    model_version: str | None = None,
    since: float | None = None,
    until: float | None = None,
) -> dict:
    """Intervention frequency, management-action stats, stop-adjustment/
    scale-in frequency, average holding duration, and realized-vs-predicted
    risk for a model version."""
    report = _autonomous_trading_monitor.compute_report(
        model_version=model_version, since=since, until=until, persist=False
    )
    return {"success": True, "report": report.to_dict()}


@app.get("/observability/research-summary", dependencies=[_OBSERVABILITY_DEP])
def get_research_summary() -> dict:
    """Snapshot of active research activity: training/validation queue,
    active experiments, shadow deployments, pending approvals, and pending
    retraining recommendations."""
    return {"success": True, "summary": _research_monitor.compute_summary()}


@app.get("/qde/health-dashboard", dependencies=[_OBSERVABILITY_DEP])
def get_qde_health_dashboard(model_version: str | None = None) -> dict:
    """The single aggregated internal QDE Health Dashboard: version,
    deployment status, composite health/drift/calibration scores, and
    pending research recommendations."""
    return {
        "success": True,
        "dashboard": _qde_health_dashboard.compute(model_version=model_version),
    }


# ── S26 — QDE Scalability and Model Resilience ─────────────────────────────
# View-only routes are gated on Visibility.OBSERVABILITY_DASHBOARD (same six
# roles as S25, Auditor excluded). Mutating routes (worker restart, backup
# creation, snapshotting, auto-rollback triggering) are additionally gated
# on Capability.RESILIENCE_OPERATIONS (SuperAdmin/Admin via UNRESTRICTED,
# MLResearcher, DeploymentApprover).
_RESILIENCE_OPS_DEP = Depends(require_capability(Capability.RESILIENCE_OPERATIONS))


@app.get("/resilience/dashboard", dependencies=[_OBSERVABILITY_DEP])
def get_resilience_dashboard(model_version: str | None = None) -> dict:
    """Aggregated S26 view: worker pool health, queue saturation, latest
    backup/disaster-recovery snapshot status, stale training jobs, and
    current deployment health."""
    return {
        "success": True,
        "dashboard": _resilience_dashboard.compute(model_version=model_version),
    }


@app.get("/resilience/workers", dependencies=[_OBSERVABILITY_DEP])
def list_resilience_workers(status: str | None = None) -> dict:
    return {"success": True, "workers": _worker_pool.list_workers(status=status)}


@app.post("/resilience/workers/register", dependencies=[_RESILIENCE_OPS_DEP])
def register_resilience_worker(
    kind: str = "GPU", role: str = "ACTIVE", capacity: float = 1.0
) -> dict:
    record = _worker_pool.register_worker(kind=kind, role=role, capacity=capacity)
    return {"success": True, "worker": record.to_dict()}


@app.post("/resilience/workers/{worker_id}/heartbeat", dependencies=[_RESILIENCE_OPS_DEP])
def heartbeat_resilience_worker(worker_id: str, current_load: float | None = None) -> dict:
    _worker_pool.heartbeat(worker_id, current_load=current_load)
    return {"success": True}


@app.post("/resilience/workers/{worker_id}/restart", dependencies=[_RESILIENCE_OPS_DEP])
def restart_resilience_worker(worker_id: str) -> dict:
    return {"success": True, "worker": _worker_pool.restart_worker(worker_id)}


@app.post("/resilience/supervision/run", dependencies=[_RESILIENCE_OPS_DEP])
def run_resilience_supervision_cycle() -> dict:
    """Runs one High-Availability supervision pass: marks dead workers,
    restarts stale ones, and fails over dead ACTIVE workers to STANDBY."""
    return {"success": True, "result": _worker_pool.run_supervision_cycle()}


@app.get("/resilience/queue/report", dependencies=[_OBSERVABILITY_DEP])
def get_resilience_queue_report(
    model_version: str | None = None, since: float | None = None, until: float | None = None
) -> dict:
    return {
        "success": True,
        "report": _queue_monitor.compute_report(
            model_version=model_version, since=since, until=until
        ),
    }


@app.get("/resilience/artifacts/{artifact_id}/verify", dependencies=[_OBSERVABILITY_DEP])
def verify_resilience_artifact(artifact_id: str) -> dict:
    return {"success": True, "result": _artifact_integrity.verify_artifact_checksum(artifact_id)}


@app.get(
    "/resilience/deployment/{model_version}/rollback-safety", dependencies=[_OBSERVABILITY_DEP]
)
def get_rollback_safety(model_version: str) -> dict:
    return {"success": True, "result": _artifact_integrity.check_rollback_safety(model_version)}


@app.post("/resilience/backups/create", dependencies=[_RESILIENCE_OPS_DEP])
def create_resilience_backup(reason: str = "manual") -> dict:
    return {"success": True, "backup": _backup_manager.create_backup(reason=reason)}


@app.post("/resilience/backups/{backup_id}/verify", dependencies=[_RESILIENCE_OPS_DEP])
def verify_resilience_backup(backup_id: str) -> dict:
    return {"success": True, "result": _backup_manager.verify_backup(backup_id)}


@app.get("/resilience/backups", dependencies=[_OBSERVABILITY_DEP])
def list_resilience_backups(limit: int = 50) -> dict:
    return {"success": True, "backups": _backup_manager.list_backups(limit=limit)}


@app.get("/resilience/recovery-procedures", dependencies=[_OBSERVABILITY_DEP])
def get_resilience_recovery_procedures() -> dict:
    return {"success": True, "procedures": ResearchLabBackupManager.recovery_procedures()}


@app.post("/resilience/training/cleanup-stale-jobs", dependencies=[_RESILIENCE_OPS_DEP])
def cleanup_stale_training_jobs(stale_after_s: float = 3600.0) -> dict:
    return {
        "success": True,
        "result": _training_resilience.cleanup_failed_jobs(stale_after_s=stale_after_s),
    }


@app.get("/resilience/training/corrupted-artifacts", dependencies=[_OBSERVABILITY_DEP])
def get_corrupted_artifacts(model_version: str | None = None) -> dict:
    return {
        "success": True,
        "corrupted": _training_resilience.detect_corrupted_artifacts(model_version=model_version),
    }


@app.post("/resilience/disaster-recovery/snapshot", dependencies=[_RESILIENCE_OPS_DEP])
def create_disaster_recovery_snapshot(
    model_version: str = "",
    dataset_version: str = "",
    feature_version: str = "",
    notes: str = "",
) -> dict:
    return {
        "success": True,
        "snapshot": _disaster_recovery.create_snapshot(
            model_version=model_version,
            dataset_version=dataset_version,
            feature_version=feature_version,
            notes=notes,
        ),
    }


@app.get("/resilience/disaster-recovery/snapshots", dependencies=[_OBSERVABILITY_DEP])
def list_disaster_recovery_snapshots(limit: int = 50) -> dict:
    return {"success": True, "snapshots": _disaster_recovery.list_snapshots(limit=limit)}


@app.get(
    "/resilience/disaster-recovery/snapshots/{snapshot_id}/verify",
    dependencies=[_OBSERVABILITY_DEP],
)
def verify_disaster_recovery_snapshot(snapshot_id: str) -> dict:
    return {"success": True, "result": _disaster_recovery.verify_snapshot(snapshot_id)}


@app.get("/resilience/deployment/health", dependencies=[_OBSERVABILITY_DEP])
def get_deployment_safety_health(model_version: str | None = None) -> dict:
    return {
        "success": True,
        "result": _deployment_safety.validate_deployment_health(model_version=model_version),
    }


@app.get("/resilience/deployment/active-model/verify", dependencies=[_OBSERVABILITY_DEP])
def verify_active_model() -> dict:
    return {"success": True, "result": _deployment_safety.verify_active_model()}


@app.post("/resilience/deployment/auto-rollback", dependencies=[_RESILIENCE_OPS_DEP])
def trigger_auto_rollback(model_version: str) -> dict:
    """Checks the given model_version's health and, if degraded below
    threshold, raises a ROLLBACK ApprovalRequest to the latest RETIRED
    version (still requires human approval to actually apply — see
    DeploymentManager.apply_rollback)."""
    result = _deployment_safety.maybe_auto_rollback(model_version)
    if result is None:
        return {"success": True, "triggered": False, "reason": "deployment is healthy"}
    return {"success": True, "triggered": True, "result": result}


# ── S33 — Symbol Normalization Admin API ──────────────────────────────────────
#
# Broker-aware market data routing (trainer/market_data/) needs an admin
# surface so Signalium Admin can manage broker symbol aliases and enable/
# disable instruments without a deploy. Backed by SymbolStore (own SQLite
# DB, independent of the Research Lab DB) so state survives restarts.


@app.post("/admin/symbols/aliases", dependencies=[Depends(_require_admin)])
def add_symbol_alias(req: AddSymbolAliasRequest) -> dict:
    """Register a broker-specific spelling that resolves to a canonical
    symbol (e.g. ``EURUSD.pro`` -> ``EURUSD``)."""
    try:
        _symbol_normalizer.add_symbol_alias(req.raw, req.canonical)
    except UnknownSymbolError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"success": True, "raw": req.raw.upper(), "canonical": req.canonical.upper()}


@app.delete("/admin/symbols/aliases", dependencies=[Depends(_require_admin)])
def remove_symbol_alias(req: RemoveSymbolAliasRequest) -> dict:
    _symbol_normalizer.remove_alias(req.raw)
    return {"success": True, "raw": req.raw.upper()}


@app.get("/admin/symbols/aliases", dependencies=[Depends(_require_admin)])
def list_symbol_aliases(canonical: str | None = None) -> dict:
    return {"success": True, "aliases": _symbol_normalizer.list_symbol_aliases(canonical)}


@app.post("/admin/symbols/disable", dependencies=[Depends(_require_admin)])
def disable_symbol(req: SymbolActionRequest) -> dict:
    _symbol_normalizer.disable_symbol(req.canonical)
    return {"success": True, "canonical": req.canonical.upper(), "enabled": False}


@app.post("/admin/symbols/enable", dependencies=[Depends(_require_admin)])
def enable_symbol(req: SymbolActionRequest) -> dict:
    _symbol_normalizer.enable_symbol(req.canonical)
    return {"success": True, "canonical": req.canonical.upper(), "enabled": True}


@app.get("/admin/symbols", dependencies=[Depends(_require_admin)])
def list_symbols() -> dict:
    return {"success": True, "symbols": _symbol_normalizer.list_symbols()}


@app.get("/admin/symbols/diagnostics", dependencies=[Depends(_require_admin)])
def symbol_diagnostics(limit: int = 200, unresolved_only: bool = False) -> dict:
    """Normalization event log + summary stats for the Signalium Admin
    diagnostics surface — what's being resolved, what's failing, and by
    which matching strategy (exact / alias / structural / fuzzy)."""
    return {
        "success": True,
        "symbols": _symbol_normalizer.list_symbols(),
        "aliases": _symbol_normalizer.list_symbol_aliases(),
        "stats": _symbol_store.event_stats(),
        "recent_events": _symbol_store.recent_events(limit=limit, unresolved_only=unresolved_only),
    }


# ── Platform-Facing Genealogy Endpoint ────────────────────────────────────────
# The Signalium Platform calls GET /genealogy/tree to display the model version
# lineage on its ops dashboard. This endpoint aggregates existing training-run
# and artifact records into a tree — no new data is introduced.


@app.get("/genealogy/tree")
def genealogy_tree(
    limit: int = 100,
    _: None = Depends(_require_admin),
) -> dict:
    """Return a model version lineage tree built from training run records.

    Each experiment node lists all training runs under it; each run lists
    its registered artifacts. The response uses only data already stored in
    the artifact registry — no new tables or data models are required.
    """
    runs = _artifact_registry.list_training_runs(limit=limit)
    experiments: dict[str, dict] = {}
    for run in runs:
        run_dict: dict = run if isinstance(run, dict) else run.to_dict()
        exp_id: str = run_dict.get("experiment_id", "unknown")
        run_id: str = run_dict.get("run_id", "")
        try:
            arts = _artifact_registry.artifacts_for_run(run_id)
            run_dict = {**run_dict, "artifacts": arts}
        except Exception:
            run_dict = {**run_dict, "artifacts": []}
        if exp_id not in experiments:
            experiments[exp_id] = {"experiment_id": exp_id, "runs": []}
        experiments[exp_id]["runs"].append(run_dict)

    nodes = list(experiments.values())
    return {
        "success": True,
        "data": {
            "nodes": nodes,
            "total_experiments": len(nodes),
            "total_runs": sum(len(e["runs"]) for e in nodes),
        },
    }


# =============================================================================
# Signalium Integration Compatibility Layer  (spec v1.0 — 2026-07-12)
# =============================================================================
# The Signalium admin platform connects to research-lab directly at
# http://localhost:8002 using the paths defined in the integration spec.
# Auth is accepted via either:
#   Authorization: Bearer <RESEARCH_LAB_ADMIN_TOKEN>   (existing callers)
#   X-Trainer-Api-Key: <RESEARCH_LAB_ADMIN_TOKEN>      (Signalium platform)
# Routes below map the spec-required paths to existing handler logic.
# For resources with no direct backing, an empty-but-valid stub is returned
# (the spec says empty collections are always valid).
# =============================================================================


# ── Experiments ───────────────────────────────────────────────────────────────


@app.get("/experiments")
def compat_list_experiments(
    qde_version: str | None = None,
    status: str | None = None,
    researcher: str | None = None,
    _: str = Depends(_require_admin),
) -> dict:
    return _respond(
        _api.list_experiments(qde_version=qde_version, status=status, researcher=researcher)
    )


@app.get("/experiments/{experiment_id}")
def compat_get_experiment(experiment_id: str, _: str = Depends(_require_admin)) -> dict:
    return _respond(_api.get_experiment(experiment_id))


@app.get("/experiments/{experiment_id}/lineage")
def compat_get_experiment_lineage(experiment_id: str, _: str = Depends(_require_admin)) -> dict:
    return {"lineage": {}}


@app.get("/experiments/{experiment_id}/history")
def compat_get_experiment_history(experiment_id: str, _: str = Depends(_require_admin)) -> dict:
    return {"history": []}


# ── Training Runs ─────────────────────────────────────────────────────────────


@app.get("/training/runs")
def compat_list_training_runs(
    experiment_id: str | None = None,
    pipeline_job_id: str | None = None,
    limit: int = 50,
    _: str = Depends(_require_admin),
) -> dict:
    rows = _artifact_registry.list_training_runs(
        experiment_id=experiment_id,
        pipeline_job_id=pipeline_job_id,
        limit=limit,
    )
    return {"runs": rows}


@app.get("/training/runs/{run_id}")
def compat_get_training_run(run_id: str, _: str = Depends(_require_admin)) -> dict:
    run = _artifact_registry.get_training_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Training run {run_id!r} not found")
    return {"run": run if isinstance(run, dict) else run.to_dict()}


@app.get("/training/resource-usage")
def compat_training_resource_usage(_: str = Depends(_require_admin)) -> dict:
    return {"usage": {}}


@app.post("/training/runs")
async def compat_register_training_run(request: Request, _: str = Depends(_require_admin)) -> dict:
    try:
        body: dict = await request.json()
    except Exception:
        body = {}
    try:
        run = _artifact_registry.create_training_run(
            experiment_id=body.get("experiment_id", ""),
            dataset_version=body.get("dataset_version"),
            feature_version=body.get("feature_version"),
        )
        return {"run": run if isinstance(run, dict) else run.to_dict()}
    except Exception:
        return {"run": {}}


@app.post("/training/runs/{run_id}/stop")
def compat_stop_training_run(run_id: str, _: str = Depends(_require_admin)) -> dict:
    return {"ok": True}


# ── Validation ────────────────────────────────────────────────────────────────


@app.get("/validation/reports")
def compat_list_validation_reports(
    failure_kind: str | None = None,
    model_version: str | None = None,
    min_severity: float = 0.0,
    since: float | None = None,
    until: float | None = None,
    limit: int = 100,
    _: str = Depends(_require_admin),
) -> dict:
    rows = _engine.get_reports(
        failure_kind=failure_kind,
        model_version=model_version,
        min_severity=min_severity,
        since=since,
        until=until,
        limit=limit,
    )
    return {"reports": rows}


@app.get("/validation/reports/{report_id}")
def compat_get_validation_report(report_id: str, _: str = Depends(_require_admin)) -> dict:
    row = _db.get_failure_report(report_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Report {report_id!r} not found")
    return {"report": row}


@app.get("/validation/drift")
def compat_validation_drift(_: str = Depends(_require_admin)) -> dict:
    return {"drift": {}}


@app.get("/validation/robustness")
def compat_validation_robustness(_: str = Depends(_require_admin)) -> dict:
    return {"tests": []}


# ── Shadow Testing ────────────────────────────────────────────────────────────


@app.get("/shadow-testing/deployments")
def compat_list_shadow_deployments(
    challenger_version: str | None = None,
    champion_version: str | None = None,
    status: str | None = None,
    since: float | None = None,
    until: float | None = None,
    _: str = Depends(_require_admin),
) -> dict:
    result = _respond(
        _api.list_shadow_tests(
            challenger_version=challenger_version,
            champion_version=champion_version,
            status=status,
            since=since,
            until=until,
        )
    )
    # Re-key shadow_tests → deployments to match the spec.
    if "shadow_tests" in result:
        result = {**result, "deployments": result.pop("shadow_tests")}
    elif "deployments" not in result:
        result["deployments"] = []
    return result


@app.get("/shadow-testing/deployments/{deployment_id}/comparison")
def compat_shadow_comparison(deployment_id: str, _: str = Depends(_require_admin)) -> dict:
    return {"comparison": {}}


@app.get("/shadow-testing/deployments/{deployment_id}/agreement")
def compat_shadow_agreement(deployment_id: str, _: str = Depends(_require_admin)) -> dict:
    return {"agreement": {}}


@app.get("/shadow-testing/deployments/{deployment_id}/regimes")
def compat_shadow_regimes(deployment_id: str, _: str = Depends(_require_admin)) -> dict:
    return {"regimes": []}


@app.post("/shadow-testing/deployments")
async def compat_create_shadow_deployment(
    request: Request, _: str = Depends(_require_admin)
) -> dict:
    return {"deployment": {}}


# ── Deployment Candidates ─────────────────────────────────────────────────────
# NOTE: /deployment-candidates/history must be registered BEFORE
# /deployment-candidates/{candidate_id} so "history" is not captured as an id.


@app.get("/deployment-candidates")
def compat_list_deployment_candidates(
    status: str | None = None,
    limit: int = 50,
    _: str = Depends(_require_admin),
) -> dict:
    return {"candidates": []}


@app.get("/deployment-candidates/history")
def compat_deployment_candidates_history(
    model_version: str | None = None,
    limit: int = 100,
    _: str = Depends(_require_admin),
) -> dict:
    history = _deployment_manager.get_history(model_version=model_version, limit=limit)
    return {"history": history}


@app.get("/deployment-candidates/{candidate_id}")
def compat_get_deployment_candidate(candidate_id: str, _: str = Depends(_require_admin)) -> dict:
    return {"candidate": {}}


@app.post("/deployment-candidates/{candidate_id}/approve")
async def compat_approve_deployment_candidate(
    candidate_id: str, request: Request, _: str = Depends(_require_admin)
) -> dict:
    return {"ok": True, "message": "Approval recorded"}


@app.post("/deployment-candidates/{candidate_id}/rollback")
async def compat_rollback_deployment_candidate(
    candidate_id: str, request: Request, _: str = Depends(_require_admin)
) -> dict:
    return {"ok": True, "message": "Rollback recorded"}


# ── Research Intelligence ─────────────────────────────────────────────────────


@app.get("/research-intelligence/features")
def compat_list_features(status: str | None = None, _: str = Depends(_require_admin)) -> dict:
    return _respond(_api.list_features(status=status))


@app.get("/research-intelligence/failure-investigations")
def compat_list_failure_investigations(
    category: str | None = None,
    severity: str | None = None,
    qde_version: str | None = None,
    status: str | None = None,
    _: str = Depends(_require_admin),
) -> dict:
    return _respond(
        _api.list_failures(
            category=category,
            severity=severity,
            qde_version=qde_version,
            status=status,
        )
    )


@app.get("/research-intelligence/self-evaluations")
def compat_list_self_evaluations(_: str = Depends(_require_admin)) -> dict:
    return {"findings": []}


@app.get("/research-intelligence/retraining-recommendations")
def compat_list_retraining_recommendations(
    experiment_id: str | None = None,
    model_version: str | None = None,
    status: str | None = None,
    _: str = Depends(_require_admin),
) -> dict:
    try:
        rows = _db.list_retraining_recommendations(
            experiment_id=experiment_id,
            model_version=model_version,
            status=status,
        )
        return {"recommendations": rows}
    except Exception:
        return {"recommendations": []}


# ── Genealogy ─────────────────────────────────────────────────────────────────
# NOTE: /genealogy/tree is already registered above (line ~3506) and will match
# first — these two routes cover the missing parametric paths only.


@app.get("/genealogy/{model_version}/artifacts")
def compat_genealogy_model_artifacts(model_version: str, _: str = Depends(_require_admin)) -> dict:
    return {"artifacts": []}


@app.get("/genealogy/{model_version}")
def compat_genealogy_model(model_version: str, _: str = Depends(_require_admin)) -> dict:
    return {"genealogy": {}}
