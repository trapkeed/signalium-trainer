"""Tests for the S22B composite QDE Health Score (trainer/research_lab/composite_health.py).

Covers:
  1. The five-category partition is a complete, disjoint cover of all
     thirteen DriftDimension values.
  2. Score computation from a mix of healthy / WARNING / CRITICAL alerts.
  3. Grade thresholds.
  4. Persistence round-trip through ResearchLabDatabase.
  5. Universal Research Lab task creation: WARNING alerts (not just CRITICAL)
     now create a RetrainingRecommendation, so no health degradation finding
     is silently dropped.
  6. Human-control audit: no QDE-adjacent code path can execute retrain/
     promotion/rollback without going through ApprovalGate's atomic consume.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest

from trainer.research_lab.composite_health import (
    CATEGORY_DIMENSIONS,
    CompositeHealthConfig,
    CompositeQDEHealthScorer,
    HealthGrade,
    grade_for_composite,
)
from trainer.research_lab.health_monitor import (
    AlertSeverity,
    DriftDimension,
    HealthMonitorConfig,
    ResearchLabHealthMonitor,
)
from trainer.research_lab.recommendation_models import RecommendationConfig
from trainer.research_lab.retraining_recommendation_engine import (
    RetrainingRecommendationEngine,
)
from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.telemetry_warehouse import TelemetryWarehouse


@pytest.fixture()
def db(tmp_path):
    path = str(tmp_path / "test_composite_health.db")
    database = ResearchLabDatabase(path)
    yield database
    database.close() if hasattr(database, "close") else None


@pytest.fixture()
def warehouse(db):
    return TelemetryWarehouse(db)


@pytest.fixture()
def rec_engine(warehouse, db):
    return RetrainingRecommendationEngine(warehouse, db, RecommendationConfig())


@pytest.fixture()
def monitor(db, warehouse, rec_engine):
    return ResearchLabHealthMonitor(db, warehouse, rec_engine=rec_engine)


@pytest.fixture()
def scorer(db, monitor):
    return CompositeQDEHealthScorer(db, monitor)


# ── 1. Category partition ────────────────────────────────────────────────────


def test_category_partition_covers_all_dimensions_exactly_once():
    all_dims = {d.value for d in DriftDimension}
    seen: dict[str, str] = {}
    for category, dims in CATEGORY_DIMENSIONS.items():
        for dim in dims:
            assert dim not in seen, (
                f"{dim} appears in both {seen.get(dim)!r} and {category!r} — "
                "composite categories must be disjoint"
            )
            seen[dim] = category
    assert set(seen.keys()) == all_dims, (
        f"missing dims: {all_dims - set(seen.keys())}; "
        f"unexpected dims: {set(seen.keys()) - all_dims}"
    )


def test_config_weights_sum_to_one():
    CompositeHealthConfig().validate()  # should not raise


def test_config_validate_rejects_bad_weights():
    bad = CompositeHealthConfig(weight_calibration=0.9)
    with pytest.raises(ValueError):
        bad.validate()


# ── 2 & 3. Scoring + grading ──────────────────────────────────────────────────


def test_grade_thresholds():
    cfg = CompositeHealthConfig()
    assert grade_for_composite(100.0, cfg) == HealthGrade.EXCELLENT
    assert grade_for_composite(90.0, cfg) == HealthGrade.EXCELLENT
    assert grade_for_composite(89.9, cfg) == HealthGrade.GOOD
    assert grade_for_composite(75.0, cfg) == HealthGrade.GOOD
    assert grade_for_composite(55.0, cfg) == HealthGrade.WATCH
    assert grade_for_composite(35.0, cfg) == HealthGrade.DEGRADED
    assert grade_for_composite(0.0, cfg) == HealthGrade.CRITICAL


def test_compute_with_no_alerts_is_excellent(scorer):
    result = scorer.compute(model_version="v1")
    assert result.composite == pytest.approx(100.0)
    assert result.grade == HealthGrade.EXCELLENT.value
    assert result.n_critical_alerts == 0
    assert result.n_warning_alerts == 0
    for cs in result.category_scores:
        assert cs.score == pytest.approx(100.0)


def _inject_alert(db, dimension: str, severity: str, model_version: str = "v1") -> None:
    from trainer.research_lab.health_monitor import DriftAlert

    alert = DriftAlert(
        alert_id=f"hal-test-{dimension}-{severity}",
        dimension=dimension,
        severity=severity,
        model_version=model_version,
        affected_symbols=[],
        affected_features=[],
        evidence={},
        recommended_action="test",
        created_at=time.time(),
    )
    db.upsert_health_alert(alert.to_dict())


def test_compute_reflects_critical_alert_in_one_category(db, scorer):
    _inject_alert(db, DriftDimension.CALIBRATION.value, AlertSeverity.CRITICAL.value)
    result = scorer.compute(model_version="v1")

    calibration = result.category_score("calibration")
    assert calibration is not None
    # 1 of 4 calibration dims critical, 3 healthy: (20 + 100*3) / 4 = 80
    assert calibration.score == pytest.approx(80.0)
    assert calibration.dimension_states[DriftDimension.CALIBRATION.value] == "CRITICAL"

    # Other categories remain fully healthy
    profitability = result.category_score("profitability")
    assert profitability.score == pytest.approx(100.0)

    assert result.composite < 100.0
    assert result.n_critical_alerts == 1
    assert result.grade in (HealthGrade.EXCELLENT.value, HealthGrade.GOOD.value)


def test_compute_multiple_critical_categories_lowers_grade(db, scorer):
    for dim in (
        DriftDimension.PERFORMANCE.value,
        DriftDimension.DECISION_QUALITY.value,
        DriftDimension.FEATURE_HEALTH.value,
        DriftDimension.UNSEEN_CONDITION.value,
        DriftDimension.BLACK_SWAN.value,
        DriftDimension.MARKET.value,
    ):
        _inject_alert(db, dim, AlertSeverity.CRITICAL.value)
    result = scorer.compute(model_version="v1")
    assert result.category_score("profitability").score == pytest.approx(20.0)
    assert result.category_score("robustness").score == pytest.approx(20.0)
    # profitability (weight 0.30) + robustness (weight 0.20) both bottom out
    # at 20, dragging the composite well below EXCELLENT/GOOD.
    assert result.grade in (
        HealthGrade.WATCH.value,
        HealthGrade.DEGRADED.value,
        HealthGrade.CRITICAL.value,
    )
    assert result.composite < 75.0


def test_compute_filters_by_model_version(db, scorer):
    _inject_alert(db, DriftDimension.CALIBRATION.value, AlertSeverity.CRITICAL.value, "v1")
    result_v2 = scorer.compute(model_version="v2")
    assert result_v2.composite == pytest.approx(100.0)


# ── 4. Persistence ────────────────────────────────────────────────────────────


def test_score_persists_and_is_queryable(db, scorer):
    result = scorer.compute(model_version="v1")

    fetched = db.get_composite_health_score(result.score_id)
    assert fetched is not None
    assert fetched["model_version"] == "v1"
    assert fetched["grade"] == result.grade

    latest = scorer.get_latest(model_version="v1")
    assert latest is not None
    assert latest["score_id"] == result.score_id

    history = scorer.list_scores(model_version="v1")
    assert len(history) == 1
    assert history[0]["score_id"] == result.score_id


def test_list_scores_orders_newest_first(db, scorer):
    first = scorer.compute(model_version="v1")
    time.sleep(0.01)
    _inject_alert(db, DriftDimension.CALIBRATION.value, AlertSeverity.WARNING.value)
    second = scorer.compute(model_version="v1")

    history = scorer.list_scores(model_version="v1")
    assert history[0]["score_id"] == second.score_id
    assert history[1]["score_id"] == first.score_id


# ── 5. Universal Research Lab task creation (WARNING findings) ──────────────


def test_warning_alert_creates_recommendation_not_just_critical(monitor, db, warehouse):
    """S22B requirement: every degradation finding, including WARNING (not
    only CRITICAL), must automatically create/update a Research Lab review
    task so nothing is silently dropped."""
    # Seed enough decisions with a confidence shift that produces a WARNING
    # (not CRITICAL) confidence-drift alert.
    now = time.time()
    cfg = monitor._cfg if hasattr(monitor, "_cfg") else HealthMonitorConfig()
    n = max(cfg.min_samples * 2, 40)
    decisions = []
    half = n // 2
    for i in range(half):
        decisions.append(
            {
                "decision_id": f"d-base-{i}",
                "model_version": "v1",
                "symbol": "EURUSD",
                "direction": "buy",
                "confidence": 0.80,
                "timestamp": now - 1000 + i,
            }
        )
    warn_drift = (cfg.confidence_warning + cfg.confidence_critical) / 2
    for i in range(half):
        decisions.append(
            {
                "decision_id": f"d-recent-{i}",
                "model_version": "v1",
                "symbol": "EURUSD",
                "direction": "buy",
                "confidence": max(0.0, 0.80 - warn_drift),
                "timestamp": now - 10 + i,
            }
        )
    for d in decisions:
        warehouse.ingest_decision(**d)

    result = monitor.run_check(model_version="v1")
    confidence_alerts = [a for a in result.alerts if a.dimension == DriftDimension.CONFIDENCE.value]
    assert confidence_alerts, "expected a confidence-drift alert to fire"
    alert = confidence_alerts[0]
    assert alert.severity == AlertSeverity.WARNING.value, (
        "test fixture must produce a WARNING (not CRITICAL) alert to prove "
        "the WARNING path also creates a recommendation"
    )
    assert alert.recommendation_id, (
        "WARNING alert must create a RetrainingRecommendation — degradation "
        "findings below CRITICAL must not be silently dropped"
    )
    rec = db.get_retraining_recommendation(alert.recommendation_id)
    assert rec is not None
    assert rec["urgency"] == "LOW"


# ── 6. Human-control audit ───────────────────────────────────────────────────


_EXECUTION_METHODS = {"apply_promotion", "apply_rollback", "run_pipeline"}


def _find_direct_calls(py_file: Path, method_names: set[str]) -> list[str]:
    """Return textual call sites of the given method names in a source file,
    via AST (more reliable than grep for distinguishing definitions from calls)."""
    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in method_names:
                hits.append(f"{py_file}:{node.lineno}")
    return hits


def test_no_qde_module_imports_deployment_manager_or_approval_gate():
    """QDE-side modules must never import the execution-authority classes
    directly — all lifecycle mutation must flow through the Research Lab
    server's admin-gated endpoints, never a direct in-process call from QDE."""
    qde_dir = Path(__file__).resolve().parents[2] / "trainer" / "qde"
    assert qde_dir.is_dir()
    offenders = []
    for py_file in qde_dir.rglob("*.py"):
        text = py_file.read_text()
        if "DeploymentManager" in text or "ApprovalGate" in text:
            offenders.append(str(py_file))
    assert (
        not offenders
    ), f"trainer/qde must never import DeploymentManager/ApprovalGate directly: {offenders}"


def test_apply_promotion_and_rollback_called_only_from_gated_server_endpoints():
    """The only two methods that actually execute a deployment stage
    transition (`apply_promotion`, `apply_rollback`) must be called
    exclusively from the admin-gated FastAPI server module — never from a
    QDE module, a recommendation engine, or a health monitor."""
    repo_root = Path(__file__).resolve().parents[2]
    trainer_dir = repo_root / "trainer"

    call_sites: list[str] = []
    for py_file in trainer_dir.rglob("*.py"):
        if py_file.name in ("deployment_manager.py", "pipeline_controller.py"):
            continue  # method definitions themselves, not calls
        call_sites.extend(_find_direct_calls(py_file, _EXECUTION_METHODS))

    disallowed = [site for site in call_sites if "research_lab_server.py" not in site]
    assert not disallowed, (
        f"apply_promotion/apply_rollback/run_pipeline must only be called from the "
        f"admin-gated server module: unexpected call sites {disallowed}"
    )
    assert call_sites, "sanity check: expected to find the known server call sites"


def test_run_pipeline_raises_without_training_start_approval(tmp_path):
    """The TRAINING stage of run_pipeline (the only path that actually
    executes training) must not proceed without a consumed TRAINING_START
    approval — proving retraining itself is gated, not just promotion."""
    from trainer.research_lab.approval_gate import ApprovalGate
    from trainer.research_lab.pipeline_controller import (
        PipelineJobSpec,
        PipelineStage,
        ResearchLabPipelineController,
    )
    from trainer.research_lab.rl_database import ResearchLabDatabase

    db = ResearchLabDatabase(str(tmp_path / "pipeline_audit.db"))
    gate = ApprovalGate(db)
    controller = ResearchLabPipelineController(db, approval_gate=gate)

    spec = PipelineJobSpec(
        requested_by="test-actor",
        experiment_id="exp-audit-1",
        stages_to_run=[PipelineStage.TRAINING.value],
    )
    job = controller.run_pipeline(spec)
    # No TRAINING_START approval was ever created/approved — the TRAINING
    # stage must fail, not silently execute.
    assert job.status == "FAILED"


def test_apply_promotion_raises_without_prior_approval(db, warehouse):
    """Direct-call audit: even if some future code called apply_promotion
    directly, it cannot execute without a matching APPROVED, unconsumed
    ApprovalRequest — the atomic-consume gate is the only path to success."""
    from trainer.research_lab.approval_gate import ApprovalGate
    from trainer.research_lab.deployment_manager import DeploymentManager

    gate = ApprovalGate(db)
    manager = DeploymentManager(db, approval_gate=gate)

    with pytest.raises(Exception):
        manager.apply_promotion(
            request_id="nonexistent-request-id",
            approved_by="test-actor",
        )


def test_apply_rollback_raises_without_prior_approval(db):
    from trainer.research_lab.approval_gate import ApprovalGate
    from trainer.research_lab.deployment_manager import DeploymentManager

    gate = ApprovalGate(db)
    manager = DeploymentManager(db, approval_gate=gate)

    with pytest.raises(Exception):
        manager.apply_rollback(
            request_id="nonexistent-request-id",
            approved_by="test-actor",
        )
