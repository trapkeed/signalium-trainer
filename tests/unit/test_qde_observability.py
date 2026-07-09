"""S25 — Tests for QDE Observability and Model Intelligence.

Covers:
  - tel_inference telemetry stream (ingest/query/stream_counts)
  - InferenceMonitor.compute_report
  - ModelDistributionMonitor.compute_report (risk-signal/position-size/hold-duration)
  - AccuracyMonitor.compute_report (directional/EV/risk-estimation/sizing/portfolio)
  - AutonomousTradingMonitor.compute_report (intervention/management-action stats)
  - ResearchMonitor.compute_summary
  - QDEHealthDashboard.compute
  - observability_reports generic table persistence
  - Server endpoints: visibility-gated (SuperAdmin passes, Auditor excluded)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from tests.unit._sso_test_utils import DEFAULT_TEST_JWT_SECRET, make_signalium_jwt
from trainer.research_lab.qde_observability import (
    AccuracyMonitor,
    AutonomousTradingMonitor,
    InferenceMonitor,
    ModelDistributionMonitor,
    QDEHealthDashboard,
    ResearchMonitor,
)
from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.telemetry_warehouse import TelemetryWarehouse

NOW = time.time()
MV = "v1.0.0"
SYM = "EURUSD"


@pytest.fixture()
def db(tmp_path: Path) -> ResearchLabDatabase:
    return ResearchLabDatabase(str(tmp_path / "test_qde_obs.db"))


@pytest.fixture()
def warehouse(db: ResearchLabDatabase) -> TelemetryWarehouse:
    return TelemetryWarehouse(db)


def _seed_inference(wh: TelemetryWarehouse, n: int = 3) -> None:
    for i in range(n):
        wh.ingest_inference(
            model_version=MV,
            symbol=SYM,
            timestamp=NOW - (n - i) * 60,
            inference_count=10 + i,
            latency_ms_p50=5.0 + i,
            latency_ms_p95=9.0 + i,
            latency_ms_max=20.0,
            queue_length=i,
            throughput_per_s=3.0 + i,
            cpu_pct=40.0 + i,
            memory_pct=50.0,
            gpu_pct=None,
        )


def _seed_decisions_and_trades(wh: TelemetryWarehouse) -> None:
    wh.ingest_decision(
        model_version=MV,
        symbol=SYM,
        timestamp=NOW - 100,
        direction="LONG",
        confidence=0.85,
        regime="TREND",
        position_size=0.5,
    )
    wh.ingest_decision(
        model_version=MV,
        symbol=SYM,
        timestamp=NOW - 90,
        direction="SHORT",
        confidence=0.3,
        regime="RANGE",
        position_size=1.5,
    )
    wh.ingest_trade(
        model_version=MV,
        symbol=SYM,
        timestamp=NOW - 80,
        direction="LONG",
        pnl=10.0,
        pnl_pct=0.02,
        entry_price=100.0,
        exit_price=102.0,
        size=1.0,
        confidence_at_entry=0.85,
        hold_duration_s=120.0,
    )
    wh.ingest_trade(
        model_version=MV,
        symbol=SYM,
        timestamp=NOW - 70,
        direction="SHORT",
        pnl=-5.0,
        pnl_pct=-0.01,
        entry_price=100.0,
        exit_price=101.0,
        size=0.5,
        confidence_at_entry=0.3,
        hold_duration_s=600.0,
        market_conditions={"stop_distance": 1.0},
    )
    wh.ingest_lifecycle(model_version=MV, symbol=SYM, timestamp=NOW - 75, kind="STOP_ADJUSTED")
    wh.ingest_lifecycle(model_version=MV, symbol=SYM, timestamp=NOW - 72, kind="SIZE_SCALED")


# ── tel_inference stream ────────────────────────────────────────────────────────


class TestInferenceTelemetryStream:
    def test_ingest_and_query(self, warehouse: TelemetryWarehouse) -> None:
        _seed_inference(warehouse, n=3)
        rows = warehouse.query_inference()
        assert len(rows) == 3
        assert rows[0]["model_version"] == MV

    def test_stream_counts_includes_inference(self, warehouse: TelemetryWarehouse) -> None:
        _seed_inference(warehouse, n=2)
        counts = warehouse.stream_counts()
        assert counts["tel_inference"] == 2

    def test_duplicate_event_id_idempotent(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase
    ) -> None:
        ev = warehouse.ingest_inference(
            model_version=MV,
            symbol=SYM,
            timestamp=NOW,
            inference_count=1,
            event_id="fixed-id",
        )
        db.insert_telemetry_event("tel_inference", "fixed-id", ev.to_dict())
        assert warehouse.stream_counts()["tel_inference"] == 1


# ── InferenceMonitor ─────────────────────────────────────────────────────────────


class TestInferenceMonitor:
    def test_compute_report_empty(self, db, warehouse) -> None:
        report = InferenceMonitor(db, warehouse).compute_report(model_version=MV)
        assert report.n_samples == 0
        assert report.avg_gpu_pct is None

    def test_compute_report_with_samples(self, db, warehouse) -> None:
        _seed_inference(warehouse, n=3)
        report = InferenceMonitor(db, warehouse).compute_report(model_version=MV)
        assert report.n_samples == 3
        assert report.total_inference_count == 10 + 11 + 12
        assert report.latency_ms_max == 20.0
        assert report.avg_cpu_pct == pytest.approx(41.0)

    def test_persist_and_get_latest(self, db, warehouse) -> None:
        _seed_inference(warehouse, n=1)
        monitor = InferenceMonitor(db, warehouse)
        monitor.compute_report(model_version=MV)
        latest = monitor.get_latest_report(model_version=MV)
        assert latest is not None
        assert latest["model_version"] == MV

    def test_list_reports(self, db, warehouse) -> None:
        _seed_inference(warehouse, n=1)
        monitor = InferenceMonitor(db, warehouse)
        monitor.compute_report(model_version=MV)
        monitor.compute_report(model_version=MV)
        assert len(monitor.list_reports(model_version=MV)) == 2


# ── ModelDistributionMonitor ─────────────────────────────────────────────────────


class TestModelDistributionMonitor:
    def test_compute_report(self, db, warehouse) -> None:
        _seed_decisions_and_trades(warehouse)
        report = ModelDistributionMonitor(db, warehouse).compute_report(model_version=MV)
        assert report.n_decisions == 2
        assert report.n_trades == 2
        assert report.decision_distribution == {"LONG": 1, "SHORT": 1}
        assert report.regime_distribution == {"TREND": 1, "RANGE": 1}
        # risk_signal derived from confidence when market_conditions lacks it
        assert report.risk_signal_distribution.get("LOW", 0) == 1  # confidence 0.85
        assert report.risk_signal_distribution.get("HIGH", 0) == 1  # confidence 0.3
        assert sum(report.hold_duration_distribution.values()) == 2

    def test_empty_warehouse(self, db, warehouse) -> None:
        report = ModelDistributionMonitor(db, warehouse).compute_report(model_version=MV)
        assert report.n_decisions == 0
        assert sum(report.confidence_distribution.values()) == 0


# ── AccuracyMonitor ───────────────────────────────────────────────────────────────


class TestAccuracyMonitor:
    def test_compute_report(self, db, warehouse) -> None:
        _seed_decisions_and_trades(warehouse)
        report = AccuracyMonitor(db, warehouse).compute_report(model_version=MV)
        assert report.n_trades == 2
        # LONG trade with pnl>0 correct, SHORT trade with pnl<0 incorrect -> 0.5
        assert report.directional_accuracy == 0.5
        assert report.expected_value_accuracy["n_trades"] == 2
        assert report.risk_estimation_accuracy["n_losing_trades_scored"] == 1
        assert report.sizing_quality["n_trades"] == 2
        assert report.portfolio_efficiency["total_pnl"] == pytest.approx(5.0)

    def test_empty_warehouse(self, db, warehouse) -> None:
        report = AccuracyMonitor(db, warehouse).compute_report(model_version=MV)
        assert report.n_trades == 0
        assert report.directional_accuracy == 0.0
        assert report.risk_estimation_accuracy["n_losing_trades_scored"] == 0


# ── AutonomousTradingMonitor ──────────────────────────────────────────────────────


class TestAutonomousTradingMonitor:
    def test_compute_report(self, db, warehouse) -> None:
        _seed_decisions_and_trades(warehouse)
        report = AutonomousTradingMonitor(db, warehouse).compute_report(model_version=MV)
        assert report.n_lifecycle_events == 2
        assert report.n_trades == 2
        assert report.management_action_counts == {"STOP_ADJUSTED": 1, "SIZE_SCALED": 1}
        assert report.intervention_frequency_per_trade == 1.0
        assert report.stop_adjustment_frequency_per_trade == 0.5
        assert report.scale_in_frequency_per_trade == 0.5
        assert report.avg_holding_duration_s == pytest.approx(360.0)

    def test_no_trades_no_division_by_zero(self, db, warehouse) -> None:
        report = AutonomousTradingMonitor(db, warehouse).compute_report(model_version=MV)
        assert report.intervention_frequency_per_trade == 0.0


# ── ResearchMonitor ────────────────────────────────────────────────────────────────


class TestResearchMonitor:
    def test_compute_summary_empty(self, db) -> None:
        from trainer.research_lab.approval_gate import ApprovalGate
        from trainer.research_lab.recommendation_models import RecommendationConfig
        from trainer.research_lab.retraining_recommendation_engine import (
            RetrainingRecommendationEngine,
        )
        from trainer.research_lab.telemetry_warehouse import TelemetryWarehouse as TW

        approval_gate = ApprovalGate(db)
        rec_engine = RetrainingRecommendationEngine(TW(db), db, RecommendationConfig())
        summary = ResearchMonitor(
            db, approval_gate=approval_gate, rec_engine=rec_engine
        ).compute_summary()
        assert summary["training_queue"]["n_pending_or_running"] == 0
        assert summary["pending_approvals"]["count"] == 0
        assert summary["pending_retraining_recommendations"]["count"] == 0
        assert "generated_at" in summary

    def test_compute_summary_counts_proposed_recommendations(self, db) -> None:
        # Recommendations are lifecycled PROPOSED -> APPROVED/DISMISSED (no
        # "PENDING" status) — "pending" here must mean PROPOSED, and this
        # test guards against regressing to a status string that never
        # matches (e.g. "PENDING"), which would silently always report 0.
        from trainer.research_lab.approval_gate import ApprovalGate
        from trainer.research_lab.recommendation_models import (
            EvaluationDimension,
            RecommendationConfig,
            RecommendationUrgency,
            RecommendedScope,
            RetrainingRecommendation,
        )
        from trainer.research_lab.retraining_recommendation_engine import (
            RetrainingRecommendationEngine,
        )

        approval_gate = ApprovalGate(db)
        rec_engine = RetrainingRecommendationEngine(
            TelemetryWarehouse(db), db, RecommendationConfig()
        )
        rec = RetrainingRecommendation.new(
            model_version=MV,
            trigger_dimension=EvaluationDimension.SHARPE_DEGRADATION,
            trigger_reason="performance_degradation",
            evidence={},
            recommended_scope=RecommendedScope.FULL_RETRAIN,
            urgency=RecommendationUrgency.HIGH,
        )
        db.upsert_retraining_recommendation(rec.to_dict())
        summary = ResearchMonitor(
            db, approval_gate=approval_gate, rec_engine=rec_engine
        ).compute_summary()
        assert summary["pending_retraining_recommendations"]["count"] == 1


# ── QDEHealthDashboard ─────────────────────────────────────────────────────────────


class TestQDEHealthDashboard:
    def test_compute_with_no_deployment(self, db, warehouse) -> None:
        from trainer.research_lab.composite_health import CompositeQDEHealthScorer
        from trainer.research_lab.health_monitor import ResearchLabHealthMonitor
        from trainer.research_lab.recommendation_models import RecommendationConfig
        from trainer.research_lab.retraining_recommendation_engine import (
            RetrainingRecommendationEngine,
        )

        health_monitor = ResearchLabHealthMonitor(db, warehouse)
        composite_scorer = CompositeQDEHealthScorer(db, health_monitor)
        rec_engine = RetrainingRecommendationEngine(warehouse, db, RecommendationConfig())

        dashboard = QDEHealthDashboard(
            deployment_manager=None,
            composite_health_scorer=composite_scorer,
            health_monitor=health_monitor,
            rec_engine=rec_engine,
        )
        result = dashboard.compute(model_version=MV)
        assert result["deployment_status"] == "NOT_DEPLOYED"
        assert result["composite_score"] == 100.0
        assert result["grade"] == "EXCELLENT"
        assert result["is_healthy"] is True


# ── Server endpoints ────────────────────────────────────────────────────────────


class TestObservabilityServerEndpoints:
    _SUPERADMIN_HDR = {
        "Authorization": "Bearer test-obs-token",
        "X-Signalium-Token": make_signalium_jwt(role="SuperAdmin"),
    }
    _AUDITOR_HDR = {
        "Authorization": "Bearer test-obs-token",
        "X-Signalium-Token": make_signalium_jwt(role="Auditor"),
    }

    @pytest.fixture()
    def client(self, tmp_path: Path):
        import os

        os.environ["RESEARCH_LAB_ADMIN_TOKEN"] = "test-obs-token"
        os.environ["SIGNALIUM_JWT_SECRET"] = DEFAULT_TEST_JWT_SECRET
        os.environ["RESEARCH_LAB_DB_PATH"] = str(tmp_path / "server_obs.db")
        for mod_name in list(sys.modules):
            if "research_lab_server" in mod_name or "telemetry_server" in mod_name:
                del sys.modules[mod_name]
        from starlette.testclient import TestClient

        import trainer.research_lab.research_lab_server as srv_mod

        return TestClient(srv_mod.app)

    def test_health_dashboard_superadmin(self, client) -> None:
        resp = client.get("/qde/health-dashboard", headers=self._SUPERADMIN_HDR)
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_health_dashboard_auditor_forbidden(self, client) -> None:
        resp = client.get("/qde/health-dashboard", headers=self._AUDITOR_HDR)
        assert resp.status_code == 403

    def test_health_dashboard_requires_auth(self, client) -> None:
        resp = client.get("/qde/health-dashboard")
        assert resp.status_code == 401

    def test_inference_compute_and_latest(self, client) -> None:
        resp = client.post(
            "/observability/inference/compute?model_version=v1.0", headers=self._SUPERADMIN_HDR
        )
        assert resp.status_code == 200
        assert resp.json()["report"]["n_samples"] == 0
        resp2 = client.get(
            "/observability/inference/latest?model_version=v1.0", headers=self._SUPERADMIN_HDR
        )
        assert resp2.status_code == 200
        assert resp2.json()["report"] is not None

    def test_model_distributions_auditor_forbidden(self, client) -> None:
        resp = client.get("/observability/model/distributions", headers=self._AUDITOR_HDR)
        assert resp.status_code == 403

    def test_accuracy_endpoint(self, client) -> None:
        resp = client.get(
            "/observability/accuracy?model_version=v1.0", headers=self._SUPERADMIN_HDR
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_autonomous_trading_endpoint(self, client) -> None:
        resp = client.get(
            "/observability/autonomous-trading?model_version=v1.0", headers=self._SUPERADMIN_HDR
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_research_summary_endpoint(self, client) -> None:
        resp = client.get("/observability/research-summary", headers=self._SUPERADMIN_HDR)
        assert resp.status_code == 200
        assert "summary" in resp.json()

    def test_telemetry_inference_ingest_endpoint(self, client) -> None:
        payload = {
            "model_version": "v1.0",
            "symbol": SYM,
            "timestamp": NOW,
            "inference_count": 5,
            "latency_ms_p50": 3.0,
        }
        resp = client.post("/telemetry/inference", json=payload, headers=self._SUPERADMIN_HDR)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True

        status_resp = client.get("/telemetry/status", headers=self._SUPERADMIN_HDR)
        assert status_resp.json()["stream_counts"]["tel_inference"] == 1
