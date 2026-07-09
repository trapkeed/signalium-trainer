"""Unit tests for S24 §2 — Automated Research Scheduling.

Covers:
- Cycle registry (daily/weekly/monthly cycles all registered)
- run_cycle: persists a run record, COMPLETED status, findings captured
- is_due / run_due_cycles: never-run cycles are due; freshly-run cycles are not
- Cadence gating: a cycle run "recently enough" is not re-run by run_due_cycles
- status(): reflects last run + due flag
- Unknown cycle name raises ValueError
- Background thread start/stop lifecycle (no real waiting — tiny poll interval)
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from trainer.research_lab.champion_challenger import ChampionChallengerTracker
from trainer.research_lab.approval_gate import ApprovalGate
from trainer.research_lab.composite_health import CompositeQDEHealthScorer
from trainer.research_lab.decision_quality import DecisionQualityTracker
from trainer.research_lab.experiment_creator import ExperimentCreator
from trainer.research_lab.failure_investigation_engine import FailureInvestigationEngine
from trainer.research_lab.feature_discovery_pipeline import FeatureDiscoveryPipeline
from trainer.research_lab.feature_health import FeatureHealthMonitor
from trainer.research_lab.health_monitor import ResearchLabHealthMonitor
from trainer.research_lab.label_discovery_engine import LabelDiscoveryEngine
from trainer.research_lab.performance_monitor import PerformanceMonitor
from trainer.research_lab.recommendation_models import RecommendationConfig
from trainer.research_lab.regime_coverage import RegimeCoverageMonitor
from trainer.research_lab.research_scheduler import CADENCE_SECONDS, ResearchScheduler
from trainer.research_lab.retraining_recommendation_engine import RetrainingRecommendationEngine
from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.telemetry_warehouse import TelemetryWarehouse


@pytest.fixture()
def db(tmp_path: Path) -> ResearchLabDatabase:
    return ResearchLabDatabase(str(tmp_path / "test_sched.db"))


@pytest.fixture()
def warehouse(db: ResearchLabDatabase) -> TelemetryWarehouse:
    return TelemetryWarehouse(db)


@pytest.fixture()
def scheduler(db: ResearchLabDatabase, warehouse: TelemetryWarehouse) -> ResearchScheduler:
    rec_engine = RetrainingRecommendationEngine(warehouse, db, RecommendationConfig())
    health_monitor = ResearchLabHealthMonitor(db, warehouse, rec_engine=rec_engine)
    gate = ApprovalGate(db)
    return ResearchScheduler(
        db,
        health_monitor=health_monitor,
        failure_engine=FailureInvestigationEngine(warehouse, db),
        experiment_creator=ExperimentCreator(db),
        feature_pipeline=FeatureDiscoveryPipeline(warehouse, db),
        label_engine=LabelDiscoveryEngine(warehouse, db),
        rec_engine=rec_engine,
        performance_monitor=PerformanceMonitor(db, warehouse),
        regime_coverage_monitor=RegimeCoverageMonitor(db, warehouse),
        feature_health_monitor=FeatureHealthMonitor(db, warehouse),
        decision_quality_tracker=DecisionQualityTracker(db, warehouse),
        composite_health_scorer=CompositeQDEHealthScorer(db, health_monitor),
        cc_tracker=ChampionChallengerTracker(db, approval_gate=gate, rec_engine=rec_engine),
        poll_interval_s=0.05,
    )


class TestCycleRegistry:
    def test_all_expected_cycles_registered(self, scheduler: ResearchScheduler) -> None:
        names = {c["name"] for c in scheduler.list_cycles()}
        assert names == {
            "performance_review", "drift_analysis", "failure_review", "regime_review",
            "self_evaluation", "feature_discovery", "counterfactual_review",
            "shadow_promotion_review", "architecture_review", "deployment_review",
        }

    def test_cadences_are_valid(self, scheduler: ResearchScheduler) -> None:
        for c in scheduler.list_cycles():
            assert c["cadence"] in CADENCE_SECONDS


class TestRunCycle:
    def test_run_cycle_persists_completed_record(
        self, scheduler: ResearchScheduler, db: ResearchLabDatabase
    ) -> None:
        record = scheduler.run_cycle("performance_review")
        assert record["status"] == "COMPLETED"
        assert record["cadence"] == "daily"
        assert record["completed_at"] >= record["started_at"]
        assert "findings" in record

        persisted = db.get_research_cycle_run(record["run_id"])
        assert persisted is not None
        assert persisted["status"] == "COMPLETED"

    def test_run_unknown_cycle_raises(self, scheduler: ResearchScheduler) -> None:
        with pytest.raises(ValueError, match="Unknown research cycle"):
            scheduler.run_cycle("not_a_real_cycle")

    def test_run_cycle_with_missing_dependency_is_skipped_not_crashed(
        self, db: ResearchLabDatabase, warehouse: TelemetryWarehouse
    ) -> None:
        bare = ResearchScheduler(db)  # no engines injected
        record = bare.run_cycle("self_evaluation")
        assert record["status"] == "COMPLETED"
        assert record["findings"]["skipped"]


class TestDueGating:
    def test_never_run_cycle_is_due(self, scheduler: ResearchScheduler) -> None:
        assert scheduler.is_due("performance_review") is True

    def test_recently_run_cycle_is_not_due(self, scheduler: ResearchScheduler) -> None:
        scheduler.run_cycle("performance_review")
        assert scheduler.is_due("performance_review") is False

    def test_run_due_cycles_runs_everything_first_time(
        self, scheduler: ResearchScheduler
    ) -> None:
        results = scheduler.run_due_cycles()
        assert len(results) == len(scheduler.list_cycles())
        assert all(r["status"] == "COMPLETED" for r in results)

    def test_run_due_cycles_second_pass_is_empty(
        self, scheduler: ResearchScheduler
    ) -> None:
        scheduler.run_due_cycles()
        second = scheduler.run_due_cycles()
        assert second == []

    def test_due_after_cadence_elapses(self, scheduler: ResearchScheduler) -> None:
        scheduler.run_cycle("performance_review")
        far_future = time.time() + CADENCE_SECONDS["daily"] + 1
        assert scheduler.is_due("performance_review", now=far_future) is True
        # A weekly cycle run at the same time should not yet be due again.
        scheduler.run_cycle("self_evaluation")
        assert scheduler.is_due("self_evaluation", now=far_future) is False

    def test_is_due_unknown_cycle_raises(self, scheduler: ResearchScheduler) -> None:
        with pytest.raises(ValueError, match="Unknown research cycle"):
            scheduler.is_due("not_a_real_cycle")


class TestStatus:
    def test_status_reflects_last_run_and_due_flag(
        self, scheduler: ResearchScheduler
    ) -> None:
        status_before = {s["name"]: s for s in scheduler.status()}
        assert status_before["drift_analysis"]["last_run"] is None
        assert status_before["drift_analysis"]["due"] is True

        scheduler.run_cycle("drift_analysis")
        status_after = {s["name"]: s for s in scheduler.status()}
        assert status_after["drift_analysis"]["last_run"] is not None
        assert status_after["drift_analysis"]["due"] is False


class TestScheduleEndpoints:
    """Exercises the /research/schedule* routes against the real server app,
    overriding admin auth (matches the pattern used for other admin-gated
    endpoints in this server)."""

    def test_endpoints_list_status_run_history(self) -> None:
        from starlette.testclient import TestClient

        from trainer.research_lab import research_lab_server as server_mod

        app = server_mod.app
        app.dependency_overrides[server_mod._require_admin] = lambda: None
        try:
            client = TestClient(app)

            resp = client.get("/research/schedule")
            assert resp.status_code == 200
            names = {c["name"] for c in resp.json()["cycles"]}
            assert "performance_review" in names

            resp = client.get("/research/schedule/status")
            assert resp.status_code == 200
            assert len(resp.json()["status"]) == len(names)

            resp = client.post("/research/schedule/run/performance_review")
            assert resp.status_code == 200
            assert resp.json()["run"]["status"] == "COMPLETED"

            resp = client.post("/research/schedule/run/not_a_real_cycle")
            assert resp.status_code == 404

            resp = client.get("/research/schedule/history?cycle_name=performance_review")
            assert resp.status_code == 200
            assert resp.json()["total"] >= 1
        finally:
            app.dependency_overrides.pop(server_mod._require_admin, None)
            server_mod._db._conn.execute(
                "DELETE FROM research_cycle_runs WHERE cycle_name = 'performance_review'"
            )
            server_mod._db._conn.commit()


class TestConcurrency:
    def test_concurrent_run_due_cycles_does_not_duplicate(
        self, scheduler: ResearchScheduler, db: ResearchLabDatabase
    ) -> None:
        """Two threads racing run_due_cycles() at the same instant must not
        both execute the same never-run cycle."""
        import threading as _threading

        errors: list[Exception] = []

        def _worker() -> None:
            try:
                scheduler.run_due_cycles()
            except Exception as exc:  # pragma: no cover - surfaced via errors list
                errors.append(exc)

        threads = [_threading.Thread(target=_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        for cycle in scheduler.list_cycles():
            runs = db.list_research_cycle_runs(cycle_name=cycle["name"], limit=100)
            assert len(runs) == 1, f"{cycle['name']} ran {len(runs)} times, expected 1"


class TestPollIntervalParsing:
    def _parse(self, raw):
        from trainer.research_lab.research_lab_server import _parse_scheduler_poll_interval
        return _parse_scheduler_poll_interval(raw)

    def test_none_uses_default(self) -> None:
        assert self._parse(None) == 3600.0

    def test_empty_string_uses_default(self) -> None:
        assert self._parse("") == 3600.0

    def test_valid_value_used(self) -> None:
        assert self._parse("120") == 120.0

    def test_non_numeric_falls_back(self) -> None:
        assert self._parse("not-a-number") == 3600.0

    def test_zero_falls_back(self) -> None:
        assert self._parse("0") == 3600.0

    def test_negative_falls_back(self) -> None:
        assert self._parse("-5") == 3600.0

    def test_nan_falls_back(self) -> None:
        assert self._parse("nan") == 3600.0

    def test_positive_infinity_falls_back(self) -> None:
        assert self._parse("inf") == 3600.0

    def test_negative_infinity_falls_back(self) -> None:
        assert self._parse("-inf") == 3600.0


class TestBackgroundThread:
    def test_start_and_stop_background_runs_cycles(
        self, scheduler: ResearchScheduler, db: ResearchLabDatabase
    ) -> None:
        scheduler.start_background()
        try:
            deadline = time.time() + 3.0
            while time.time() < deadline:
                if db.list_research_cycle_runs(limit=1):
                    break
                time.sleep(0.05)
            runs = db.list_research_cycle_runs(limit=100)
            assert runs, "background thread should have run at least one cycle"
        finally:
            scheduler.stop_background()
        assert scheduler._thread is None

    def test_stop_background_without_start_is_safe(
        self, scheduler: ResearchScheduler
    ) -> None:
        scheduler.stop_background()  # must not raise
