"""S22B — Tests for Feature Health Monitoring.

Covers:
  - Pure helpers: _is_corrupted, compute_feature_stats
  - build_feature_health_report: missing/stale/corrupted/delayed detection
  - FeatureHealthReport persistence + querying via ResearchLabDatabase
  - FeatureHealthMonitor.compute_report end-to-end
  - check_feature_health: no alert / WARNING / CRITICAL / insufficient data
  - Wiring into ResearchLabHealthMonitor's alerting path (FEATURE_HEALTH dimension)
  - Server endpoints: /feature-health/compute, /latest, /reports
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from trainer.research_lab.feature_health import (
    FeatureHealthConfig,
    FeatureHealthMonitor,
    FeatureHealthReport,
    _is_corrupted,
    build_feature_health_report,
    check_feature_health,
    compute_feature_stats,
)
from trainer.research_lab.health_monitor import DriftDimension, HealthMonitorConfig, ResearchLabHealthMonitor
from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.telemetry_warehouse import TelemetryWarehouse
from tests.unit._sso_test_utils import make_signalium_jwt, DEFAULT_TEST_JWT_SECRET

_NOW = time.time()


@pytest.fixture()
def db(tmp_path: Path) -> ResearchLabDatabase:
    return ResearchLabDatabase(str(tmp_path / "feat_test.db"))


@pytest.fixture()
def warehouse(db: ResearchLabDatabase) -> TelemetryWarehouse:
    return TelemetryWarehouse(db)


def _event(ts: float, feature_values: dict, ingested_at: float | None = None) -> dict:
    return {
        "timestamp": ts,
        "ingested_at": ingested_at if ingested_at is not None else ts,
        "model_version": "v1.0",
        "symbol": "XAUUSD",
        "feature_values": feature_values,
    }


def _healthy_events(n: int, ts_start: float) -> list[dict]:
    return [
        _event(ts_start + i * 60, {"atr": 5.0 + i * 0.1, "rsi": 50.0 + i})
        for i in range(n)
    ]


# ── Pure helpers ──────────────────────────────────────────────────────────────


class TestIsCorrupted:
    def test_valid_number(self) -> None:
        assert _is_corrupted(5.0, None) is False

    def test_bool_is_valid(self) -> None:
        assert _is_corrupted(True, None) is False

    def test_nan_is_corrupted(self) -> None:
        assert _is_corrupted(float("nan"), None) is True

    def test_inf_is_corrupted(self) -> None:
        assert _is_corrupted(float("inf"), None) is True

    def test_non_numeric_is_corrupted(self) -> None:
        assert _is_corrupted("not-a-number", None) is True
        assert _is_corrupted(None, None) is True

    def test_out_of_bounds(self) -> None:
        assert _is_corrupted(150.0, (0.0, 100.0)) is True
        assert _is_corrupted(50.0, (0.0, 100.0)) is False


class TestComputeFeatureStats:
    def test_empty_events(self) -> None:
        stats, start, end = compute_feature_stats([], FeatureHealthConfig())
        assert stats == {}
        assert start == 0.0
        assert end == 0.0

    def test_tracks_presence_and_window(self) -> None:
        events = _healthy_events(5, _NOW - 5 * 60)
        stats, start, end = compute_feature_stats(events, FeatureHealthConfig())
        assert stats["atr"].n_present == 5
        assert stats["rsi"].n_present == 5
        assert end > start

    def test_tracks_delay(self) -> None:
        events = [_event(_NOW, {"atr": 5.0}, ingested_at=_NOW + 10.0)]
        stats, _, _ = compute_feature_stats(events, FeatureHealthConfig())
        assert stats["atr"].delay_max == pytest.approx(10.0)

    def test_tracks_corruption(self) -> None:
        events = [
            _event(_NOW, {"atr": 5.0}),
            _event(_NOW + 60, {"atr": float("nan")}),
        ]
        stats, _, _ = compute_feature_stats(events, FeatureHealthConfig())
        assert stats["atr"].n_corrupted == 1
        assert stats["atr"].n_present == 2


# ── build_feature_health_report ──────────────────────────────────────────────


class TestBuildFeatureHealthReport:
    def test_no_missing_when_no_expected_features(self) -> None:
        events = _healthy_events(5, _NOW - 5 * 60)
        report = build_feature_health_report(events, FeatureHealthConfig())
        assert report.missing_features == []

    def test_detects_missing_features(self) -> None:
        events = _healthy_events(5, _NOW - 5 * 60)
        cfg = FeatureHealthConfig(expected_features=["atr", "rsi", "macd"])
        report = build_feature_health_report(events, cfg)
        assert report.missing_features == ["macd"]

    def test_detects_stale_feature(self) -> None:
        # "atr" stops appearing after the first half of the window.
        events = []
        for i in range(10):
            fv = {"atr": 5.0} if i < 5 else {}
            fv["rsi"] = 50.0
            events.append(_event(_NOW - (10 - i) * 3600, fv))
        cfg = FeatureHealthConfig(staleness_warning_s=3600 * 4)
        report = build_feature_health_report(events, cfg)
        assert "atr" in report.stale_features
        assert "rsi" not in report.stale_features

    def test_detects_corrupted_feature(self) -> None:
        events = [_event(_NOW - i * 60, {"atr": float("nan") if i < 3 else 5.0}) for i in range(10)]
        cfg = FeatureHealthConfig(corruption_rate_warning=0.2)
        report = build_feature_health_report(events, cfg)
        assert "atr" in report.corrupted_features

    def test_detects_delayed_feature(self) -> None:
        events = [_event(_NOW - i * 60, {"atr": 5.0}, ingested_at=_NOW - i * 60 + 60.0) for i in range(5)]
        cfg = FeatureHealthConfig(delay_warning_s=30.0)
        report = build_feature_health_report(events, cfg)
        assert "atr" in report.delayed_features

    def test_to_dict_from_dict_roundtrip(self) -> None:
        report = FeatureHealthReport(
            report_id="feat-1", model_version="v1.0", created_at=_NOW,
            window_start=_NOW - 100, window_end=_NOW, n_events=5,
            per_feature={"atr": {"n_present": 5}},
            missing_features=["macd"],
        )
        d = report.to_dict()
        restored = FeatureHealthReport.from_dict(d)
        assert restored.report_id == "feat-1"
        assert restored.missing_features == ["macd"]
        assert restored.per_feature["atr"]["n_present"] == 5


# ── DB persistence ────────────────────────────────────────────────────────────


class TestFeatureHealthReportDB:
    def test_upsert_and_get(self, db: ResearchLabDatabase) -> None:
        db.upsert_feature_health_report({"report_id": "f1", "model_version": "v1.0", "created_at": _NOW, "n_events": 3})
        fetched = db.get_feature_health_report("f1")
        assert fetched is not None
        assert fetched["n_events"] == 3

    def test_get_nonexistent_returns_none(self, db: ResearchLabDatabase) -> None:
        assert db.get_feature_health_report("missing") is None

    def test_list_filter_model_version(self, db: ResearchLabDatabase) -> None:
        db.upsert_feature_health_report({"report_id": "r1", "model_version": "v1.0", "created_at": _NOW})
        db.upsert_feature_health_report({"report_id": "r2", "model_version": "v2.0", "created_at": _NOW})
        results = db.list_feature_health_reports(model_version="v1.0")
        assert len(results) == 1
        assert results[0]["report_id"] == "r1"

    def test_upsert_overwrites(self, db: ResearchLabDatabase) -> None:
        db.upsert_feature_health_report({"report_id": "r1", "model_version": "v1.0", "created_at": _NOW, "n_events": 1})
        db.upsert_feature_health_report({"report_id": "r1", "model_version": "v1.0", "created_at": _NOW, "n_events": 99})
        assert db.get_feature_health_report("r1")["n_events"] == 99


# ── FeatureHealthMonitor (end-to-end via warehouse) ──────────────────────────


class TestFeatureHealthMonitor:
    def _seed(self, warehouse: TelemetryWarehouse, events: list[dict]) -> None:
        for e in events:
            warehouse.ingest_reasoning(
                model_version=e["model_version"], symbol=e["symbol"],
                timestamp=e["timestamp"], decision_id=f"dec-{e['timestamp']}",
                feature_values=e["feature_values"],
            )

    def test_compute_report_persists_and_is_queryable(self, db, warehouse) -> None:
        monitor = FeatureHealthMonitor(db, warehouse)
        self._seed(warehouse, _healthy_events(20, _NOW - 20 * 60))
        report = monitor.compute_report(model_version="v1.0")
        assert report.n_events == 20
        assert "atr" in report.per_feature

        latest = monitor.get_latest_report(model_version="v1.0")
        assert latest is not None
        assert latest["report_id"] == report.report_id

    def test_compute_report_empty_warehouse(self, db, warehouse) -> None:
        monitor = FeatureHealthMonitor(db, warehouse)
        report = monitor.compute_report(model_version="v1.0")
        assert report.n_events == 0
        assert report.per_feature == {}

    def test_list_reports_respects_limit(self, db, warehouse) -> None:
        monitor = FeatureHealthMonitor(db, warehouse)
        for _ in range(3):
            monitor.compute_report(model_version="v1.0")
        assert len(monitor.list_reports(model_version="v1.0", limit=2)) == 2


# ── check_feature_health (pure function) ─────────────────────────────────────


class TestCheckFeatureHealth:
    def test_no_alert_when_healthy(self) -> None:
        events = _healthy_events(20, _NOW - 20 * 60)
        result = check_feature_health(events, FeatureHealthConfig())
        assert result is None

    def test_alert_on_missing_feature_is_critical(self) -> None:
        events = _healthy_events(10, _NOW - 10 * 60)
        cfg = FeatureHealthConfig(expected_features=["atr", "rsi", "macd"])
        result = check_feature_health(events, cfg)
        assert result is not None
        assert result["severity"] == "CRITICAL"
        assert "missing:macd" in result["triggers"]
        assert "macd" in result["evidence"]["missing_features"]

    def test_warning_on_moderate_corruption(self) -> None:
        events = [_event(_NOW - i * 60, {"atr": float("nan") if i < 1 else 5.0}) for i in range(10)]
        cfg = FeatureHealthConfig(corruption_rate_warning=0.05, corruption_rate_critical=0.50)
        result = check_feature_health(events, cfg)
        assert result is not None
        assert result["severity"] == "WARNING"

    def test_critical_on_severe_corruption(self) -> None:
        events = [_event(_NOW - i * 60, {"atr": float("nan") if i < 5 else 5.0}) for i in range(10)]
        cfg = FeatureHealthConfig(corruption_rate_warning=0.05, corruption_rate_critical=0.30)
        result = check_feature_health(events, cfg)
        assert result is not None
        assert result["severity"] == "CRITICAL"

    def test_no_data_returns_none(self) -> None:
        assert check_feature_health([], FeatureHealthConfig()) is None

    def test_intermittent_missingness_flagged_even_when_feature_not_expected(self) -> None:
        """A feature present in the observed set (so the expected-vs-observed
        set difference can't catch it) but missing from most events must
        still be flagged — this is the "missing values" case distinct from
        "never observed at all"."""
        events = [
            _event(_NOW - i * 60, {"atr": 5.0, **({"rsi": 50.0} if i % 5 == 0 else {})})
            for i in range(10)
        ]
        cfg = FeatureHealthConfig(missing_rate_warning=0.95, missing_rate_critical=0.80)
        result = check_feature_health(events, cfg)
        assert result is not None
        assert "intermittent_missing:rsi" in result["triggers"]
        assert "rsi" in result["evidence"]["missing_features"]
        # rsi present in only 2/10 events (20% presence) — well under the
        # critical threshold.
        assert result["severity"] == "CRITICAL"

    def test_intermittent_missingness_warning_not_critical(self) -> None:
        events = [
            _event(_NOW - i * 60, {"atr": 5.0, **({} if i == 0 else {"rsi": 50.0})})
            for i in range(10)
        ]
        cfg = FeatureHealthConfig(missing_rate_warning=0.95, missing_rate_critical=0.80)
        result = check_feature_health(events, cfg)
        assert result is not None
        assert "intermittent_missing:rsi" in result["triggers"]
        # rsi present in 9/10 events (90%) — below warning (95%) but above
        # critical (80%).
        assert result["severity"] == "WARNING"

    def test_stream_stale_flagged_against_wall_clock_not_window(self) -> None:
        """The whole reasoning stream can go dark while every feature inside
        the (old) window still looks perfectly fresh relative to itself —
        this requires an explicit wall-clock `now` to detect."""
        old_window_end = _NOW - 20_000  # window itself is old
        events = _healthy_events(5, old_window_end - 5 * 60)
        cfg = FeatureHealthConfig(
            stream_staleness_warning_s=3600.0, stream_staleness_critical_s=14400.0
        )
        # Per-feature staleness relative to the window itself is zero for
        # the most recent event — only the wall-clock check can catch this.
        result_no_now = check_feature_health(events, cfg, now=old_window_end + 1.0)
        assert result_no_now is None

        result_stale = check_feature_health(events, cfg, now=_NOW)
        assert result_stale is not None
        assert "stream_stale" in result_stale["triggers"]
        assert result_stale["severity"] == "CRITICAL"
        assert result_stale["evidence"]["stream_staleness_s"] >= 14400.0

    def test_stream_staleness_wired_into_health_monitor(self, db, warehouse) -> None:
        from trainer.research_lab.health_monitor import (
            HealthMonitorConfig,
            ResearchLabHealthMonitor,
        )

        cfg = HealthMonitorConfig(
            feature_health_stream_staleness_warning_s=3600.0,
            feature_health_stream_staleness_critical_s=14400.0,
        )
        monitor = ResearchLabHealthMonitor(db, warehouse, config=cfg)
        old_events = _healthy_events(5, _NOW - 20_000)
        alert = monitor._check_feature_health(old_events, "v1.0", now=_NOW)
        assert alert is not None
        assert "stream_stale" in alert.affected_features


# ── Wired into ResearchLabHealthMonitor ──────────────────────────────────────


class TestFeatureHealthMonitorWiring:
    def test_no_alert_when_healthy(self, db, warehouse) -> None:
        monitor = ResearchLabHealthMonitor(db, warehouse, config=HealthMonitorConfig())
        events = _healthy_events(20, _NOW - 20 * 60)
        alert = monitor._check_feature_health(events, "v1.0")
        assert alert is None

    def test_alert_on_missing_feature(self, db, warehouse) -> None:
        cfg = HealthMonitorConfig(feature_health_expected_features=["atr", "rsi", "macd"])
        monitor = ResearchLabHealthMonitor(db, warehouse, config=cfg)
        events = _healthy_events(10, _NOW - 10 * 60)
        alert = monitor._check_feature_health(events, "v1.0")
        assert alert is not None
        assert alert.dimension == DriftDimension.FEATURE_HEALTH.value
        assert alert.severity == "CRITICAL"

    def test_run_check_includes_feature_health_dimension(self, db, warehouse) -> None:
        monitor = ResearchLabHealthMonitor(db, warehouse, config=HealthMonitorConfig())
        for e in _healthy_events(10, _NOW - 10 * 60):
            warehouse.ingest_reasoning(
                model_version="v1.0", symbol="XAUUSD",
                timestamp=e["timestamp"], decision_id=f"dec-{e['timestamp']}",
                feature_values=e["feature_values"],
            )
        result = monitor.run_check(model_version="v1.0")
        assert DriftDimension.FEATURE_HEALTH.value in result.to_dict()["dimensions_checked"]

    def test_critical_alert_creates_recommendation_without_colliding_with_data_quality(
        self, db, warehouse
    ) -> None:
        from unittest.mock import MagicMock
        rec_engine = MagicMock()
        rec_engine._db = db
        cfg = HealthMonitorConfig(feature_health_expected_features=["atr", "rsi", "macd"])
        monitor = ResearchLabHealthMonitor(db, warehouse, rec_engine=rec_engine, config=cfg)
        events = _healthy_events(10, _NOW - 10 * 60)
        alert = monitor._check_feature_health(events, "v1.0")
        assert alert is not None
        rec_id = monitor._maybe_create_recommendation(alert)
        assert rec_id
        assert db.get_retraining_recommendation(rec_id) is not None

    def test_no_dedup_collision_with_feature_dimension_sharing_reasoning_drift(
        self, db, warehouse
    ) -> None:
        # FEATURE and FEATURE_HEALTH both map to EvaluationDimension.REASONING_DRIFT
        # (see the comment in health_monitor.py's _DIM_TO_EVAL). A CRITICAL alert
        # from one must not silently suppress a distinct CRITICAL alert from the
        # other for the same model_version — dedup must key on the fine-grained
        # `_health_dimension`, not the coarse EvaluationDimension alone.
        from unittest.mock import MagicMock

        from trainer.research_lab.health_monitor import AlertSeverity, DriftAlert

        rec_engine = MagicMock()
        rec_engine._db = db
        monitor = ResearchLabHealthMonitor(db, warehouse, rec_engine=rec_engine)

        feature_alert = DriftAlert(
            alert_id="hal-feature-1",
            dimension=DriftDimension.FEATURE.value,
            severity=AlertSeverity.CRITICAL.value,
            model_version="v1.0",
            affected_symbols=[],
            affected_features=[],
            evidence={},
            recommended_action="feature distribution shifted",
            created_at=_NOW,
        )
        feature_health_alert = DriftAlert(
            alert_id="hal-feature-health-1",
            dimension=DriftDimension.FEATURE_HEALTH.value,
            severity=AlertSeverity.CRITICAL.value,
            model_version="v1.0",
            affected_symbols=[],
            affected_features=["missing_features"],
            evidence={},
            recommended_action="features missing",
            created_at=_NOW,
        )

        rec_id_1 = monitor._maybe_create_recommendation(feature_alert)
        rec_id_2 = monitor._maybe_create_recommendation(feature_health_alert)

        assert rec_id_1 and rec_id_2
        assert rec_id_1 != rec_id_2
        assert db.get_retraining_recommendation(rec_id_1) is not None
        assert db.get_retraining_recommendation(rec_id_2) is not None


# ── Server endpoints ──────────────────────────────────────────────────────────


class TestFeatureHealthServerEndpoints:
    _HDR = {
        "Authorization": "Bearer test-feat-token",
        "X-Signalium-Token": make_signalium_jwt(),
    }

    @pytest.fixture()
    def client(self, tmp_path: Path):
        import os
        os.environ["RESEARCH_LAB_ADMIN_TOKEN"] = "test-feat-token"
        os.environ["SIGNALIUM_JWT_SECRET"] = DEFAULT_TEST_JWT_SECRET
        os.environ["RESEARCH_LAB_DB_PATH"] = str(tmp_path / "server_feat.db")
        for mod_name in list(sys.modules):
            if "research_lab_server" in mod_name or "telemetry_server" in mod_name:
                del sys.modules[mod_name]
        import trainer.research_lab.research_lab_server as srv_mod
        from starlette.testclient import TestClient
        return TestClient(srv_mod.app)

    def test_compute_report_empty_warehouse(self, client) -> None:
        resp = client.post("/feature-health/compute?model_version=v1.0", headers=self._HDR)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["report"]["n_events"] == 0

    def test_latest_report_none_before_compute(self, client) -> None:
        resp = client.get("/feature-health/latest?model_version=v1.0", headers=self._HDR)
        assert resp.status_code == 200
        assert resp.json()["report"] is None

    def test_latest_report_after_compute(self, client) -> None:
        client.post("/feature-health/compute?model_version=v1.0", headers=self._HDR)
        resp = client.get("/feature-health/latest?model_version=v1.0", headers=self._HDR)
        assert resp.status_code == 200
        assert resp.json()["report"] is not None

    def test_list_reports(self, client) -> None:
        client.post("/feature-health/compute?model_version=v1.0", headers=self._HDR)
        client.post("/feature-health/compute?model_version=v1.0", headers=self._HDR)
        resp = client.get("/feature-health/reports?model_version=v1.0", headers=self._HDR)
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    def test_compute_requires_auth(self, client) -> None:
        resp = client.post("/feature-health/compute")
        assert resp.status_code == 401

    def test_latest_requires_auth(self, client) -> None:
        resp = client.get("/feature-health/latest")
        assert resp.status_code == 401

    def test_reports_requires_auth(self, client) -> None:
        resp = client.get("/feature-health/reports")
        assert resp.status_code == 401
