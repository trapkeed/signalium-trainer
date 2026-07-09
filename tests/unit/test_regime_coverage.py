"""S22B — Tests for Market Regime Coverage Detection.

Covers:
  - Pure helpers: _regime_histogram, detect_unseen_conditions,
    detect_distribution_shift, detect_black_swans
  - build_regime_coverage_report end-to-end
  - RegimeCoverageReport persistence + querying via ResearchLabDatabase
  - RegimeCoverageMonitor.compute_report end-to-end
  - check_unseen_conditions / check_distribution_shift / check_black_swans:
    no alert / WARNING / CRITICAL / insufficient data
  - Wiring into ResearchLabHealthMonitor's alerting path
    (UNSEEN_CONDITION, BLACK_SWAN dimensions)
  - Server endpoints: /regime-coverage/compute, /latest, /reports
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from tests.unit._sso_test_utils import DEFAULT_TEST_JWT_SECRET, make_signalium_jwt
from trainer.research_lab.health_monitor import (
    DriftDimension,
    HealthMonitorConfig,
    ResearchLabHealthMonitor,
)
from trainer.research_lab.regime_coverage import (
    RegimeCoverageConfig,
    RegimeCoverageMonitor,
    RegimeCoverageReport,
    build_regime_coverage_report,
    check_black_swans,
    check_distribution_shift,
    check_unseen_conditions,
    detect_black_swans,
    detect_distribution_shift,
    detect_unseen_conditions,
)
from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.telemetry_warehouse import TelemetryWarehouse

_NOW = time.time()


@pytest.fixture()
def db(tmp_path: Path) -> ResearchLabDatabase:
    return ResearchLabDatabase(str(tmp_path / "regime_test.db"))


@pytest.fixture()
def warehouse(db: ResearchLabDatabase) -> TelemetryWarehouse:
    return TelemetryWarehouse(db)


def _trade(ts: float, regime: str, pnl: float, symbol: str = "XAUUSD") -> dict:
    return {
        "timestamp": ts,
        "regime": regime,
        "pnl": pnl,
        "symbol": symbol,
        "model_version": "v1.0",
    }


def _stable_trades(n: int, ts_start: float, regime: str = "ranging") -> list[dict]:
    return [_trade(ts_start + i * 60, regime, 10.0 if i % 2 == 0 else -8.0) for i in range(n)]


# ── Pure helpers ──────────────────────────────────────────────────────────────


class TestDetectUnseenConditions:
    def test_empty(self) -> None:
        result = detect_unseen_conditions([], None)
        assert result["unseen_rate"] == 0.0

    def test_all_known(self) -> None:
        records = _stable_trades(10, _NOW, regime="ranging")
        result = detect_unseen_conditions(records, None)
        assert result["unseen_rate"] == 0.0

    def test_detects_unseen_regime(self) -> None:
        records = _stable_trades(8, _NOW, regime="ranging") + [
            _trade(_NOW + 500, "flash_crash_cascade", -100.0) for _ in range(2)
        ]
        result = detect_unseen_conditions(records, None)
        assert result["unseen_rate"] == pytest.approx(0.2)
        assert "flash_crash_cascade" in result["unseen_regimes"]

    def test_custom_known_regimes_override(self) -> None:
        records = _stable_trades(10, _NOW, regime="my_custom_regime")
        # Not in DEFAULT_KNOWN_REGIMES -> unseen by default
        assert detect_unseen_conditions(records, None)["unseen_rate"] == 1.0
        # But recognized when caller supplies an override
        result = detect_unseen_conditions(records, frozenset({"my_custom_regime"}))
        assert result["unseen_rate"] == 0.0


class TestTvDistanceParityWithHealthMonitor:
    """regime_coverage._tv_distance must stay numerically identical to
    health_monitor's private _tv_distance (both compute MARKET-dimension-style
    drift over the same telemetry) — verified directly rather than assumed."""

    def test_formula_matches_health_monitor_private_helper(self) -> None:
        from trainer.research_lab.health_monitor import _tv_distance as hm_tv_distance
        from trainer.research_lab.regime_coverage import _tv_distance as rc_tv_distance

        cases = [
            ({}, {}),
            ({"a": 1.0}, {"a": 1.0}),
            ({"a": 0.6, "b": 0.4}, {"a": 0.3, "b": 0.7}),
            ({"a": 1.0}, {"b": 1.0}),
            ({"a": 0.5, "b": 0.3, "c": 0.2}, {"a": 0.5, "c": 0.5}),
        ]
        for a, b in cases:
            assert rc_tv_distance(a, b) == pytest.approx(hm_tv_distance(a, b))


class TestRegimeHistogramCasingParityWithHealthMonitor:
    """regime_coverage's histogram (feeding persisted reports) and
    health_monitor's MARKET-dimension histogram (feeding live DriftAlerts)
    must normalize regime-label casing identically, or the same telemetry
    can silently disagree between the two — a real found-and-fixed bug
    (health_monitor's MARKET histogram wasn't lowercasing) rather than a
    hypothetical."""

    def test_mixed_case_regimes_do_not_manufacture_shift_in_either_path(self) -> None:
        from trainer.research_lab.health_monitor import _regime_histogram as hm_hist
        from trainer.research_lab.regime_coverage import _regime_histogram as rc_hist

        baseline = [{"regime": "RANGING"} for _ in range(10)]
        recent = [{"regime": "ranging"} for _ in range(10)]
        # Same underlying regime, different casing — both paths must treat
        # this as zero distribution shift.
        assert hm_hist(baseline) == hm_hist(recent)
        assert rc_hist(baseline) == rc_hist(recent)


class TestDetectDistributionShift:
    def test_empty_returns_zero(self) -> None:
        result = detect_distribution_shift([], [])
        assert result["tv_distance"] == 0.0

    def test_identical_distributions_zero_shift(self) -> None:
        baseline = _stable_trades(10, _NOW - 20 * 60, regime="ranging")
        recent = _stable_trades(10, _NOW - 10 * 60, regime="ranging")
        result = detect_distribution_shift(baseline, recent)
        assert result["tv_distance"] == pytest.approx(0.0)

    def test_full_shift(self) -> None:
        baseline = _stable_trades(10, _NOW - 20 * 60, regime="ranging")
        recent = _stable_trades(10, _NOW - 10 * 60, regime="trending_strong")
        result = detect_distribution_shift(baseline, recent)
        assert result["tv_distance"] == pytest.approx(1.0)


class TestDetectBlackSwans:
    def test_insufficient_baseline_returns_zero(self) -> None:
        result = detect_black_swans([_trade(_NOW, "ranging", 10.0)], [], 4.0)
        assert result["black_swan_rate"] == 0.0

    def test_zero_variance_baseline_returns_zero(self) -> None:
        baseline = [_trade(_NOW - 100, "ranging", 10.0) for _ in range(5)]
        recent = [_trade(_NOW, "ranging", 10.0)]
        result = detect_black_swans(recent, baseline, 4.0)
        assert result["black_swan_rate"] == 0.0

    def test_detects_extreme_move(self) -> None:
        baseline = _stable_trades(30, _NOW - 60 * 60, regime="ranging")  # mean~1, small std
        recent = baseline[-5:] + [_trade(_NOW, "ranging", -5000.0)]
        result = detect_black_swans(recent, baseline, 4.0)
        assert result["black_swan_rate"] > 0.0
        assert len(result["black_swan_events"]) >= 1
        assert abs(result["black_swan_events"][-1]["zscore"]) >= 4.0

    def test_baseline_not_contaminated_by_recent_extreme(self) -> None:
        # The extreme move must be scored against the baseline's own
        # mean/std, not a combined set that includes the extreme value
        # itself (which would suppress its own z-score).
        baseline = _stable_trades(30, _NOW - 60 * 60, regime="ranging")
        recent = [_trade(_NOW, "ranging", -5000.0)]
        result = detect_black_swans(recent, baseline, 4.0)
        assert result["black_swan_rate"] == 1.0


# ── build_regime_coverage_report ─────────────────────────────────────────────


class TestBuildRegimeCoverageReport:
    def _cfg(self, **overrides) -> RegimeCoverageConfig:
        base = dict(baseline_window=20, rolling_window=10, min_samples=5)
        base.update(overrides)
        return RegimeCoverageConfig(**base)

    def test_empty_records(self) -> None:
        report = build_regime_coverage_report([], self._cfg())
        assert report.n_records == 0
        assert report.unseen_conditions["unseen_rate"] == 0.0

    def test_insufficient_data_for_split_still_checks_unseen_and_black_swan(self) -> None:
        records = _stable_trades(3, _NOW)
        report = build_regime_coverage_report(records, self._cfg())
        assert report.n_records == 3
        assert report.distribution_shift["tv_distance"] == 0.0

    def test_full_report_with_all_three_conditions(self) -> None:
        baseline = _stable_trades(20, _NOW - 30 * 60, regime="ranging")
        recent = _stable_trades(8, _NOW - 5 * 60, regime="trending_strong") + [
            _trade(_NOW, "flash_crash_cascade", -9999.0)
        ]
        records = baseline + recent
        report = build_regime_coverage_report(records, self._cfg())
        assert report.unseen_conditions["unseen_rate"] > 0.0
        assert report.distribution_shift["tv_distance"] > 0.0
        assert report.black_swans["black_swan_rate"] > 0.0

    def test_to_dict_from_dict_roundtrip(self) -> None:
        report = RegimeCoverageReport(
            report_id="regc-1",
            model_version="v1.0",
            created_at=_NOW,
            window_start=_NOW - 100,
            window_end=_NOW,
            n_records=5,
            unseen_conditions={"unseen_rate": 0.1},
            distribution_shift={"tv_distance": 0.3},
            black_swans={"black_swan_rate": 0.05},
        )
        d = report.to_dict()
        restored = RegimeCoverageReport.from_dict(d)
        assert restored.report_id == "regc-1"
        assert restored.unseen_conditions["unseen_rate"] == pytest.approx(0.1)


# ── DB persistence ────────────────────────────────────────────────────────────


class TestRegimeCoverageReportDB:
    def test_upsert_and_get(self, db: ResearchLabDatabase) -> None:
        db.upsert_regime_coverage_report(
            {"report_id": "r1", "model_version": "v1.0", "created_at": _NOW, "n_records": 3}
        )
        fetched = db.get_regime_coverage_report("r1")
        assert fetched is not None
        assert fetched["n_records"] == 3

    def test_get_nonexistent_returns_none(self, db: ResearchLabDatabase) -> None:
        assert db.get_regime_coverage_report("missing") is None

    def test_list_filter_model_version(self, db: ResearchLabDatabase) -> None:
        db.upsert_regime_coverage_report(
            {"report_id": "r1", "model_version": "v1.0", "created_at": _NOW}
        )
        db.upsert_regime_coverage_report(
            {"report_id": "r2", "model_version": "v2.0", "created_at": _NOW}
        )
        results = db.list_regime_coverage_reports(model_version="v1.0")
        assert len(results) == 1
        assert results[0]["report_id"] == "r1"

    def test_upsert_overwrites(self, db: ResearchLabDatabase) -> None:
        db.upsert_regime_coverage_report(
            {"report_id": "r1", "model_version": "v1.0", "created_at": _NOW, "n_records": 1}
        )
        db.upsert_regime_coverage_report(
            {"report_id": "r1", "model_version": "v1.0", "created_at": _NOW, "n_records": 99}
        )
        assert db.get_regime_coverage_report("r1")["n_records"] == 99


# ── RegimeCoverageMonitor (end-to-end via warehouse) ─────────────────────────


class TestRegimeCoverageMonitor:
    def _seed(self, warehouse: TelemetryWarehouse, trades: list[dict]) -> None:
        for t in trades:
            warehouse.ingest_trade(
                direction="LONG",
                entry_price=1.0,
                exit_price=1.0,
                size=1.0,
                pnl_pct=0.0,
                hold_duration_s=60.0,
                **t,
            )

    def test_compute_report_persists_and_is_queryable(self, db, warehouse) -> None:
        monitor = RegimeCoverageMonitor(db, warehouse)
        self._seed(warehouse, _stable_trades(20, _NOW - 20 * 60))
        report = monitor.compute_report(model_version="v1.0")
        assert report.n_records == 20

        latest = monitor.get_latest_report(model_version="v1.0")
        assert latest is not None
        assert latest["report_id"] == report.report_id

    def test_compute_report_empty_warehouse(self, db, warehouse) -> None:
        monitor = RegimeCoverageMonitor(db, warehouse)
        report = monitor.compute_report(model_version="v1.0")
        assert report.n_records == 0

    def test_list_reports_respects_limit(self, db, warehouse) -> None:
        monitor = RegimeCoverageMonitor(db, warehouse)
        for _ in range(3):
            monitor.compute_report(model_version="v1.0")
        assert len(monitor.list_reports(model_version="v1.0", limit=2)) == 2

    def test_list_reports_filters_by_time_window(self, db, warehouse) -> None:
        """Persisted findings must be queryable per time window, not just
        model version — list_reports/list_regime_coverage_reports must
        accept and apply since/until against each report's created_at."""
        monitor = RegimeCoverageMonitor(db, warehouse)
        db.upsert_regime_coverage_report(
            {"report_id": "old", "model_version": "v1.0", "created_at": _NOW - 10_000}
        )
        db.upsert_regime_coverage_report(
            {"report_id": "new", "model_version": "v1.0", "created_at": _NOW}
        )
        only_new = monitor.list_reports(model_version="v1.0", since=_NOW - 100)
        assert [r["report_id"] for r in only_new] == ["new"]

        only_old = monitor.list_reports(model_version="v1.0", until=_NOW - 100)
        assert [r["report_id"] for r in only_old] == ["old"]


# ── check_* pure functions ────────────────────────────────────────────────────


class TestCheckUnseenConditions:
    def _cfg(self, **overrides) -> RegimeCoverageConfig:
        base = dict(
            min_samples=5, unseen_condition_rate_warning=0.1, unseen_condition_rate_critical=0.3
        )
        base.update(overrides)
        return RegimeCoverageConfig(**base)

    def test_no_alert_when_all_known(self) -> None:
        records = _stable_trades(10, _NOW, regime="ranging")
        assert check_unseen_conditions(records, self._cfg()) is None

    def test_warning_when_moderate_unseen(self) -> None:
        records = _stable_trades(8, _NOW, regime="ranging") + [
            _trade(_NOW + 500, "novel_regime", 1.0)
        ]
        result = check_unseen_conditions(records, self._cfg())
        assert result is not None
        assert result["severity"] == "WARNING"
        assert result["kind"] == "UNSEEN_CONDITION"

    def test_critical_when_severe_unseen(self) -> None:
        records = _stable_trades(5, _NOW, regime="ranging") + [
            _trade(_NOW + 500 + i, "novel_regime", 1.0) for i in range(5)
        ]
        result = check_unseen_conditions(records, self._cfg())
        assert result is not None
        assert result["severity"] == "CRITICAL"

    def test_insufficient_data_returns_none(self) -> None:
        assert check_unseen_conditions([], self._cfg()) is None


class TestCheckDistributionShift:
    def _cfg(self, **overrides) -> RegimeCoverageConfig:
        base = dict(
            baseline_window=20,
            rolling_window=10,
            min_samples=5,
            distribution_shift_warning=0.3,
            distribution_shift_critical=0.6,
        )
        base.update(overrides)
        return RegimeCoverageConfig(**base)

    def test_no_alert_when_stable(self) -> None:
        records = _stable_trades(30, _NOW - 30 * 60, regime="ranging")
        assert check_distribution_shift(records, self._cfg()) is None

    def test_alert_when_shifted(self) -> None:
        baseline = _stable_trades(20, _NOW - 30 * 60, regime="ranging")
        recent = _stable_trades(10, _NOW - 10 * 60, regime="trending_strong")
        result = check_distribution_shift(baseline + recent, self._cfg())
        assert result is not None
        assert result["kind"] == "DISTRIBUTION_SHIFT"

    def test_insufficient_data_returns_none(self) -> None:
        assert check_distribution_shift(_stable_trades(3, _NOW), self._cfg()) is None


class TestCheckBlackSwans:
    def _cfg(self, **overrides) -> RegimeCoverageConfig:
        base = dict(
            baseline_window=20,
            rolling_window=10,
            min_samples=5,
            black_swan_zscore_threshold=4.0,
            black_swan_rate_warning=0.05,
            black_swan_rate_critical=0.5,
        )
        base.update(overrides)
        return RegimeCoverageConfig(**base)

    def test_no_alert_when_stable(self) -> None:
        records = _stable_trades(30, _NOW - 30 * 60, regime="ranging")
        assert check_black_swans(records, self._cfg()) is None

    def test_alert_on_extreme_move(self) -> None:
        baseline = _stable_trades(20, _NOW - 30 * 60, regime="ranging")
        recent = [_trade(_NOW - 100 + i, "ranging", 10.0) for i in range(9)] + [
            _trade(_NOW, "ranging", -50000.0)
        ]
        result = check_black_swans(baseline + recent, self._cfg())
        assert result is not None
        assert result["kind"] == "BLACK_SWAN"

    def test_insufficient_data_returns_none(self) -> None:
        assert check_black_swans(_stable_trades(3, _NOW), self._cfg()) is None


# ── Wired into ResearchLabHealthMonitor ──────────────────────────────────────


class TestRegimeCoverageHealthMonitorWiring:
    def _cfg(self, **overrides) -> HealthMonitorConfig:
        base = dict(
            baseline_window=20,
            rolling_window=10,
            min_samples=5,
            unseen_condition_rate_warning=0.1,
            unseen_condition_rate_critical=0.3,
            black_swan_rate_warning=0.05,
            black_swan_rate_critical=0.5,
        )
        base.update(overrides)
        return HealthMonitorConfig(**base)

    def test_no_alert_when_stable(self, db, warehouse) -> None:
        monitor = ResearchLabHealthMonitor(db, warehouse, config=self._cfg())
        decisions = [
            {"timestamp": _NOW - i * 60, "regime": "ranging", "model_version": "v1.0"}
            for i in range(10)
        ]
        assert monitor._check_unseen_condition(decisions, "v1.0") is None

    def test_alert_on_unseen_condition(self, db, warehouse) -> None:
        monitor = ResearchLabHealthMonitor(db, warehouse, config=self._cfg())
        decisions = [
            {"timestamp": _NOW - i * 60, "regime": "ranging", "model_version": "v1.0"}
            for i in range(5)
        ] + [
            {"timestamp": _NOW + i, "regime": "novel_regime", "model_version": "v1.0"}
            for i in range(5)
        ]
        alert = monitor._check_unseen_condition(decisions, "v1.0")
        assert alert is not None
        assert alert.dimension == DriftDimension.UNSEEN_CONDITION.value

    def test_alert_on_black_swan(self, db, warehouse) -> None:
        monitor = ResearchLabHealthMonitor(db, warehouse, config=self._cfg())
        baseline = _stable_trades(20, _NOW - 30 * 60, regime="ranging")
        recent = [_trade(_NOW - 100 + i, "ranging", 10.0) for i in range(9)] + [
            _trade(_NOW, "ranging", -50000.0)
        ]
        alert = monitor._check_black_swan(baseline + recent, "v1.0")
        assert alert is not None
        assert alert.dimension == DriftDimension.BLACK_SWAN.value

    def test_run_check_includes_new_dimensions(self, db, warehouse) -> None:
        monitor = ResearchLabHealthMonitor(db, warehouse, config=self._cfg())
        for t in _stable_trades(10, _NOW - 10 * 60):
            warehouse.ingest_trade(
                direction="LONG",
                entry_price=1.0,
                exit_price=1.0,
                size=1.0,
                pnl_pct=0.0,
                hold_duration_s=60.0,
                **t,
            )
        result = monitor.run_check(model_version="v1.0")
        checked = result.to_dict()["dimensions_checked"]
        assert DriftDimension.UNSEEN_CONDITION.value in checked
        assert DriftDimension.BLACK_SWAN.value in checked

    def test_critical_unseen_condition_creates_recommendation(self, db, warehouse) -> None:
        from unittest.mock import MagicMock

        rec_engine = MagicMock()
        rec_engine._db = db
        monitor = ResearchLabHealthMonitor(db, warehouse, rec_engine=rec_engine, config=self._cfg())
        decisions = [
            {"timestamp": _NOW - i, "regime": "novel_regime", "model_version": "v1.0"}
            for i in range(5)
        ]
        alert = monitor._check_unseen_condition(decisions, "v1.0")
        assert alert is not None
        assert alert.severity == "CRITICAL"
        rec_id = monitor._maybe_create_recommendation(alert)
        assert rec_id
        assert db.get_retraining_recommendation(rec_id) is not None


# ── Server endpoints ──────────────────────────────────────────────────────────


class TestRegimeCoverageServerEndpoints:
    _HDR = {
        "Authorization": "Bearer test-regime-token",
        "X-Signalium-Token": make_signalium_jwt(),
    }

    @pytest.fixture()
    def client(self, tmp_path: Path):
        import os

        os.environ["RESEARCH_LAB_ADMIN_TOKEN"] = "test-regime-token"
        os.environ["SIGNALIUM_JWT_SECRET"] = DEFAULT_TEST_JWT_SECRET
        os.environ["RESEARCH_LAB_DB_PATH"] = str(tmp_path / "server_regime.db")
        for mod_name in list(sys.modules):
            if "research_lab_server" in mod_name or "telemetry_server" in mod_name:
                del sys.modules[mod_name]
        from starlette.testclient import TestClient

        import trainer.research_lab.research_lab_server as srv_mod

        return TestClient(srv_mod.app)

    def test_compute_report_empty_warehouse(self, client) -> None:
        resp = client.post("/regime-coverage/compute?model_version=v1.0", headers=self._HDR)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["report"]["n_records"] == 0

    def test_latest_report_none_before_compute(self, client) -> None:
        resp = client.get("/regime-coverage/latest?model_version=v1.0", headers=self._HDR)
        assert resp.status_code == 200
        assert resp.json()["report"] is None

    def test_latest_report_after_compute(self, client) -> None:
        client.post("/regime-coverage/compute?model_version=v1.0", headers=self._HDR)
        resp = client.get("/regime-coverage/latest?model_version=v1.0", headers=self._HDR)
        assert resp.status_code == 200
        assert resp.json()["report"] is not None

    def test_list_reports(self, client) -> None:
        client.post("/regime-coverage/compute?model_version=v1.0", headers=self._HDR)
        client.post("/regime-coverage/compute?model_version=v1.0", headers=self._HDR)
        resp = client.get("/regime-coverage/reports?model_version=v1.0", headers=self._HDR)
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    def test_compute_requires_auth(self, client) -> None:
        resp = client.post("/regime-coverage/compute")
        assert resp.status_code == 401

    def test_latest_requires_auth(self, client) -> None:
        resp = client.get("/regime-coverage/latest")
        assert resp.status_code == 401

    def test_reports_requires_auth(self, client) -> None:
        resp = client.get("/regime-coverage/reports")
        assert resp.status_code == 401

    def test_list_reports_time_window_filtering_via_endpoint(self, client) -> None:
        """Validates the since/until query-param contract end-to-end through
        the HTTP endpoint, not just at the monitor/DB level."""
        import time as _time

        client.post("/regime-coverage/compute?model_version=v1.0", headers=self._HDR)
        cutoff = _time.time()
        client.post("/regime-coverage/compute?model_version=v1.0", headers=self._HDR)

        resp_since = client.get(
            f"/regime-coverage/reports?model_version=v1.0&since={cutoff}",
            headers=self._HDR,
        )
        assert resp_since.status_code == 200
        assert resp_since.json()["total"] == 1

        resp_until = client.get(
            f"/regime-coverage/reports?model_version=v1.0&until={cutoff}",
            headers=self._HDR,
        )
        assert resp_until.status_code == 200
        assert resp_until.json()["total"] == 1
