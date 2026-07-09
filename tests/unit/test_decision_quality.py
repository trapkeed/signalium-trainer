"""S22B — Tests for Decision Quality Monitoring.

Covers:
  - Confidence-vs-outcome correlation helper
  - Expectancy stats + grouping helpers (regime/symbol/session/volatility)
  - Volatility bucketing (explicit key vs ATR-derived vs unknown)
  - DecisionQualityReport persistence + querying via ResearchLabDatabase
  - DecisionQualityTracker.compute_report end-to-end
  - check_decision_quality_drift: no alert / WARNING / CRITICAL / insufficient data
  - Wiring into ResearchLabHealthMonitor's alerting path (DECISION_QUALITY dimension)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from tests.unit._sso_test_utils import DEFAULT_TEST_JWT_SECRET, make_signalium_jwt
from trainer.research_lab.decision_quality import (
    DecisionQualityConfig,
    DecisionQualityReport,
    DecisionQualityTracker,
    _expectancy_stats,
    _group_expectancy,
    _volatility_bucket,
    check_decision_quality_drift,
    confidence_outcome_correlation,
)
from trainer.research_lab.dimension_evaluators import _split_windows
from trainer.research_lab.health_monitor import (
    DriftDimension,
    HealthMonitorConfig,
    ResearchLabHealthMonitor,
)
from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.telemetry_warehouse import TelemetryWarehouse

_NOW = time.time()


@pytest.fixture()
def db(tmp_path: Path) -> ResearchLabDatabase:
    return ResearchLabDatabase(str(tmp_path / "dq_test.db"))


@pytest.fixture()
def warehouse(db: ResearchLabDatabase) -> TelemetryWarehouse:
    return TelemetryWarehouse(db)


def _well_calibrated_trades(n: int, ts_start: float, **extra) -> list[dict]:
    out = []
    for i in range(n):
        win = (i % 10) < 7
        row = {
            "confidence_at_entry": 0.8 if win else 0.3,
            "pnl": 10.0 if win else -8.0,
            "timestamp": ts_start + i * 60,
            "regime": "RANGING",
            "symbol": "XAUUSD",
            "session": "LONDON",
            "market_conditions": {"atr": 5.0},
            "model_version": "v1.0",
        }
        row.update(extra)
        out.append(row)
    return out


def _degraded_trades(n: int, ts_start: float, **extra) -> list[dict]:
    """High confidence, poor/uncorrelated outcomes, negative expectancy."""
    out = []
    for i in range(n):
        win = (i % 10) < 2  # only 20% win rate, expectancy goes negative
        row = {
            "confidence_at_entry": 0.85,  # stays high regardless of outcome
            "pnl": 5.0 if win else -12.0,
            "timestamp": ts_start + i * 60,
            "regime": "TRENDING",
            "symbol": "EURUSD",
            "session": "NY",
            "market_conditions": {"atr": 30.0},
            "model_version": "v1.0",
        }
        row.update(extra)
        out.append(row)
    return out


# ── Pure helpers ──────────────────────────────────────────────────────────────


class TestExpectancyStats:
    def test_empty(self) -> None:
        stats = _expectancy_stats([])
        assert stats == {
            "n_trades": 0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    def test_mixed_trades(self) -> None:
        trades = [{"pnl": 10.0}, {"pnl": 10.0}, {"pnl": -5.0}]
        stats = _expectancy_stats(trades)
        assert stats["n_trades"] == 3
        assert stats["win_rate"] == pytest.approx(2 / 3, abs=1e-3)
        assert stats["avg_win"] == pytest.approx(10.0)
        assert stats["avg_loss"] == pytest.approx(-5.0)
        expected_exp = (2 / 3) * 10.0 + (1 / 3) * (-5.0)
        assert stats["expectancy"] == pytest.approx(expected_exp)

    def test_breakeven_counts_as_loss(self) -> None:
        # pnl == 0 should not count as a "win" (win_rate would overstate quality)
        stats = _expectancy_stats([{"pnl": 0.0}, {"pnl": 10.0}])
        assert stats["win_rate"] == pytest.approx(0.5)
        assert stats["avg_loss"] == pytest.approx(0.0)


class TestGroupExpectancy:
    def test_groups_by_key(self) -> None:
        trades = [
            {"pnl": 10.0, "regime": "TRENDING"},
            {"pnl": -5.0, "regime": "TRENDING"},
            {"pnl": 8.0, "regime": "RANGING"},
        ]
        groups = _group_expectancy(trades, lambda t: t.get("regime"))
        assert set(groups.keys()) == {"TRENDING", "RANGING"}
        assert groups["TRENDING"]["n_trades"] == 2
        assert groups["RANGING"]["n_trades"] == 1

    def test_missing_key_buckets_as_unknown(self) -> None:
        trades = [{"pnl": 1.0}]
        groups = _group_expectancy(trades, lambda t: t.get("symbol"))
        assert "UNKNOWN" in groups


class TestVolatilityBucket:
    cfg = DecisionQualityConfig(low_atr_threshold=10.0, high_atr_threshold=25.0)

    def test_explicit_key_wins(self) -> None:
        trade = {"market_conditions": {"volatility_state": "extreme", "atr": 1.0}}
        assert _volatility_bucket(trade, self.cfg) == "EXTREME"

    def test_low_atr_bucket(self) -> None:
        trade = {"market_conditions": {"atr": 3.0}}
        assert _volatility_bucket(trade, self.cfg) == "LOW"

    def test_medium_atr_bucket(self) -> None:
        trade = {"market_conditions": {"atr": 15.0}}
        assert _volatility_bucket(trade, self.cfg) == "MEDIUM"

    def test_high_atr_bucket(self) -> None:
        trade = {"market_conditions": {"atr": 40.0}}
        assert _volatility_bucket(trade, self.cfg) == "HIGH"

    def test_no_market_conditions_is_unknown(self) -> None:
        assert _volatility_bucket({}, self.cfg) == "UNKNOWN"


class TestConfidenceOutcomeCorrelation:
    def test_none_with_insufficient_data(self) -> None:
        assert confidence_outcome_correlation([]) is None
        assert confidence_outcome_correlation([{"confidence_at_entry": 0.5, "pnl": 1.0}]) is None

    def test_none_without_confidence_field(self) -> None:
        assert confidence_outcome_correlation([{"pnl": 1.0}, {"pnl": -1.0}]) is None

    def test_positive_correlation_when_well_calibrated(self) -> None:
        trades = _well_calibrated_trades(50, _NOW)
        corr = confidence_outcome_correlation(trades)
        assert corr is not None
        assert corr > 0.3

    def test_low_correlation_when_confidence_uninformative(self) -> None:
        # Confidence identical across all trades → zero variance → undefined.
        trades = [
            {"confidence_at_entry": 0.5, "pnl": 10.0 if i % 2 == 0 else -10.0} for i in range(10)
        ]
        assert confidence_outcome_correlation(trades) is None


# ── DecisionQualityReport dataclass ──────────────────────────────────────────


class TestDecisionQualityReport:
    def test_to_dict_from_dict_roundtrip(self) -> None:
        report = DecisionQualityReport(
            report_id="dqr-1",
            model_version="v1.0",
            created_at=_NOW,
            window_start=_NOW - 100,
            window_end=_NOW,
            n_trades=5,
            n_decisions=6,
            overall_expectancy=1.234,
            overall_win_rate=0.6,
            confidence_outcome_correlation=0.42,
            expectancy_by_regime={"RANGING": {"n_trades": 5, "expectancy": 1.234}},
        )
        d = report.to_dict()
        restored = DecisionQualityReport.from_dict(d)
        assert restored.report_id == "dqr-1"
        assert restored.overall_expectancy == pytest.approx(1.234)
        assert restored.confidence_outcome_correlation == pytest.approx(0.42)
        assert restored.expectancy_by_regime["RANGING"]["n_trades"] == 5

    def test_none_correlation_serializes_as_none(self) -> None:
        report = DecisionQualityReport(
            report_id="dqr-2",
            model_version="",
            created_at=_NOW,
            window_start=_NOW,
            window_end=_NOW,
            n_trades=0,
            n_decisions=0,
            overall_expectancy=0.0,
            overall_win_rate=0.0,
            confidence_outcome_correlation=None,
        )
        assert report.to_dict()["confidence_outcome_correlation"] is None


# ── DB persistence ────────────────────────────────────────────────────────────


class TestDecisionQualityReportDB:
    def test_upsert_and_get(self, db: ResearchLabDatabase) -> None:
        record = {"report_id": "dqr-a", "model_version": "v1.0", "created_at": _NOW, "n_trades": 3}
        db.upsert_decision_quality_report(record)
        fetched = db.get_decision_quality_report("dqr-a")
        assert fetched is not None
        assert fetched["n_trades"] == 3

    def test_get_nonexistent_returns_none(self, db: ResearchLabDatabase) -> None:
        assert db.get_decision_quality_report("missing") is None

    def test_list_filter_model_version(self, db: ResearchLabDatabase) -> None:
        db.upsert_decision_quality_report(
            {"report_id": "r1", "model_version": "v1.0", "created_at": _NOW}
        )
        db.upsert_decision_quality_report(
            {"report_id": "r2", "model_version": "v2.0", "created_at": _NOW}
        )
        results = db.list_decision_quality_reports(model_version="v1.0")
        assert len(results) == 1
        assert results[0]["report_id"] == "r1"

    def test_list_orders_newest_first(self, db: ResearchLabDatabase) -> None:
        db.upsert_decision_quality_report(
            {"report_id": "old", "model_version": "v1.0", "created_at": _NOW - 100}
        )
        db.upsert_decision_quality_report(
            {"report_id": "new", "model_version": "v1.0", "created_at": _NOW}
        )
        results = db.list_decision_quality_reports(model_version="v1.0")
        assert [r["report_id"] for r in results] == ["new", "old"]

    def test_upsert_overwrites(self, db: ResearchLabDatabase) -> None:
        db.upsert_decision_quality_report(
            {"report_id": "r1", "model_version": "v1.0", "created_at": _NOW, "n_trades": 1}
        )
        db.upsert_decision_quality_report(
            {"report_id": "r1", "model_version": "v1.0", "created_at": _NOW, "n_trades": 99}
        )
        assert db.get_decision_quality_report("r1")["n_trades"] == 99

    def test_stats_includes_decision_quality_reports(self, db: ResearchLabDatabase) -> None:
        db.upsert_decision_quality_report(
            {"report_id": "r1", "model_version": "v1.0", "created_at": _NOW}
        )
        stats = db.telemetry_stream_counts() if hasattr(db, "telemetry_stream_counts") else None
        # table existence + basic count sanity via list
        assert len(db.list_decision_quality_reports()) == 1


# ── DecisionQualityTracker (end-to-end via warehouse) ────────────────────────


class TestDecisionQualityTracker:
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
        tracker = DecisionQualityTracker(db, warehouse)
        self._seed(warehouse, _well_calibrated_trades(20, _NOW - 20 * 60))
        report = tracker.compute_report(model_version="v1.0")
        assert report.n_trades == 20
        assert report.overall_expectancy != 0.0
        assert "RANGING" in report.expectancy_by_regime
        assert "XAUUSD" in report.expectancy_by_symbol
        assert "LONDON" in report.expectancy_by_session
        assert "LOW" in report.expectancy_by_volatility_state

        latest = tracker.get_latest_report(model_version="v1.0")
        assert latest is not None
        assert latest["report_id"] == report.report_id

    def test_compute_report_empty_warehouse(self, db, warehouse) -> None:
        tracker = DecisionQualityTracker(db, warehouse)
        report = tracker.compute_report(model_version="v1.0")
        assert report.n_trades == 0
        assert report.overall_expectancy == 0.0
        assert report.confidence_outcome_correlation is None

    def test_list_reports_respects_limit(self, db, warehouse) -> None:
        tracker = DecisionQualityTracker(db, warehouse)
        for _ in range(3):
            tracker.compute_report(model_version="v1.0")
        assert len(tracker.list_reports(model_version="v1.0", limit=2)) == 2


# ── Drift check (pure function) ──────────────────────────────────────────────


class TestCheckDecisionQualityDrift:
    def _cfg(self, **overrides) -> DecisionQualityConfig:
        base = dict(baseline_window=20, rolling_window=10, min_samples=5)
        base.update(overrides)
        return DecisionQualityConfig(**base)

    def test_no_alert_when_stable(self) -> None:
        trades = _well_calibrated_trades(20, _NOW - 30 * 60) + _well_calibrated_trades(
            10, _NOW - 10 * 60
        )
        result = check_decision_quality_drift(trades, self._cfg(), _split_windows)
        assert result is None

    def test_alert_when_degraded(self) -> None:
        trades = _well_calibrated_trades(20, _NOW - 30 * 60) + _degraded_trades(10, _NOW - 10 * 60)
        result = check_decision_quality_drift(trades, self._cfg(), _split_windows)
        assert result is not None
        assert result["severity"] in ("WARNING", "CRITICAL")
        assert "expectancy" in result["triggers"]
        assert result["evidence"]["recent_expectancy"] < result["evidence"]["baseline_expectancy"]

    def test_critical_when_severely_degraded(self) -> None:
        trades = _well_calibrated_trades(20, _NOW - 30 * 60) + _degraded_trades(10, _NOW - 10 * 60)
        result = check_decision_quality_drift(
            trades,
            self._cfg(expectancy_drift_warning=0.05, expectancy_drift_critical=0.10),
            _split_windows,
        )
        assert result is not None
        assert result["severity"] == "CRITICAL"

    def test_alert_when_correlation_becomes_undefined(self) -> None:
        # Baseline: confidence meaningfully tracks outcomes (varies, correlated).
        # Recent: confidence flatlines (zero variance) -> correlation undefined.
        # This must still register as degradation, not silently pass.
        baseline = _well_calibrated_trades(20, _NOW - 30 * 60)
        recent = [
            {
                "confidence_at_entry": 0.6,  # constant -> zero variance -> undefined corr
                "pnl": 10.0 if i % 2 == 0 else -10.0,
                "timestamp": _NOW - 10 * 60 + i * 60,
                "regime": "RANGING",
                "symbol": "XAUUSD",
                "session": "LONDON",
                "market_conditions": {"atr": 5.0},
                "model_version": "v1.0",
            }
            for i in range(10)
        ]
        result = check_decision_quality_drift(baseline + recent, self._cfg(), _split_windows)
        assert result is not None
        assert "confidence_outcome_correlation" in result["triggers"]
        assert result["evidence"]["confidence_outcome_correlation_became_undefined"] is True

    def test_no_alert_when_baseline_correlation_also_undefined(self) -> None:
        # Both baseline and recent have undefined correlation (no signal either
        # way) -> should not spuriously trigger the correlation check.
        flat_conf_trades = [
            {
                "confidence_at_entry": 0.5,
                "pnl": 10.0 if i % 2 == 0 else -10.0,
                "timestamp": _NOW - 30 * 60 + i * 60,
                "regime": "RANGING",
                "model_version": "v1.0",
            }
            for i in range(30)
        ]
        result = check_decision_quality_drift(flat_conf_trades, self._cfg(), _split_windows)
        if result is not None:
            assert "confidence_outcome_correlation" not in result["triggers"]

    def test_insufficient_data_returns_none(self) -> None:
        assert check_decision_quality_drift([], self._cfg(), _split_windows) is None
        assert (
            check_decision_quality_drift(
                _well_calibrated_trades(3, _NOW), self._cfg(), _split_windows
            )
            is None
        )


# ── Wired into ResearchLabHealthMonitor ──────────────────────────────────────


class TestDecisionQualityHealthMonitorWiring:
    def _cfg(self, **overrides) -> HealthMonitorConfig:
        base = dict(
            baseline_window=20,
            rolling_window=10,
            min_samples=5,
            decision_quality_expectancy_warning=0.15,
            decision_quality_expectancy_critical=0.35,
        )
        base.update(overrides)
        return HealthMonitorConfig(**base)

    def test_no_alert_when_stable(self, db, warehouse) -> None:
        monitor = ResearchLabHealthMonitor(db, warehouse, config=self._cfg())
        trades = _well_calibrated_trades(20, _NOW - 30 * 60) + _well_calibrated_trades(
            10, _NOW - 10 * 60
        )
        alert = monitor._check_decision_quality(trades, "v1.0")
        assert alert is None

    def test_alert_on_degradation(self, db, warehouse) -> None:
        monitor = ResearchLabHealthMonitor(db, warehouse, config=self._cfg())
        trades = _well_calibrated_trades(20, _NOW - 30 * 60) + _degraded_trades(10, _NOW - 10 * 60)
        alert = monitor._check_decision_quality(trades, "v1.0")
        assert alert is not None
        assert alert.dimension == DriftDimension.DECISION_QUALITY.value
        assert alert.model_version == "v1.0"
        assert "expectancy" in alert.affected_features

    def test_run_check_includes_decision_quality_dimension(self, db, warehouse) -> None:
        monitor = ResearchLabHealthMonitor(db, warehouse, config=self._cfg())
        for t in _degraded_trades(10, _NOW - 5 * 60):
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
        assert DriftDimension.DECISION_QUALITY.value in result.to_dict()["dimensions_checked"]

    def test_critical_decision_quality_alert_creates_recommendation(self, db, warehouse) -> None:
        from unittest.mock import MagicMock

        rec_engine = MagicMock()
        rec_engine._db = db
        monitor = ResearchLabHealthMonitor(
            db,
            warehouse,
            rec_engine=rec_engine,
            config=self._cfg(
                decision_quality_expectancy_warning=0.05, decision_quality_expectancy_critical=0.10
            ),
        )
        trades = _well_calibrated_trades(20, _NOW - 30 * 60) + _degraded_trades(10, _NOW - 10 * 60)
        alert = monitor._check_decision_quality(trades, "v1.0")
        assert alert is not None
        assert alert.severity == "CRITICAL"
        rec_id = monitor._maybe_create_recommendation(alert)
        assert rec_id  # non-empty recommendation id was created
        assert db.get_retraining_recommendation(rec_id) is not None


# ── Server endpoints ──────────────────────────────────────────────────────────


class TestDecisionQualityServerEndpoints:
    _HDR = {
        "Authorization": "Bearer test-dq-token",
        "X-Signalium-Token": make_signalium_jwt(),
    }

    @pytest.fixture()
    def client(self, tmp_path: Path):
        import os

        os.environ["RESEARCH_LAB_ADMIN_TOKEN"] = "test-dq-token"
        os.environ["SIGNALIUM_JWT_SECRET"] = DEFAULT_TEST_JWT_SECRET
        os.environ["RESEARCH_LAB_DB_PATH"] = str(tmp_path / "server_dq.db")
        for mod_name in list(sys.modules):
            if "research_lab_server" in mod_name or "telemetry_server" in mod_name:
                del sys.modules[mod_name]
        from starlette.testclient import TestClient

        import trainer.research_lab.research_lab_server as srv_mod

        return TestClient(srv_mod.app)

    def test_compute_report_empty_warehouse(self, client) -> None:
        resp = client.post("/decision-quality/compute?model_version=v1.0", headers=self._HDR)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["report"]["n_trades"] == 0

    def test_latest_report_none_before_compute(self, client) -> None:
        resp = client.get("/decision-quality/latest?model_version=v1.0", headers=self._HDR)
        assert resp.status_code == 200
        assert resp.json()["report"] is None

    def test_latest_report_after_compute(self, client) -> None:
        client.post("/decision-quality/compute?model_version=v1.0", headers=self._HDR)
        resp = client.get("/decision-quality/latest?model_version=v1.0", headers=self._HDR)
        assert resp.status_code == 200
        assert resp.json()["report"] is not None

    def test_list_reports(self, client) -> None:
        client.post("/decision-quality/compute?model_version=v1.0", headers=self._HDR)
        client.post("/decision-quality/compute?model_version=v1.0", headers=self._HDR)
        resp = client.get("/decision-quality/reports?model_version=v1.0", headers=self._HDR)
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    def test_compute_requires_auth(self, client) -> None:
        resp = client.post("/decision-quality/compute")
        assert resp.status_code == 401

    def test_latest_requires_auth(self, client) -> None:
        resp = client.get("/decision-quality/latest")
        assert resp.status_code == 401

    def test_reports_requires_auth(self, client) -> None:
        resp = client.get("/decision-quality/reports")
        assert resp.status_code == 401
