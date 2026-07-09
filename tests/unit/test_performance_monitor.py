"""S22B — Tests for Autonomous Performance Monitoring.

Covers:
  - Pure metric helpers: profit factor, hit rate, max drawdown, capital
    efficiency, risk-adjusted return
  - Cohort derivation helper
  - Grouping by symbol/regime/cohort
  - PerformanceReport persistence + querying via ResearchLabDatabase
  - PerformanceMonitor.compute_report end-to-end
  - check_performance_drift: no alert / WARNING / CRITICAL / insufficient data
  - Wiring into ResearchLabHealthMonitor's alerting path (PERFORMANCE dimension)
  - Server endpoints: /performance/compute, /performance/latest, /performance/reports
"""

from __future__ import annotations

# `trainer.validation.trading` has no internal trainer dependencies (only
# numpy), but importing it normally (`from trainer.validation.trading import
# ...`) executes `trainer/validation/__init__.py`, which pulls in a
# torch-dependent chain that's broken in this environment (see memory:
# torch-broken-lazy-registry-import). Load the module file directly,
# bypassing the package `__init__.py`, the same isolation pattern used for
# `trainer.mie` (see memory: mie-test-isolation).
import importlib.util as _importlib_util
import sys
import sys as _sys
import time
from pathlib import Path
from pathlib import Path as _Path

import pytest

from tests.unit._sso_test_utils import DEFAULT_TEST_JWT_SECRET, make_signalium_jwt
from trainer.research_lab.dimension_evaluators import _split_windows
from trainer.research_lab.health_monitor import (
    DriftDimension,
    HealthMonitorConfig,
    ResearchLabHealthMonitor,
)
from trainer.research_lab.performance_monitor import (
    PerformanceMonitor,
    PerformanceMonitorConfig,
    PerformanceReport,
    _capital_efficiency,
    _cohort_of,
    _group_performance,
    _hit_rate,
    _max_drawdown,
    _profit_factor,
    _risk_adjusted_return,
    _sharpe_ratio,
    check_performance_drift,
    performance_stats,
)
from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.telemetry_warehouse import TelemetryWarehouse

_trading_path = _Path(__file__).resolve().parents[2] / "trainer" / "validation" / "trading.py"
_trading_spec = _importlib_util.spec_from_file_location("trainer.validation.trading", _trading_path)
_trading_mod = _importlib_util.module_from_spec(_trading_spec)
_sys.modules["trainer.validation.trading"] = _trading_mod
_trading_spec.loader.exec_module(_trading_mod)
TradeResult = _trading_mod.TradeResult
TradingMetricsEngine = _trading_mod.TradingMetricsEngine

_NOW = time.time()


@pytest.fixture()
def db(tmp_path: Path) -> ResearchLabDatabase:
    return ResearchLabDatabase(str(tmp_path / "perf_test.db"))


@pytest.fixture()
def warehouse(db: ResearchLabDatabase) -> TelemetryWarehouse:
    return TelemetryWarehouse(db)


def _profitable_trades(n: int, ts_start: float, **extra) -> list[dict]:
    out = []
    for i in range(n):
        win = (i % 10) < 7  # 70% hit rate, net positive
        row = {
            "pnl": 15.0 if win else -8.0,
            "timestamp": ts_start + i * 60,
            "regime": "TRENDING",
            "symbol": "XAUUSD",
            "market_conditions": {"cohort": "premium"},
            "model_version": "v1.0",
        }
        row.update(extra)
        out.append(row)
    return out


def _degraded_trades(n: int, ts_start: float, **extra) -> list[dict]:
    """Losing streak: low hit rate, deep drawdown, negative expectancy."""
    out = []
    for i in range(n):
        win = (i % 10) < 2  # 20% hit rate
        row = {
            "pnl": 10.0 if win else -15.0,
            "timestamp": ts_start + i * 60,
            "regime": "CHOPPY",
            "symbol": "EURUSD",
            "market_conditions": {"cohort": "standard"},
            "model_version": "v1.0",
        }
        row.update(extra)
        out.append(row)
    return out


# ── Pure helpers ──────────────────────────────────────────────────────────────


class TestProfitFactor:
    def test_no_losses_is_inf(self) -> None:
        assert _profit_factor([10.0, 5.0]) == float("inf")

    def test_mixed(self) -> None:
        pf = _profit_factor([10.0, 10.0, -5.0])
        assert pf == pytest.approx(20.0 / 5.0)

    def test_empty(self) -> None:
        assert _profit_factor([]) == float("inf")


class TestHitRate:
    def test_empty(self) -> None:
        assert _hit_rate([]) == 0.0

    def test_mixed(self) -> None:
        assert _hit_rate([10.0, -5.0, 10.0, -5.0]) == pytest.approx(0.5)

    def test_breakeven_not_a_win(self) -> None:
        assert _hit_rate([0.0, 10.0]) == pytest.approx(0.5)


class TestMaxDrawdown:
    def test_empty(self) -> None:
        assert _max_drawdown([]) == 0.0

    def test_monotonic_gains_zero_drawdown(self) -> None:
        assert _max_drawdown([10.0, 10.0, 10.0]) == pytest.approx(0.0)

    def test_drawdown_after_peak(self) -> None:
        # equity: 10 -> 20 -> 10 -> 5; peak 20, trough 5 => dd = (20-5)/20 = 0.75
        dd = _max_drawdown([10.0, 10.0, -10.0, -5.0])
        assert dd == pytest.approx(0.75)

    def test_loss_first_sequence_no_phantom_drawdown_at_start(self) -> None:
        # equity: -5 -> -10 -> -15 (never recovers). Peak is the running
        # cummax of equity itself (i.e. -5, the first point), NOT a floored
        # 0.0 baseline — so drawdown reflects the loss *within* the losing
        # streak (relative to its own best point), not relative to a
        # phantom zero peak that was never actually reached.
        dd = _max_drawdown([-5.0, -5.0, -5.0])
        # peak sequence: -5, -5, -5; equity: -5, -10, -15
        # dd at each point: 0, (-5 - -10)/5 = 1.0, (-5 - -15)/5 = 2.0
        assert dd == pytest.approx(2.0)

    def test_first_point_always_zero_drawdown(self) -> None:
        # A single losing trade must show zero drawdown (peak == equity at
        # the only point) — it must not be measured against a floored 0.0.
        assert _max_drawdown([-100.0]) == pytest.approx(0.0)

    def test_all_losses_bounded_by_own_peak_not_inflated_by_zero_floor(self) -> None:
        # Regression for the "peak floored at 0.0" bug: with an all-losses
        # sequence starting negative, drawdown must be computed relative to
        # the running cummax of equity (which starts negative), not
        # relative to a phantom zero that was never the actual peak.
        pnls = [-10.0, -20.0, -5.0]
        dd = _max_drawdown(pnls)
        # peak: -10, -10, -10; equity: -10, -30, -35
        # dd: 0, (-10--30)/10=2.0, (-10--35)/10=2.5
        assert dd == pytest.approx(2.5)


class TestCapitalEfficiency:
    def test_no_losses(self) -> None:
        ce = _capital_efficiency([10.0, 5.0])
        assert ce > 1e6  # total_return / ~epsilon

    def test_mixed(self) -> None:
        ce = _capital_efficiency([20.0, -10.0])
        assert ce == pytest.approx(10.0 / 10.0, rel=1e-3)


class TestRiskAdjustedReturn:
    def test_zero_drawdown_positive_return_is_inf(self) -> None:
        assert _risk_adjusted_return([10.0, 10.0]) == float("inf")

    def test_zero_drawdown_no_return_is_zero(self) -> None:
        assert _risk_adjusted_return([]) == 0.0

    def test_mixed(self) -> None:
        # equity 10 -> 20 -> 10 -> 15; peak 20, trough 10 => dd=0.5; total_return=15
        rar = _risk_adjusted_return([10.0, 10.0, -10.0, 5.0])
        assert rar == pytest.approx(15.0 / 0.5)


class TestSharpeRatio:
    def test_matches_trading_metrics_engine_on_shared_data(self) -> None:
        """`_sharpe_ratio` must be formula-identical to
        `TradingMetricsEngine`'s Sharpe step — live and offline numbers
        must agree on the same trade data."""
        pnls = [15.0, -8.0, 15.0, 15.0, -8.0, 15.0, 15.0, -8.0, 15.0, 15.0]
        live_sharpe = _sharpe_ratio(pnls, annualisation_factor=252.0, risk_free_rate=0.0)

        trades = [
            TradeResult(direction=1, pnl_pips=p, entry_price=1.0, stop_loss=0.9, take_profit=1.1)
            for p in pnls
        ]
        engine = TradingMetricsEngine(annualisation_factor=252.0, risk_free_rate=0.0)
        offline_metrics = engine.compute(trades)

        assert live_sharpe == pytest.approx(offline_metrics.sharpe_ratio, rel=1e-6)

    def test_matches_trading_metrics_engine_when_losses_dominate(self) -> None:
        """Regression: `TradingMetricsEngine` normalizes by
        `max(abs(pnls.max()), 1.0)` — the absolute value of the single
        largest (most positive) pnl, NOT the largest-*magnitude* pnl. A
        loss-dominated series (biggest loss bigger than any gain), combined
        with a non-zero risk-free rate and non-default annualisation (so
        the result is denominator-sensitive rather than scale-invariant),
        is the case where `max(abs(p) for p in pnls)` would silently
        diverge from the correct `max(abs(max(pnls)), 1.0)`."""
        pnls = [3.0, -50.0, 3.0, -2.0, 3.0, -1.0, 3.0, -3.0]
        live_sharpe = _sharpe_ratio(pnls, annualisation_factor=52.0, risk_free_rate=0.03)

        trades = [
            TradeResult(direction=1, pnl_pips=p, entry_price=1.0, stop_loss=0.9, take_profit=1.1)
            for p in pnls
        ]
        engine = TradingMetricsEngine(annualisation_factor=52.0, risk_free_rate=0.03)
        offline_metrics = engine.compute(trades)

        assert live_sharpe == pytest.approx(offline_metrics.sharpe_ratio, rel=1e-6)
        # Sanity: this case must actually be denominator-sensitive — the old
        # (buggy) `max(abs(p) for p in pnls)` denominator would be 50.0,
        # not `max(abs(max(pnls)), 1.0) == 3.0`, and produce a different
        # number given the non-zero risk-free rate.
        buggy_denom = max(abs(p) for p in pnls)
        correct_denom = max(abs(max(pnls)), 1.0)
        assert buggy_denom != correct_denom

    def test_matches_trading_metrics_engine_all_losses(self) -> None:
        """All-negative series: `pnls.max()` is itself negative, so the
        denominator is `max(abs(negative_max), 1.0)` — still must match
        the engine exactly, with a non-zero risk-free rate so the
        comparison is denominator-sensitive."""
        pnls = [-5.0, -20.0, -3.0, -8.0]
        live_sharpe = _sharpe_ratio(pnls, annualisation_factor=52.0, risk_free_rate=0.03)

        trades = [
            TradeResult(direction=1, pnl_pips=p, entry_price=1.0, stop_loss=0.9, take_profit=1.1)
            for p in pnls
        ]
        engine = TradingMetricsEngine(annualisation_factor=52.0, risk_free_rate=0.03)
        offline_metrics = engine.compute(trades)

        assert live_sharpe == pytest.approx(offline_metrics.sharpe_ratio, rel=1e-6)

    def test_matches_trading_metrics_engine_single_trade(self) -> None:
        """A single trade is not special-cased: population std of one
        point is exactly 0.0, so both live and offline formulas fall
        through to the same epsilon-stabilized large value — they must
        still agree exactly rather than live returning an arbitrary 0.0."""
        pnls = [10.0]
        live_sharpe = _sharpe_ratio(pnls, annualisation_factor=252.0, risk_free_rate=0.0)

        trades = [
            TradeResult(direction=1, pnl_pips=p, entry_price=1.0, stop_loss=0.9, take_profit=1.1)
            for p in pnls
        ]
        engine = TradingMetricsEngine(annualisation_factor=252.0, risk_free_rate=0.0)
        offline_metrics = engine.compute(trades)

        assert live_sharpe == pytest.approx(offline_metrics.sharpe_ratio, rel=1e-6)

    def test_empty_is_zero(self) -> None:
        assert _sharpe_ratio([]) == 0.0


class TestPerformanceStats:
    def test_empty(self) -> None:
        stats = performance_stats([])
        assert stats["n_trades"] == 0
        assert stats["profit_factor"] == 0.0
        assert stats["hit_rate"] == 0.0

    def test_inf_ratios_are_capped(self) -> None:
        stats = performance_stats([{"pnl": 10.0}, {"pnl": 5.0}])
        assert stats["profit_factor"] == pytest.approx(999.0)
        assert stats["risk_adjusted_return"] == pytest.approx(999.0)


class TestCohortOf:
    def test_explicit_cohort_key(self) -> None:
        assert _cohort_of({"market_conditions": {"cohort": "vip"}}) == "vip"

    def test_explicit_user_cohort_key(self) -> None:
        assert _cohort_of({"market_conditions": {"user_cohort": "beta"}}) == "beta"

    def test_falls_back_to_account_id(self) -> None:
        assert _cohort_of({"account_id": "acct-1"}) == "acct-1"

    def test_unknown_when_nothing_present(self) -> None:
        assert _cohort_of({}) == "UNKNOWN"


class TestGroupPerformance:
    def test_groups_by_symbol(self) -> None:
        trades = [{"pnl": 10.0, "symbol": "XAUUSD"}, {"pnl": -5.0, "symbol": "EURUSD"}]
        groups = _group_performance(trades, lambda t: t.get("symbol"))
        assert set(groups.keys()) == {"XAUUSD", "EURUSD"}


# ── PerformanceReport dataclass ──────────────────────────────────────────────


class TestPerformanceReport:
    def test_to_dict_from_dict_roundtrip(self) -> None:
        report = PerformanceReport(
            report_id="perf-1",
            model_version="v1.0",
            created_at=_NOW,
            window_start=_NOW - 100,
            window_end=_NOW,
            n_trades=5,
            overall={"profit_factor": 1.5},
            performance_by_symbol={"XAUUSD": {"n_trades": 5}},
        )
        d = report.to_dict()
        restored = PerformanceReport.from_dict(d)
        assert restored.report_id == "perf-1"
        assert restored.overall["profit_factor"] == pytest.approx(1.5)
        assert restored.performance_by_symbol["XAUUSD"]["n_trades"] == 5


# ── DB persistence ────────────────────────────────────────────────────────────


class TestPerformanceReportDB:
    def test_upsert_and_get(self, db: ResearchLabDatabase) -> None:
        db.upsert_performance_report(
            {"report_id": "p1", "model_version": "v1.0", "created_at": _NOW, "n_trades": 3}
        )
        fetched = db.get_performance_report("p1")
        assert fetched is not None
        assert fetched["n_trades"] == 3

    def test_get_nonexistent_returns_none(self, db: ResearchLabDatabase) -> None:
        assert db.get_performance_report("missing") is None

    def test_list_filter_model_version(self, db: ResearchLabDatabase) -> None:
        db.upsert_performance_report(
            {"report_id": "r1", "model_version": "v1.0", "created_at": _NOW}
        )
        db.upsert_performance_report(
            {"report_id": "r2", "model_version": "v2.0", "created_at": _NOW}
        )
        results = db.list_performance_reports(model_version="v1.0")
        assert len(results) == 1
        assert results[0]["report_id"] == "r1"

    def test_upsert_overwrites(self, db: ResearchLabDatabase) -> None:
        db.upsert_performance_report(
            {"report_id": "r1", "model_version": "v1.0", "created_at": _NOW, "n_trades": 1}
        )
        db.upsert_performance_report(
            {"report_id": "r1", "model_version": "v1.0", "created_at": _NOW, "n_trades": 99}
        )
        assert db.get_performance_report("r1")["n_trades"] == 99


# ── PerformanceMonitor (end-to-end via warehouse) ────────────────────────────


class TestPerformanceMonitor:
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
        monitor = PerformanceMonitor(db, warehouse)
        self._seed(warehouse, _profitable_trades(20, _NOW - 20 * 60))
        report = monitor.compute_report(model_version="v1.0")
        assert report.n_trades == 20
        assert "XAUUSD" in report.performance_by_symbol
        assert "TRENDING" in report.performance_by_regime
        assert "premium" in report.performance_by_cohort

        latest = monitor.get_latest_report(model_version="v1.0")
        assert latest is not None
        assert latest["report_id"] == report.report_id

    def test_compute_report_empty_warehouse(self, db, warehouse) -> None:
        monitor = PerformanceMonitor(db, warehouse)
        report = monitor.compute_report(model_version="v1.0")
        assert report.n_trades == 0
        assert report.overall["profit_factor"] == 0.0

    def test_list_reports_respects_limit(self, db, warehouse) -> None:
        monitor = PerformanceMonitor(db, warehouse)
        for _ in range(3):
            monitor.compute_report(model_version="v1.0")
        assert len(monitor.list_reports(model_version="v1.0", limit=2)) == 2


# ── Drift check (pure function) ──────────────────────────────────────────────


class TestCheckPerformanceDrift:
    def _cfg(self, **overrides) -> PerformanceMonitorConfig:
        base = dict(baseline_window=20, rolling_window=10, min_samples=5)
        base.update(overrides)
        return PerformanceMonitorConfig(**base)

    def test_no_alert_when_stable(self) -> None:
        trades = _profitable_trades(20, _NOW - 30 * 60) + _profitable_trades(10, _NOW - 10 * 60)
        result = check_performance_drift(trades, self._cfg(), _split_windows)
        assert result is None

    def test_alert_when_degraded(self) -> None:
        trades = _profitable_trades(20, _NOW - 30 * 60) + _degraded_trades(10, _NOW - 10 * 60)
        result = check_performance_drift(trades, self._cfg(), _split_windows)
        assert result is not None
        assert result["severity"] in ("WARNING", "CRITICAL")
        assert len(result["triggers"]) > 0
        assert result["evidence"]["recent_hit_rate"] < result["evidence"]["baseline_hit_rate"]

    def test_critical_when_severely_degraded(self) -> None:
        trades = _profitable_trades(20, _NOW - 30 * 60) + _degraded_trades(10, _NOW - 10 * 60)
        result = check_performance_drift(
            trades,
            self._cfg(hit_rate_drop_warning=0.05, hit_rate_drop_critical=0.10),
            _split_windows,
        )
        assert result is not None
        assert result["severity"] == "CRITICAL"

    def test_insufficient_data_returns_none(self) -> None:
        assert check_performance_drift([], self._cfg(), _split_windows) is None
        assert (
            check_performance_drift(_profitable_trades(3, _NOW), self._cfg(), _split_windows)
            is None
        )


# ── Wired into ResearchLabHealthMonitor ──────────────────────────────────────


class TestPerformanceHealthMonitorWiring:
    def _cfg(self, **overrides) -> HealthMonitorConfig:
        base = dict(
            baseline_window=20,
            rolling_window=10,
            min_samples=5,
            performance_hit_rate_drop_warning=0.10,
            performance_hit_rate_drop_critical=0.20,
        )
        base.update(overrides)
        return HealthMonitorConfig(**base)

    def test_no_alert_when_stable(self, db, warehouse) -> None:
        monitor = ResearchLabHealthMonitor(db, warehouse, config=self._cfg())
        trades = _profitable_trades(20, _NOW - 30 * 60) + _profitable_trades(10, _NOW - 10 * 60)
        alert = monitor._check_performance(trades, "v1.0")
        assert alert is None

    def test_alert_on_degradation(self, db, warehouse) -> None:
        monitor = ResearchLabHealthMonitor(db, warehouse, config=self._cfg())
        trades = _profitable_trades(20, _NOW - 30 * 60) + _degraded_trades(10, _NOW - 10 * 60)
        alert = monitor._check_performance(trades, "v1.0")
        assert alert is not None
        assert alert.dimension == DriftDimension.PERFORMANCE.value
        assert alert.model_version == "v1.0"

    def test_run_check_includes_performance_dimension(self, db, warehouse) -> None:
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
        assert DriftDimension.PERFORMANCE.value in result.to_dict()["dimensions_checked"]

    def test_critical_performance_alert_creates_recommendation(self, db, warehouse) -> None:
        from unittest.mock import MagicMock

        rec_engine = MagicMock()
        rec_engine._db = db
        monitor = ResearchLabHealthMonitor(
            db,
            warehouse,
            rec_engine=rec_engine,
            config=self._cfg(
                performance_hit_rate_drop_warning=0.05, performance_hit_rate_drop_critical=0.10
            ),
        )
        trades = _profitable_trades(20, _NOW - 30 * 60) + _degraded_trades(10, _NOW - 10 * 60)
        alert = monitor._check_performance(trades, "v1.0")
        assert alert is not None
        assert alert.severity == "CRITICAL"
        rec_id = monitor._maybe_create_recommendation(alert)
        assert rec_id
        assert db.get_retraining_recommendation(rec_id) is not None


# ── Server endpoints ──────────────────────────────────────────────────────────


class TestPerformanceServerEndpoints:
    _HDR = {
        "Authorization": "Bearer test-perf-token",
        "X-Signalium-Token": make_signalium_jwt(),
    }

    @pytest.fixture()
    def client(self, tmp_path: Path):
        import os

        os.environ["RESEARCH_LAB_ADMIN_TOKEN"] = "test-perf-token"
        os.environ["SIGNALIUM_JWT_SECRET"] = DEFAULT_TEST_JWT_SECRET
        os.environ["RESEARCH_LAB_DB_PATH"] = str(tmp_path / "server_perf.db")
        for mod_name in list(sys.modules):
            if "research_lab_server" in mod_name or "telemetry_server" in mod_name:
                del sys.modules[mod_name]
        from starlette.testclient import TestClient

        import trainer.research_lab.research_lab_server as srv_mod

        return TestClient(srv_mod.app)

    def test_compute_report_empty_warehouse(self, client) -> None:
        resp = client.post("/performance/compute?model_version=v1.0", headers=self._HDR)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["report"]["n_trades"] == 0

    def test_latest_report_none_before_compute(self, client) -> None:
        resp = client.get("/performance/latest?model_version=v1.0", headers=self._HDR)
        assert resp.status_code == 200
        assert resp.json()["report"] is None

    def test_latest_report_after_compute(self, client) -> None:
        client.post("/performance/compute?model_version=v1.0", headers=self._HDR)
        resp = client.get("/performance/latest?model_version=v1.0", headers=self._HDR)
        assert resp.status_code == 200
        assert resp.json()["report"] is not None

    def test_list_reports(self, client) -> None:
        client.post("/performance/compute?model_version=v1.0", headers=self._HDR)
        client.post("/performance/compute?model_version=v1.0", headers=self._HDR)
        resp = client.get("/performance/reports?model_version=v1.0", headers=self._HDR)
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    def test_compute_requires_auth(self, client) -> None:
        resp = client.post("/performance/compute")
        assert resp.status_code == 401

    def test_latest_requires_auth(self, client) -> None:
        resp = client.get("/performance/latest")
        assert resp.status_code == 401

    def test_reports_requires_auth(self, client) -> None:
        resp = client.get("/performance/reports")
        assert resp.status_code == 401
