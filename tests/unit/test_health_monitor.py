"""S20/S22B — Tests for Model Health Monitoring (thirteen-dimension drift detection).

Covers:
  - Each drift dimension: threshold crossing (below / above WARNING / CRITICAL)
  - DriftAlert schema: to_dict / from_dict round-trip
  - CRITICAL alert → RetrainingRecommendation auto-creation
  - Deduplication: second CRITICAL alert does not create a second recommendation
  - DB CRUD: health_alerts table
  - Server endpoints: POST /health/check, GET /health/status, GET /health/alerts
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from tests.unit._sso_test_utils import DEFAULT_TEST_JWT_SECRET, make_signalium_jwt
from trainer.research_lab.health_monitor import (
    AlertSeverity,
    DriftAlert,
    DriftDimension,
    HealthMonitorConfig,
    ResearchLabHealthMonitor,
    _class_distribution,
    _decision_entropy,
    _expected_calibration_error,
    _regime_histogram,
    _stdev,
    _tv_distance,
)
from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.telemetry_warehouse import TelemetryWarehouse

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def db(tmp_path: Path) -> ResearchLabDatabase:
    return ResearchLabDatabase(str(tmp_path / "health_test.db"))


@pytest.fixture()
def warehouse(db: ResearchLabDatabase) -> TelemetryWarehouse:
    return TelemetryWarehouse(db)


def _make_monitor(
    db: ResearchLabDatabase,
    warehouse: TelemetryWarehouse,
    rec_engine=None,
    config: HealthMonitorConfig | None = None,
) -> ResearchLabHealthMonitor:
    return ResearchLabHealthMonitor(db, warehouse, rec_engine=rec_engine, config=config)


# ── Decision / trade helpers ──────────────────────────────────────────────────

_NOW = time.time()


def _decisions(
    n: int,
    *,
    confidence: float = 0.7,
    buy_prob: float = 0.5,
    sell_prob: float = 0.3,
    no_trade_prob: float = 0.2,
    regime: str = "RANGING",
    model_version: str = "v1.0",
    ts_start: float | None = None,
) -> list[dict]:
    start = ts_start if ts_start is not None else _NOW - n * 60
    return [
        {
            "confidence": confidence,
            "buy_prob": buy_prob,
            "sell_prob": sell_prob,
            "no_trade_prob": no_trade_prob,
            "regime": regime,
            "model_version": model_version,
            "symbol": "XAUUSD",
            "timestamp": start + i * 60,
        }
        for i in range(n)
    ]


def _trades(
    n: int,
    *,
    pnl: float = 10.0,
    regime: str = "RANGING",
    model_version: str = "v1.0",
) -> list[dict]:
    start = _NOW - n * 60
    return [
        {
            "pnl": pnl,
            "pnl_pct": pnl / 1000.0,
            "regime": regime,
            "model_version": model_version,
            "symbol": "XAUUSD",
            "confidence_at_entry": 0.7,
            "timestamp": start + i * 60,
        }
        for i in range(n)
    ]


def _mixed_decisions(
    n_baseline: int,
    n_recent: int,
    baseline_conf: float,
    recent_conf: float,
    baseline_regime: str = "RANGING",
    recent_regime: str = "RANGING",
) -> list[dict]:
    baseline = _decisions(n_baseline, confidence=baseline_conf, regime=baseline_regime)
    recent = _decisions(
        n_recent,
        confidence=recent_conf,
        regime=recent_regime,
        ts_start=_NOW - n_recent * 60,
    )
    return baseline + recent


# ── DriftAlert model ──────────────────────────────────────────────────────────


class TestDriftAlertModel:
    def _make_alert(self, dim: str = "CONFIDENCE", sev: str = "WARNING") -> DriftAlert:
        return DriftAlert(
            alert_id="hal-test001",
            dimension=dim,
            severity=sev,
            model_version="v1.0",
            affected_symbols=["XAUUSD"],
            affected_features=["confidence"],
            evidence={"drift": 0.08},
            recommended_action="Monitor confidence trend.",
            created_at=_NOW,
        )

    def test_roundtrip(self) -> None:
        alert = self._make_alert()
        restored = DriftAlert.from_dict(alert.to_dict())
        assert restored.alert_id == "hal-test001"
        assert restored.dimension == "CONFIDENCE"
        assert restored.severity == "WARNING"
        assert restored.affected_symbols == ["XAUUSD"]
        assert restored.evidence == {"drift": 0.08}

    def test_critical_severity(self) -> None:
        alert = self._make_alert(sev="CRITICAL")
        d = alert.to_dict()
        assert d["severity"] == "CRITICAL"

    def test_all_dimensions_valid(self) -> None:
        for dim in DriftDimension:
            alert = self._make_alert(dim=dim.value)
            assert alert.dimension == dim.value


# ── Helpers ───────────────────────────────────────────────────────────────────


class TestHelpers:
    def test_tv_distance_identical(self) -> None:
        a = {"buy": 0.5, "sell": 0.3, "no_trade": 0.2}
        assert _tv_distance(a, a) == 0.0

    def test_tv_distance_max(self) -> None:
        # Two completely non-overlapping dists → TV = 1
        a = {"x": 1.0}
        b = {"y": 1.0}
        assert abs(_tv_distance(a, b) - 1.0) < 1e-9

    def test_tv_distance_partial(self) -> None:
        a = {"buy": 0.5, "sell": 0.5}
        b = {"buy": 0.3, "sell": 0.7}
        tv = _tv_distance(a, b)
        assert 0.0 < tv < 1.0

    def test_class_distribution_empty(self) -> None:
        dist = _class_distribution([])
        assert "buy" in dist and "sell" in dist and "no_trade" in dist

    def test_regime_histogram_no_regimes(self) -> None:
        decisions = [{"buy_prob": 0.5}]  # no regime key
        hist = _regime_histogram(decisions)
        assert hist == {}

    def test_regime_histogram_counts(self) -> None:
        # Regime labels are lowercased before counting (see
        # memory: regime-label-casing) — casing alone must never
        # manufacture a false MARKET distribution shift.
        decisions = [{"regime": "RANGING"}, {"regime": "RANGING"}, {"regime": "TRENDING_BULL"}]
        hist = _regime_histogram(decisions)
        assert abs(hist["ranging"] - 2 / 3) < 1e-9
        assert abs(hist["trending_bull"] - 1 / 3) < 1e-9


# ── Confidence drift ──────────────────────────────────────────────────────────


class TestConfidenceDrift:
    def _cfg(self, warning=0.06, critical=0.12) -> HealthMonitorConfig:
        return HealthMonitorConfig(
            baseline_window=20,
            rolling_window=10,
            min_samples=5,
            confidence_warning=warning,
            confidence_critical=critical,
        )

    def test_no_alert_below_threshold(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg())
        decisions = _decisions(20, confidence=0.72) + _decisions(10, confidence=0.73, ts_start=_NOW)
        alert = monitor._check_confidence(decisions, "v1.0")
        assert alert is None

    def test_warning_above_threshold(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg())
        # baseline=0.72, recent=0.65 → drift=0.07 > 0.06
        decisions = _decisions(20, confidence=0.72) + _decisions(10, confidence=0.65, ts_start=_NOW)
        alert = monitor._check_confidence(decisions, "v1.0")
        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING.value
        assert alert.dimension == DriftDimension.CONFIDENCE.value

    def test_critical_above_threshold(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg())
        # drift = 0.72 - 0.55 = 0.17 > 0.12 → CRITICAL
        decisions = _decisions(20, confidence=0.72) + _decisions(10, confidence=0.55, ts_start=_NOW)
        alert = monitor._check_confidence(decisions, "v1.0")
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL.value

    def test_insufficient_data_no_alert(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg())
        decisions = _decisions(3, confidence=0.4)  # < min_samples*2
        alert = monitor._check_confidence(decisions, "v1.0")
        assert alert is None

    def test_evidence_fields(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg())
        decisions = _decisions(20, confidence=0.72) + _decisions(10, confidence=0.58, ts_start=_NOW)
        alert = monitor._check_confidence(decisions, "v1.0")
        assert alert is not None
        assert "baseline_mean_confidence" in alert.evidence
        assert "recent_mean_confidence" in alert.evidence
        assert "drift" in alert.evidence


# ── Feature drift ─────────────────────────────────────────────────────────────


class TestFeatureDrift:
    def _cfg(self, warning=0.08, critical=0.16) -> HealthMonitorConfig:
        return HealthMonitorConfig(
            baseline_window=20,
            rolling_window=10,
            min_samples=5,
            feature_warning=warning,
            feature_critical=critical,
        )

    def test_no_alert_stable_distribution(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg())
        # Identical distribution across windows
        decisions = _decisions(20, buy_prob=0.5, sell_prob=0.3, no_trade_prob=0.2) + _decisions(
            10, buy_prob=0.51, sell_prob=0.29, no_trade_prob=0.20, ts_start=_NOW
        )
        alert = monitor._check_feature(decisions, "v1.0")
        assert alert is None

    def test_warning_shifted_distribution(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg())
        # Large shift: baseline mostly buy, recent mostly sell
        baseline = _decisions(20, buy_prob=0.7, sell_prob=0.2, no_trade_prob=0.1)
        recent = _decisions(10, buy_prob=0.2, sell_prob=0.7, no_trade_prob=0.1, ts_start=_NOW)
        alert = monitor._check_feature(baseline + recent, "v1.0")
        assert alert is not None
        assert alert.dimension == DriftDimension.FEATURE.value

    def test_critical_extreme_shift(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg(critical=0.05))
        baseline = _decisions(20, buy_prob=0.9, sell_prob=0.05, no_trade_prob=0.05)
        recent = _decisions(10, buy_prob=0.05, sell_prob=0.9, no_trade_prob=0.05, ts_start=_NOW)
        alert = monitor._check_feature(baseline + recent, "v1.0")
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL.value

    def test_evidence_includes_distributions(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg())
        baseline = _decisions(20, buy_prob=0.7, sell_prob=0.2, no_trade_prob=0.1)
        recent = _decisions(10, buy_prob=0.2, sell_prob=0.7, no_trade_prob=0.1, ts_start=_NOW)
        alert = monitor._check_feature(baseline + recent, "v1.0")
        if alert:
            assert "baseline_distribution" in alert.evidence
            assert "recent_distribution" in alert.evidence
            assert "tv_distance" in alert.evidence


# ── Market drift ──────────────────────────────────────────────────────────────


class TestMarketDrift:
    def _cfg(self, warning=0.12, critical=0.25) -> HealthMonitorConfig:
        return HealthMonitorConfig(
            baseline_window=20,
            rolling_window=10,
            min_samples=5,
            market_warning=warning,
            market_critical=critical,
        )

    def test_no_alert_stable_regimes(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg())
        decisions = _decisions(20, regime="RANGING") + _decisions(
            10, regime="RANGING", ts_start=_NOW
        )
        alert = monitor._check_market(decisions, "v1.0")
        assert alert is None

    def test_warning_regime_shift(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg(warning=0.05, critical=0.50))
        # baseline: all RANGING; recent: mix of CRISIS and TRENDING_BULL
        baseline = _decisions(20, regime="RANGING")
        recent = [
            {**d, "regime": "CRISIS" if i < 5 else "TRENDING_BULL", "timestamp": _NOW + i * 60}
            for i, d in enumerate(_decisions(10, regime="RANGING"))
        ]
        alert = monitor._check_market(baseline + recent, "v1.0")
        assert alert is not None
        assert alert.dimension == DriftDimension.MARKET.value

    def test_critical_market_shift(self, db, warehouse) -> None:
        """Market drift CRITICAL: threshold crossed by a complete regime flip."""
        monitor = _make_monitor(db, warehouse, config=self._cfg(warning=0.05, critical=0.15))
        # baseline: all RANGING; recent: all CRISIS → max TV distance
        baseline = _decisions(20, regime="RANGING")
        recent = _decisions(10, regime="CRISIS", ts_start=_NOW)
        alert = monitor._check_market(baseline + recent, "v1.0")
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL.value

    def test_no_alert_no_regime_labels(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg())
        decisions = [{"confidence": 0.7, "timestamp": _NOW + i} for i in range(30)]
        alert = monitor._check_market(decisions, "v1.0")
        assert alert is None

    def test_new_regimes_in_evidence(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg(warning=0.01, critical=0.90))
        baseline = _decisions(20, regime="RANGING")
        recent = _decisions(10, regime="CRISIS", ts_start=_NOW)
        alert = monitor._check_market(baseline + recent, "v1.0")
        if alert:
            assert "baseline_regime_dist" in alert.evidence
            assert "recent_regime_dist" in alert.evidence


# ── Data quality ──────────────────────────────────────────────────────────────


class TestDataQualityDrift:
    def _cfg(
        self, warn_s=3600.0, crit_s=7200.0, drop_warn=0.5, drop_crit=0.8
    ) -> HealthMonitorConfig:
        return HealthMonitorConfig(
            baseline_window=20,
            rolling_window=10,
            min_samples=5,
            staleness_warning_s=warn_s,
            staleness_critical_s=crit_s,
            event_rate_drop_warning=drop_warn,
            event_rate_drop_critical=drop_crit,
        )

    def test_no_alert_fresh_data(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg())
        now = time.time()
        decisions = [{"confidence": 0.7, "timestamp": now - i * 5} for i in range(30)]
        decisions.sort(key=lambda d: d["timestamp"])
        alert = monitor._check_data_quality(decisions, "v1.0", now)
        assert alert is None

    def test_warning_stale_data(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg(warn_s=100.0, crit_s=200.0))
        now = time.time()
        # last event 150s ago → WARNING (between warn=100, crit=200)
        decisions = [{"confidence": 0.7, "timestamp": now - 150 - i * 60} for i in range(30)]
        decisions.sort(key=lambda d: d["timestamp"])
        alert = monitor._check_data_quality(decisions, "v1.0", now)
        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING.value
        assert alert.dimension == DriftDimension.DATA_QUALITY.value

    def test_critical_very_stale_data(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg(warn_s=100.0, crit_s=200.0))
        now = time.time()
        # last event 250s ago → CRITICAL
        decisions = [{"confidence": 0.7, "timestamp": now - 250 - i * 60} for i in range(30)]
        decisions.sort(key=lambda d: d["timestamp"])
        alert = monitor._check_data_quality(decisions, "v1.0", now)
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL.value

    def test_no_alert_empty_decisions(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg())
        alert = monitor._check_data_quality([], "v1.0", time.time())
        assert alert is None


# ── Regime drift ──────────────────────────────────────────────────────────────


class TestRegimeDrift:
    def _cfg(self, warning=0.08, critical=0.15) -> HealthMonitorConfig:
        return HealthMonitorConfig(
            baseline_window=20,
            rolling_window=10,
            min_samples=5,
            regime_win_rate_warning=warning,
            regime_win_rate_critical=critical,
        )

    def test_no_alert_stable_win_rate(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg())
        trades = _trades(20, pnl=10.0) + _trades(10, pnl=10.0)
        alert = monitor._check_regime(trades, "v1.0")
        assert alert is None

    def test_warning_degraded_win_rate(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg(warning=0.05, critical=0.30))
        # baseline: all wins; recent: ~50% loss → large drop
        baseline = _trades(20, pnl=50.0, regime="RANGING")
        recent = [
            {**t, "pnl": -5.0 if i % 2 == 0 else 5.0, "timestamp": _NOW + i * 60}
            for i, t in enumerate(_trades(10, regime="RANGING"))
        ]
        alert = monitor._check_regime(baseline + recent, "v1.0")
        assert alert is not None
        assert alert.dimension == DriftDimension.REGIME.value

    def test_critical_win_rate_collapse(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg(critical=0.10))
        # baseline: all wins; recent: all losses
        baseline = _trades(20, pnl=10.0, regime="RANGING")
        recent = [
            {**t, "pnl": -5.0, "timestamp": _NOW + i * 60}
            for i, t in enumerate(_trades(10, regime="RANGING"))
        ]
        alert = monitor._check_regime(baseline + recent, "v1.0")
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL.value

    def test_insufficient_data_no_alert(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg())
        alert = monitor._check_regime([], "v1.0")
        assert alert is None

    def test_per_regime_evidence(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg(warning=0.05, critical=0.90))
        baseline = _trades(20, pnl=50.0, regime="RANGING")
        recent = [
            {**t, "pnl": -5.0, "timestamp": _NOW + i * 60}
            for i, t in enumerate(_trades(10, regime="RANGING"))
        ]
        alert = monitor._check_regime(baseline + recent, "v1.0")
        if alert:
            assert "degraded_regimes" in alert.evidence
            assert "worst_regime" in alert.evidence


# ── Entropy drift ─────────────────────────────────────────────────────────────


class TestEntropyDrift:
    def _cfg(self, warning=0.15, critical=0.30) -> HealthMonitorConfig:
        return HealthMonitorConfig(
            baseline_window=20,
            rolling_window=10,
            min_samples=5,
            entropy_warning=warning,
            entropy_critical=critical,
        )

    def test_no_alert_stable_entropy(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg())
        decisions = _decisions(20, buy_prob=0.7, sell_prob=0.2, no_trade_prob=0.1) + _decisions(
            10, buy_prob=0.72, sell_prob=0.19, no_trade_prob=0.09, ts_start=_NOW
        )
        alert = monitor._check_entropy(decisions, "v1.0")
        assert alert is None

    def test_warning_rising_entropy(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg())
        # baseline: decisive (low entropy); recent: near-uniform (high entropy)
        baseline = _decisions(20, buy_prob=0.9, sell_prob=0.05, no_trade_prob=0.05)
        recent = _decisions(10, buy_prob=0.34, sell_prob=0.33, no_trade_prob=0.33, ts_start=_NOW)
        alert = monitor._check_entropy(baseline + recent, "v1.0")
        assert alert is not None
        assert alert.dimension == DriftDimension.ENTROPY.value

    def test_critical_extreme_entropy_rise(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg(critical=0.20))
        baseline = _decisions(20, buy_prob=0.95, sell_prob=0.025, no_trade_prob=0.025)
        recent = _decisions(10, buy_prob=0.34, sell_prob=0.33, no_trade_prob=0.33, ts_start=_NOW)
        alert = monitor._check_entropy(baseline + recent, "v1.0")
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL.value

    def test_insufficient_data_no_alert(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg())
        alert = monitor._check_entropy(_decisions(3), "v1.0")
        assert alert is None

    def test_decision_entropy_helper_uniform_is_max(self) -> None:
        uniform = _decision_entropy({"buy_prob": 1 / 3, "sell_prob": 1 / 3, "no_trade_prob": 1 / 3})
        certain = _decision_entropy({"buy_prob": 1.0, "sell_prob": 0.0, "no_trade_prob": 0.0})
        assert uniform > certain
        assert certain == 0.0 or certain < 1e-6

    def test_decision_entropy_fallback_from_confidence(self) -> None:
        e = _decision_entropy({"confidence": 0.9, "direction": "buy"})
        assert e >= 0.0


# ── Uncertainty drift ─────────────────────────────────────────────────────────


class TestUncertaintyDrift:
    def _cfg(self, warning=0.05, critical=0.10) -> HealthMonitorConfig:
        return HealthMonitorConfig(
            baseline_window=20,
            rolling_window=10,
            min_samples=5,
            uncertainty_warning=warning,
            uncertainty_critical=critical,
        )

    def test_no_alert_stable_spread(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg())
        decisions = [
            {"confidence": 0.7 + (i % 2) * 0.02, "timestamp": _NOW - (30 - i) * 60}
            for i in range(20)
        ] + [
            {"confidence": 0.7 + (i % 2) * 0.02, "timestamp": _NOW - (10 - i) * 60}
            for i in range(10)
        ]
        alert = monitor._check_uncertainty(decisions, "v1.0")
        assert alert is None

    def test_warning_widening_spread(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg())
        baseline = [{"confidence": 0.7, "timestamp": _NOW - (30 - i) * 60} for i in range(20)]
        # Alternate high/low confidence → much higher std
        recent = [
            {"confidence": 0.95 if i % 2 == 0 else 0.35, "timestamp": _NOW - (10 - i) * 60}
            for i in range(10)
        ]
        alert = monitor._check_uncertainty(baseline + recent, "v1.0")
        assert alert is not None
        assert alert.dimension == DriftDimension.UNCERTAINTY.value

    def test_insufficient_data_no_alert(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg())
        alert = monitor._check_uncertainty(_decisions(3), "v1.0")
        assert alert is None

    def test_stdev_helper(self) -> None:
        assert _stdev([1.0]) == 0.0
        assert _stdev([1.0, 1.0, 1.0]) == 0.0
        assert _stdev([0.0, 1.0]) > 0.0


# ── Calibration drift ─────────────────────────────────────────────────────────


class TestCalibrationDrift:
    def _cfg(self, warning=0.08, critical=0.15) -> HealthMonitorConfig:
        return HealthMonitorConfig(
            baseline_window=20,
            rolling_window=10,
            min_samples=5,
            calibration_warning=warning,
            calibration_critical=critical,
        )

    def _well_calibrated_trades(self, n: int, ts_start: float) -> list[dict]:
        # confidence ≈ win rate: alternate so that ~conf fraction win
        out = []
        for i in range(n):
            conf = 0.7
            win = (i % 10) < 7  # 70% win rate matches 0.7 confidence
            out.append(
                {
                    "confidence_at_entry": conf,
                    "pnl": 10.0 if win else -10.0,
                    "timestamp": ts_start + i * 60,
                    "regime": "RANGING",
                    "model_version": "v1.0",
                }
            )
        return out

    def _poorly_calibrated_trades(self, n: int, ts_start: float) -> list[dict]:
        # confidence stays high but win rate collapses → large calibration gap
        out = []
        for i in range(n):
            out.append(
                {
                    "confidence_at_entry": 0.9,
                    "pnl": -10.0 if i % 10 < 8 else 10.0,  # only 20% win rate
                    "timestamp": ts_start + i * 60,
                    "regime": "RANGING",
                    "model_version": "v1.0",
                }
            )
        return out

    def test_no_alert_stable_calibration(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg())
        trades = self._well_calibrated_trades(20, _NOW - 30 * 60) + self._well_calibrated_trades(
            10, _NOW - 10 * 60
        )
        alert = monitor._check_calibration(trades, "v1.0")
        assert alert is None

    def test_alert_worsening_calibration(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg())
        trades = self._well_calibrated_trades(20, _NOW - 30 * 60) + self._poorly_calibrated_trades(
            10, _NOW - 10 * 60
        )
        alert = monitor._check_calibration(trades, "v1.0")
        assert alert is not None
        assert alert.dimension == DriftDimension.CALIBRATION.value
        assert alert.evidence["recent_ece"] > alert.evidence["baseline_ece"]

    def test_insufficient_data_no_alert(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg())
        alert = monitor._check_calibration([], "v1.0")
        assert alert is None

    def test_ece_helper_none_without_confidence_field(self) -> None:
        assert _expected_calibration_error([{"pnl": 1.0}], 5) is None

    def test_ece_helper_perfect_calibration_near_zero(self) -> None:
        trades = self._well_calibrated_trades(50, _NOW)
        ece = _expected_calibration_error(trades, 5)
        assert ece is not None
        assert ece < 0.3  # loose bound; binning coarsens exact match


# ── Full run_check integration ────────────────────────────────────────────────


class TestRunCheck:
    def _cfg(self) -> HealthMonitorConfig:
        return HealthMonitorConfig(
            baseline_window=20,
            rolling_window=10,
            min_samples=5,
            confidence_warning=0.06,
            confidence_critical=0.12,
        )

    def test_healthy_model_no_alerts(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg())
        # Stable data — no drift in any dimension
        for i in range(30):
            warehouse.ingest_decision(
                model_version="v1.0",
                symbol="XAUUSD",
                timestamp=_NOW - (30 - i) * 60,
                direction="buy",
                confidence=0.72,
                regime="RANGING",
            )
        result = monitor.run_check(model_version="v1.0")
        assert result.is_healthy
        assert len(result.alerts) == 0
        assert DriftDimension.CONFIDENCE.value in result.dimensions_checked

    def test_drifting_model_produces_alert(self, db, warehouse) -> None:
        cfg = HealthMonitorConfig(
            baseline_window=20,
            rolling_window=10,
            min_samples=5,
            confidence_warning=0.06,
            confidence_critical=0.20,
        )
        monitor = _make_monitor(db, warehouse, config=cfg)
        # Ingest 20 high-confidence baseline decisions
        for i in range(20):
            warehouse.ingest_decision(
                model_version="v2.0",
                symbol="EURUSD",
                timestamp=_NOW - (30 - i) * 60,
                direction="buy",
                confidence=0.82,
                regime="TRENDING_BULL",
            )
        # Ingest 10 low-confidence recent decisions
        for i in range(10):
            warehouse.ingest_decision(
                model_version="v2.0",
                symbol="EURUSD",
                timestamp=_NOW - (10 - i) * 60,
                direction="no_trade",
                confidence=0.55,
                regime="TRENDING_BULL",
            )
        result = monitor.run_check(model_version="v2.0")
        confidence_alerts = [
            a for a in result.alerts if a.dimension == DriftDimension.CONFIDENCE.value
        ]
        assert len(confidence_alerts) >= 1

    def test_alerts_persisted_to_db(self, db, warehouse) -> None:
        cfg = HealthMonitorConfig(
            baseline_window=20,
            rolling_window=10,
            min_samples=5,
            confidence_warning=0.01,
            confidence_critical=0.20,
        )
        monitor = _make_monitor(db, warehouse, config=cfg)
        for i in range(20):
            warehouse.ingest_decision(
                model_version="v3.0",
                symbol="XAUUSD",
                timestamp=_NOW - (30 - i) * 60,
                direction="buy",
                confidence=0.80,
                regime="RANGING",
            )
        for i in range(10):
            warehouse.ingest_decision(
                model_version="v3.0",
                symbol="XAUUSD",
                timestamp=_NOW - (10 - i) * 60,
                direction="buy",
                confidence=0.60,
                regime="RANGING",
            )
        monitor.run_check(model_version="v3.0")
        alerts_in_db = db.list_health_alerts(model_version="v3.0")
        assert len(alerts_in_db) > 0

    def test_result_dimensions_tracked(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, config=self._cfg())
        result = monitor.run_check(model_version="v-empty")
        assert set(result.dimensions_checked) == {d.value for d in DriftDimension}


# ── CRITICAL → RetrainingRecommendation ───────────────────────────────────────


class TestCriticalAlertCreatesRecommendation:
    def _cfg(self) -> HealthMonitorConfig:
        return HealthMonitorConfig(
            baseline_window=20,
            rolling_window=10,
            min_samples=5,
            confidence_warning=0.01,
            confidence_critical=0.02,  # very low → force CRITICAL
        )

    def _mock_rec_engine(self, db: ResearchLabDatabase):
        """Thin mock that delegates to the DB so dedup logic works."""
        from trainer.research_lab.retraining_recommendation_engine import (
            RetrainingRecommendationEngine,
        )
        from trainer.research_lab.telemetry_warehouse import TelemetryWarehouse

        wh = TelemetryWarehouse(db)
        from trainer.research_lab.recommendation_models import RecommendationConfig

        return RetrainingRecommendationEngine(wh, db, RecommendationConfig())

    def test_critical_alert_creates_recommendation(self, db, warehouse) -> None:
        rec_engine = self._mock_rec_engine(db)
        monitor = _make_monitor(db, warehouse, rec_engine=rec_engine, config=self._cfg())

        # Force a CRITICAL confidence alert
        decisions = _decisions(20, confidence=0.80) + _decisions(10, confidence=0.55, ts_start=_NOW)
        alert = monitor._check_confidence(decisions, "v-crit-test")
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL.value

        rec_id = monitor._maybe_create_recommendation(alert)
        assert rec_id.startswith("rr-")

        # Verify persisted
        recs = db.list_retraining_recommendations(model_version="v-crit-test")
        assert len(recs) == 1
        assert recs[0]["recommendation_id"] == rec_id

    def test_deduplication_prevents_duplicate_recommendation(self, db, warehouse) -> None:
        rec_engine = self._mock_rec_engine(db)
        monitor = _make_monitor(db, warehouse, rec_engine=rec_engine, config=self._cfg())

        decisions = _decisions(20, confidence=0.80) + _decisions(10, confidence=0.55, ts_start=_NOW)
        alert = monitor._check_confidence(decisions, "v-dedup")
        rec_id_1 = monitor._maybe_create_recommendation(alert)
        rec_id_2 = monitor._maybe_create_recommendation(alert)

        # Must return same recommendation, not create a second
        assert rec_id_1 == rec_id_2
        recs = db.list_retraining_recommendations(model_version="v-dedup")
        assert len(recs) == 1

    def test_distinct_health_dimensions_both_create_recommendations(self, db, warehouse) -> None:
        """ENTROPY and CALIBRATION both map to the same coarse EvaluationDimension
        (CONFIDENCE_DRIFT) for categorization, but dedup must not collapse them —
        an open entropy recommendation must not suppress a distinct calibration one.
        """
        rec_engine = self._mock_rec_engine(db)
        monitor = _make_monitor(db, warehouse, rec_engine=rec_engine, config=self._cfg())

        entropy_alert = DriftAlert(
            alert_id="hal-ent",
            dimension=DriftDimension.ENTROPY.value,
            severity="CRITICAL",
            model_version="v-multi",
            affected_symbols=[],
            affected_features=[],
            evidence={},
            recommended_action="test entropy",
            created_at=time.time(),
        )
        calibration_alert = DriftAlert(
            alert_id="hal-cal",
            dimension=DriftDimension.CALIBRATION.value,
            severity="CRITICAL",
            model_version="v-multi",
            affected_symbols=[],
            affected_features=[],
            evidence={},
            recommended_action="test calibration",
            created_at=time.time(),
        )
        rec_id_1 = monitor._maybe_create_recommendation(entropy_alert)
        rec_id_2 = monitor._maybe_create_recommendation(calibration_alert)

        assert rec_id_1 and rec_id_2
        assert rec_id_1 != rec_id_2
        recs = db.list_retraining_recommendations(model_version="v-multi")
        assert len(recs) == 2
        assert {r["recommended_scope"] for r in recs} == {"FINE_TUNE", "CALIBRATION_ONLY"}

    def test_no_rec_engine_returns_empty(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse, rec_engine=None, config=self._cfg())
        alert = DriftAlert(
            alert_id="hal-x",
            dimension="CONFIDENCE",
            severity="CRITICAL",
            model_version="v-test",
            affected_symbols=[],
            affected_features=[],
            evidence={},
            recommended_action="test",
            created_at=time.time(),
        )
        rec_id = monitor._maybe_create_recommendation(alert)
        assert rec_id == ""


# ── DB CRUD ───────────────────────────────────────────────────────────────────


class TestHealthAlertDB:
    def _alert(self, alert_id: str = "hal-db01", severity: str = "WARNING") -> dict:
        return {
            "alert_id": alert_id,
            "dimension": "CONFIDENCE",
            "severity": severity,
            "model_version": "v1.0",
            "affected_symbols": [],
            "affected_features": ["confidence"],
            "evidence": {"drift": 0.07},
            "recommended_action": "Monitor.",
            "created_at": time.time(),
            "recommendation_id": "",
        }

    def test_upsert_and_get(self, db: ResearchLabDatabase) -> None:
        a = self._alert()
        db.upsert_health_alert(a)
        fetched = db.get_health_alert("hal-db01")
        assert fetched is not None
        assert fetched["severity"] == "WARNING"

    def test_upsert_overwrites(self, db: ResearchLabDatabase) -> None:
        a = self._alert()
        db.upsert_health_alert(a)
        a["severity"] = "CRITICAL"
        db.upsert_health_alert(a)
        fetched = db.get_health_alert("hal-db01")
        assert fetched["severity"] == "CRITICAL"

    def test_list_filter_severity(self, db: ResearchLabDatabase) -> None:
        db.upsert_health_alert(self._alert("hal-w", "WARNING"))
        db.upsert_health_alert(self._alert("hal-c", "CRITICAL"))
        crits = db.list_health_alerts(severity="CRITICAL")
        assert len(crits) == 1
        assert crits[0]["alert_id"] == "hal-c"

    def test_list_filter_dimension(self, db: ResearchLabDatabase) -> None:
        a = self._alert()
        a["dimension"] = "MARKET"
        db.upsert_health_alert(a)
        db.upsert_health_alert(self._alert("hal-conf", "WARNING"))
        market = db.list_health_alerts(dimension="MARKET")
        assert len(market) == 1

    def test_list_filter_model_version(self, db: ResearchLabDatabase) -> None:
        a1 = self._alert("hal-v1")
        a1["model_version"] = "v2.0"
        a2 = self._alert("hal-v2")
        a2["model_version"] = "v3.0"
        db.upsert_health_alert(a1)
        db.upsert_health_alert(a2)
        v2_alerts = db.list_health_alerts(model_version="v2.0")
        assert len(v2_alerts) == 1
        assert v2_alerts[0]["model_version"] == "v2.0"

    def test_list_limit(self, db: ResearchLabDatabase) -> None:
        for i in range(5):
            db.upsert_health_alert(self._alert(f"hal-limit-{i}"))
        rows = db.list_health_alerts(limit=3)
        assert len(rows) == 3

    def test_stats_includes_health_alerts(self, db: ResearchLabDatabase) -> None:
        db.upsert_health_alert(self._alert())
        stats = db.stats()
        assert stats.get("health_alerts", 0) == 1

    def test_get_nonexistent_returns_none(self, db: ResearchLabDatabase) -> None:
        assert db.get_health_alert("hal-not-exist") is None

    def test_scalar_columns_refreshed_on_upsert(self, db: ResearchLabDatabase) -> None:
        """Scalar filter/index columns must stay in sync with data_json on update.

        Regression guard for the ON CONFLICT clause — if only data_json is
        refreshed, list_health_alerts(severity=X) returns stale rows.
        """
        a = self._alert("hal-update-test", severity="WARNING")
        db.upsert_health_alert(a)

        # Now update to CRITICAL via another upsert with the same alert_id
        a2 = dict(a)
        a2["severity"] = "CRITICAL"
        a2["dimension"] = "MARKET"
        a2["model_version"] = "v2.0"
        db.upsert_health_alert(a2)

        # Scalar-column filter must reflect the new values
        crits = db.list_health_alerts(severity="CRITICAL")
        assert any(
            r["alert_id"] == "hal-update-test" for r in crits
        ), "Updated severity not reflected in scalar column — filter returned nothing"
        warns = db.list_health_alerts(severity="WARNING")
        assert not any(
            r["alert_id"] == "hal-update-test" for r in warns
        ), "Old severity still visible in scalar index after update"
        market = db.list_health_alerts(dimension="MARKET")
        assert any(r["alert_id"] == "hal-update-test" for r in market)
        v2 = db.list_health_alerts(model_version="v2.0")
        assert any(r["alert_id"] == "hal-update-test" for r in v2)


# ── get_status ────────────────────────────────────────────────────────────────


class TestGetStatus:
    def test_status_empty(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse)
        status = monitor.get_status(model_version="v-empty")
        assert status["is_healthy"] is True
        assert status["n_alerts_total"] == 0

    def test_status_with_alerts(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse)
        db.upsert_health_alert(
            {
                "alert_id": "hal-s1",
                "dimension": "CONFIDENCE",
                "severity": "CRITICAL",
                "model_version": "v1.0",
                "created_at": time.time(),
                "affected_symbols": [],
                "affected_features": [],
                "evidence": {},
                "recommended_action": "",
                "recommendation_id": "",
            }
        )
        db.upsert_health_alert(
            {
                "alert_id": "hal-s2",
                "dimension": "MARKET",
                "severity": "WARNING",
                "model_version": "v1.0",
                "created_at": time.time() - 1,
                "affected_symbols": [],
                "affected_features": [],
                "evidence": {},
                "recommended_action": "",
                "recommendation_id": "",
            }
        )
        status = monitor.get_status(model_version="v1.0")
        assert status["is_healthy"] is False
        assert status["n_critical"] == 1
        assert status["n_warning"] == 1
        assert "CONFIDENCE" in status["by_dimension"]

    def test_status_dimensions_listed(self, db, warehouse) -> None:
        monitor = _make_monitor(db, warehouse)
        status = monitor.get_status()
        assert set(status["dimensions"]) == {d.value for d in DriftDimension}


# ── Server endpoints ──────────────────────────────────────────────────────────


class TestHealthServerEndpoints:
    @property
    def _HDR(self) -> dict:
        # Minted fresh on each access (not a class-level constant) since the
        # JWT's `iat` is checked against a clock-skew window and would go
        # stale if computed once at class-definition/import time.
        return {
            "Authorization": "Bearer test-health-token",
            "X-Signalium-Token": make_signalium_jwt(),
        }

    @pytest.fixture()
    def client(self, tmp_path: Path):
        import os

        os.environ["RESEARCH_LAB_ADMIN_TOKEN"] = "test-health-token"
        os.environ["SIGNALIUM_JWT_SECRET"] = DEFAULT_TEST_JWT_SECRET
        os.environ["RESEARCH_LAB_DB_PATH"] = str(tmp_path / "server_health.db")
        for mod_name in list(sys.modules):
            if "research_lab_server" in mod_name or "telemetry_server" in mod_name:
                del sys.modules[mod_name]
        from starlette.testclient import TestClient

        import trainer.research_lab.research_lab_server as srv_mod

        return TestClient(srv_mod.app)

    def test_run_check_empty_warehouse(self, client) -> None:
        resp = client.post("/health/check", headers=self._HDR)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        hc = body["health_check"]
        assert "is_healthy" in hc
        assert "dimensions_checked" in hc
        assert len(hc["dimensions_checked"]) == 13

    def test_run_check_model_version_param(self, client) -> None:
        resp = client.post("/health/check?model_version=v2.0", headers=self._HDR)
        assert resp.status_code == 200
        assert resp.json()["health_check"]["model_version"] == "v2.0"

    def test_get_status_empty(self, client) -> None:
        resp = client.get("/health/status", headers=self._HDR)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["status"]["is_healthy"] is True
        assert body["status"]["n_alerts_total"] == 0

    def test_list_alerts_empty(self, client) -> None:
        resp = client.get("/health/alerts", headers=self._HDR)
        assert resp.status_code == 200
        assert resp.json()["total"] == 0
        assert resp.json()["alerts"] == []

    def test_alerts_after_check(self, client) -> None:
        """After running a check with stable data, list should have 0 alerts."""
        client.post("/health/check?model_version=v1.0", headers=self._HDR)
        resp = client.get("/health/alerts?model_version=v1.0", headers=self._HDR)
        assert resp.status_code == 200
        # No data → no alerts
        assert resp.json()["total"] == 0

    def test_health_check_requires_auth(self, client) -> None:
        resp = client.post("/health/check")
        assert resp.status_code == 401  # no token → 401 Unauthorized

    def test_health_status_requires_auth(self, client) -> None:
        resp = client.get("/health/status")
        assert resp.status_code == 401  # no token → 401 Unauthorized

    def test_health_alerts_requires_auth(self, client) -> None:
        resp = client.get("/health/alerts")
        assert resp.status_code == 401  # no token → 401 Unauthorized

    def test_list_alerts_filter_severity(self, client) -> None:
        resp = client.get("/health/alerts?severity=CRITICAL", headers=self._HDR)
        assert resp.status_code == 200

    def test_list_alerts_filter_dimension(self, client) -> None:
        resp = client.get("/health/alerts?dimension=CONFIDENCE", headers=self._HDR)
        assert resp.status_code == 200
