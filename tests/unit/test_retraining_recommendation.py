"""Unit tests for S20 Retraining Recommendation Engine.

Covers:
- RecommendationConfig defaults + field structure
- SharpeDegradationEvaluator: threshold crossing and non-trigger
- ConfidenceDriftEvaluator: threshold crossing and non-trigger
- ReasoningDriftEvaluator: threshold crossing and non-trigger
- RegimeUnderperformEvaluator: per-regime Sharpe
- PortfolioInefficiencyEvaluator: HHI + scale-in rate
- RetrainingRecommendationEngine: orchestration, deduplication, approve, dismiss
- DB: upsert/get/list retraining_recommendations
- Server endpoints: GET /research/recommendations, POST evaluate, approve, dismiss
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from trainer.research_lab.dimension_evaluators import (
    ConfidenceDriftEvaluator,
    PortfolioInefficiencyEvaluator,
    ReasoningDriftEvaluator,
    RegimeUnderperformEvaluator,
    SharpeDegradationEvaluator,
)
from trainer.research_lab.recommendation_models import (
    EvaluationDimension,
    RecommendationConfig,
    RecommendationStatus,
    RecommendationUrgency,
    RecommendedScope,
    RetrainingRecommendation,
    _elevate_urgency,
    _urgency_from_score,
)
from trainer.research_lab.retraining_recommendation_engine import (
    RetrainingRecommendationEngine,
)
from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.telemetry_warehouse import TelemetryWarehouse


# ── Module-scope Pydantic models (Pydantic v2 + `from __future__ import annotations`) ──

class _EvaluateReq(BaseModel):
    model_version: str | None = None
    since: float | None = None
    until: float | None = None


class _ResolveReq(BaseModel):
    resolved_by: str = "admin"
    notes: str = ""


# ── Fixtures ───────────────────────────────────────────────────────────────────

NOW = time.time()
MV = "v3.0.0"


@pytest.fixture()
def db(tmp_path: Path) -> ResearchLabDatabase:
    return ResearchLabDatabase(str(tmp_path / "test_rr.db"))


@pytest.fixture()
def warehouse(db: ResearchLabDatabase) -> TelemetryWarehouse:
    return TelemetryWarehouse(db)


@pytest.fixture()
def cfg() -> RecommendationConfig:
    return RecommendationConfig(
        min_trades_for_evaluation=3,
        baseline_window=20,
        rolling_window=10,
        sharpe_degradation_threshold=0.25,
        confidence_drift_threshold=0.08,
        reasoning_drift_threshold=0.15,
        regime_sharpe_threshold=0.0,
        portfolio_concentration_threshold=0.5,
    )


@pytest.fixture()
def engine(warehouse: TelemetryWarehouse, db: ResearchLabDatabase, cfg: RecommendationConfig):
    return RetrainingRecommendationEngine(warehouse, db, config=cfg)


# ── Telemetry helpers ──────────────────────────────────────────────────────────


def _trade(wh: TelemetryWarehouse, pnl_pct: float, mv: str = MV,
           conf: float = 0.7, symbol: str = "XAUUSD",
           regime: str = "TRENDING_BULL", ts: float | None = None):
    return wh.ingest_trade(
        model_version=mv, symbol=symbol, timestamp=ts or NOW,
        direction="LONG", entry_price=1950.0,
        exit_price=1950.0 + pnl_pct * 19.5,
        size=0.1, pnl=pnl_pct * 195, pnl_pct=pnl_pct,
        hold_duration_s=3600.0, confidence_at_entry=conf, regime=regime,
    )


def _decision(wh: TelemetryWarehouse, conf: float, mv: str = MV,
              symbol: str = "XAUUSD", ts: float | None = None):
    return wh.ingest_decision(
        model_version=mv, symbol=symbol, timestamp=ts or NOW,
        direction="LONG", confidence=conf,
    )


def _reasoning(wh: TelemetryWarehouse, output_heads: dict[str, float],
               mv: str = MV, ts: float | None = None):
    return wh.ingest_reasoning(
        model_version=mv, symbol="XAUUSD", timestamp=ts or NOW,
        decision_id="dec-test", output_heads=output_heads,
    )


def _position(wh: TelemetryWarehouse, cum_pnl: float, action: str = "SCALE_IN",
              mv: str = MV):
    return wh.ingest_position(
        model_version=mv, symbol="XAUUSD", timestamp=NOW,
        action=action, size=0.1, price=1950.0, cumulative_pnl=cum_pnl,
    )


# ── RecommendationConfig ───────────────────────────────────────────────────────


class TestRecommendationConfig:
    def test_defaults_are_reasonable(self) -> None:
        cfg = RecommendationConfig()
        assert 0 < cfg.sharpe_degradation_threshold < 1
        assert 0 < cfg.confidence_drift_threshold < 0.5
        assert 0 < cfg.reasoning_drift_threshold < 1
        assert cfg.min_trades_for_evaluation >= 1
        assert cfg.baseline_window > cfg.rolling_window

    def test_urgency_from_score(self) -> None:
        assert _urgency_from_score(0.0) == RecommendationUrgency.LOW
        assert _urgency_from_score(0.3) == RecommendationUrgency.LOW
        assert _urgency_from_score(0.6) == RecommendationUrgency.MEDIUM
        assert _urgency_from_score(2.0) == RecommendationUrgency.HIGH
        assert _urgency_from_score(4.0) == RecommendationUrgency.CRITICAL

    def test_elevate_urgency_bumps_on_multiple_dimensions(self) -> None:
        base = RecommendationUrgency.LOW
        assert _elevate_urgency(base, 1) == RecommendationUrgency.LOW
        assert _elevate_urgency(base, 2) == RecommendationUrgency.MEDIUM
        assert _elevate_urgency(base, 3) == RecommendationUrgency.HIGH
        assert _elevate_urgency(base, 4) == RecommendationUrgency.CRITICAL


# ── RetrainingRecommendation model ────────────────────────────────────────────


class TestRetrainingRecommendation:
    def test_new_starts_in_proposed(self) -> None:
        r = RetrainingRecommendation.new(
            model_version=MV,
            trigger_dimension=EvaluationDimension.SHARPE_DEGRADATION,
            trigger_reason="test reason",
            evidence={"baseline_sharpe": 1.2, "recent_sharpe": 0.4},
            recommended_scope=RecommendedScope.FINE_TUNE,
            urgency=RecommendationUrgency.MEDIUM,
        )
        assert r.status == RecommendationStatus.PROPOSED.value
        assert r.recommendation_id.startswith("rr-")
        assert r.model_version == MV
        assert r.resolved_at is None

    def test_approve_changes_status(self) -> None:
        r = RetrainingRecommendation.new(
            model_version=MV,
            trigger_dimension=EvaluationDimension.CONFIDENCE_DRIFT,
            trigger_reason="drift",
            evidence={}, recommended_scope=RecommendedScope.CALIBRATION_ONLY,
            urgency=RecommendationUrgency.LOW,
        )
        r.approve(resolved_by="head_quant")
        assert r.status == RecommendationStatus.APPROVED.value
        assert r.resolved_by == "head_quant"
        assert r.resolved_at is not None

    def test_dismiss_changes_status(self) -> None:
        r = RetrainingRecommendation.new(
            model_version=MV,
            trigger_dimension=EvaluationDimension.REGIME_UNDERPERFORMANCE,
            trigger_reason="regime",
            evidence={}, recommended_scope=RecommendedScope.FINE_TUNE,
            urgency=RecommendationUrgency.LOW,
        )
        r.dismiss(resolved_by="admin", notes="false alarm")
        assert r.status == RecommendationStatus.DISMISSED.value
        assert r.notes == "false alarm"

    def test_approve_non_proposed_raises(self) -> None:
        r = RetrainingRecommendation.new(
            model_version=MV,
            trigger_dimension=EvaluationDimension.SHARPE_DEGRADATION,
            trigger_reason="x", evidence={},
            recommended_scope=RecommendedScope.FINE_TUNE,
            urgency=RecommendationUrgency.LOW,
        )
        r.approve()
        with pytest.raises(ValueError, match="PROPOSED"):
            r.approve()

    def test_to_dict_has_required_keys(self) -> None:
        r = RetrainingRecommendation.new(
            model_version=MV,
            trigger_dimension=EvaluationDimension.PORTFOLIO_INEFFICIENCY,
            trigger_reason="hhi",
            evidence={"herfindahl_index": 0.8},
            recommended_scope=RecommendedScope.FINE_TUNE,
            urgency=RecommendationUrgency.HIGH,
        )
        d = r.to_dict()
        required = {
            "recommendation_id", "model_version", "trigger_dimension",
            "trigger_reason", "evidence", "recommended_scope", "urgency",
            "status", "created_at", "updated_at",
        }
        assert required.issubset(d.keys())


# ── SharpeDegradationEvaluator ────────────────────────────────────────────────


class TestSharpeDegradationEvaluator:
    def test_triggers_when_sharpe_degrades(
        self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig
    ) -> None:
        # Baseline: consistently profitable
        for i in range(20):
            _trade(warehouse, pnl_pct=0.5, ts=NOW + i)
        # Recent: consistently losing
        for i in range(10):
            _trade(warehouse, pnl_pct=-1.0, ts=NOW + 100 + i)
        ev = SharpeDegradationEvaluator(warehouse, cfg)
        result = ev.evaluate()
        assert result.triggered is True
        assert result.evidence["recent_sharpe"] < result.evidence["baseline_sharpe"]
        assert result.dimension == EvaluationDimension.SHARPE_DEGRADATION

    def test_does_not_trigger_when_sharpe_stable(
        self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig
    ) -> None:
        for i in range(30):
            _trade(warehouse, pnl_pct=0.5, ts=NOW + i)
        ev = SharpeDegradationEvaluator(warehouse, cfg)
        result = ev.evaluate()
        assert result.triggered is False

    def test_insufficient_data_returns_not_triggered(
        self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig
    ) -> None:
        _trade(warehouse, pnl_pct=0.5)
        ev = SharpeDegradationEvaluator(warehouse, cfg)
        result = ev.evaluate()
        assert result.triggered is False
        assert result.score == 0.0

    def test_critical_degradation_scopes_to_full_retrain(
        self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig
    ) -> None:
        for i in range(20):
            _trade(warehouse, pnl_pct=2.0, ts=NOW + i)
        for i in range(10):
            _trade(warehouse, pnl_pct=-5.0, ts=NOW + 100 + i)
        ev = SharpeDegradationEvaluator(warehouse, cfg)
        result = ev.evaluate()
        assert result.triggered
        if result.urgency == RecommendationUrgency.CRITICAL:
            assert result.recommended_scope == RecommendedScope.FULL_RETRAIN

    def test_evidence_fields_present(
        self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig
    ) -> None:
        for i in range(25):
            pnl = 0.5 if i < 20 else -1.0
            _trade(warehouse, pnl_pct=pnl, ts=NOW + i)
        ev = SharpeDegradationEvaluator(warehouse, cfg)
        result = ev.evaluate()
        for key in ("baseline_sharpe", "recent_sharpe", "degradation_fraction",
                    "n_baseline_trades", "n_recent_trades"):
            assert key in result.evidence


# ── ConfidenceDriftEvaluator ──────────────────────────────────────────────────


class TestConfidenceDriftEvaluator:
    def test_triggers_when_confidence_shifts(
        self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig
    ) -> None:
        for i in range(20):
            _decision(warehouse, conf=0.85, ts=NOW + i)
        for i in range(10):
            _decision(warehouse, conf=0.55, ts=NOW + 100 + i)
        ev = ConfidenceDriftEvaluator(warehouse, cfg)
        result = ev.evaluate()
        assert result.triggered is True
        assert result.dimension == EvaluationDimension.CONFIDENCE_DRIFT
        assert result.recommended_scope == RecommendedScope.CALIBRATION_ONLY

    def test_does_not_trigger_when_stable(
        self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig
    ) -> None:
        for i in range(30):
            _decision(warehouse, conf=0.70, ts=NOW + i)
        ev = ConfidenceDriftEvaluator(warehouse, cfg)
        result = ev.evaluate()
        assert result.triggered is False

    def test_evidence_includes_calibration_corr_when_trades_present(
        self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig
    ) -> None:
        for i in range(20):
            _decision(warehouse, conf=0.85, ts=NOW + i)
            _trade(warehouse, pnl_pct=0.5, conf=0.85, ts=NOW + i + 0.1)
        for i in range(10):
            _decision(warehouse, conf=0.55, ts=NOW + 100 + i)
        ev = ConfidenceDriftEvaluator(warehouse, cfg)
        result = ev.evaluate()
        assert "confidence_outcome_correlation" in result.evidence

    def test_insufficient_data_not_triggered(
        self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig
    ) -> None:
        _decision(warehouse, conf=0.9)
        ev = ConfidenceDriftEvaluator(warehouse, cfg)
        result = ev.evaluate()
        assert result.triggered is False

    def test_score_proportional_to_drift(
        self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig
    ) -> None:
        # Big drift should produce high score
        for i in range(20):
            _decision(warehouse, conf=0.95, ts=NOW + i)
        for i in range(10):
            _decision(warehouse, conf=0.05, ts=NOW + 100 + i)
        ev = ConfidenceDriftEvaluator(warehouse, cfg)
        result = ev.evaluate()
        assert result.score > 5.0  # 0.9 drift / 0.08 threshold ≈ 11


# ── ReasoningDriftEvaluator ───────────────────────────────────────────────────


class TestReasoningDriftEvaluator:
    def test_triggers_when_heads_shift(
        self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig
    ) -> None:
        for i in range(20):
            _reasoning(warehouse, {"entry_signal": 0.8, "exit_signal": 0.2}, ts=NOW + i)
        for i in range(10):
            _reasoning(warehouse, {"entry_signal": 0.1, "exit_signal": 0.9}, ts=NOW + 100 + i)
        ev = ReasoningDriftEvaluator(warehouse, cfg)
        result = ev.evaluate()
        assert result.triggered is True
        assert result.dimension == EvaluationDimension.REASONING_DRIFT
        assert result.recommended_scope == RecommendedScope.HEAD_RETRAIN

    def test_does_not_trigger_when_stable(
        self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig
    ) -> None:
        for i in range(30):
            _reasoning(warehouse, {"entry_signal": 0.7, "exit_signal": 0.3}, ts=NOW + i)
        ev = ReasoningDriftEvaluator(warehouse, cfg)
        result = ev.evaluate()
        assert result.triggered is False

    def test_no_shared_heads_returns_not_triggered(
        self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig
    ) -> None:
        for i in range(20):
            _reasoning(warehouse, {"head_a": 0.5}, ts=NOW + i)
        for i in range(10):
            _reasoning(warehouse, {"head_b": 0.5}, ts=NOW + 100 + i)
        ev = ReasoningDriftEvaluator(warehouse, cfg)
        result = ev.evaluate()
        # No shared heads — cannot compute drift
        assert result.triggered is False

    def test_evidence_contains_top_drifting_heads(
        self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig
    ) -> None:
        for i in range(20):
            _reasoning(warehouse, {"a": 0.9, "b": 0.1, "c": 0.5}, ts=NOW + i)
        for i in range(10):
            _reasoning(warehouse, {"a": 0.1, "b": 0.9, "c": 0.5}, ts=NOW + 100 + i)
        ev = ReasoningDriftEvaluator(warehouse, cfg)
        result = ev.evaluate()
        assert "top_drifting_heads" in result.evidence
        assert "avg_head_drift" in result.evidence


# ── RegimeUnderperformEvaluator ───────────────────────────────────────────────


class TestRegimeUnderperformEvaluator:
    def test_triggers_for_negative_regime_sharpe(
        self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig
    ) -> None:
        # Good regime: positive PnL
        for i in range(10):
            _trade(warehouse, pnl_pct=1.0, regime="TRENDING_BULL", ts=NOW + i)
        # Bad regime: negative PnL
        for i in range(8):
            _trade(warehouse, pnl_pct=-2.0, regime="HIGH_VOLATILITY", ts=NOW + 100 + i)
        ev = RegimeUnderperformEvaluator(warehouse, cfg)
        result = ev.evaluate()
        assert result.triggered is True
        assert result.dimension == EvaluationDimension.REGIME_UNDERPERFORMANCE
        assert "HIGH_VOLATILITY" in result.evidence["underperforming_regimes"]

    def test_does_not_trigger_when_all_regimes_positive(
        self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig
    ) -> None:
        for i in range(10):
            _trade(warehouse, pnl_pct=0.5, regime="TRENDING_BULL", ts=NOW + i)
        for i in range(8):
            _trade(warehouse, pnl_pct=0.3, regime="RANGING", ts=NOW + 100 + i)
        ev = RegimeUnderperformEvaluator(warehouse, cfg)
        result = ev.evaluate()
        assert result.triggered is False

    def test_evidence_per_regime_sharpe(
        self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig
    ) -> None:
        for i in range(5):
            _trade(warehouse, pnl_pct=-1.0, regime="CRISIS", ts=NOW + i)
        ev = RegimeUnderperformEvaluator(warehouse, cfg)
        result = ev.evaluate()
        if result.triggered:
            assert "per_regime_sharpe" in result.evidence
            assert "CRISIS" in result.evidence["per_regime_sharpe"]

    def test_insufficient_data_not_triggered(
        self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig
    ) -> None:
        _trade(warehouse, pnl_pct=-1.0)
        ev = RegimeUnderperformEvaluator(warehouse, cfg)
        result = ev.evaluate()
        assert result.triggered is False

    def test_regime_with_single_trade_excluded_from_sharpe(
        self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig
    ) -> None:
        # Need ≥2 trades per regime to compute Sharpe
        for i in range(5):
            _trade(warehouse, pnl_pct=0.5, regime="TRENDING_BULL", ts=NOW + i)
        _trade(warehouse, pnl_pct=-2.0, regime="CRISIS", ts=NOW + 10)  # single trade
        ev = RegimeUnderperformEvaluator(warehouse, cfg)
        result = ev.evaluate()
        # CRISIS has only 1 trade, should not appear in per_regime_sharpe
        per_regime = result.evidence.get("per_regime_sharpe", {})
        assert "CRISIS" not in per_regime


# ── PortfolioInefficiencyEvaluator ────────────────────────────────────────────


class TestPortfolioInefficiencyEvaluator:
    def test_triggers_on_high_concentration(
        self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig
    ) -> None:
        # All trades in one symbol → HHI = 1.0 > 0.5
        for i in range(10):
            _trade(warehouse, pnl_pct=0.5, symbol="XAUUSD", ts=NOW + i)
        ev = PortfolioInefficiencyEvaluator(warehouse, cfg)
        result = ev.evaluate()
        assert result.triggered is True
        assert result.dimension == EvaluationDimension.PORTFOLIO_INEFFICIENCY
        assert result.evidence["herfindahl_index"] == pytest.approx(1.0)

    def test_triggers_on_low_scale_in_success_rate(
        self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig
    ) -> None:
        # Diversified trades (low HHI)
        for sym in ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"]:
            for i in range(3):
                _trade(warehouse, pnl_pct=0.2, symbol=sym, ts=NOW + i)
        # Mostly losing scale-ins
        for i in range(10):
            _position(warehouse, cum_pnl=-50.0, action="SCALE_IN")
        ev = PortfolioInefficiencyEvaluator(warehouse, cfg)
        result = ev.evaluate()
        assert result.triggered is True
        assert result.evidence["scale_in_success_rate"] < 0.4

    def test_does_not_trigger_when_diversified_and_scale_ins_ok(
        self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig
    ) -> None:
        for sym in ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "BTCUSD"]:
            for i in range(4):
                _trade(warehouse, pnl_pct=0.2, symbol=sym, ts=NOW + i)
        for i in range(10):
            _position(warehouse, cum_pnl=100.0, action="SCALE_IN")
        ev = PortfolioInefficiencyEvaluator(warehouse, cfg)
        result = ev.evaluate()
        assert result.triggered is False

    def test_evidence_fields_present(
        self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig
    ) -> None:
        for i in range(5):
            _trade(warehouse, pnl_pct=0.5, symbol="XAUUSD", ts=NOW + i)
        ev = PortfolioInefficiencyEvaluator(warehouse, cfg)
        result = ev.evaluate()
        for key in ("herfindahl_index", "dominant_symbol", "dominant_symbol_pct",
                    "scale_in_success_rate", "n_symbols"):
            assert key in result.evidence

    def test_empty_data_not_triggered(
        self, warehouse: TelemetryWarehouse, cfg: RecommendationConfig
    ) -> None:
        ev = PortfolioInefficiencyEvaluator(warehouse, cfg)
        result = ev.evaluate()
        assert result.triggered is False


# ── RetrainingRecommendationEngine ────────────────────────────────────────────


class TestRetrainingRecommendationEngine:
    def test_evaluate_returns_five_results(
        self, engine: RetrainingRecommendationEngine, warehouse: TelemetryWarehouse
    ) -> None:
        for i in range(25):
            _trade(warehouse, pnl_pct=0.3, ts=NOW + i)
        results = engine.evaluate()
        assert len(results) == 5
        dimensions = {r.dimension for r in results}
        assert len(dimensions) == 5  # all unique

    def test_triggered_dimensions_create_recommendations(
        self, engine: RetrainingRecommendationEngine, warehouse: TelemetryWarehouse,
        db: ResearchLabDatabase
    ) -> None:
        # Induce Sharpe degradation
        for i in range(20):
            _trade(warehouse, pnl_pct=1.0, ts=NOW + i)
        for i in range(10):
            _trade(warehouse, pnl_pct=-2.0, ts=NOW + 100 + i)
        engine.evaluate(model_version=MV)
        recs = db.list_retraining_recommendations(status="PROPOSED")
        assert len(recs) >= 1
        dims = {r["trigger_dimension"] for r in recs}
        assert "sharpe_degradation" in dims

    def test_deduplication_prevents_duplicate_open_recommendations(
        self, engine: RetrainingRecommendationEngine, warehouse: TelemetryWarehouse,
        db: ResearchLabDatabase
    ) -> None:
        # Populate with degraded Sharpe
        for i in range(20):
            _trade(warehouse, pnl_pct=1.0, ts=NOW + i)
        for i in range(10):
            _trade(warehouse, pnl_pct=-2.0, ts=NOW + 100 + i)

        engine.evaluate(model_version=MV)
        count_after_first = len(db.list_retraining_recommendations(status="PROPOSED"))
        engine.evaluate(model_version=MV)
        count_after_second = len(db.list_retraining_recommendations(status="PROPOSED"))
        # Second run should not create duplicates
        assert count_after_second == count_after_first

    def test_approve_changes_status_to_approved(
        self, engine: RetrainingRecommendationEngine, warehouse: TelemetryWarehouse,
        db: ResearchLabDatabase
    ) -> None:
        for i in range(20):
            _trade(warehouse, pnl_pct=1.0, ts=NOW + i)
        for i in range(10):
            _trade(warehouse, pnl_pct=-2.0, ts=NOW + 100 + i)
        engine.evaluate(model_version=MV)
        recs = db.list_retraining_recommendations(status="PROPOSED")
        assert recs
        rec_id = recs[0]["recommendation_id"]
        engine.approve(rec_id, resolved_by="head_quant")
        updated = db.get_retraining_recommendation(rec_id)
        assert updated["status"] == "APPROVED"
        assert updated["resolved_by"] == "head_quant"
        assert updated["resolved_at"] is not None

    def test_dismiss_changes_status_to_dismissed(
        self, engine: RetrainingRecommendationEngine, warehouse: TelemetryWarehouse,
        db: ResearchLabDatabase
    ) -> None:
        for i in range(20):
            _trade(warehouse, pnl_pct=1.0, ts=NOW + i)
        for i in range(10):
            _trade(warehouse, pnl_pct=-2.0, ts=NOW + 100 + i)
        engine.evaluate(model_version=MV)
        recs = db.list_retraining_recommendations(status="PROPOSED")
        assert recs
        rec_id = recs[0]["recommendation_id"]
        engine.dismiss(rec_id, resolved_by="admin", notes="false positive")
        updated = db.get_retraining_recommendation(rec_id)
        assert updated["status"] == "DISMISSED"
        assert updated["notes"] == "false positive"

    def test_approve_creates_linked_experiment(
        self, engine: RetrainingRecommendationEngine, warehouse: TelemetryWarehouse,
        db: ResearchLabDatabase
    ) -> None:
        """S24 §3: approving a recommendation must bridge to an actual
        ExperimentRecord (PENDING), not just flip a status flag."""
        for i in range(20):
            _trade(warehouse, pnl_pct=1.0, ts=NOW + i)
        for i in range(10):
            _trade(warehouse, pnl_pct=-2.0, ts=NOW + 100 + i)
        engine.evaluate(model_version=MV)
        recs = db.list_retraining_recommendations(status="PROPOSED")
        assert recs
        rec_id = recs[0]["recommendation_id"]

        result = engine.approve(rec_id, resolved_by="head_quant")
        assert result["experiment_id"]

        exp = db.get_experiment(result["experiment_id"])
        assert exp is not None
        assert exp["status"] in ("PENDING", "PROPOSED")
        assert exp["qde_version"] == MV
        assert f"from_recommendation:{rec_id}" in exp["tags"]
        # Bridging must not itself execute anything — no training timestamps.
        assert exp.get("training_started_at") is None

    def test_approve_experiment_link_is_idempotent(
        self, engine: RetrainingRecommendationEngine, warehouse: TelemetryWarehouse,
        db: ResearchLabDatabase
    ) -> None:
        """Repeated internal experiment-creation calls (e.g. retried approve
        logic) must not create duplicate experiments for the same
        recommendation."""
        for i in range(20):
            _trade(warehouse, pnl_pct=1.0, ts=NOW + i)
        for i in range(10):
            _trade(warehouse, pnl_pct=-2.0, ts=NOW + 100 + i)
        engine.evaluate(model_version=MV)
        rec_id = db.list_retraining_recommendations(status="PROPOSED")[0]["recommendation_id"]
        result = engine.approve(rec_id, resolved_by="head_quant")

        from trainer.research_lab.retraining_recommendation_engine import _from_dict
        rec_obj = _from_dict(db.get_retraining_recommendation(rec_id))
        # Directly re-invoke the internal bridge to simulate a retry/duplicate
        # call — it must resolve to the same experiment, not create a new one.
        second_experiment_id = engine._maybe_create_linked_experiment(rec_obj)
        assert second_experiment_id == result["experiment_id"]
        matching = [
            e for e in db.list_experiments(status="PENDING")
            if f"from_recommendation:{rec_id}" in e.get("tags", [])
        ]
        assert len(matching) == 1

    def test_approve_experiment_link_dedups_across_all_statuses(
        self, engine: RetrainingRecommendationEngine, warehouse: TelemetryWarehouse,
        db: ResearchLabDatabase
    ) -> None:
        """Once the linked experiment advances past PENDING/PROPOSED (e.g. to
        RUNNING or COMPLETED), a retried approval must still resolve to the
        same experiment rather than creating a second one."""
        for i in range(20):
            _trade(warehouse, pnl_pct=1.0, ts=NOW + i)
        for i in range(10):
            _trade(warehouse, pnl_pct=-2.0, ts=NOW + 100 + i)
        engine.evaluate(model_version=MV)
        rec_id = db.list_retraining_recommendations(status="PROPOSED")[0]["recommendation_id"]
        result = engine.approve(rec_id, resolved_by="head_quant")
        experiment_id = result["experiment_id"]

        # Simulate the experiment progressing to RUNNING then COMPLETED.
        exp = db.get_experiment(experiment_id)
        exp["status"] = "RUNNING"
        db.upsert_experiment(exp)
        exp["status"] = "COMPLETED"
        db.upsert_experiment(exp)

        from trainer.research_lab.retraining_recommendation_engine import _from_dict
        rec_obj = _from_dict(db.get_retraining_recommendation(rec_id))
        retried_id = engine._maybe_create_linked_experiment(rec_obj)
        assert retried_id == experiment_id

        all_tagged = [
            e for status in ("PROPOSED", "PENDING", "RUNNING", "COMPLETED", "FAILED", "ARCHIVED")
            for e in db.list_experiments(status=status)
            if f"from_recommendation:{rec_id}" in e.get("tags", [])
        ]
        assert len(all_tagged) == 1

    def test_approve_nonexistent_raises(
        self, engine: RetrainingRecommendationEngine
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            engine.approve("nonexistent-id")

    def test_dismiss_nonexistent_raises(
        self, engine: RetrainingRecommendationEngine
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            engine.dismiss("nonexistent-id")

    def test_dedup_cleared_after_approval(
        self, engine: RetrainingRecommendationEngine, warehouse: TelemetryWarehouse,
        db: ResearchLabDatabase
    ) -> None:
        """After approving a recommendation, a second evaluation run should
        create a fresh PROPOSED recommendation for the same dimension."""
        for i in range(20):
            _trade(warehouse, pnl_pct=1.0, ts=NOW + i)
        for i in range(10):
            _trade(warehouse, pnl_pct=-2.0, ts=NOW + 100 + i)
        engine.evaluate(model_version=MV)
        recs = db.list_retraining_recommendations(
            status="PROPOSED", model_version=MV
        )
        # Approve all open recommendations
        for r in recs:
            engine.approve(r["recommendation_id"])
        # Second evaluate should create new PROPOSED ones
        engine.evaluate(model_version=MV)
        new_proposed = db.list_retraining_recommendations(
            status="PROPOSED", model_version=MV
        )
        assert len(new_proposed) >= 1

    def test_evaluate_filters_by_model_version(
        self, engine: RetrainingRecommendationEngine, warehouse: TelemetryWarehouse,
        db: ResearchLabDatabase
    ) -> None:
        # Add degrading trades for v1.0.0
        for i in range(20):
            _trade(warehouse, pnl_pct=1.0, mv="v1.0.0", ts=NOW + i)
        for i in range(10):
            _trade(warehouse, pnl_pct=-2.0, mv="v1.0.0", ts=NOW + 100 + i)
        # Add healthy trades for v2.0.0
        for i in range(30):
            _trade(warehouse, pnl_pct=0.5, mv="v2.0.0", ts=NOW + i)

        engine.evaluate(model_version="v1.0.0")
        v1_recs = db.list_retraining_recommendations(model_version="v1.0.0")
        v2_recs = db.list_retraining_recommendations(model_version="v2.0.0")
        # Only v1.0.0 should have triggered recommendations
        assert len(v1_recs) >= 1
        assert len(v2_recs) == 0

    def test_list_recommendations_filter_by_status(
        self, engine: RetrainingRecommendationEngine, warehouse: TelemetryWarehouse
    ) -> None:
        for i in range(20):
            _trade(warehouse, pnl_pct=1.0, ts=NOW + i)
        for i in range(10):
            _trade(warehouse, pnl_pct=-2.0, ts=NOW + 100 + i)
        engine.evaluate(model_version=MV)
        proposed = engine.list_recommendations(status="PROPOSED")
        assert all(r["status"] == "PROPOSED" for r in proposed)
        dismissed = engine.list_recommendations(status="DISMISSED")
        assert dismissed == []

    def test_exception_isolation_one_evaluator_failing(
        self, engine: RetrainingRecommendationEngine, warehouse: TelemetryWarehouse
    ) -> None:
        """One evaluator raising should not crash the whole run; others still return."""
        from unittest.mock import patch

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated evaluator crash")

        with patch.object(engine._evaluators[0], "evaluate", side_effect=_boom):
            results = engine.evaluate()
        # One evaluator failed, remaining four should still return
        assert len(results) == 4

    def test_multi_trigger_urgency_elevation_persisted(
        self, engine: RetrainingRecommendationEngine, warehouse: TelemetryWarehouse,
        db: ResearchLabDatabase
    ) -> None:
        """When multiple dimensions trigger in one run, urgency should be elevated
        (≥ base urgency) and that elevated value must be persisted."""
        # Induce Sharpe degradation (strong enough to trigger)
        for i in range(20):
            _trade(warehouse, pnl_pct=2.0, ts=NOW + i)
        for i in range(10):
            _trade(warehouse, pnl_pct=-4.0, ts=NOW + 200 + i)
        # Induce confidence drift
        for i in range(20):
            _decision(warehouse, conf=0.9, ts=NOW + i)
        for i in range(10):
            _decision(warehouse, conf=0.1, ts=NOW + 200 + i)
        # Induce bad regime
        for i in range(5):
            _trade(warehouse, pnl_pct=-3.0, regime="CRISIS", ts=NOW + 300 + i)
        for i in range(3):
            _trade(warehouse, pnl_pct=-3.0, regime="HIGH_VOLATILITY", ts=NOW + 300 + i)

        results = engine.evaluate(model_version=MV)
        triggered = [r for r in results if r.triggered]
        if len(triggered) >= 2:
            recs = db.list_retraining_recommendations(status="PROPOSED", model_version=MV)
            urgencies = [r["urgency"] for r in recs]
            # With multiple triggered dimensions, at least one rec should be >= MEDIUM
            urgency_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
            assert any(urgency_order.index(u) >= urgency_order.index("MEDIUM") for u in urgencies)


class TestSplitWindows:
    """Unit tests for the _split_windows disjoint window helper."""

    def test_full_windows_disjoint(self) -> None:
        from trainer.research_lab.dimension_evaluators import _split_windows
        data = list(range(30))  # baseline=20, rolling=10 → no overlap
        baseline, recent = _split_windows(data, 20, 10, 3)
        assert baseline == list(range(20))
        assert recent == list(range(20, 30))
        # Guarantee disjoint
        assert set(baseline) & set(recent) == set()

    def test_short_data_splits_67_33(self) -> None:
        from trainer.research_lab.dimension_evaluators import _split_windows
        data = list(range(12))  # less than 30, but ≥ 2*3
        baseline, recent = _split_windows(data, 20, 10, 3)
        assert baseline is not None and recent is not None
        # Disjoint
        assert set(baseline) & set(recent) == set()
        # Older portion in baseline, newer in recent
        assert max(baseline) < min(recent)

    def test_insufficient_data_returns_none(self) -> None:
        from trainer.research_lab.dimension_evaluators import _split_windows
        data = list(range(3))  # less than 2 * min_samples=3
        baseline, recent = _split_windows(data, 20, 10, 3)
        assert baseline is None
        assert recent is None

    def test_exactly_2x_min_samples(self) -> None:
        from trainer.research_lab.dimension_evaluators import _split_windows
        data = list(range(6))  # exactly 2 * min_samples=3
        baseline, recent = _split_windows(data, 20, 10, 3)
        assert baseline is not None and recent is not None
        assert len(baseline) >= 3
        assert len(recent) >= 3


# ── DB: retraining_recommendations table ──────────────────────────────────────


class TestRetrainingRecommendationsDB:
    def _make_rec(self, mv: str = MV, dim: str = "sharpe_degradation",
                  status: str = "PROPOSED") -> dict:
        r = RetrainingRecommendation.new(
            model_version=mv,
            trigger_dimension=EvaluationDimension.SHARPE_DEGRADATION,
            trigger_reason="test",
            evidence={"baseline_sharpe": 1.2, "recent_sharpe": 0.3},
            recommended_scope=RecommendedScope.FINE_TUNE,
            urgency=RecommendationUrgency.MEDIUM,
        )
        d = r.to_dict()
        d["trigger_dimension"] = dim
        d["status"] = status
        return d

    def test_upsert_and_get(self, db: ResearchLabDatabase) -> None:
        rec = self._make_rec()
        db.upsert_retraining_recommendation(rec)
        fetched = db.get_retraining_recommendation(rec["recommendation_id"])
        assert fetched is not None
        assert fetched["urgency"] == "MEDIUM"
        assert fetched["trigger_dimension"] == "sharpe_degradation"

    def test_upsert_updates_status(self, db: ResearchLabDatabase) -> None:
        rec = self._make_rec()
        db.upsert_retraining_recommendation(rec)
        rec["status"] = "APPROVED"
        rec["resolved_by"] = "head_quant"
        db.upsert_retraining_recommendation(rec)
        fetched = db.get_retraining_recommendation(rec["recommendation_id"])
        assert fetched["status"] == "APPROVED"

    def test_list_filter_by_status(self, db: ResearchLabDatabase) -> None:
        proposed = self._make_rec(status="PROPOSED")
        approved = self._make_rec(status="APPROVED")
        approved["recommendation_id"] = "rr-different"
        db.upsert_retraining_recommendation(proposed)
        db.upsert_retraining_recommendation(approved)
        assert len(db.list_retraining_recommendations(status="PROPOSED")) == 1
        assert len(db.list_retraining_recommendations(status="APPROVED")) == 1

    def test_list_filter_by_model_version(self, db: ResearchLabDatabase) -> None:
        r1 = self._make_rec(mv="v1.0.0")
        r2 = self._make_rec(mv="v2.0.0")
        r2["recommendation_id"] = "rr-v2"
        db.upsert_retraining_recommendation(r1)
        db.upsert_retraining_recommendation(r2)
        assert len(db.list_retraining_recommendations(model_version="v1.0.0")) == 1
        assert len(db.list_retraining_recommendations(model_version="v2.0.0")) == 1

    def test_list_sorted_newest_first(self, db: ResearchLabDatabase) -> None:
        import time as _t
        for i in range(3):
            r = self._make_rec()
            r["recommendation_id"] = f"rr-{i:04d}"
            r["created_at"] = NOW + i
            r["updated_at"] = NOW + i
            db.upsert_retraining_recommendation(r)
            _t.sleep(0.01)
        rows = db.list_retraining_recommendations()
        ts = [r["created_at"] for r in rows]
        assert ts == sorted(ts, reverse=True)

    def test_get_nonexistent_returns_none(self, db: ResearchLabDatabase) -> None:
        assert db.get_retraining_recommendation("nonexistent") is None

    def test_stats_includes_recommendations(self, db: ResearchLabDatabase) -> None:
        db.upsert_retraining_recommendation(self._make_rec())
        stats = db.stats()
        assert "retraining_recommendations" in stats
        assert stats["retraining_recommendations"] >= 1


# ── Server endpoints ───────────────────────────────────────────────────────────


class TestRecommendationEndpoints:
    def _make_app(self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase,
                  cfg: RecommendationConfig):
        from fastapi import FastAPI, HTTPException

        app = FastAPI()
        eng = RetrainingRecommendationEngine(warehouse, db, config=cfg)

        @app.post("/research/evaluate")
        def evaluate(req: _EvaluateReq) -> dict:
            try:
                results = eng.evaluate(
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
        def list_recs(status: str | None = None,
                      model_version: str | None = None,
                      limit: int = 50) -> dict:
            rows = eng.list_recommendations(status=status,
                                            model_version=model_version, limit=limit)
            return {"success": True, "total": len(rows), "recommendations": rows}

        @app.post("/research/recommendations/{rec_id}/approve")
        def approve(rec_id: str, req: _ResolveReq) -> dict:
            try:
                rec = eng.approve(rec_id, resolved_by=req.resolved_by)
                return {"success": True, "recommendation": rec}
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))

        @app.post("/research/recommendations/{rec_id}/dismiss")
        def dismiss(rec_id: str, req: _ResolveReq) -> dict:
            try:
                rec = eng.dismiss(rec_id, resolved_by=req.resolved_by, notes=req.notes)
                return {"success": True, "recommendation": rec}
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))

        @app.get("/research/recommendations/{rec_id}")
        def get_rec(rec_id: str) -> dict:
            rec = eng.get_recommendation(rec_id)
            if rec is None:
                raise HTTPException(status_code=404, detail=f"Recommendation {rec_id!r} not found")
            return {"success": True, "recommendation": rec}

        return app

    def test_evaluate_returns_five_dimensions(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase,
        cfg: RecommendationConfig
    ) -> None:
        from starlette.testclient import TestClient
        app = self._make_app(warehouse, db, cfg)
        client = TestClient(app)
        resp = client.post("/research/evaluate", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["n_evaluated"] == 5

    def test_evaluate_triggered_dimensions_in_results(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase,
        cfg: RecommendationConfig
    ) -> None:
        from starlette.testclient import TestClient
        for i in range(20):
            _trade(warehouse, pnl_pct=1.0, ts=NOW + i)
        for i in range(10):
            _trade(warehouse, pnl_pct=-2.0, ts=NOW + 100 + i)
        app = self._make_app(warehouse, db, cfg)
        client = TestClient(app)
        resp = client.post("/research/evaluate", json={"model_version": MV})
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_triggered"] >= 1

    def test_list_recommendations_empty(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase,
        cfg: RecommendationConfig
    ) -> None:
        from starlette.testclient import TestClient
        app = self._make_app(warehouse, db, cfg)
        client = TestClient(app)
        resp = client.get("/research/recommendations")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_list_recommendations_after_evaluate(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase,
        cfg: RecommendationConfig
    ) -> None:
        from starlette.testclient import TestClient
        for i in range(20):
            _trade(warehouse, pnl_pct=1.0, ts=NOW + i)
        for i in range(10):
            _trade(warehouse, pnl_pct=-2.0, ts=NOW + 100 + i)
        app = self._make_app(warehouse, db, cfg)
        client = TestClient(app)
        client.post("/research/evaluate", json={})
        resp = client.get("/research/recommendations?status=PROPOSED")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 1
        assert all(r["status"] == "PROPOSED" for r in body["recommendations"])

    def test_approve_endpoint(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase,
        cfg: RecommendationConfig
    ) -> None:
        from starlette.testclient import TestClient
        for i in range(20):
            _trade(warehouse, pnl_pct=1.0, ts=NOW + i)
        for i in range(10):
            _trade(warehouse, pnl_pct=-2.0, ts=NOW + 100 + i)
        app = self._make_app(warehouse, db, cfg)
        client = TestClient(app)
        client.post("/research/evaluate", json={})
        recs = client.get("/research/recommendations?status=PROPOSED").json()["recommendations"]
        assert recs
        rec_id = recs[0]["recommendation_id"]
        resp = client.post(f"/research/recommendations/{rec_id}/approve",
                           json={"resolved_by": "head_quant"})
        assert resp.status_code == 200
        assert resp.json()["recommendation"]["status"] == "APPROVED"

    def test_dismiss_endpoint(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase,
        cfg: RecommendationConfig
    ) -> None:
        from starlette.testclient import TestClient
        for i in range(20):
            _trade(warehouse, pnl_pct=1.0, ts=NOW + i)
        for i in range(10):
            _trade(warehouse, pnl_pct=-2.0, ts=NOW + 100 + i)
        app = self._make_app(warehouse, db, cfg)
        client = TestClient(app)
        client.post("/research/evaluate", json={})
        recs = client.get("/research/recommendations?status=PROPOSED").json()["recommendations"]
        assert recs
        rec_id = recs[0]["recommendation_id"]
        resp = client.post(f"/research/recommendations/{rec_id}/dismiss",
                           json={"resolved_by": "admin", "notes": "not actionable"})
        assert resp.status_code == 200
        rec = resp.json()["recommendation"]
        assert rec["status"] == "DISMISSED"
        assert rec["notes"] == "not actionable"

    def test_approve_nonexistent_returns_400(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase,
        cfg: RecommendationConfig
    ) -> None:
        from starlette.testclient import TestClient
        app = self._make_app(warehouse, db, cfg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/research/recommendations/nope/approve",
                           json={"resolved_by": "admin"})
        assert resp.status_code == 400

    def test_dismiss_nonexistent_returns_400(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase,
        cfg: RecommendationConfig
    ) -> None:
        from starlette.testclient import TestClient
        app = self._make_app(warehouse, db, cfg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/research/recommendations/nope/dismiss",
                           json={"resolved_by": "admin"})
        assert resp.status_code == 400

    def test_double_approve_returns_400(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase,
        cfg: RecommendationConfig
    ) -> None:
        from starlette.testclient import TestClient
        for i in range(20):
            _trade(warehouse, pnl_pct=1.0, ts=NOW + i)
        for i in range(10):
            _trade(warehouse, pnl_pct=-2.0, ts=NOW + 100 + i)
        app = self._make_app(warehouse, db, cfg)
        client = TestClient(app, raise_server_exceptions=False)
        client.post("/research/evaluate", json={})
        recs = client.get("/research/recommendations?status=PROPOSED").json()["recommendations"]
        rec_id = recs[0]["recommendation_id"]
        client.post(f"/research/recommendations/{rec_id}/approve", json={"resolved_by": "admin"})
        resp = client.post(f"/research/recommendations/{rec_id}/approve",
                           json={"resolved_by": "admin"})
        assert resp.status_code == 400

    def test_get_recommendation_by_id(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase,
        cfg: RecommendationConfig
    ) -> None:
        from starlette.testclient import TestClient

        for i in range(20):
            _trade(warehouse, pnl_pct=1.0, ts=NOW + i)
        for i in range(10):
            _trade(warehouse, pnl_pct=-2.0, ts=NOW + 100 + i)
        app = self._make_app(warehouse, db, cfg)
        client = TestClient(app)
        client.post("/research/evaluate", json={})
        recs = client.get("/research/recommendations?status=PROPOSED").json()["recommendations"]
        assert recs
        rec_id = recs[0]["recommendation_id"]
        resp = client.get(f"/research/recommendations/{rec_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["recommendation"]["recommendation_id"] == rec_id

    def test_get_recommendation_not_found_returns_404(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase,
        cfg: RecommendationConfig
    ) -> None:
        from starlette.testclient import TestClient

        app = self._make_app(warehouse, db, cfg)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/research/recommendations/nonexistent-id")
        assert resp.status_code == 404
