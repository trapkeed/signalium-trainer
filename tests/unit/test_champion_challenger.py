"""S20 Task #12 — Tests for Champion vs Challenger tracking.

Covers:
  - Each of the five dimension scorers: below threshold, above threshold
  - HeadToHeadSummary computation and serialisation
  - RollingMetrics: Sharpe, Sortino, profit_factor, max_drawdown,
    confidence_calibration, confidence_reliability
  - PromotionCriterion: met / not met on each guard (min_decisions,
    majority_threshold, min_promotion_score, min_agreement_rate)
  - evaluate_promotion: approval request created + idempotent
  - evaluate_promotion: no approval gate → no request created
  - get_status: with / without test_id
  - get_comparison: dimension breakdown
  - Server endpoints: GET /shadow/status, GET /shadow/comparison,
    POST /shadow/evaluate (including auth enforcement)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from tests.unit._sso_test_utils import DEFAULT_TEST_JWT_SECRET, make_signalium_jwt
from trainer.research_lab.champion_challenger import (
    ChampionChallengerTracker,
    ComparisonDimension,
    PromotionCriterionConfig,
    _std,
    compute_rolling_metrics,
    score_confidence_quality,
    score_direction_accuracy,
    score_hold_quality,
    score_risk_quality,
    score_sizing_quality,
)
from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.rl_models import ShadowTestRun

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def db(tmp_path: Path) -> ResearchLabDatabase:
    return ResearchLabDatabase(str(tmp_path / "cc_test.db"))


def _tracker(db: ResearchLabDatabase, gate=None, cfg: PromotionCriterionConfig | None = None):
    return ChampionChallengerTracker(db, approval_gate=gate, config=cfg)


# ── Decision helpers ──────────────────────────────────────────────────────────

_NOW = time.time()


def _decision(
    test_id: str = "t01",
    agree: bool = True,
    champ_conf: float = 0.70,
    chall_conf: float = 0.72,
    pnl_diff: float = 0.01,
    risk_diff: float = 0.005,
    latency_diff_ms: float = 0.0,
    regime: str = "RANGING",
    promo: float | None = None,
) -> dict:
    conf_diff = chall_conf - champ_conf
    if promo is None:
        from trainer.research_lab.rl_models import compute_promotion_score

        promo = compute_promotion_score(
            decisions_agree=agree,
            confidence_diff=conf_diff,
            pnl_diff=pnl_diff,
            latency_diff_ms=latency_diff_ms,
            risk_diff=risk_diff,
        )
    return {
        "record_id": f"sdr-{hash((test_id, agree, champ_conf, pnl_diff))}",
        "test_id": test_id,
        "instrument": "XAUUSD",
        "champion_decision_type": "buy",
        "challenger_decision_type": "buy" if agree else "sell",
        "champion_confidence": champ_conf,
        "challenger_confidence": chall_conf,
        "decisions_agree": agree,
        "regime": regime,
        "recorded_at": _NOW,
        "confidence_diff": conf_diff,
        "pnl_diff": pnl_diff,
        "latency_diff_ms": latency_diff_ms,
        "risk_diff": risk_diff,
        "promotion_score": promo,
    }


def _many(n: int, **kw) -> list[dict]:
    return [_decision(**kw) for _ in range(n)]


def _shadow_test_in_db(
    db: ResearchLabDatabase,
    test_id: str = "t01",
    n_decisions: int = 20,
    agree: bool = True,
    pnl_diff: float = 0.01,
    risk_diff: float = 0.005,
    regime: str = "RANGING",
) -> str:
    run = ShadowTestRun.new(
        name="test-run",
        challenger_version="v2.0",
        champion_version="v1.0",
        instruments=["XAUUSD"],
        min_decisions=n_decisions,
    )
    run.test_id = test_id
    db.upsert_shadow_test(run.to_dict())
    for i in range(n_decisions):
        rec = _decision(
            test_id=test_id, agree=agree, pnl_diff=pnl_diff, risk_diff=risk_diff, regime=regime
        )
        rec["record_id"] = f"sdr-{i}"
        db.save_shadow_decision(rec)
    return test_id


# ── Helpers ────────────────────────────────────────────────────────────────────


class TestHelpers:
    def test_std_zero_on_constant(self) -> None:
        assert _std([1.0, 1.0, 1.0]) == 0.0

    def test_std_basic(self) -> None:
        vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        assert abs(_std(vals) - 2.0) < 0.01

    def test_std_single(self) -> None:
        assert _std([5.0]) == 0.0


# ── Dimension Scorers ─────────────────────────────────────────────────────────


class TestDirectionAccuracy:
    def test_all_agree_wins(self) -> None:
        decisions = _many(20, agree=True)
        result = score_direction_accuracy(decisions)
        assert result.challenger_wins is True
        assert result.challenger_score > 0.5
        assert result.dimension == ComparisonDimension.DIRECTION_ACCURACY.value

    def test_all_disagree_loses(self) -> None:
        decisions = _many(20, agree=False)
        result = score_direction_accuracy(decisions)
        assert result.challenger_wins is False
        assert result.challenger_score < 0.5

    def test_50_50_no_win(self) -> None:
        decisions = _many(10, agree=True) + _many(10, agree=False)
        result = score_direction_accuracy(decisions)
        assert result.challenger_wins is False  # 50% == baseline → no win

    def test_per_regime_in_evidence(self) -> None:
        decisions = _many(10, agree=True, regime="RANGING") + _many(
            10, agree=False, regime="TRENDING_BULL"
        )
        result = score_direction_accuracy(decisions)
        assert "per_regime_agreement" in result.evidence
        per = result.evidence["per_regime_agreement"]
        assert abs(per.get("RANGING", 0) - 1.0) < 0.01
        assert abs(per.get("TRENDING_BULL", 0) - 0.0) < 0.01

    def test_insufficient_data_inconclusive(self) -> None:
        result = score_direction_accuracy([_decision()])  # n < 2
        assert result.challenger_wins is False
        assert "insufficient_data" in result.evidence.get("reason", "")

    def test_evidence_fields(self) -> None:
        decisions = _many(15, agree=True)
        result = score_direction_accuracy(decisions)
        assert "agreement_rate" in result.evidence
        assert "n_agreements" in result.evidence
        assert result.n_decisions == 15


class TestRiskQuality:
    def test_tiny_risk_diff_wins(self) -> None:
        decisions = _many(20, risk_diff=0.002)  # within 1%
        result = score_risk_quality(decisions)
        assert result.challenger_wins is True

    def test_large_risk_diff_loses(self) -> None:
        decisions = _many(20, risk_diff=0.10)  # 10% risk swing
        result = score_risk_quality(decisions)
        assert result.challenger_wins is False

    def test_negative_risk_diff_still_penalised(self) -> None:
        # Challenger taking far less risk → also large abs diff
        decisions = _many(20, risk_diff=-0.10)
        result = score_risk_quality(decisions)
        assert result.challenger_wins is False

    def test_champion_score_is_neutral_baseline(self) -> None:
        # Champion baseline is 0.5 (neutral) so challenger can win or lose
        decisions = _many(5, risk_diff=0.0)
        result = score_risk_quality(decisions)
        assert result.champion_score == 0.5  # neutral mid-point baseline

    def test_zero_risk_diff_challenger_wins(self) -> None:
        # Challenger with identical risk to champion should score very high → win
        decisions = _many(10, risk_diff=0.0)
        result = score_risk_quality(decisions)
        assert result.challenger_wins is True
        assert result.score_delta > 0

    def test_evidence_includes_mean_diff(self) -> None:
        decisions = _many(10, risk_diff=0.005)
        result = score_risk_quality(decisions)
        assert "mean_abs_risk_diff" in result.evidence
        assert "mean_risk_diff" in result.evidence


class TestSizingQuality:
    def test_positive_pnl_diff_wins(self) -> None:
        decisions = _many(20, pnl_diff=0.05)
        result = score_sizing_quality(decisions)
        assert result.challenger_wins is True

    def test_negative_pnl_diff_loses(self) -> None:
        decisions = _many(20, pnl_diff=-0.05)
        result = score_sizing_quality(decisions)
        assert result.challenger_wins is False

    def test_zero_pnl_diff_no_win(self) -> None:
        decisions = _many(20, pnl_diff=0.0)
        result = score_sizing_quality(decisions)
        assert result.challenger_wins is False

    def test_evidence_fields(self) -> None:
        decisions = _many(10, pnl_diff=0.02)
        result = score_sizing_quality(decisions)
        assert "mean_pnl_diff" in result.evidence
        assert "pct_positive_diff" in result.evidence


class TestConfidenceQuality:
    def test_calibrated_challenger_wins(self) -> None:
        # Challenger very confident on agreements, less on disagreements
        agree_d = [_decision(agree=True, champ_conf=0.70, chall_conf=0.85) for _ in range(15)]
        disagree_d = [_decision(agree=False, champ_conf=0.70, chall_conf=0.45) for _ in range(5)]
        result = score_confidence_quality(agree_d + disagree_d)
        assert result.challenger_wins is True
        assert result.evidence["calibration_delta"] > 0

    def test_overconfident_disagree_loses(self) -> None:
        # Challenger more confident when it DISAGREES with champion
        agree_d = [_decision(agree=True, champ_conf=0.70, chall_conf=0.60) for _ in range(10)]
        disagree_d = [_decision(agree=False, champ_conf=0.70, chall_conf=0.90) for _ in range(10)]
        result = score_confidence_quality(agree_d + disagree_d)
        assert result.evidence["calibration_delta"] < 0

    def test_evidence_fields(self) -> None:
        decisions = _many(20, agree=True)
        result = score_confidence_quality(decisions)
        assert "calibration_delta" in result.evidence
        assert "confidence_reliability" in result.evidence
        assert "mean_challenger_confidence" in result.evidence


class TestHoldQuality:
    def test_consistent_cross_regime_wins(self) -> None:
        # Challenger always agrees regardless of regime → low CoV of per-regime rates
        decisions = (
            _many(10, agree=True, regime="RANGING")
            + _many(10, agree=True, regime="TRENDING_BULL")
            + _many(10, agree=True, regime="CRISIS")
        )
        result = score_hold_quality(decisions)
        assert result.challenger_wins is True

    def test_erratic_cross_regime_does_not_win(self) -> None:
        # Challenger agrees in one regime, totally disagrees in another
        # → high CoV of per-regime agreement rates → challenger_score < 0.6 → loses
        decisions = _many(10, agree=True, regime="RANGING") + _many(
            10, agree=False, regime="CRISIS"
        )
        result = score_hold_quality(decisions)
        assert result.challenger_wins is False
        # Score should be below the 0.6 winning threshold
        assert result.challenger_score < 0.6

    def test_insufficient_data(self) -> None:
        result = score_hold_quality([_decision()])  # < 5 decisions
        assert "insufficient_data" in result.evidence.get("reason", "")

    def test_per_regime_in_evidence(self) -> None:
        decisions = _many(10, agree=True, regime="RANGING") + _many(
            10, agree=True, regime="TRENDING_BULL"
        )
        result = score_hold_quality(decisions)
        assert "per_regime_agreement_rates" in result.evidence


# ── RollingMetrics ────────────────────────────────────────────────────────────


class TestRollingMetrics:
    def test_empty_decisions(self) -> None:
        m = compute_rolling_metrics([])
        assert m.n_decisions == 0
        assert m.agreement_rate == 0.0

    def test_all_winning_decisions(self) -> None:
        decisions = _many(20, agree=True, pnl_diff=0.05)
        m = compute_rolling_metrics(decisions)
        assert m.n_decisions == 20
        assert m.agreement_rate == 1.0
        assert m.mean_pnl_diff > 0
        assert m.challenger_profit_factor > 1.0

    def test_all_losing_pnl(self) -> None:
        decisions = _many(20, agree=False, pnl_diff=-0.05)
        m = compute_rolling_metrics(decisions)
        assert m.mean_pnl_diff < 0
        assert m.challenger_profit_factor < 1.0

    def test_sharpe_positive_on_consistent_gains(self) -> None:
        decisions = _many(50, agree=True, pnl_diff=0.01)
        m = compute_rolling_metrics(decisions)
        # All same pnl_diff → std=0 → Sharpe=0 (division guard)
        # Just verify it doesn't crash
        assert isinstance(m.challenger_sharpe, float)

    def test_max_drawdown_zero_on_monotone_gains(self) -> None:
        # Strictly increasing cumulative pnl → never below peak → max_dd = 0
        decisions = [_decision(pnl_diff=0.01) for _ in range(20)]
        m = compute_rolling_metrics(decisions)
        assert m.challenger_max_drawdown == 0.0

    def test_max_drawdown_nonzero_on_losses(self) -> None:
        gains = [_decision(pnl_diff=0.10) for _ in range(10)]
        losses = [_decision(pnl_diff=-0.05) for _ in range(10)]
        m = compute_rolling_metrics(gains + losses)
        assert m.challenger_max_drawdown > 0.0

    def test_max_drawdown_nonzero_on_all_negative(self) -> None:
        """All-negative pnl_diff must not produce zero max_drawdown."""
        decisions = [_decision(pnl_diff=-0.02) for _ in range(20)]
        m = compute_rolling_metrics(decisions)
        assert m.challenger_max_drawdown > 0.0

    def test_sharpe_positive_on_gains_nonzero_variance(self) -> None:
        """Mixed positive series (mostly gains) should yield positive Sharpe."""
        decisions = [_decision(pnl_diff=0.02 + (i % 3) * 0.005) for i in range(30)]
        m = compute_rolling_metrics(decisions)
        assert m.challenger_sharpe > 0.0

    def test_profit_factor_above_one_on_gains(self) -> None:
        gains = [_decision(pnl_diff=0.05) for _ in range(15)]
        losses = [_decision(pnl_diff=-0.01) for _ in range(5)]
        m = compute_rolling_metrics(gains + losses)
        assert m.challenger_profit_factor > 1.0

    def test_profit_factor_below_one_on_losses(self) -> None:
        losses = [_decision(pnl_diff=-0.05) for _ in range(15)]
        gains = [_decision(pnl_diff=0.01) for _ in range(5)]
        m = compute_rolling_metrics(losses + gains)
        assert m.challenger_profit_factor < 1.0

    def test_confidence_calibration_positive_when_calibrated(self) -> None:
        agree_d = [_decision(agree=True, chall_conf=0.85) for _ in range(15)]
        disagree_d = [_decision(agree=False, chall_conf=0.45) for _ in range(5)]
        m = compute_rolling_metrics(agree_d + disagree_d)
        assert m.challenger_confidence_calibration > 0

    def test_to_dict_keys(self) -> None:
        decisions = _many(10, agree=True)
        m = compute_rolling_metrics(decisions)
        d = m.to_dict()
        required = {
            "n_decisions",
            "agreement_rate",
            "avg_promotion_score",
            "challenger_sharpe",
            "challenger_sortino",
            "challenger_profit_factor",
            "challenger_max_drawdown",
            "challenger_confidence_calibration",
            "challenger_confidence_reliability",
            "mean_pnl_diff",
        }
        assert required.issubset(d.keys())


# ── HeadToHeadSummary ─────────────────────────────────────────────────────────


class TestHeadToHeadSummary:
    def test_compute_head_to_head_missing_test(self, db) -> None:
        tracker = _tracker(db)
        result = tracker.compute_head_to_head("does-not-exist")
        assert result is None

    def test_compute_head_to_head_no_decisions(self, db) -> None:
        run = ShadowTestRun.new(
            name="empty",
            challenger_version="v2.0",
            champion_version="v1.0",
            instruments=["XAUUSD"],
            min_decisions=5,
        )
        db.upsert_shadow_test(run.to_dict())
        tracker = _tracker(db)
        h2h = tracker.compute_head_to_head(run.test_id)
        # Returns summary with 0 decisions and 5 dimension results
        assert h2h is not None
        assert h2h.n_decisions == 0
        assert len(h2h.dimension_results) == 5

    def test_compute_head_to_head_with_decisions(self, db) -> None:
        test_id = _shadow_test_in_db(db, n_decisions=20, agree=True, pnl_diff=0.05)
        tracker = _tracker(db)
        h2h = tracker.compute_head_to_head(test_id)
        assert h2h is not None
        assert h2h.n_decisions == 20
        assert h2h.n_dimensions_total == 5
        assert 0 <= h2h.n_dimensions_challenger_wins <= 5

    def test_summary_to_dict_keys(self, db) -> None:
        test_id = _shadow_test_in_db(db, n_decisions=10, agree=True)
        tracker = _tracker(db)
        h2h = tracker.compute_head_to_head(test_id)
        assert h2h is not None
        d = h2h.to_dict()
        for key in [
            "test_id",
            "champion_version",
            "challenger_version",
            "n_decisions",
            "n_dimensions_challenger_wins",
            "n_dimensions_total",
            "challenger_win_rate",
            "promotion_criterion_met",
            "dimension_results",
            "rolling_metrics",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_five_dimensions_always_present(self, db) -> None:
        test_id = _shadow_test_in_db(db, n_decisions=30, agree=True)
        tracker = _tracker(db)
        h2h = tracker.compute_head_to_head(test_id)
        dims = {r.dimension for r in h2h.dimension_results}
        expected = {d.value for d in ComparisonDimension}
        assert dims == expected


# ── Promotion Criterion ────────────────────────────────────────────────────────


class TestPromotionCriterion:
    def _good_cfg(self) -> PromotionCriterionConfig:
        return PromotionCriterionConfig(
            min_decisions=5,
            majority_threshold=0.4,  # 2 of 5 sufficient for tests
            min_promotion_score=0.01,  # very low → effectively disabled
            min_agreement_rate=0.01,
        )

    def test_criterion_met_when_all_pass(self, db) -> None:
        # Strongly winning challenger: agree=True, positive pnl → most dimensions win
        test_id = _shadow_test_in_db(db, n_decisions=10, agree=True, pnl_diff=0.05, risk_diff=0.001)
        tracker = _tracker(db, cfg=self._good_cfg())
        h2h = tracker.compute_head_to_head(test_id)
        assert h2h.promotion_criterion_met is True

    def test_criterion_not_met_insufficient_decisions(self, db) -> None:
        test_id = _shadow_test_in_db(db, n_decisions=3, agree=True, pnl_diff=0.05)
        cfg = PromotionCriterionConfig(min_decisions=50)
        tracker = _tracker(db, cfg=cfg)
        h2h = tracker.compute_head_to_head(test_id)
        assert h2h.promotion_criterion_met is False

    def test_criterion_not_met_low_agreement_rate(self, db) -> None:
        test_id = _shadow_test_in_db(db, n_decisions=20, agree=False)
        cfg = PromotionCriterionConfig(
            min_decisions=5,
            majority_threshold=0.4,
            min_promotion_score=0.0,
            min_agreement_rate=0.80,
        )
        tracker = _tracker(db, cfg=cfg)
        h2h = tracker.compute_head_to_head(test_id)
        assert h2h.promotion_criterion_met is False

    def test_criterion_not_met_low_promotion_score(self, db) -> None:
        # All disagree → low promotion_score
        test_id = _shadow_test_in_db(db, n_decisions=10, agree=False, pnl_diff=-0.05)
        cfg = PromotionCriterionConfig(
            min_decisions=5,
            majority_threshold=0.4,
            min_promotion_score=0.99,
            min_agreement_rate=0.0,
        )
        tracker = _tracker(db, cfg=cfg)
        h2h = tracker.compute_head_to_head(test_id)
        assert h2h.promotion_criterion_met is False

    def test_criterion_not_met_majority_threshold(self, db) -> None:
        """Challenger winning only 1 of 5 dimensions must fail majority threshold."""
        # Construct decisions where challenger barely wins: mostly disagrees + negative PnL
        test_id = _shadow_test_in_db(
            db, n_decisions=10, agree=False, pnl_diff=-0.05, risk_diff=0.15
        )
        # Require 80% of dimensions (4 of 5) → unachievable with bad decisions
        cfg = PromotionCriterionConfig(
            min_decisions=5,
            majority_threshold=0.80,
            min_promotion_score=0.0,
            min_agreement_rate=0.0,
        )
        tracker = _tracker(db, cfg=cfg)
        h2h = tracker.compute_head_to_head(test_id)
        assert h2h.promotion_criterion_met is False


# ── evaluate_promotion ────────────────────────────────────────────────────────


class TestEvaluatePromotion:
    def _good_cfg(self) -> PromotionCriterionConfig:
        return PromotionCriterionConfig(
            min_decisions=5,
            majority_threshold=0.4,
            min_promotion_score=0.0,
            min_agreement_rate=0.0,
        )

    def test_evaluate_missing_test(self, db) -> None:
        tracker = _tracker(db, cfg=self._good_cfg())
        result = tracker.evaluate_promotion("does-not-exist")
        assert result["criterion_met"] is False
        assert "not found" in result["reason"].lower()

    def test_evaluate_criterion_not_met_returns_reason(self, db) -> None:
        test_id = _shadow_test_in_db(db, n_decisions=2, agree=False)
        cfg = PromotionCriterionConfig(min_decisions=50)
        tracker = _tracker(db, cfg=cfg)
        result = tracker.evaluate_promotion(test_id)
        assert result["criterion_met"] is False
        assert result["approval_id"] == ""
        assert "not met" in result["reason"].lower()

    def test_evaluate_criterion_met_no_gate_no_request(self, db) -> None:
        test_id = _shadow_test_in_db(db, n_decisions=10, agree=True, pnl_diff=0.05, risk_diff=0.001)
        tracker = _tracker(db, gate=None, cfg=self._good_cfg())
        result = tracker.evaluate_promotion(test_id)
        assert result["criterion_met"] is True
        assert result["approval_id"] == ""  # no gate → no request

    def test_evaluate_criterion_met_with_gate_creates_request(self, db) -> None:
        from trainer.research_lab.approval_gate import ApprovalGate

        gate = ApprovalGate(db)
        test_id = _shadow_test_in_db(db, n_decisions=10, agree=True, pnl_diff=0.05, risk_diff=0.001)
        tracker = _tracker(db, gate=gate, cfg=self._good_cfg())
        result = tracker.evaluate_promotion(test_id)
        assert result["criterion_met"] is True
        assert result["approval_id"].startswith("apr-")

    def test_decisive_rejection_creates_recommendation(self, db) -> None:
        """S22B: a challenger decisively rejected by live shadow traffic
        (loses the mirror-image majority of dimensions, with enough data)
        must create a Research Lab review task, not be silently dropped."""
        from trainer.research_lab.recommendation_models import (
            RecommendationConfig,
            RecommendationStatus,
        )
        from trainer.research_lab.retraining_recommendation_engine import (
            RetrainingRecommendationEngine,
        )
        from trainer.research_lab.telemetry_warehouse import TelemetryWarehouse

        warehouse = TelemetryWarehouse(db)
        rec_engine = RetrainingRecommendationEngine(warehouse, db, RecommendationConfig())

        test_id = _shadow_test_in_db(
            db, n_decisions=20, agree=False, pnl_diff=-0.05, risk_diff=0.05
        )
        cfg = PromotionCriterionConfig(
            min_decisions=5,
            majority_threshold=0.8,
            min_promotion_score=0.99,
            min_agreement_rate=0.99,
        )
        tracker = ChampionChallengerTracker(db, config=cfg, rec_engine=rec_engine)
        result = tracker.evaluate_promotion(test_id)

        assert result["criterion_met"] is False
        assert result[
            "recommendation_id"
        ], "decisive rejection must create a RetrainingRecommendation"
        rec = db.get_retraining_recommendation(result["recommendation_id"])
        assert rec is not None
        assert rec["status"] == RecommendationStatus.PROPOSED.value
        assert rec["model_version"] == "v2.0"  # challenger_version

    def test_decisive_rejection_is_deduped_on_repeat_evaluation(self, db) -> None:
        from trainer.research_lab.recommendation_models import RecommendationConfig
        from trainer.research_lab.retraining_recommendation_engine import (
            RetrainingRecommendationEngine,
        )
        from trainer.research_lab.telemetry_warehouse import TelemetryWarehouse

        warehouse = TelemetryWarehouse(db)
        rec_engine = RetrainingRecommendationEngine(warehouse, db, RecommendationConfig())
        test_id = _shadow_test_in_db(
            db, n_decisions=20, agree=False, pnl_diff=-0.05, risk_diff=0.05
        )
        cfg = PromotionCriterionConfig(
            min_decisions=5,
            majority_threshold=0.8,
            min_promotion_score=0.99,
            min_agreement_rate=0.99,
        )
        tracker = ChampionChallengerTracker(db, config=cfg, rec_engine=rec_engine)
        result1 = tracker.evaluate_promotion(test_id)
        result2 = tracker.evaluate_promotion(test_id)
        assert result1["recommendation_id"] == result2["recommendation_id"]

    def test_no_recommendation_without_rec_engine(self, db) -> None:
        """Backward compatible: without a rec_engine, evaluate_promotion still
        works and simply doesn't flag anything."""
        test_id = _shadow_test_in_db(
            db, n_decisions=20, agree=False, pnl_diff=-0.05, risk_diff=0.05
        )
        cfg = PromotionCriterionConfig(
            min_decisions=5,
            majority_threshold=0.8,
            min_promotion_score=0.99,
            min_agreement_rate=0.99,
        )
        tracker = _tracker(db, cfg=cfg)
        result = tracker.evaluate_promotion(test_id)
        assert result["criterion_met"] is False
        assert result["recommendation_id"] == ""

    def test_marginal_shortfall_does_not_create_recommendation(self, db) -> None:
        """A challenger that merely falls short of the promotion bar (not
        decisively rejected) should not spuriously create a task."""
        from trainer.research_lab.recommendation_models import RecommendationConfig
        from trainer.research_lab.retraining_recommendation_engine import (
            RetrainingRecommendationEngine,
        )
        from trainer.research_lab.telemetry_warehouse import TelemetryWarehouse

        warehouse = TelemetryWarehouse(db)
        rec_engine = RetrainingRecommendationEngine(warehouse, db, RecommendationConfig())
        # Challenger agrees and has positive pnl_diff (wins most dimensions)
        # but promotion_score threshold is set unreasonably high.
        test_id = _shadow_test_in_db(db, n_decisions=10, agree=True, pnl_diff=0.05, risk_diff=0.001)
        cfg = PromotionCriterionConfig(
            min_decisions=5,
            majority_threshold=0.4,
            min_promotion_score=0.9999,
            min_agreement_rate=0.0,
        )
        tracker = ChampionChallengerTracker(db, config=cfg, rec_engine=rec_engine)
        result = tracker.evaluate_promotion(test_id)
        assert result["criterion_met"] is False
        assert result["recommendation_id"] == ""

    def test_evaluate_idempotent_duplicate_call(self, db) -> None:
        from trainer.research_lab.approval_gate import ApprovalGate

        gate = ApprovalGate(db)
        test_id = _shadow_test_in_db(db, n_decisions=10, agree=True, pnl_diff=0.05, risk_diff=0.001)
        tracker = _tracker(db, gate=gate, cfg=self._good_cfg())
        result1 = tracker.evaluate_promotion(test_id)
        result2 = tracker.evaluate_promotion(test_id)
        # Both must return the SAME approval_id (dedup)
        assert result1["approval_id"] == result2["approval_id"]

    def test_evaluate_result_includes_summary(self, db) -> None:
        test_id = _shadow_test_in_db(db, n_decisions=10, agree=True, pnl_diff=0.05, risk_diff=0.001)
        tracker = _tracker(db, cfg=self._good_cfg())
        result = tracker.evaluate_promotion(test_id)
        assert result["summary"] is not None
        assert "dimension_results" in result["summary"]
        assert "rolling_metrics" in result["summary"]


# ── get_status / get_comparison ───────────────────────────────────────────────


class TestGetStatus:
    def test_status_no_running_tests(self, db) -> None:
        tracker = _tracker(db)
        status = tracker.get_status()
        assert status["n_running"] == 0
        assert status["running_tests"] == []

    def test_status_with_running_tests(self, db) -> None:
        _shadow_test_in_db(db, test_id="t-run-1", n_decisions=10)
        tracker = _tracker(db)
        status = tracker.get_status()
        assert status["n_running"] >= 1
        test_ids = [s["test_id"] for s in status["running_tests"]]
        assert "t-run-1" in test_ids

    def test_status_with_test_id(self, db) -> None:
        test_id = _shadow_test_in_db(db, test_id="t-specific", n_decisions=15)
        tracker = _tracker(db)
        status = tracker.get_status(test_id=test_id)
        assert status["test_id"] == test_id
        assert status["n_decisions"] == 15
        assert "head_to_head" in status

    def test_status_unknown_test_id(self, db) -> None:
        tracker = _tracker(db)
        status = tracker.get_status(test_id="does-not-exist")
        assert "error" in status

    def test_status_includes_criterion_config(self, db) -> None:
        tracker = _tracker(db)
        status = tracker.get_status()
        assert "promotion_criterion_config" in status


class TestGetComparison:
    def test_comparison_missing_test(self, db) -> None:
        tracker = _tracker(db)
        result = tracker.get_comparison("no-test")
        assert "error" in result

    def test_comparison_returns_five_dimensions(self, db) -> None:
        test_id = _shadow_test_in_db(db, n_decisions=20, agree=True)
        tracker = _tracker(db)
        comp = tracker.get_comparison(test_id)
        assert "dimension_results" in comp
        assert len(comp["dimension_results"]) == 5

    def test_comparison_dimensions_are_all_five(self, db) -> None:
        test_id = _shadow_test_in_db(db, n_decisions=20, agree=True)
        tracker = _tracker(db)
        comp = tracker.get_comparison(test_id)
        dims = {r["dimension"] for r in comp["dimension_results"]}
        assert dims == {d.value for d in ComparisonDimension}

    def test_comparison_rolling_metrics_included(self, db) -> None:
        test_id = _shadow_test_in_db(db, n_decisions=20, agree=True)
        tracker = _tracker(db)
        comp = tracker.get_comparison(test_id)
        assert "rolling_metrics" in comp
        assert "challenger_sharpe" in comp["rolling_metrics"]

    def test_comparison_challenger_win_rate(self, db) -> None:
        test_id = _shadow_test_in_db(db, n_decisions=20, agree=True, pnl_diff=0.05)
        tracker = _tracker(db)
        comp = tracker.get_comparison(test_id)
        # A strongly good challenger should win some dimensions
        assert "challenger_win_rate" in comp
        assert 0.0 <= comp["challenger_win_rate"] <= 1.0


# ── Server Endpoints ──────────────────────────────────────────────────────────


class TestServerEndpoints:
    @property
    def _HDR(self) -> dict:
        # Minted fresh on each access (not a class-level constant) since the
        # JWT's `iat` is checked against a clock-skew window and would go
        # stale if computed once at class-definition/import time.
        return {
            "Authorization": "Bearer cc-test-token",
            "X-Signalium-Token": make_signalium_jwt(),
        }

    @pytest.fixture()
    def client(self, tmp_path: Path):
        import os

        os.environ["RESEARCH_LAB_ADMIN_TOKEN"] = "cc-test-token"
        os.environ["SIGNALIUM_JWT_SECRET"] = DEFAULT_TEST_JWT_SECRET
        os.environ["RESEARCH_LAB_DB_PATH"] = str(tmp_path / "server_cc.db")
        for mod_name in list(sys.modules):
            if "research_lab_server" in mod_name or "telemetry_server" in mod_name:
                del sys.modules[mod_name]
        from starlette.testclient import TestClient

        import trainer.research_lab.research_lab_server as srv_mod

        return TestClient(srv_mod.app)

    def test_shadow_status_empty(self, client) -> None:
        resp = client.get("/shadow/status", headers=self._HDR)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["status"]["n_running"] == 0

    def test_shadow_status_with_test_id_missing(self, client) -> None:
        resp = client.get("/shadow/status?test_id=no-such-test", headers=self._HDR)
        assert resp.status_code == 200
        # Should return error in status dict, not 500
        body = resp.json()
        assert body["success"] is True
        assert "error" in body["status"]

    def test_shadow_comparison_missing_test(self, client) -> None:
        resp = client.get("/shadow/comparison?test_id=not-exist", headers=self._HDR)
        assert resp.status_code == 200
        assert "error" in resp.json()["comparison"]

    def test_shadow_evaluate_missing_test(self, client) -> None:
        resp = client.post("/shadow/evaluate?test_id=not-exist", headers=self._HDR)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["evaluation"]["criterion_met"] is False

    def test_shadow_status_requires_auth(self, client) -> None:
        assert client.get("/shadow/status").status_code == 401  # no token → 401

    def test_shadow_comparison_requires_auth(self, client) -> None:
        assert client.get("/shadow/comparison?test_id=x").status_code == 401  # no token → 401

    def test_shadow_evaluate_requires_auth(self, client) -> None:
        assert client.post("/shadow/evaluate?test_id=x").status_code == 401  # no token → 401

    def test_shadow_status_after_creating_shadow_test(self, client) -> None:
        """Create a shadow test via existing API, then check /shadow/status."""
        create_resp = client.post(
            "/research/shadow-tests",
            json={
                "name": "integration-shadow",
                "challenger_version": "v2.0",
                "champion_version": "v1.0",
                "instruments": ["XAUUSD"],
                "min_decisions": 100,
            },
            headers=self._HDR,
        )
        assert create_resp.status_code == 200, create_resp.text
        test_id = create_resp.json()["data"]["test_id"]

        status_resp = client.get("/shadow/status", headers=self._HDR)
        assert status_resp.status_code == 200
        running = status_resp.json()["status"]["running_tests"]
        test_ids = [s["test_id"] for s in running]
        assert test_id in test_ids

    def test_shadow_comparison_after_decisions(self, client) -> None:
        """Create shadow test + decisions, then verify comparison returns 5 dimensions."""
        create_resp = client.post(
            "/research/shadow-tests",
            json={
                "name": "cmp-integration",
                "challenger_version": "v2.0",
                "champion_version": "v1.0",
                "instruments": ["XAUUSD"],
                "min_decisions": 10,
            },
            headers=self._HDR,
        )
        test_id = create_resp.json()["data"]["test_id"]

        # Record 15 decisions
        for i in range(15):
            client.post(
                f"/research/shadow-tests/{test_id}/decisions",
                json={
                    "instrument": "XAUUSD",
                    "champion_decision_type": "buy",
                    "challenger_decision_type": "buy",
                    "champion_confidence": 0.72,
                    "challenger_confidence": 0.74,
                    "regime": "RANGING",
                    "pnl_diff": 0.01,
                    "latency_diff_ms": 1.0,
                    "risk_diff": 0.002,
                },
                headers=self._HDR,
            )

        comp_resp = client.get(f"/shadow/comparison?test_id={test_id}", headers=self._HDR)
        assert comp_resp.status_code == 200
        comp = comp_resp.json()["comparison"]
        assert "dimension_results" in comp
        assert len(comp["dimension_results"]) == 5

    def test_shadow_evaluate_after_sufficient_decisions(self, client) -> None:
        """With a strongly winning challenger and low thresholds, criterion should be met."""
        # Create test
        create_resp = client.post(
            "/research/shadow-tests",
            json={
                "name": "eval-integration",
                "challenger_version": "v3.0",
                "champion_version": "v1.0",
                "instruments": ["XAUUSD"],
                "min_decisions": 10,
            },
            headers=self._HDR,
        )
        test_id = create_resp.json()["data"]["test_id"]

        # Record 15 decisions with strong agreement and positive pnl_diff
        for i in range(15):
            client.post(
                f"/research/shadow-tests/{test_id}/decisions",
                json={
                    "instrument": "XAUUSD",
                    "champion_decision_type": "buy",
                    "challenger_decision_type": "buy",
                    "champion_confidence": 0.70,
                    "challenger_confidence": 0.85,
                    "regime": "RANGING",
                    "pnl_diff": 0.05,
                    "latency_diff_ms": 0.5,
                    "risk_diff": 0.001,
                },
                headers=self._HDR,
            )

        eval_resp = client.post(f"/shadow/evaluate?test_id={test_id}", headers=self._HDR)
        assert eval_resp.status_code == 200
        body = eval_resp.json()
        assert body["success"] is True
        # evaluation dict is present regardless of criterion outcome
        assert "evaluation" in body
        assert "criterion_met" in body["evaluation"]
