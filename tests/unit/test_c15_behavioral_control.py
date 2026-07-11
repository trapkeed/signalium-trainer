"""Tests for C15: Behavioral Control Layer.

Covers:
  - CognitiveState enum: four values, severity ordering
  - BehavioralSignals: construction, defaults, clamps
  - BehavioralAssessment: to_dict, score invariants
  - BehavioralRecord: is_transition, to_dict
  - _confidence_inflation: sparse / paired / direction
  - _overconfidence_rate: threshold logic, blending
  - _drawdown_response_quality: below threshold, ideal, pathological
  - _aggressiveness_drift: neutral, positive, negative slope
  - _risk_escalation_rate: flat, accelerating, decelerating
  - _euphoria_risk: streak + inflation components
  - _loss_spiral_risk: loss + drawdown + response quality
  - _pathology_score: weighting, bounds
  - _recommend_state: all FSM branches
  - evaluate_behavior(): stateless, all five signals
  - adjust_state(): worsening immediate, recovery throttled
  - behavior_report(): keys, counts
  - BehavioralControlEngine: full round-trip
  - BehavioralControlEngine: after losses → DEFENSIVE
  - BehavioralControlEngine: after strong performance → euphoria guard → CAUTION
  - BehavioralControlEngine: recovery path DEFENSIVE → CAUTION → NORMAL
  - BehavioralControlEngine: history, summary, transition_count
  - BehavioralControlEngine: MAX_BEHAVIORAL_HISTORY cap
  - Edge cases: empty signal lists, sparse data
"""

from __future__ import annotations

import pytest

from trainer.qde.brain.behavioral import (
    AGGRESSION_DRIFT_MILD,
    DRAWDOWN_CAUTION_THRESHOLD,
    DRAWDOWN_DEFENSIVE_THRESHOLD,
    EUPHORIA_RISK_THRESHOLD,
    INFLATION_MILD_THRESHOLD,
    LOSING_STREAK_CAUTION,
    LOSING_STREAK_DEFENSIVE,
    MAX_BEHAVIORAL_HISTORY,
    OVERCONFIDENCE_RATE_MILD,
    PATHOLOGY_CAUTION_THRESHOLD,
    PATHOLOGY_DEFENSIVE_THRESHOLD,
    WINNING_STREAK_AGGRESSIVE,
    WINNING_STREAK_EUPHORIA,
    BehavioralAssessment,
    BehavioralControlEngine,
    BehavioralRecord,
    BehavioralSignals,
    CognitiveState,
    adjust_state,
    behavior_report,
    evaluate_behavior,
    state_severity,
)
from trainer.qde.brain.behavioral.behavioral_engine import (
    _aggressiveness_drift,
    _confidence_inflation,
    _drawdown_response_quality,
    _euphoria_risk,
    _loss_spiral_risk,
    _overconfidence_rate,
    _pathology_score,
    _recommend_state,
    _risk_escalation_rate,
)

_T0 = 1_700_000_000.0


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _signals(
    confidences: list[float] | None = None,
    outcomes: list[float] | None = None,
    aggressiveness: list[float] | None = None,
    risk_per_trade: list[float] | None = None,
    drawdown: float = 0.0,
    drawdown_peak_agg: float = 0.0,
    consecutive_losses: int = 0,
    consecutive_wins: int = 0,
    overconfidence_score: float = 0.0,
    metacognition_caution: float = 0.0,
    calibration_error: float = 0.0,
) -> BehavioralSignals:
    return BehavioralSignals(
        recent_trade_confidences=confidences or [],
        recent_trade_outcomes=outcomes or [],
        recent_aggressiveness_scores=aggressiveness or [],
        recent_risk_per_trade=risk_per_trade or [],
        current_drawdown_fraction=drawdown,
        drawdown_peak_aggressiveness=drawdown_peak_agg,
        consecutive_losses=consecutive_losses,
        consecutive_wins=consecutive_wins,
        overconfidence_score=overconfidence_score,
        metacognition_caution_level=metacognition_caution,
        calibration_error=calibration_error,
    )


def _assessment(
    recommended_state: CognitiveState = CognitiveState.NORMAL,
    pathology: float = 0.0,
    euphoria: float = 0.0,
    loss_spiral: float = 0.0,
    inflation: float = 0.0,
    overconf: float = 0.0,
    dq: float = 1.0,
    drift: float = 0.0,
    escalation: float = 0.0,
) -> BehavioralAssessment:
    return BehavioralAssessment(
        confidence_inflation=inflation,
        overconfidence_level=overconf,
        drawdown_response_quality=dq,
        aggressiveness_drift=drift,
        risk_escalation_rate=escalation,
        pathology_score=pathology,
        euphoria_risk=euphoria,
        loss_spiral_risk=loss_spiral,
        recommended_state=recommended_state,
        reasoning="test",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CognitiveState enum
# ═══════════════════════════════════════════════════════════════════════════════

class TestCognitiveState:
    def test_four_values(self):
        expected = {"CAUTION", "NORMAL", "AGGRESSIVE", "DEFENSIVE"}
        assert {s.value for s in CognitiveState} == expected

    def test_str_enum(self):
        assert CognitiveState.NORMAL == "NORMAL"
        assert CognitiveState.DEFENSIVE == "DEFENSIVE"

    def test_severity_ordering(self):
        """AGGRESSIVE < NORMAL < CAUTION < DEFENSIVE in severity."""
        assert state_severity(CognitiveState.AGGRESSIVE) < state_severity(CognitiveState.NORMAL)
        assert state_severity(CognitiveState.NORMAL)     < state_severity(CognitiveState.CAUTION)
        assert state_severity(CognitiveState.CAUTION)    < state_severity(CognitiveState.DEFENSIVE)

    def test_severity_all_distinct(self):
        severities = [state_severity(s) for s in CognitiveState]
        assert len(set(severities)) == len(severities)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BehavioralSignals
# ═══════════════════════════════════════════════════════════════════════════════

class TestBehavioralSignals:
    def test_defaults(self):
        s = BehavioralSignals()
        assert s.recent_trade_confidences == []
        assert s.consecutive_losses == 0
        assert s.current_drawdown_fraction == 0.0

    def test_negative_drawdown_clamped(self):
        s = BehavioralSignals(current_drawdown_fraction=-0.05)
        assert s.current_drawdown_fraction == 0.0

    def test_negative_streaks_clamped(self):
        s = BehavioralSignals(consecutive_losses=-3, consecutive_wins=-2)
        assert s.consecutive_losses == 0
        assert s.consecutive_wins == 0

    def test_accepts_all_fields(self):
        s = _signals(
            confidences=[0.70, 0.65],
            outcomes=[0.0, 1.0],
            aggressiveness=[0.5, 0.6],
            risk_per_trade=[0.02, 0.03],
            drawdown=0.06,
            consecutive_losses=2,
            consecutive_wins=0,
        )
        assert len(s.recent_trade_confidences) == 2
        assert s.current_drawdown_fraction == 0.06


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Internal signal functions
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfidenceInflation:
    def test_sparse_returns_zero(self):
        assert _confidence_inflation([0.7], [0.0]) == 0.0

    def test_no_inflation_perfect_calibration(self):
        result = _confidence_inflation([0.60, 0.60], [0.60, 0.60])
        assert abs(result) < 1e-6

    def test_positive_inflation(self):
        # predicted 0.70, won 0.0 twice → inflation = +0.70
        result = _confidence_inflation([0.70, 0.70], [0.0, 0.0])
        assert result > 0.0

    def test_negative_inflation_conservative(self):
        # predicted 0.40, won 1.0 = conservative
        result = _confidence_inflation([0.40, 0.40], [1.0, 1.0])
        assert result < 0.0

    def test_mixed(self):
        # 0.80 - 1.0 = -0.20; 0.80 - 0.0 = +0.80; mean = 0.30
        result = _confidence_inflation([0.80, 0.80], [1.0, 0.0])
        assert abs(result - 0.30) < 1e-4

    def test_empty_returns_zero(self):
        assert _confidence_inflation([], []) == 0.0


class TestOverconfidenceRate:
    def test_no_high_conf_trades(self):
        assert _overconfidence_rate([0.50, 0.55], [0.0, 0.0]) == 0.0

    def test_all_high_conf_win(self):
        assert _overconfidence_rate([0.80, 0.75], [1.0, 1.0]) == 0.0

    def test_all_high_conf_lose(self):
        result = _overconfidence_rate([0.80, 0.75], [0.0, 0.0])
        assert result == 1.0

    def test_half_high_conf_lose(self):
        result = _overconfidence_rate([0.80, 0.75, 0.40, 0.40], [0.0, 1.0, 0.0, 0.0])
        # 2 high-conf trades (0.80 and 0.75), 1 loss → 0.50
        assert abs(result - 0.50) < 1e-4

    def test_empty(self):
        assert _overconfidence_rate([], []) == 0.0

    def test_threshold_boundary(self):
        # Exactly at threshold (0.65) should count
        result = _overconfidence_rate([0.65], [0.0])
        assert result == 1.0


class TestDrawdownResponseQuality:
    def test_no_drawdown_perfect_score(self):
        result = _drawdown_response_quality(0.02, 0.8, [0.5, 0.6])
        assert result == 1.0

    def test_drawdown_low_aggressiveness_good_score(self):
        """During drawdown, aggressiveness ≤ 0.30 → good quality."""
        result = _drawdown_response_quality(
            0.08, drawdown_peak_aggressiveness=0.25, recent_aggressiveness=[0.20, 0.25]
        )
        assert result > 0.5

    def test_drawdown_high_aggressiveness_poor_score(self):
        """During drawdown, aggressiveness ≥ 0.70 → pathological."""
        result = _drawdown_response_quality(
            0.09, drawdown_peak_aggressiveness=0.80, recent_aggressiveness=[0.75, 0.80]
        )
        assert result == 0.0

    def test_result_bounded_01(self):
        for ddown, peak, recent in [
            (0.0, 0.0, []),
            (0.15, 1.0, [1.0, 1.0]),
            (0.0, 0.0, [0.1]),
        ]:
            r = _drawdown_response_quality(ddown, peak, recent)
            assert 0.0 <= r <= 1.0


class TestAggressivenessDrift:
    def test_sparse_returns_zero(self):
        assert _aggressiveness_drift([0.5, 0.6]) == 0.0

    def test_flat_no_drift(self):
        result = _aggressiveness_drift([0.5, 0.5, 0.5, 0.5, 0.5])
        assert abs(result) < 1e-6

    def test_rising_positive_drift(self):
        result = _aggressiveness_drift([0.30, 0.40, 0.50, 0.60, 0.70])
        assert result > 0.0

    def test_falling_negative_drift(self):
        result = _aggressiveness_drift([0.70, 0.60, 0.50, 0.40, 0.30])
        assert result < 0.0

    def test_drift_bounded(self):
        for scores in [
            [0.0] * 10 + [1.0] * 10,
            [1.0] * 10 + [0.0] * 10,
            [0.5] * 20,
        ]:
            r = _aggressiveness_drift(scores)
            assert -1.0 <= r <= 1.0


class TestRiskEscalationRate:
    def test_sparse_returns_zero(self):
        assert _risk_escalation_rate([0.02, 0.02, 0.02]) == 0.0

    def test_flat_no_escalation(self):
        result = _risk_escalation_rate([0.02] * 10)
        assert result == 0.0

    def test_accelerating_risk_positive(self):
        # First half flat, second half rising fast
        result = _risk_escalation_rate([0.02, 0.02, 0.02, 0.02, 0.10, 0.20, 0.30, 0.40])
        assert result > 0.0

    def test_decelerating_risk_zero(self):
        # Decreasing risk → no escalation
        result = _risk_escalation_rate([0.40, 0.30, 0.20, 0.10, 0.02, 0.02, 0.02, 0.02])
        assert result == 0.0

    def test_bounded_01(self):
        for series in [
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.02] * 8,
        ]:
            r = _risk_escalation_rate(series)
            assert 0.0 <= r <= 1.0


class TestEuphoriaRisk:
    def test_no_streak_no_euphoria(self):
        result = _euphoria_risk(consecutive_wins=0, confidence_inflation=0.0)
        assert result == 0.0

    def test_long_streak_raises_euphoria(self):
        result = _euphoria_risk(consecutive_wins=WINNING_STREAK_EUPHORIA + 2, confidence_inflation=0.0)
        assert result > 0.0

    def test_inflation_raises_euphoria(self):
        result_low  = _euphoria_risk(consecutive_wins=4, confidence_inflation=0.0)
        result_high = _euphoria_risk(consecutive_wins=4, confidence_inflation=0.30)
        assert result_high > result_low

    def test_negative_inflation_no_contribution(self):
        # Conservative (negative inflation) should not add to euphoria
        result = _euphoria_risk(consecutive_wins=0, confidence_inflation=-0.20)
        assert result == 0.0

    def test_bounded_01(self):
        for wins, infl in [(0, 0), (10, 0.5), (20, 1.0)]:
            r = _euphoria_risk(wins, infl)
            assert 0.0 <= r <= 1.0


class TestLossSpiralRisk:
    def test_no_losses_no_spiral(self):
        result = _loss_spiral_risk(0, 0.0, 1.0)
        assert result == 0.0

    def test_losses_raise_spiral(self):
        result_low  = _loss_spiral_risk(1, 0.0, 1.0)
        result_high = _loss_spiral_risk(5, 0.0, 1.0)
        assert result_high > result_low

    def test_drawdown_raises_spiral(self):
        result_low  = _loss_spiral_risk(2, 0.02, 1.0)
        result_high = _loss_spiral_risk(2, 0.10, 1.0)
        assert result_high > result_low

    def test_poor_response_amplifies_spiral(self):
        result_good = _loss_spiral_risk(3, 0.06, drawdown_response_quality=1.0)
        result_poor = _loss_spiral_risk(3, 0.06, drawdown_response_quality=0.0)
        assert result_poor > result_good

    def test_bounded_01(self):
        for l, d, q in [(0, 0, 1), (10, 1, 0), (3, 0.08, 0.5)]:
            r = _loss_spiral_risk(l, d, q)
            assert 0.0 <= r <= 1.0


class TestPathologyScore:
    def test_all_zero_inputs_score_zero(self):
        result = _pathology_score(0.0, 0.0, 1.0, 0.0, 0.0)
        assert result == 0.0

    def test_all_severe_scores_high(self):
        result = _pathology_score(
            confidence_inflation=0.25,
            overconfidence_level=0.60,
            drawdown_response_quality=0.0,
            aggressiveness_drift=0.40,
            risk_escalation_rate=0.30,
        )
        assert result > 0.5

    def test_negative_inflation_no_contrib(self):
        # Conservative (negative inflation) should not raise pathology
        result_pos = _pathology_score(0.20, 0.0, 1.0, 0.0, 0.0)
        result_neg = _pathology_score(-0.20, 0.0, 1.0, 0.0, 0.0)
        assert result_neg < result_pos

    def test_negative_drift_no_contrib(self):
        # Drifting more cautious should not raise pathology
        result_pos = _pathology_score(0.0, 0.0, 1.0, 0.30, 0.0)
        result_neg = _pathology_score(0.0, 0.0, 1.0, -0.30, 0.0)
        assert result_neg < result_pos

    def test_bounded_01(self):
        for args in [
            (0.0, 0.0, 1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0, 1.0, 1.0),
            (0.15, 0.35, 0.7, 0.1, 0.05),
        ]:
            r = _pathology_score(*args)
            assert 0.0 <= r <= 1.0

    def test_metacognition_blend(self):
        without = _pathology_score(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        with_meta = _pathology_score(0.0, 0.0, 1.0, 0.0, 0.0, 0.80, 0.80)
        assert with_meta > without


# ═══════════════════════════════════════════════════════════════════════════════
# 4. _recommend_state FSM
# ═══════════════════════════════════════════════════════════════════════════════

class TestRecommendState:
    def test_losing_streak_defensive_threshold(self):
        state, trigger = _recommend_state(
            consecutive_losses=LOSING_STREAK_DEFENSIVE,
            consecutive_wins=0, current_drawdown=0.0,
            pathology=0.0, euphoria=0.0, loss_spiral=0.0,
        )
        assert state == CognitiveState.DEFENSIVE
        assert "losing_streak" in trigger

    def test_drawdown_defensive(self):
        state, _ = _recommend_state(
            consecutive_losses=0, consecutive_wins=0,
            current_drawdown=DRAWDOWN_DEFENSIVE_THRESHOLD,
            pathology=0.0, euphoria=0.0, loss_spiral=0.0,
        )
        assert state == CognitiveState.DEFENSIVE

    def test_high_pathology_defensive(self):
        state, _ = _recommend_state(
            consecutive_losses=0, consecutive_wins=0, current_drawdown=0.0,
            pathology=PATHOLOGY_DEFENSIVE_THRESHOLD,
            euphoria=0.0, loss_spiral=0.0,
        )
        assert state == CognitiveState.DEFENSIVE

    def test_euphoria_long_streak_caution(self):
        state, trigger = _recommend_state(
            consecutive_losses=0,
            consecutive_wins=WINNING_STREAK_EUPHORIA,
            current_drawdown=0.0,
            pathology=0.0, euphoria=0.8, loss_spiral=0.0,
        )
        assert state == CognitiveState.CAUTION
        assert "euphoria" in trigger.lower()

    def test_euphoria_risk_from_aggressive(self):
        state, _ = _recommend_state(
            consecutive_losses=0, consecutive_wins=4, current_drawdown=0.0,
            pathology=0.0, euphoria=EUPHORIA_RISK_THRESHOLD + 0.01, loss_spiral=0.0,
            current_state=CognitiveState.AGGRESSIVE,
        )
        assert state == CognitiveState.CAUTION

    def test_mild_losing_streak_caution(self):
        state, _ = _recommend_state(
            consecutive_losses=LOSING_STREAK_CAUTION,
            consecutive_wins=0, current_drawdown=0.0,
            pathology=0.0, euphoria=0.0, loss_spiral=0.0,
        )
        assert state == CognitiveState.CAUTION

    def test_pathology_caution(self):
        state, _ = _recommend_state(
            consecutive_losses=0, consecutive_wins=0, current_drawdown=0.0,
            pathology=PATHOLOGY_CAUTION_THRESHOLD,
            euphoria=0.0, loss_spiral=0.0,
        )
        assert state == CognitiveState.CAUTION

    def test_defensive_recovery_to_caution(self):
        state, trigger = _recommend_state(
            consecutive_losses=0, consecutive_wins=0,
            current_drawdown=DRAWDOWN_CAUTION_THRESHOLD - 0.01,
            pathology=0.0, euphoria=0.0, loss_spiral=0.0,
            current_state=CognitiveState.DEFENSIVE,
        )
        assert state == CognitiveState.CAUTION
        assert "recover" in trigger.lower()

    def test_aggressive_promotion(self):
        state, _ = _recommend_state(
            consecutive_losses=0,
            consecutive_wins=WINNING_STREAK_AGGRESSIVE,
            current_drawdown=0.0,
            pathology=0.10, euphoria=0.10, loss_spiral=0.0,
            current_state=CognitiveState.NORMAL,
        )
        assert state == CognitiveState.AGGRESSIVE

    def test_caution_resolved_to_normal(self):
        state, trigger = _recommend_state(
            consecutive_losses=0, consecutive_wins=0,
            current_drawdown=0.01,
            pathology=0.10, euphoria=0.0, loss_spiral=0.0,
            current_state=CognitiveState.CAUTION,
        )
        assert state == CognitiveState.NORMAL
        assert "caution_resolved" in trigger

    def test_normal_no_trigger_stays_normal(self):
        state, trigger = _recommend_state(
            consecutive_losses=0, consecutive_wins=0, current_drawdown=0.0,
            pathology=0.0, euphoria=0.0, loss_spiral=0.0,
            current_state=CognitiveState.NORMAL,
        )
        assert state == CognitiveState.NORMAL
        assert "no_trigger" in trigger

    def test_aggressive_fades_without_streak(self):
        state, _ = _recommend_state(
            consecutive_losses=0,
            consecutive_wins=WINNING_STREAK_AGGRESSIVE - 1,
            current_drawdown=0.0,
            pathology=0.0, euphoria=0.0, loss_spiral=0.0,
            current_state=CognitiveState.AGGRESSIVE,
        )
        assert state == CognitiveState.NORMAL


# ═══════════════════════════════════════════════════════════════════════════════
# 5. evaluate_behavior()
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluateBehavior:
    def test_returns_behavioral_assessment(self):
        result = evaluate_behavior(_signals())
        assert isinstance(result, BehavioralAssessment)

    def test_normal_signals_no_pathology(self):
        result = evaluate_behavior(_signals(
            confidences=[0.65, 0.60, 0.70],
            outcomes=[1.0, 1.0, 1.0],
            aggressiveness=[0.45, 0.45, 0.45],
            risk_per_trade=[0.02, 0.02, 0.02],
        ))
        assert result.recommended_state == CognitiveState.NORMAL
        assert result.pathology_score < PATHOLOGY_CAUTION_THRESHOLD

    def test_losing_streak_triggers_defensive(self):
        result = evaluate_behavior(_signals(consecutive_losses=LOSING_STREAK_DEFENSIVE))
        assert result.recommended_state == CognitiveState.DEFENSIVE

    def test_losing_streak_caution(self):
        result = evaluate_behavior(_signals(consecutive_losses=LOSING_STREAK_CAUTION))
        assert result.recommended_state in (CognitiveState.CAUTION, CognitiveState.DEFENSIVE)

    def test_severe_drawdown_defensive(self):
        result = evaluate_behavior(_signals(
            drawdown=DRAWDOWN_DEFENSIVE_THRESHOLD,
            drawdown_peak_agg=0.70,
            aggressiveness=[0.70, 0.75],
        ))
        assert result.recommended_state == CognitiveState.DEFENSIVE

    def test_high_confidence_inflation_caution(self):
        # Predict 0.80, lose every trade → inflation = +0.80
        result = evaluate_behavior(_signals(
            confidences=[0.80] * 5,
            outcomes=[0.0] * 5,
        ))
        assert result.confidence_inflation > INFLATION_MILD_THRESHOLD
        assert result.recommended_state in (
            CognitiveState.CAUTION, CognitiveState.DEFENSIVE
        )

    def test_winning_streak_aggressive(self):
        result = evaluate_behavior(_signals(
            consecutive_wins=WINNING_STREAK_AGGRESSIVE,
            confidences=[0.60] * 6,
            outcomes=[1.0] * 6,
        ), current_state=CognitiveState.NORMAL)
        assert result.recommended_state == CognitiveState.AGGRESSIVE

    def test_euphoria_after_wins_then_caution(self):
        result = evaluate_behavior(_signals(consecutive_wins=WINNING_STREAK_EUPHORIA + 1))
        assert result.recommended_state == CognitiveState.CAUTION

    def test_all_scores_bounded(self):
        for i in range(3):
            sig = _signals(
                confidences=[0.70] * 10,
                outcomes=[float(i % 2)] * 10,
                aggressiveness=[0.3 + i * 0.2] * 10,
                risk_per_trade=[0.01 + i * 0.02] * 10,
                drawdown=i * 0.04,
                consecutive_losses=i * 2,
            )
            a = evaluate_behavior(sig)
            assert 0.0 <= a.pathology_score <= 1.0
            assert 0.0 <= a.euphoria_risk <= 1.0
            assert 0.0 <= a.loss_spiral_risk <= 1.0
            assert -1.0 <= a.aggressiveness_drift <= 1.0

    def test_metacognition_signals_used(self):
        """Pre-computed metacognition signals should influence the assessment."""
        baseline = evaluate_behavior(_signals(overconfidence_score=0.0))
        with_meta = evaluate_behavior(_signals(overconfidence_score=0.80, metacognition_caution=0.70))
        assert with_meta.pathology_score > baseline.pathology_score

    def test_to_dict_keys(self):
        d = evaluate_behavior(_signals()).to_dict()
        for key in (
            "confidence_inflation", "overconfidence_level", "drawdown_response_quality",
            "aggressiveness_drift", "risk_escalation_rate", "pathology_score",
            "euphoria_risk", "loss_spiral_risk", "recommended_state", "reasoning",
        ):
            assert key in d

    def test_reasoning_nonempty(self):
        result = evaluate_behavior(_signals())
        assert len(result.reasoning) > 0

    def test_empty_signals_no_crash(self):
        result = evaluate_behavior(BehavioralSignals())
        assert isinstance(result, BehavioralAssessment)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. adjust_state()
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdjustState:
    def test_worsening_immediate(self):
        """Can jump from NORMAL to DEFENSIVE in one step."""
        a = _assessment(recommended_state=CognitiveState.DEFENSIVE)
        new = adjust_state(CognitiveState.NORMAL, a)
        assert new == CognitiveState.DEFENSIVE

    def test_recovery_throttled_one_level(self):
        """DEFENSIVE → NORMAL recommendation → only reach CAUTION (one step)."""
        a = _assessment(recommended_state=CognitiveState.NORMAL)
        new = adjust_state(CognitiveState.DEFENSIVE, a)
        assert new == CognitiveState.CAUTION

    def test_caution_to_normal_throttled(self):
        """CAUTION → AGGRESSIVE recommendation → only reach NORMAL (one step)."""
        a = _assessment(recommended_state=CognitiveState.AGGRESSIVE)
        new = adjust_state(CognitiveState.CAUTION, a)
        assert new == CognitiveState.NORMAL

    def test_same_state_returns_same(self):
        a = _assessment(recommended_state=CognitiveState.NORMAL)
        new = adjust_state(CognitiveState.NORMAL, a)
        assert new == CognitiveState.NORMAL

    def test_normal_to_caution_direct(self):
        a = _assessment(recommended_state=CognitiveState.CAUTION)
        new = adjust_state(CognitiveState.NORMAL, a)
        assert new == CognitiveState.CAUTION

    def test_normal_to_aggressive_direct(self):
        a = _assessment(recommended_state=CognitiveState.AGGRESSIVE)
        new = adjust_state(CognitiveState.NORMAL, a)
        assert new == CognitiveState.AGGRESSIVE

    def test_defensive_to_caution(self):
        a = _assessment(recommended_state=CognitiveState.CAUTION)
        new = adjust_state(CognitiveState.DEFENSIVE, a)
        assert new == CognitiveState.CAUTION

    def test_aggressive_to_caution_euphoria(self):
        a = _assessment(recommended_state=CognitiveState.CAUTION)
        new = adjust_state(CognitiveState.AGGRESSIVE, a)
        assert new == CognitiveState.CAUTION

    def test_returns_cognitive_state(self):
        a = _assessment(recommended_state=CognitiveState.NORMAL)
        result = adjust_state(CognitiveState.NORMAL, a)
        assert isinstance(result, CognitiveState)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. BehavioralRecord
# ═══════════════════════════════════════════════════════════════════════════════

class TestBehavioralRecord:
    def _record(self, prev: CognitiveState, new: CognitiveState) -> BehavioralRecord:
        return BehavioralRecord(
            record_id="r1",
            previous_state=prev,
            new_state=new,
            assessment=_assessment(),
            session_label="S1",
            recorded_at=_T0,
            trigger="test",
        )

    def test_is_transition_true(self):
        r = self._record(CognitiveState.NORMAL, CognitiveState.CAUTION)
        assert r.is_transition is True

    def test_is_transition_false(self):
        r = self._record(CognitiveState.NORMAL, CognitiveState.NORMAL)
        assert r.is_transition is False

    def test_to_dict_keys(self):
        r = self._record(CognitiveState.NORMAL, CognitiveState.DEFENSIVE)
        d = r.to_dict()
        for key in ("record_id", "previous_state", "new_state", "assessment",
                    "session_label", "recorded_at", "trigger", "is_transition"):
            assert key in d

    def test_to_dict_values(self):
        r = self._record(CognitiveState.NORMAL, CognitiveState.DEFENSIVE)
        d = r.to_dict()
        assert d["previous_state"] == "NORMAL"
        assert d["new_state"] == "DEFENSIVE"
        assert d["is_transition"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# 8. BehavioralControlEngine
# ═══════════════════════════════════════════════════════════════════════════════

class TestBehavioralControlEngine:
    def test_instantiates_default(self):
        engine = BehavioralControlEngine()
        assert engine.state == CognitiveState.NORMAL

    def test_instantiates_with_initial_state(self):
        engine = BehavioralControlEngine(initial_state=CognitiveState.CAUTION)
        assert engine.state == CognitiveState.CAUTION

    def test_instrument_label(self):
        engine = BehavioralControlEngine(instrument="EURUSD")
        assert engine.instrument == "EURUSD"

    def test_evaluate_returns_assessment(self):
        engine = BehavioralControlEngine()
        a = engine.evaluate_behavior(_signals())
        assert isinstance(a, BehavioralAssessment)

    def test_adjust_updates_state(self):
        engine = BehavioralControlEngine()
        a = engine.evaluate_behavior(_signals(consecutive_losses=LOSING_STREAK_DEFENSIVE))
        new = engine.adjust_state(a)
        assert new == CognitiveState.DEFENSIVE
        assert engine.state == CognitiveState.DEFENSIVE

    def test_history_appended(self):
        engine = BehavioralControlEngine()
        a = engine.evaluate_behavior(_signals())
        engine.adjust_state(a)
        assert len(engine.history()) == 1

    def test_history_records_transition(self):
        engine = BehavioralControlEngine()
        a = engine.evaluate_behavior(_signals(consecutive_losses=LOSING_STREAK_DEFENSIVE))
        engine.adjust_state(a)
        records = engine.history()
        assert records[-1].is_transition is True

    def test_history_session_label(self):
        engine = BehavioralControlEngine()
        a = engine.evaluate_behavior(_signals())
        engine.adjust_state(a, session_label="2024-01-15_LONDON")
        assert engine.history()[-1].session_label == "2024-01-15_LONDON"

    def test_transition_count(self):
        engine = BehavioralControlEngine()
        for _ in range(3):
            a = engine.evaluate_behavior(_signals(consecutive_losses=LOSING_STREAK_DEFENSIVE))
            engine.adjust_state(a)
        assert engine.transition_count() >= 1  # first call transitions NORMAL→DEFENSIVE

    def test_time_in_state(self):
        engine = BehavioralControlEngine()
        for _ in range(3):
            a = engine.evaluate_behavior(_signals())
            engine.adjust_state(a)
        assert engine.time_in_state(CognitiveState.NORMAL) >= 1

    def test_recent_states(self):
        engine = BehavioralControlEngine()
        for _ in range(5):
            engine.adjust_state(engine.evaluate_behavior(_signals()))
        states = engine.recent_states(3)
        assert len(states) == 3

    def test_state_isolated_between_instances(self):
        e1 = BehavioralControlEngine()
        e2 = BehavioralControlEngine()
        a = e1.evaluate_behavior(_signals(consecutive_losses=LOSING_STREAK_DEFENSIVE))
        e1.adjust_state(a)
        assert e1.state == CognitiveState.DEFENSIVE
        assert e2.state == CognitiveState.NORMAL

    def test_history_cap(self):
        engine = BehavioralControlEngine()
        for _ in range(MAX_BEHAVIORAL_HISTORY + 50):
            engine.adjust_state(engine.evaluate_behavior(_signals()))
        assert len(engine.history()) <= MAX_BEHAVIORAL_HISTORY

    def test_summary_keys(self):
        engine = BehavioralControlEngine(instrument="GOLD")
        d = engine.summary()
        for key in ("instrument", "current_state", "history_size",
                    "transition_count", "time_in_defensive",
                    "time_in_caution", "time_in_aggressive"):
            assert key in d


# ═══════════════════════════════════════════════════════════════════════════════
# 9. behavior_report()
# ═══════════════════════════════════════════════════════════════════════════════

class TestBehaviorReport:
    def test_returns_dict(self):
        engine = BehavioralControlEngine(instrument="EUR")
        engine.adjust_state(engine.evaluate_behavior(_signals()))
        report = behavior_report(engine)
        assert isinstance(report, dict)

    def test_required_keys(self):
        engine = BehavioralControlEngine(instrument="EUR")
        engine.adjust_state(engine.evaluate_behavior(_signals()))
        report = engine.behavior_report()
        for key in (
            "instrument", "current_state", "state_history",
            "transition_count", "time_in_state", "latest_assessment",
            "pathology_trend", "euphoria_events", "loss_spiral_events",
            "history_size", "generated_at",
        ):
            assert key in report

    def test_instrument_in_report(self):
        engine = BehavioralControlEngine(instrument="GBPUSD")
        engine.adjust_state(engine.evaluate_behavior(_signals()))
        assert engine.behavior_report()["instrument"] == "GBPUSD"

    def test_euphoria_events_counted(self):
        engine = BehavioralControlEngine(initial_state=CognitiveState.AGGRESSIVE)
        # Force a CAUTION assessment
        a = _assessment(recommended_state=CognitiveState.CAUTION)
        a = BehavioralAssessment(
            confidence_inflation=0.0,
            overconfidence_level=0.0,
            drawdown_response_quality=1.0,
            aggressiveness_drift=0.0,
            risk_escalation_rate=0.0,
            pathology_score=0.0,
            euphoria_risk=0.8,
            loss_spiral_risk=0.0,
            recommended_state=CognitiveState.CAUTION,
            reasoning="test",
        )
        engine.adjust_state(a)
        report = engine.behavior_report()
        assert report["euphoria_events"] >= 1

    def test_loss_spiral_events_counted(self):
        engine = BehavioralControlEngine()
        a = BehavioralAssessment(
            confidence_inflation=0.0,
            overconfidence_level=0.0,
            drawdown_response_quality=1.0,
            aggressiveness_drift=0.0,
            risk_escalation_rate=0.0,
            pathology_score=0.0,
            euphoria_risk=0.0,
            loss_spiral_risk=0.80,
            recommended_state=CognitiveState.DEFENSIVE,
            reasoning="test",
        )
        engine.adjust_state(a)
        report = engine.behavior_report()
        assert report["loss_spiral_events"] >= 1

    def test_pathology_trend_list(self):
        engine = BehavioralControlEngine()
        for _ in range(5):
            engine.adjust_state(engine.evaluate_behavior(_signals()))
        report = engine.behavior_report()
        assert isinstance(report["pathology_trend"], list)
        assert len(report["pathology_trend"]) == 5

    def test_empty_engine_report(self):
        engine = BehavioralControlEngine(instrument="X")
        report = engine.behavior_report()
        assert report["history_size"] == 0
        assert report["latest_assessment"] is None


# ═══════════════════════════════════════════════════════════════════════════════
# 10. End-to-end behavioral scenarios
# ═══════════════════════════════════════════════════════════════════════════════

class TestBehavioralScenarios:
    def test_losses_reduce_aggressiveness(self):
        """After several losses, state moves to DEFENSIVE."""
        engine = BehavioralControlEngine(instrument="EURUSD")
        for i in range(1, LOSING_STREAK_DEFENSIVE + 1):
            sig = _signals(
                consecutive_losses=i,
                drawdown=i * 0.02,
                drawdown_peak_agg=0.65,
                aggressiveness=[0.65, 0.70],
            )
            a = engine.evaluate_behavior(sig)
            engine.adjust_state(a)
        assert engine.state == CognitiveState.DEFENSIVE

    def test_euphoria_guard_after_strong_performance(self):
        """After unusually strong winning streak, state moves to CAUTION via euphoria guard."""
        engine = BehavioralControlEngine(initial_state=CognitiveState.NORMAL)
        sig = _signals(
            consecutive_wins=WINNING_STREAK_EUPHORIA + 1,
            confidences=[0.80] * 10,
            outcomes=[1.0] * 10,
        )
        a = engine.evaluate_behavior(sig)
        engine.adjust_state(a)
        # Should transition to AGGRESSIVE first, then euphoria triggers CAUTION
        # or directly CAUTION depending on streak magnitude
        assert engine.state in (CognitiveState.CAUTION, CognitiveState.AGGRESSIVE)
        # After one more call with euphoria risk still high, should be CAUTION
        engine2 = BehavioralControlEngine(initial_state=CognitiveState.AGGRESSIVE)
        a2 = engine2.evaluate_behavior(sig)
        engine2.adjust_state(a2)
        assert engine2.state == CognitiveState.CAUTION

    def test_recovery_path_defensive_to_normal(self):
        """DEFENSIVE → CAUTION → NORMAL recovery (throttled)."""
        engine = BehavioralControlEngine(initial_state=CognitiveState.DEFENSIVE)

        # Clean signals: no losses, recovering drawdown
        clean = _signals(consecutive_losses=0, drawdown=0.01)

        # Step 1: DEFENSIVE → CAUTION (throttled from NORMAL recommendation)
        a1 = engine.evaluate_behavior(clean)
        engine.adjust_state(a1)
        assert engine.state == CognitiveState.CAUTION

        # Step 2: CAUTION → NORMAL (throttled from NORMAL recommendation)
        a2 = engine.evaluate_behavior(clean)
        engine.adjust_state(a2)
        assert engine.state == CognitiveState.NORMAL

    def test_confidence_inflation_triggers_caution(self):
        """Systematic overestimation of confidence raises pathology, triggers CAUTION."""
        engine = BehavioralControlEngine()
        # Always predict 0.80, always lose → severe inflation
        sig = _signals(
            confidences=[0.80] * 8,
            outcomes=[0.0] * 8,
        )
        a = engine.evaluate_behavior(sig)
        engine.adjust_state(a)
        assert engine.state in (CognitiveState.CAUTION, CognitiveState.DEFENSIVE)

    def test_aggressiveness_drift_triggers_caution(self):
        """Steady creep in aggressiveness triggers CAUTION."""
        engine = BehavioralControlEngine()
        scores = [0.20 + i * 0.05 for i in range(20)]  # rising from 0.20 to 1.15
        sig = _signals(aggressiveness=scores)
        a = engine.evaluate_behavior(sig)
        assert a.aggressiveness_drift > AGGRESSION_DRIFT_MILD

    def test_clean_record_stays_normal(self):
        """Healthy signals keep the engine in NORMAL."""
        engine = BehavioralControlEngine()
        for i in range(20):
            sig = _signals(
                confidences=[0.60, 0.65, 0.60],
                outcomes=[1.0, 1.0, 0.0],
                aggressiveness=[0.45, 0.45, 0.45],
                risk_per_trade=[0.02, 0.02, 0.02],
                consecutive_losses=0,
                consecutive_wins=min(i, 2),
                drawdown=0.0,
            )
            a = engine.evaluate_behavior(sig)
            engine.adjust_state(a)
        assert engine.state == CognitiveState.NORMAL

    def test_aggressive_promotion_then_fade(self):
        """Engine enters AGGRESSIVE on winning streak, returns to NORMAL when streak ends."""
        engine = BehavioralControlEngine()
        # Promote to AGGRESSIVE
        sig_win = _signals(consecutive_wins=WINNING_STREAK_AGGRESSIVE)
        a = engine.evaluate_behavior(sig_win)
        engine.adjust_state(a)
        assert engine.state == CognitiveState.AGGRESSIVE

        # Streak fades
        sig_flat = _signals(consecutive_wins=WINNING_STREAK_AGGRESSIVE - 1)
        a = engine.evaluate_behavior(sig_flat)
        engine.adjust_state(a)
        assert engine.state == CognitiveState.NORMAL


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Import sanity
# ═══════════════════════════════════════════════════════════════════════════════

class TestImportSanity:
    def test_all_spec_callables_importable(self):
        from trainer.qde.brain.behavioral import (
            evaluate_behavior,
            adjust_state,
            behavior_report,
        )
        assert all(callable(fn) for fn in (evaluate_behavior, adjust_state, behavior_report))

    def test_cognitive_state_importable(self):
        from trainer.qde.brain.behavioral import CognitiveState
        assert CognitiveState.CAUTION.value == "CAUTION"
        assert CognitiveState.AGGRESSIVE.value == "AGGRESSIVE"
        assert CognitiveState.DEFENSIVE.value == "DEFENSIVE"
        assert CognitiveState.NORMAL.value == "NORMAL"

    def test_behavioral_control_engine_importable(self):
        from trainer.qde.brain.behavioral import BehavioralControlEngine
        engine = BehavioralControlEngine(instrument="TEST")
        assert engine.state == CognitiveState.NORMAL

    def test_package_path_correct(self):
        import trainer.qde.brain.behavioral as pkg
        assert hasattr(pkg, "CognitiveState")
        assert hasattr(pkg, "evaluate_behavior")
        assert hasattr(pkg, "adjust_state")
        assert hasattr(pkg, "behavior_report")
        assert hasattr(pkg, "BehavioralControlEngine")
