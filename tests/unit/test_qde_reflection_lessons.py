"""QDE Brain — post-trade reflection/surprise/lesson-extraction tests.

Covers `surprise_engine.compute_surprise` (prediction-error scoring),
`reflection_engine.TradeReflectionEngine` (per-trade post-mortem — distinct
from the periodic multi-decision `reflection.ReflectionEngine`), and
`lesson_extractor.LessonExtractor` (routing findings into symbol
personalities, conviction/uncertainty calibration, and hypotheses) — plus
`QDEBrain.reflect_on_trade` wiring all three together.
"""

from __future__ import annotations

from trainer.memory.decision_models import (
    DecisionOutcome,
    DecisionReasoning,
    DecisionRecord,
    DecisionScore,
    DecisionSnapshot,
)
from trainer.memory.decision_types import DecisionGrade, DecisionType, OutcomeType
from trainer.qde.brain.brain import QDEBrain
from trainer.qde.brain.brain_models import BrainState, Hypothesis
from trainer.qde.brain.brain_types import BeliefDirection
from trainer.qde.brain.lesson_extractor import LessonExtractor
from trainer.qde.brain.reflection_engine import TradeReflectionEngine
from trainer.qde.brain.surprise_engine import SurpriseLevel, compute_surprise

SYM = "EURUSD"


def _record(
    *,
    direction_correct: bool | None,
    expected_move: float = 0.0050,
    actual_pnl: float = 0.0050,
    expected_holding_h: float = 24.0,
    holding_time_h: float = 24.0,
    outcome_type: OutcomeType = OutcomeType.CORRECT_DIRECTION,
    primary_reasons: list[str] | None = None,
    contradicting_factors: list[str] | None = None,
    supporting_factors: list[str] | None = None,
    score_overall: float | None = 0.7,
) -> DecisionRecord:
    snapshot = DecisionSnapshot(
        forecast_confidence=0.7,
        expected_move=expected_move,
        expected_holding_h=expected_holding_h,
    )
    reasoning = DecisionReasoning(
        primary_reasons=primary_reasons or ["trend continuation"],
        supporting_factors=supporting_factors or ["momentum aligned"],
        contradicting_factors=contradicting_factors or [],
    )
    outcome = DecisionOutcome(
        outcome_type=outcome_type,
        actual_direction_correct=direction_correct,
        actual_pnl=actual_pnl,
        holding_time_h=holding_time_h,
        hit_tp=direction_correct is True,
    )
    score = (
        DecisionScore(overall=score_overall, grade=DecisionGrade.GOOD)
        if score_overall is not None
        else None
    )
    return DecisionRecord(
        decision_id="d1",
        decision_type=DecisionType.BUY,
        instrument=SYM,
        timestamp=0.0,
        snapshot=snapshot,
        reasoning=reasoning,
        outcome=outcome,
        score=score,
    )


# ── surprise_engine ──────────────────────────────────────────────────────────


def test_no_outcome_yields_no_surprise():
    record = _record(direction_correct=True)
    record.outcome = None
    assert compute_surprise(record) is None


def test_expected_outcome_has_low_surprise():
    assessment = compute_surprise(_record(direction_correct=True))
    assert assessment is not None
    assert assessment.level == SurpriseLevel.EXPECTED
    assert assessment.magnitude < 0.3


def test_wrong_confident_direction_is_highly_surprising():
    assessment = compute_surprise(
        _record(
            direction_correct=False,
            actual_pnl=-0.0050,
            outcome_type=OutcomeType.INCORRECT_DIRECTION,
        )
    )
    assert assessment is not None
    assert assessment.signals.direction_surprise > 0
    assert assessment.magnitude > 0.3
    assert assessment.level in (SurpriseLevel.MODERATE, SurpriseLevel.HIGH, SurpriseLevel.SHOCKING)


def test_unexpected_market_event_raises_surprise():
    baseline = compute_surprise(_record(direction_correct=True))
    shocked = compute_surprise(
        _record(direction_correct=True, outcome_type=OutcomeType.UNEXPECTED_MARKET_EVENT)
    )
    assert baseline is not None and shocked is not None
    assert shocked.magnitude > baseline.magnitude


# ── reflection_engine ────────────────────────────────────────────────────────


def test_reflect_on_trade_returns_none_without_outcome():
    record = _record(direction_correct=True)
    record.outcome = None
    assert TradeReflectionEngine().reflect_on_trade(record) is None


def test_reflect_on_trade_flags_ignored_evidence_on_loss():
    record = _record(
        direction_correct=False,
        actual_pnl=-0.0060,
        outcome_type=OutcomeType.INCORRECT_DIRECTION,
        contradicting_factors=["momentum divergence on H4"],
    )
    reflection = TradeReflectionEngine().reflect_on_trade(record)
    assert reflection is not None
    assert reflection.was_win is False
    assert "momentum divergence on H4" in reflection.ignored_evidence
    assert reflection.false_evidence  # primary reasons didn't hold up
    assert any("contradicting evidence" in lesson for lesson in reflection.lessons)


def test_reflect_on_trade_clean_win_has_no_false_or_ignored_evidence():
    record = _record(direction_correct=True)
    reflection = TradeReflectionEngine().reflect_on_trade(record)
    assert reflection is not None
    assert reflection.was_win is True
    assert reflection.ignored_evidence == []
    assert reflection.false_evidence == []
    assert reflection.reasoning_quality == 0.7


def test_reflect_on_trade_flags_lucky_win_with_low_quality_reasoning():
    record = _record(direction_correct=True, score_overall=0.2)
    reflection = TradeReflectionEngine().reflect_on_trade(record)
    assert reflection is not None
    assert any("luck" in lesson.lower() for lesson in reflection.lessons)


# ── lesson_extractor ─────────────────────────────────────────────────────────


def test_lesson_extractor_updates_personality_and_calibration_on_loss():
    state = BrainState.new(SYM)
    record = _record(
        direction_correct=False,
        actual_pnl=-0.0080,
        outcome_type=OutcomeType.INCORRECT_DIRECTION,
        contradicting_factors=["momentum divergence"],
    )
    reflection = TradeReflectionEngine().reflect_on_trade(record)
    assert reflection is not None

    extractor = LessonExtractor()
    application = extractor.apply(state, reflection, regime="TREND_UP")

    assert state.personality.sample_count == 1
    assert state.personality.regime_outcome_quality.get("TREND_UP", 0.0) < 0.5
    # Losing surprisingly should push conviction bias down and uncertainty up.
    assert state.calibration.conviction_bias <= 0.0
    assert state.calibration.uncertainty_bias >= 0.0
    assert application.experience.instrument == SYM
    assert any(item.kind == "LESSON" for item in state.working_memory)


def test_lesson_extractor_updates_active_hypothesis_contradicting_evidence():
    state = BrainState.new(SYM)
    hyp = Hypothesis(
        instrument=SYM,
        thesis="uptrend continuation",
        direction=BeliefDirection.BULLISH,
        confidence=0.6,
    )
    state.hypotheses[hyp.hypothesis_id] = hyp

    record = _record(
        direction_correct=False,
        actual_pnl=-0.0080,
        outcome_type=OutcomeType.INCORRECT_DIRECTION,
        contradicting_factors=["momentum divergence"],
    )
    reflection = TradeReflectionEngine().reflect_on_trade(record)
    assert reflection is not None

    application = LessonExtractor().apply(state, reflection)

    assert "momentum divergence" in hyp.contradicting_evidence
    assert hyp.hypothesis_id in application.hypotheses_updated


def test_lesson_extractor_does_not_touch_hypotheses_on_win():
    state = BrainState.new(SYM)
    hyp = Hypothesis(
        instrument=SYM,
        thesis="uptrend continuation",
        direction=BeliefDirection.BULLISH,
        confidence=0.6,
    )
    state.hypotheses[hyp.hypothesis_id] = hyp

    record = _record(direction_correct=True)
    reflection = TradeReflectionEngine().reflect_on_trade(record)
    assert reflection is not None

    application = LessonExtractor().apply(state, reflection)

    assert hyp.contradicting_evidence == []
    assert application.hypotheses_updated == []


# ── QDEBrain integration ─────────────────────────────────────────────────────


def test_brain_reflect_on_trade_returns_none_without_outcome():
    brain = QDEBrain()
    record = _record(direction_correct=True)
    record.outcome = None
    assert brain.reflect_on_trade(record) is None


def test_brain_reflect_on_trade_applies_lessons_to_state():
    brain = QDEBrain()
    record = _record(
        direction_correct=False,
        actual_pnl=-0.0080,
        outcome_type=OutcomeType.INCORRECT_DIRECTION,
        contradicting_factors=["momentum divergence"],
    )

    result = brain.reflect_on_trade(record, regime="TREND_UP")
    assert result is not None
    reflection, application = result

    state = brain.get_state(SYM)
    assert state.personality.sample_count == 1
    assert application.instrument == SYM
    assert reflection.instrument == SYM


def test_brain_state_serializes_calibration_profile():
    brain = QDEBrain()
    record = _record(
        direction_correct=False,
        actual_pnl=-0.0080,
        outcome_type=OutcomeType.INCORRECT_DIRECTION,
    )
    brain.reflect_on_trade(record)

    state = brain.get_state(SYM)
    as_dict = state.to_dict()
    assert "calibration" in as_dict
    assert as_dict["calibration"]["instrument"] == SYM
