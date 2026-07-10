"""QDE Brain — metacognition/self-awareness tests.

Covers `metacognition_engine.MetacognitionEngine` (continuous numeric
self-evaluation: performance, calibration, overconfidence, regime/symbol
weakness) and `self_awareness_engine.SelfAwarenessEngine` (bounded dynamic
adjustment of aggression/risk appetite/trade frequency/evidence
requirements) — plus `QDEBrain.evaluate_self` wiring both together.
"""

from __future__ import annotations

from trainer.memory.decision_memory import DecisionMemory
from trainer.memory.decision_models import (
    DecisionOutcome,
    DecisionReasoning,
    DecisionRecord,
    DecisionSnapshot,
)
from trainer.memory.decision_types import DecisionType, OutcomeType
from trainer.qde.brain.brain import QDEBrain
from trainer.qde.brain.brain_models import CalibrationProfile, SymbolPersonality
from trainer.qde.brain.metacognition_engine import MetacognitionEngine
from trainer.qde.brain.self_awareness_engine import (
    MAX_AGGRESSION,
    MAX_RISK_APPETITE,
    MIN_AGGRESSION,
    MIN_EVIDENCE_REQUIREMENT,
    MIN_RISK_APPETITE,
    SelfAwarenessEngine,
    SelfAwarenessState,
)

SYM = "EURUSD"


def _record(
    idx: int, *, win: bool, confidence: float = 0.6, regime: str = "TREND"
) -> DecisionRecord:
    snapshot = DecisionSnapshot(
        forecast_confidence=confidence,
        market_regime=regime,
    )
    outcome = DecisionOutcome(
        outcome_type=OutcomeType.CORRECT_DIRECTION if win else OutcomeType.INCORRECT_DIRECTION,
        actual_direction_correct=win,
        actual_pnl=0.005 if win else -0.005,
    )
    return DecisionRecord(
        decision_id=f"d{idx}",
        decision_type=DecisionType.BUY,
        instrument=SYM,
        timestamp=float(idx),
        snapshot=snapshot,
        reasoning=DecisionReasoning(),
        outcome=outcome,
    )


def _fill_memory(mem: DecisionMemory, results: list[bool], **kwargs) -> None:
    for i, win in enumerate(results):
        mem.store(_record(i, win=win, **kwargs))


# ── metacognition_engine ─────────────────────────────────────────────────────


class TestMetacognitionEngine:
    def test_insufficient_samples_stays_neutral(self):
        mem = DecisionMemory()
        _fill_memory(mem, [True, True])  # below MIN_SAMPLES_FOR_SIGNAL
        assessment = MetacognitionEngine().assess(SYM, mem)
        assert assessment.sample_count == 2
        assert assessment.recent_win_rate is None
        assert assessment.caution_level == 0.0

    def test_no_decision_memory_stays_neutral(self):
        assessment = MetacognitionEngine().assess(SYM, decision_memory=None)
        assert assessment.sample_count == 0
        assert assessment.caution_level == 0.0

    def test_poor_recent_performance_raises_caution_and_symbol_weakness(self):
        mem = DecisionMemory()
        # Strong baseline, then a recent losing streak.
        _fill_memory(mem, [True] * 10 + [False] * 10)
        assessment = MetacognitionEngine().assess(SYM, mem, recent_window=10)
        assert assessment.recent_win_rate == 0.0
        assert assessment.baseline_win_rate == 1.0
        assert assessment.performance_trend < 0
        assert assessment.symbol_weakness > 0.5
        assert assessment.caution_level > 0.3

    def test_strong_performance_yields_low_caution(self):
        mem = DecisionMemory()
        _fill_memory(mem, [True] * 20)
        assessment = MetacognitionEngine().assess(SYM, mem, recent_window=10)
        assert assessment.recent_win_rate == 1.0
        assert assessment.symbol_weakness == 0.0
        assert assessment.caution_level < 0.3

    def test_overconfidence_detected_on_high_confidence_losses(self):
        mem = DecisionMemory()
        _fill_memory(mem, [False] * 10, confidence=0.9)
        assessment = MetacognitionEngine().assess(SYM, mem, recent_window=10)
        assert assessment.overconfidence_score > 0.0

    def test_weak_regimes_from_symbol_personality_not_recomputed(self):
        personality = SymbolPersonality(instrument=SYM)
        personality.regime_outcome_quality = {"CHOP": 0.2, "TREND": 0.8}
        mem = DecisionMemory()
        _fill_memory(mem, [True] * 6)
        assessment = MetacognitionEngine().assess(SYM, mem, personality=personality)
        assert assessment.weak_regimes == ["CHOP"]

    def test_calibration_bias_compounds_calibration_error(self):
        mem = DecisionMemory()
        _fill_memory(mem, [True] * 10, confidence=0.6)
        baseline = MetacognitionEngine().assess(SYM, mem, recent_window=10)

        overstated = CalibrationProfile(instrument=SYM, confidence_bias=-0.4)
        with_bias = MetacognitionEngine().assess(SYM, mem, calibration=overstated, recent_window=10)
        assert with_bias.calibration_error >= baseline.calibration_error


# ── self_awareness_engine ────────────────────────────────────────────────────


class TestSelfAwarenessEngine:
    def _assess(
        self,
        caution: float,
        weak_regimes=None,
        symbol_weakness: float = 0.0,
        recent_win_rate: float | None = None,
        performance_trend: float = 0.0,
    ):
        from trainer.qde.brain.metacognition_engine import MetacognitiveAssessment

        return MetacognitiveAssessment(
            instrument=SYM,
            sample_count=20,
            caution_level=caution,
            weak_regimes=weak_regimes or [],
            symbol_weakness=symbol_weakness,
            recent_win_rate=recent_win_rate,
            performance_trend=performance_trend,
        )

    def test_high_caution_reduces_all_multipliers_and_raises_evidence_bar(self):
        engine = SelfAwarenessEngine()
        state = engine.adjust(None, self._assess(caution=0.95))
        assert state.aggression < 1.0
        assert state.risk_appetite < 1.0
        assert state.trade_frequency < 1.0
        assert state.evidence_requirement > 1.0

    def test_low_caution_increases_multipliers_but_stays_bounded(self):
        engine = SelfAwarenessEngine()
        state = engine.adjust(
            None,
            self._assess(caution=0.0, recent_win_rate=0.8, performance_trend=0.2),
        )
        assert state.aggression > 1.0
        assert state.aggression <= MAX_AGGRESSION
        assert state.risk_appetite <= MAX_RISK_APPETITE
        assert state.evidence_requirement >= MIN_EVIDENCE_REQUIREMENT

    def test_hot_streak_cannot_compound_past_ceiling(self):
        """Repeated ticks of maximal confidence must never push aggression or
        risk appetite past their hard ceilings — this is the literal
        'strong performance must never cause uncontrolled risk expansion'
        requirement.
        """
        engine = SelfAwarenessEngine()
        state: SelfAwarenessState | None = None
        for _ in range(200):
            state = engine.adjust(
                state,
                self._assess(caution=0.0, recent_win_rate=0.9, performance_trend=0.3),
            )
        assert state.aggression <= MAX_AGGRESSION
        assert state.risk_appetite <= MAX_RISK_APPETITE
        assert state.aggression > 1.0  # it did expand, just not unboundedly

    def test_losing_streak_saturates_at_floor_not_below(self):
        engine = SelfAwarenessEngine()
        state: SelfAwarenessState | None = None
        for _ in range(200):
            state = engine.adjust(state, self._assess(caution=1.0))
        assert state.aggression >= MIN_AGGRESSION
        assert state.risk_appetite >= MIN_RISK_APPETITE

    def test_weak_regime_bumps_caution_further(self):
        engine = SelfAwarenessEngine()
        neutral = engine.adjust(None, self._assess(caution=0.5))
        in_weak_regime = engine.adjust(
            None, self._assess(caution=0.5, weak_regimes=["CHOP"]), current_regime="CHOP"
        )
        assert in_weak_regime.caution_level > neutral.caution_level
        assert in_weak_regime.aggression <= neutral.aggression

    def test_symbol_weakness_bumps_caution_further(self):
        engine = SelfAwarenessEngine()
        neutral = engine.adjust(None, self._assess(caution=0.5))
        weak_symbol = engine.adjust(None, self._assess(caution=0.5, symbol_weakness=1.0))
        assert weak_symbol.caution_level > neutral.caution_level

    def test_serialization_round_trip(self):
        engine = SelfAwarenessEngine()
        state = engine.adjust(None, self._assess(caution=0.7))
        restored = SelfAwarenessState.from_dict(state.to_dict())
        assert restored.aggression == state.aggression
        assert restored.risk_appetite == state.risk_appetite
        assert restored.trade_frequency == state.trade_frequency
        assert restored.evidence_requirement == state.evidence_requirement


# ── QDEBrain.evaluate_self integration ──────────────────────────────────────


class TestQDEBrainEvaluateSelf:
    def test_evaluate_self_updates_and_persists_state(self):
        brain = QDEBrain()
        mem = DecisionMemory()
        _fill_memory(mem, [True] * 10 + [False] * 10)

        assessment, self_awareness = brain.evaluate_self(SYM, decision_memory=mem)

        assert assessment.instrument == SYM
        assert self_awareness.instrument == SYM
        state = brain.get_state(SYM)
        assert state.self_awareness is self_awareness
        assert state.self_awareness.aggression < 1.0  # losing streak -> more caution

    def test_evaluate_self_defaults_neutral_with_no_history(self):
        brain = QDEBrain()
        assessment, self_awareness = brain.evaluate_self(SYM, decision_memory=None)
        assert assessment.sample_count == 0
        assert self_awareness.aggression == 1.0

    def test_state_serialization_round_trips_self_awareness(self):
        brain = QDEBrain()
        mem = DecisionMemory()
        _fill_memory(mem, [True] * 15)
        brain.evaluate_self(SYM, decision_memory=mem)

        state = brain.get_state(SYM)
        raw = state.to_dict()
        assert "self_awareness" in raw

        from trainer.qde.brain.brain import _state_from_dict

        restored = _state_from_dict(raw)
        assert restored.self_awareness.aggression == state.self_awareness.aggression
        assert restored.self_awareness.instrument == SYM
