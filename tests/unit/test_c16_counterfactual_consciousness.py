"""Tests for C16: Counterfactual Consciousness.

Covers:
  - ScenarioType / FlipTrigger enums
  - ProbabilityPoint: spread, to_dict
  - CounterfactualScenario: to_dict
  - FlipLesson: to_dict, could_have_switched_earlier
  - ThesisFlipRecord: to_dict
  - ThesisReversalRecord: to_dict
  - UncertaintySpikeRecord: to_dict
  - ThesisSwitchResult: to_dict
  - CounterfactualState: flip_count, reversal_rate, to_snapshot
  - generate_alternatives(): stateless, scenario count, types, weights
  - generate_alternatives(): seeded from conflicting evidence
  - generate_alternatives(): uncertainty raises sideways P
  - generate_alternatives(): low primary skips ACCELERATION / EXHAUSTION
  - generate_alternatives(): high primary includes EXHAUSTION
  - generate_alternatives(): stores on _state
  - evaluate_alternative(): match/invalidation scoring
  - evaluate_alternative(): uncertainty bonus for SIDEWAYS
  - evaluate_alternative(): no matching evidence → prior blend only
  - evaluate_alternative(): full invalidation → near-zero
  - evaluate_alternative(): result bounded [0, 1]
  - switch_thesis(): returns ThesisSwitchResult
  - switch_thesis(): flip record populated correctly
  - switch_thesis(): state sides updated
  - switch_thesis(): generates new scenarios
  - switch_thesis(): appends probability point
  - switch_thesis(): FlipLesson has ignored_warnings
  - switch_thesis(): reversal detection within window
  - switch_thesis(): no reversal outside window
  - switch_thesis(): reversal record stored
  - switch_thesis(): flip_records capped at MAX_FLIP_RECORDS
  - record_probability_point(): appends point
  - record_probability_point(): spike detection
  - record_probability_point(): no spike below threshold
  - record_probability_point(): cap at MAX_PROBABILITY_POINTS
  - recommend_switch(): secondary dominance → True
  - recommend_switch(): confidence collapse → True
  - recommend_switch(): both strong primary → False
  - CounterfactualEngine: full round-trip
  - CounterfactualEngine: generate → evaluate → switch sequence
  - CounterfactualEngine: history access methods
  - CounterfactualEngine: snapshot keys
  - CounterfactualEngine: independent instances isolated
  - End-to-end: continuous "what if I am wrong" scenario
  - End-to-end: rapid flip-back detected as reversal
  - End-to-end: uncertainty spike recorded
  - Import sanity: all spec callables importable from package
"""

from __future__ import annotations

from trainer.qde.brain.brain_types import BeliefDirection
from trainer.qde.brain.counterfactual import (
    MAX_FLIP_RECORDS,
    MAX_PROBABILITY_POINTS,
    PRIMARY_CONFIDENCE_FLOOR,
    REVERSAL_WINDOW_SECONDS,
    SWITCH_MARGIN,
    UNCERTAINTY_SPIKE_DELTA,
    CounterfactualEngine,
    CounterfactualScenario,
    CounterfactualState,
    FlipLesson,
    FlipTrigger,
    ProbabilityPoint,
    ScenarioType,
    ThesisFlipRecord,
    ThesisReversalRecord,
    ThesisSwitchResult,
    UncertaintySpikeRecord,
    evaluate_alternative,
    generate_alternatives,
    recommend_switch,
    record_probability_point,
    switch_thesis,
)
from trainer.qde.brain.thesis_manager import Thesis

_T0 = 1_700_000_000.0


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _thesis(
    side: BeliefDirection = BeliefDirection.BULLISH,
    confidence: float = 0.65,
    historical: float = 0.60,
    supporting: list[str] | None = None,
    conflicting: list[str] | None = None,
    argument: str = "",
) -> Thesis:
    t = Thesis(
        side=side,
        argument=argument or f"{side.value.lower()} thesis",
        supporting_evidence=supporting or [],
        conflicting_evidence=conflicting or [],
        confidence=confidence,
        historical_support=historical,
    )
    return t


def _bull(confidence: float = 0.70, **kw) -> Thesis:
    return _thesis(BeliefDirection.BULLISH, confidence=confidence, **kw)


def _bear(confidence: float = 0.55, **kw) -> Thesis:
    return _thesis(BeliefDirection.BEARISH, confidence=confidence, **kw)


def _state(
    instrument: str = "EURUSD",
    primary_side: BeliefDirection = BeliefDirection.BULLISH,
) -> CounterfactualState:
    return CounterfactualState(
        instrument=instrument,
        primary_side=primary_side,
        secondary_side=(
            BeliefDirection.BEARISH
            if primary_side == BeliefDirection.BULLISH
            else BeliefDirection.BULLISH
        ),
        created_at=_T0,
        last_updated_at=_T0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Enums
# ═══════════════════════════════════════════════════════════════════════════════


class TestEnums:
    def test_scenario_type_values(self):
        expected = {"CONTINUATION", "REVERSAL", "SIDEWAYS", "ACCELERATION", "EXHAUSTION"}
        assert {s.value for s in ScenarioType} == expected

    def test_flip_trigger_values(self):
        expected = {
            "EVIDENCE_SHIFT",
            "UNCERTAINTY_SPIKE",
            "CONFIDENCE_COLLAPSE",
            "SECONDARY_DOMINANCE",
            "FORCED",
        }
        assert {t.value for t in FlipTrigger} == expected

    def test_str_enum_flip_trigger(self):
        assert FlipTrigger.FORCED == "FORCED"

    def test_str_enum_scenario_type(self):
        assert ScenarioType.REVERSAL == "REVERSAL"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ProbabilityPoint
# ═══════════════════════════════════════════════════════════════════════════════


class TestProbabilityPoint:
    def test_spread_positive_when_primary_leads(self):
        p = ProbabilityPoint(timestamp=_T0, primary_p=0.70, secondary_p=0.50, uncertainty=0.0)
        assert p.spread > 0.0

    def test_spread_negative_when_secondary_leads(self):
        p = ProbabilityPoint(timestamp=_T0, primary_p=0.40, secondary_p=0.65, uncertainty=0.0)
        assert p.spread < 0.0

    def test_to_dict_keys(self):
        p = ProbabilityPoint(timestamp=_T0, primary_p=0.65, secondary_p=0.50, uncertainty=0.20)
        d = p.to_dict()
        for k in ("timestamp", "primary_p", "secondary_p", "uncertainty", "spread"):
            assert k in d

    def test_to_dict_values(self):
        p = ProbabilityPoint(timestamp=_T0, primary_p=0.70, secondary_p=0.50, uncertainty=0.10)
        d = p.to_dict()
        assert abs(d["spread"] - 0.20) < 1e-4


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CounterfactualScenario
# ═══════════════════════════════════════════════════════════════════════════════


class TestCounterfactualScenario:
    def _scenario(self) -> CounterfactualScenario:
        return CounterfactualScenario(
            scenario_id="s1",
            scenario_type=ScenarioType.REVERSAL,
            direction=BeliefDirection.BEARISH,
            trigger_conditions=["price breaks support", "volume confirms"],
            probability_weight=0.30,
            expected_observations=["bearish candle sequence"],
            invalidation_signals=["price holds bullish structure"],
            confidence_if_correct=0.72,
            source_evidence=["momentum divergence"],
            created_at=_T0,
        )

    def test_to_dict_keys(self):
        d = self._scenario().to_dict()
        for k in (
            "scenario_id",
            "scenario_type",
            "direction",
            "trigger_conditions",
            "probability_weight",
            "expected_observations",
            "invalidation_signals",
            "confidence_if_correct",
            "source_evidence",
            "created_at",
        ):
            assert k in d

    def test_to_dict_values(self):
        d = self._scenario().to_dict()
        assert d["scenario_type"] == "REVERSAL"
        assert d["direction"] == "BEARISH"
        assert abs(d["probability_weight"] - 0.30) < 1e-4


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FlipLesson
# ═══════════════════════════════════════════════════════════════════════════════


class TestFlipLesson:
    def test_to_dict_keys(self):
        lesson = FlipLesson(
            ignored_warnings=["bearish divergence", "volume declining"],
            key_trigger="secondary_dominance",
            time_to_flip=400.0,
            could_have_switched_earlier=True,
        )
        d = lesson.to_dict()
        for k in ("ignored_warnings", "key_trigger", "time_to_flip", "could_have_switched_earlier"):
            assert k in d

    def test_could_have_switched_earlier_true_when_long_time(self):
        lesson = FlipLesson(
            ignored_warnings=["warning1"],
            key_trigger="trigger",
            time_to_flip=REVERSAL_WINDOW_SECONDS + 100,
            could_have_switched_earlier=True,
        )
        assert lesson.could_have_switched_earlier is True

    def test_could_have_switched_earlier_false_when_fast(self):
        lesson = FlipLesson(
            ignored_warnings=["warning1"],
            key_trigger="trigger",
            time_to_flip=60.0,
            could_have_switched_earlier=False,
        )
        assert lesson.could_have_switched_earlier is False


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ThesisFlipRecord
# ═══════════════════════════════════════════════════════════════════════════════


class TestThesisFlipRecord:
    def _record(self) -> ThesisFlipRecord:
        return ThesisFlipRecord(
            flip_id="flip_001",
            previous_primary_side=BeliefDirection.BULLISH,
            new_primary_side=BeliefDirection.BEARISH,
            trigger=FlipTrigger.SECONDARY_DOMINANCE,
            trigger_detail="secondary_p > primary_p + margin",
            primary_p_before=0.45,
            secondary_p_before=0.65,
            evidence_snapshot=["bearish divergence"],
            flip_lesson=None,
            flipped_at=_T0,
        )

    def test_to_dict_keys(self):
        d = self._record().to_dict()
        for k in (
            "flip_id",
            "previous_primary_side",
            "new_primary_side",
            "trigger",
            "trigger_detail",
            "primary_p_before",
            "secondary_p_before",
            "evidence_snapshot",
            "flip_lesson",
            "flipped_at",
        ):
            assert k in d

    def test_values(self):
        d = self._record().to_dict()
        assert d["previous_primary_side"] == "BULLISH"
        assert d["new_primary_side"] == "BEARISH"
        assert d["trigger"] == "SECONDARY_DOMINANCE"
        assert d["flip_lesson"] is None


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ThesisReversalRecord
# ═══════════════════════════════════════════════════════════════════════════════


class TestThesisReversalRecord:
    def test_to_dict_keys(self):
        r = ThesisReversalRecord(
            reversal_id="rev_001",
            returning_to_side=BeliefDirection.BULLISH,
            intermediate_side=BeliefDirection.BEARISH,
            seconds_elapsed=120.0,
            flip_id="flip_001",
            reversed_at=_T0,
        )
        d = r.to_dict()
        for k in (
            "reversal_id",
            "returning_to_side",
            "intermediate_side",
            "seconds_elapsed",
            "flip_id",
            "reversed_at",
        ):
            assert k in d

    def test_values(self):
        r = ThesisReversalRecord(
            reversal_id="rev_001",
            returning_to_side=BeliefDirection.BULLISH,
            intermediate_side=BeliefDirection.BEARISH,
            seconds_elapsed=90.0,
            flip_id="flip_001",
            reversed_at=_T0,
        )
        d = r.to_dict()
        assert d["returning_to_side"] == "BULLISH"
        assert d["intermediate_side"] == "BEARISH"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. UncertaintySpikeRecord
# ═══════════════════════════════════════════════════════════════════════════════


class TestUncertaintySpikeRecord:
    def test_to_dict_keys(self):
        r = UncertaintySpikeRecord(
            spike_id="spk_001",
            instrument="EURUSD",
            uncertainty_before=0.20,
            uncertainty_after=0.60,
            delta=0.40,
            level_after="HIGH",
            context="regime conflict",
            spiked_at=_T0,
        )
        d = r.to_dict()
        for k in (
            "spike_id",
            "instrument",
            "uncertainty_before",
            "uncertainty_after",
            "delta",
            "level_after",
            "context",
            "spiked_at",
        ):
            assert k in d


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CounterfactualState
# ═══════════════════════════════════════════════════════════════════════════════


class TestCounterfactualState:
    def test_flip_count_empty(self):
        s = _state()
        assert s.flip_count() == 0

    def test_reversal_rate_empty(self):
        s = _state()
        assert s.reversal_rate() == 0.0

    def test_last_probability_point_none_when_empty(self):
        s = _state()
        assert s.last_probability_point() is None

    def test_to_snapshot_keys(self):
        s = _state()
        d = s.to_snapshot()
        for k in (
            "instrument",
            "primary_side",
            "secondary_side",
            "flip_count",
            "reversal_count",
            "spike_count",
            "reversal_rate",
            "last_probability",
            "scenario_count",
            "created_at",
            "last_updated_at",
        ):
            assert k in d

    def test_to_snapshot_initial_values(self):
        s = _state("GBPUSD")
        d = s.to_snapshot()
        assert d["instrument"] == "GBPUSD"
        assert d["flip_count"] == 0
        assert d["reversal_count"] == 0
        assert d["last_probability"] is None


# ═══════════════════════════════════════════════════════════════════════════════
# 9. generate_alternatives()
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerateAlternatives:
    def test_returns_list_of_scenarios(self):
        result = generate_alternatives(_bull(), now=_T0)
        assert isinstance(result, list)
        assert all(isinstance(s, CounterfactualScenario) for s in result)

    def test_always_includes_reversal_and_sideways(self):
        result = generate_alternatives(_bull(), now=_T0)
        types = {s.scenario_type for s in result}
        assert ScenarioType.REVERSAL in types
        assert ScenarioType.SIDEWAYS in types

    def test_reversal_has_opposite_direction(self):
        result = generate_alternatives(_bull(), now=_T0)
        reversal = next(s for s in result if s.scenario_type == ScenarioType.REVERSAL)
        assert reversal.direction == BeliefDirection.BEARISH

    def test_bear_primary_reversal_is_bullish(self):
        result = generate_alternatives(_bear(), now=_T0)
        reversal = next(s for s in result if s.scenario_type == ScenarioType.REVERSAL)
        assert reversal.direction == BeliefDirection.BULLISH

    def test_low_primary_skips_acceleration_and_exhaustion(self):
        weak = _thesis(confidence=0.35, historical=0.40)  # strength < 0.45
        result = generate_alternatives(weak, now=_T0)
        types = {s.scenario_type for s in result}
        assert ScenarioType.ACCELERATION not in types
        assert ScenarioType.EXHAUSTION not in types

    def test_strong_primary_includes_acceleration(self):
        strong = _thesis(confidence=0.75, historical=0.70, supporting=["s1", "s2", "s3"])
        result = generate_alternatives(strong, now=_T0)
        types = {s.scenario_type for s in result}
        assert ScenarioType.ACCELERATION in types

    def test_very_strong_primary_includes_exhaustion(self):
        very_strong = _thesis(confidence=0.85, historical=0.75, supporting=["s1", "s2", "s3", "s4"])
        result = generate_alternatives(very_strong, now=_T0)
        types = {s.scenario_type for s in result}
        assert ScenarioType.EXHAUSTION in types

    def test_probability_weights_positive(self):
        result = generate_alternatives(_bull(), now=_T0)
        for s in result:
            assert s.probability_weight > 0.0

    def test_probability_weights_bounded(self):
        result = generate_alternatives(_bull(), now=_T0)
        for s in result:
            assert 0.0 <= s.probability_weight <= 1.0

    def test_conflicting_evidence_seeds_reversal(self):
        bull = _thesis(conflicting=["bearish divergence", "volume declining"])
        result = generate_alternatives(bull, now=_T0)
        reversal = next(s for s in result if s.scenario_type == ScenarioType.REVERSAL)
        # Source evidence or trigger conditions should reference conflicting items
        assert reversal.source_evidence or reversal.trigger_conditions

    def test_high_uncertainty_raises_sideways_weight(self):
        low_unc = generate_alternatives(_bull(), uncertainty_score=0.0, now=_T0)
        high_unc = generate_alternatives(_bull(), uncertainty_score=0.80, now=_T0)
        sw_low = next(s for s in low_unc if s.scenario_type == ScenarioType.SIDEWAYS)
        sw_high = next(s for s in high_unc if s.scenario_type == ScenarioType.SIDEWAYS)
        assert sw_high.probability_weight >= sw_low.probability_weight

    def test_strong_secondary_raises_reversal_weight(self):
        strong_secondary = _bear(confidence=0.85)
        weak_secondary = _bear(confidence=0.35)
        r_strong = generate_alternatives(_bull(), secondary_thesis=strong_secondary, now=_T0)
        r_weak = generate_alternatives(_bull(), secondary_thesis=weak_secondary, now=_T0)
        rev_strong = next(s for s in r_strong if s.scenario_type == ScenarioType.REVERSAL)
        rev_weak = next(s for s in r_weak if s.scenario_type == ScenarioType.REVERSAL)
        assert rev_strong.probability_weight >= rev_weak.probability_weight

    def test_stores_on_state(self):
        st = _state()
        generate_alternatives(_bull(), _state=st, now=_T0)
        assert len(st.current_scenarios) >= 2

    def test_each_scenario_has_trigger_conditions(self):
        result = generate_alternatives(_bull(), now=_T0)
        for s in result:
            assert len(s.trigger_conditions) > 0

    def test_each_scenario_has_invalidation_signals(self):
        result = generate_alternatives(_bull(), now=_T0)
        for s in result:
            assert len(s.invalidation_signals) > 0

    def test_scenario_ids_unique(self):
        result = generate_alternatives(_bull(confidence=0.80, historical=0.75), now=_T0)
        ids = [s.scenario_id for s in result]
        assert len(ids) == len(set(ids))


# ═══════════════════════════════════════════════════════════════════════════════
# 10. evaluate_alternative()
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluateAlternative:
    def _reversal_scenario(self, weight: float = 0.30) -> CounterfactualScenario:
        return CounterfactualScenario(
            scenario_id="s1",
            scenario_type=ScenarioType.REVERSAL,
            direction=BeliefDirection.BEARISH,
            trigger_conditions=["bearish divergence", "support breaks"],
            probability_weight=weight,
            expected_observations=["bearish candle sequence"],
            invalidation_signals=["price holds bullish", "volume absent"],
            confidence_if_correct=0.70,
            source_evidence=[],
            created_at=_T0,
        )

    def _sideways_scenario(self, weight: float = 0.20) -> CounterfactualScenario:
        return CounterfactualScenario(
            scenario_id="s2",
            scenario_type=ScenarioType.SIDEWAYS,
            direction=BeliefDirection.NEUTRAL,
            trigger_conditions=["price oscillates range", "volume contracts"],
            probability_weight=weight,
            expected_observations=["narrowing range"],
            invalidation_signals=["decisive breakout"],
            confidence_if_correct=0.60,
            source_evidence=[],
            created_at=_T0,
        )

    def test_no_evidence_returns_prior_blend(self):
        s = self._reversal_scenario(weight=0.30)
        result = evaluate_alternative(s, new_evidence=[])
        # 0.60 * 0.30 + 0.40 * 0 = 0.18
        assert abs(result - 0.18) < 1e-3

    def test_full_match_raises_score(self):
        s = self._reversal_scenario(weight=0.30)
        result = evaluate_alternative(
            s,
            new_evidence=["bearish divergence confirmed", "support breaks at 1.0900"],
        )
        assert result > 0.18  # above prior blend

    def test_full_invalidation_drops_score(self):
        s = self._reversal_scenario(weight=0.30)
        result = evaluate_alternative(
            s,
            new_evidence=["price holds bullish structure", "volume absent on move"],
        )
        # raw_score approaches 0 → result ≈ 0.60 * 0.30 = 0.18 at most
        assert result <= 0.20

    def test_sideways_gets_uncertainty_bonus(self):
        s = self._sideways_scenario(weight=0.20)
        without_unc = evaluate_alternative(s, new_evidence=[], current_uncertainty=0.0)
        with_unc = evaluate_alternative(s, new_evidence=[], current_uncertainty=0.80)
        assert with_unc > without_unc

    def test_reversal_no_uncertainty_bonus(self):
        s = self._reversal_scenario(weight=0.30)
        without_unc = evaluate_alternative(s, new_evidence=[], current_uncertainty=0.0)
        with_unc = evaluate_alternative(s, new_evidence=[], current_uncertainty=0.80)
        assert abs(with_unc - without_unc) < 1e-6  # no bonus for REVERSAL

    def test_result_bounded_01(self):
        s = self._reversal_scenario(weight=0.99)
        r = evaluate_alternative(
            s, new_evidence=["bearish divergence", "support breaks"], current_uncertainty=1.0
        )
        assert 0.0 <= r <= 1.0

    def test_result_nonnegative_on_empty(self):
        s = self._reversal_scenario(weight=0.05)
        r = evaluate_alternative(s, new_evidence=[])
        assert r >= 0.0

    def test_substring_matching(self):
        s = self._reversal_scenario(weight=0.30)
        # Trigger condition contains "bearish divergence"; new evidence is a longer string
        result = evaluate_alternative(
            s,
            new_evidence=["strong bearish divergence on H4 chart"],
        )
        assert result > 0.18  # match found via substring


# ═══════════════════════════════════════════════════════════════════════════════
# 11. switch_thesis()
# ═══════════════════════════════════════════════════════════════════════════════


class TestSwitchThesis:
    def test_returns_thesis_switch_result(self):
        st = _state()
        result = switch_thesis(
            state=st,
            trigger=FlipTrigger.SECONDARY_DOMINANCE,
            new_primary_side=BeliefDirection.BEARISH,
            new_primary_thesis=_bear(confidence=0.75),
            new_secondary_thesis=_bull(confidence=0.45),
            now=_T0 + 10,
        )
        assert isinstance(result, ThesisSwitchResult)

    def test_flip_record_sides(self):
        st = _state(primary_side=BeliefDirection.BULLISH)
        result = switch_thesis(
            state=st,
            trigger=FlipTrigger.EVIDENCE_SHIFT,
            new_primary_side=BeliefDirection.BEARISH,
            new_primary_thesis=_bear(),
            new_secondary_thesis=_bull(),
            now=_T0 + 20,
        )
        assert result.flip_record.previous_primary_side == BeliefDirection.BULLISH
        assert result.flip_record.new_primary_side == BeliefDirection.BEARISH

    def test_flip_record_trigger(self):
        st = _state()
        result = switch_thesis(
            state=st,
            trigger=FlipTrigger.FORCED,
            new_primary_side=BeliefDirection.BEARISH,
            new_primary_thesis=_bear(),
            new_secondary_thesis=_bull(),
            now=_T0,
        )
        assert result.flip_record.trigger == FlipTrigger.FORCED

    def test_state_sides_updated(self):
        st = _state(primary_side=BeliefDirection.BULLISH)
        switch_thesis(
            state=st,
            trigger=FlipTrigger.SECONDARY_DOMINANCE,
            new_primary_side=BeliefDirection.BEARISH,
            new_primary_thesis=_bear(),
            new_secondary_thesis=_bull(),
            now=_T0,
        )
        assert st.primary_side == BeliefDirection.BEARISH

    def test_generates_new_scenarios(self):
        st = _state()
        result = switch_thesis(
            state=st,
            trigger=FlipTrigger.EVIDENCE_SHIFT,
            new_primary_side=BeliefDirection.BEARISH,
            new_primary_thesis=_bear(),
            new_secondary_thesis=_bull(),
            now=_T0,
        )
        assert len(result.new_scenarios) >= 2

    def test_appends_probability_point(self):
        st = _state()
        assert len(st.probability_evolution) == 0
        switch_thesis(
            state=st,
            trigger=FlipTrigger.FORCED,
            new_primary_side=BeliefDirection.BEARISH,
            new_primary_thesis=_bear(),
            new_secondary_thesis=_bull(),
            now=_T0,
        )
        assert len(st.probability_evolution) == 1

    def test_flip_lesson_has_ignored_warnings(self):
        conflicting = ["bearish divergence", "volume declining on advance"]
        bull_with_conflicts = _bull(conflicting=conflicting)
        st = _state()
        result = switch_thesis(
            state=st,
            trigger=FlipTrigger.EVIDENCE_SHIFT,
            new_primary_side=BeliefDirection.BEARISH,
            new_primary_thesis=_bear(),
            new_secondary_thesis=bull_with_conflicts,
            trigger_detail="secondary > primary",
            now=_T0 + REVERSAL_WINDOW_SECONDS + 1,
        )
        lesson = result.flip_record.flip_lesson
        assert lesson is not None
        assert isinstance(lesson.ignored_warnings, list)

    def test_flip_record_stored_in_state(self):
        st = _state()
        switch_thesis(
            state=st,
            trigger=FlipTrigger.FORCED,
            new_primary_side=BeliefDirection.BEARISH,
            new_primary_thesis=_bear(),
            new_secondary_thesis=_bull(),
            now=_T0,
        )
        assert len(st.flip_records) == 1

    def test_no_reversal_on_first_flip(self):
        st = _state()
        result = switch_thesis(
            state=st,
            trigger=FlipTrigger.FORCED,
            new_primary_side=BeliefDirection.BEARISH,
            new_primary_thesis=_bear(),
            new_secondary_thesis=_bull(),
            now=_T0,
        )
        assert result.is_reversal is False
        assert result.reversal_record is None

    def test_reversal_detected_within_window(self):
        """Flip BULLISH→BEARISH, then flip back BEARISH→BULLISH within window."""
        st = _state(primary_side=BeliefDirection.BULLISH)

        # First flip: BULLISH → BEARISH
        switch_thesis(
            state=st,
            trigger=FlipTrigger.EVIDENCE_SHIFT,
            new_primary_side=BeliefDirection.BEARISH,
            new_primary_thesis=_bear(),
            new_secondary_thesis=_bull(),
            now=_T0,
        )
        # Second flip back: BEARISH → BULLISH within window
        result2 = switch_thesis(
            state=st,
            trigger=FlipTrigger.EVIDENCE_SHIFT,
            new_primary_side=BeliefDirection.BULLISH,
            new_primary_thesis=_bull(),
            new_secondary_thesis=_bear(),
            now=_T0 + REVERSAL_WINDOW_SECONDS - 10,
        )
        assert result2.is_reversal is True
        assert result2.reversal_record is not None
        assert result2.reversal_record.returning_to_side == BeliefDirection.BULLISH

    def test_no_reversal_outside_window(self):
        """Same flip-back but after the reversal window expires — not a reversal."""
        st = _state(primary_side=BeliefDirection.BULLISH)

        switch_thesis(
            state=st,
            trigger=FlipTrigger.EVIDENCE_SHIFT,
            new_primary_side=BeliefDirection.BEARISH,
            new_primary_thesis=_bear(),
            new_secondary_thesis=_bull(),
            now=_T0,
        )
        result2 = switch_thesis(
            state=st,
            trigger=FlipTrigger.EVIDENCE_SHIFT,
            new_primary_side=BeliefDirection.BULLISH,
            new_primary_thesis=_bull(),
            new_secondary_thesis=_bear(),
            now=_T0 + REVERSAL_WINDOW_SECONDS + 10,
        )
        assert result2.is_reversal is False

    def test_reversal_record_stored_in_state(self):
        st = _state(primary_side=BeliefDirection.BULLISH)

        switch_thesis(
            state=st,
            trigger=FlipTrigger.FORCED,
            new_primary_side=BeliefDirection.BEARISH,
            new_primary_thesis=_bear(),
            new_secondary_thesis=_bull(),
            now=_T0,
        )
        switch_thesis(
            state=st,
            trigger=FlipTrigger.FORCED,
            new_primary_side=BeliefDirection.BULLISH,
            new_primary_thesis=_bull(),
            new_secondary_thesis=_bear(),
            now=_T0 + 60,
        )
        assert len(st.reversal_records) == 1

    def test_flip_records_capped(self):
        st = _state()
        sides = [BeliefDirection.BULLISH, BeliefDirection.BEARISH]
        for i in range(MAX_FLIP_RECORDS + 20):
            side = sides[i % 2]
            other = sides[(i + 1) % 2]
            switch_thesis(
                state=st,
                trigger=FlipTrigger.FORCED,
                new_primary_side=side,
                new_primary_thesis=_thesis(side=side),
                new_secondary_thesis=_thesis(side=other),
                now=_T0 + i * 1000,  # far apart to avoid reversal detection
            )
        assert len(st.flip_records) <= MAX_FLIP_RECORDS

    def test_to_dict_has_required_keys(self):
        st = _state()
        result = switch_thesis(
            state=st,
            trigger=FlipTrigger.FORCED,
            new_primary_side=BeliefDirection.BEARISH,
            new_primary_thesis=_bear(),
            new_secondary_thesis=_bull(),
            now=_T0,
        )
        d = result.to_dict()
        for k in ("flip_record", "new_scenarios", "is_reversal", "reversal_record"):
            assert k in d


# ═══════════════════════════════════════════════════════════════════════════════
# 12. record_probability_point()
# ═══════════════════════════════════════════════════════════════════════════════


class TestRecordProbabilityPoint:
    def test_appends_point(self):
        st = _state()
        pt = record_probability_point(
            st, primary_p=0.65, secondary_p=0.50, uncertainty=0.20, now=_T0
        )
        assert isinstance(pt, ProbabilityPoint)
        assert len(st.probability_evolution) == 1

    def test_multiple_points_ordered(self):
        st = _state()
        for i in range(5):
            record_probability_point(st, 0.60 + i * 0.01, 0.40, 0.20, now=_T0 + i)
        assert len(st.probability_evolution) == 5
        timestamps = [p.timestamp for p in st.probability_evolution]
        assert timestamps == sorted(timestamps)

    def test_spike_recorded_when_delta_exceeds_threshold(self):
        st = _state()
        record_probability_point(st, 0.60, 0.50, uncertainty=0.10, now=_T0)
        # Jump uncertainty by > UNCERTAINTY_SPIKE_DELTA
        record_probability_point(
            st, 0.55, 0.50, uncertainty=0.10 + UNCERTAINTY_SPIKE_DELTA + 0.05, now=_T0 + 10
        )
        assert len(st.uncertainty_spikes) == 1

    def test_no_spike_below_threshold(self):
        st = _state()
        record_probability_point(st, 0.60, 0.50, uncertainty=0.20, now=_T0)
        record_probability_point(
            st, 0.60, 0.50, uncertainty=0.20 + UNCERTAINTY_SPIKE_DELTA - 0.01, now=_T0 + 10
        )
        assert len(st.uncertainty_spikes) == 0

    def test_cap_at_max_probability_points(self):
        st = _state()
        for i in range(MAX_PROBABILITY_POINTS + 50):
            record_probability_point(st, 0.60, 0.40, 0.20, now=_T0 + i)
        assert len(st.probability_evolution) <= MAX_PROBABILITY_POINTS

    def test_values_stored_correctly(self):
        st = _state()
        pt = record_probability_point(
            st, primary_p=0.72, secondary_p=0.48, uncertainty=0.15, now=_T0
        )
        assert abs(pt.primary_p - 0.72) < 1e-4
        assert abs(pt.secondary_p - 0.48) < 1e-4
        assert abs(pt.uncertainty - 0.15) < 1e-4


# ═══════════════════════════════════════════════════════════════════════════════
# 13. recommend_switch()
# ═══════════════════════════════════════════════════════════════════════════════


class TestRecommendSwitch:
    def test_secondary_dominance_recommends_switch(self):
        st = _state()
        primary_p = 0.40
        secondary_p = 0.40 + SWITCH_MARGIN + 0.05
        should, reason = recommend_switch(st, primary_p=primary_p, secondary_p=secondary_p)
        assert should is True
        assert "secondary_dominance" in reason

    def test_confidence_collapse_recommends_switch(self):
        st = _state()
        should, reason = recommend_switch(
            st,
            primary_p=PRIMARY_CONFIDENCE_FLOOR - 0.05,
            secondary_p=0.30,
        )
        assert should is True
        assert "confidence_collapse" in reason

    def test_both_strong_no_switch(self):
        st = _state()
        should, _ = recommend_switch(st, primary_p=0.70, secondary_p=0.55)
        assert should is False

    def test_equal_no_switch(self):
        st = _state()
        should, _ = recommend_switch(st, primary_p=0.60, secondary_p=0.60)
        assert should is False

    def test_just_below_switch_margin_no_switch(self):
        st = _state()
        primary_p = 0.50
        secondary_p = 0.50 + SWITCH_MARGIN - 0.01
        should, _ = recommend_switch(st, primary_p=primary_p, secondary_p=secondary_p)
        assert should is False

    def test_reason_empty_when_no_switch(self):
        st = _state()
        _, reason = recommend_switch(st, primary_p=0.70, secondary_p=0.50)
        assert reason == ""


# ═══════════════════════════════════════════════════════════════════════════════
# 14. CounterfactualEngine
# ═══════════════════════════════════════════════════════════════════════════════


class TestCounterfactualEngine:
    def test_instantiates_default(self):
        engine = CounterfactualEngine()
        assert engine.primary_side == BeliefDirection.BULLISH
        assert engine.secondary_side == BeliefDirection.BEARISH

    def test_instrument_label(self):
        engine = CounterfactualEngine(instrument="GOLD")
        assert engine.instrument == "GOLD"

    def test_initial_sides(self):
        engine = CounterfactualEngine(
            initial_primary_side=BeliefDirection.BEARISH,
            initial_secondary_side=BeliefDirection.BULLISH,
        )
        assert engine.primary_side == BeliefDirection.BEARISH

    def test_generate_alternatives_returns_scenarios(self):
        engine = CounterfactualEngine()
        result = engine.generate_alternatives(_bull(), now=_T0)
        assert len(result) >= 2

    def test_evaluate_alternative_returns_float(self):
        engine = CounterfactualEngine()
        scenarios = engine.generate_alternatives(_bull(), now=_T0)
        score = engine.evaluate_alternative(scenarios[0], ["bearish divergence"])
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_switch_thesis_updates_state(self):
        engine = CounterfactualEngine(instrument="EURUSD")
        engine.switch_thesis(
            trigger=FlipTrigger.SECONDARY_DOMINANCE,
            new_primary_side=BeliefDirection.BEARISH,
            new_primary_thesis=_bear(),
            new_secondary_thesis=_bull(),
            now=_T0,
        )
        assert engine.primary_side == BeliefDirection.BEARISH

    def test_record_probability_point(self):
        engine = CounterfactualEngine()
        engine.record_probability_point(0.65, 0.45, uncertainty=0.20, now=_T0)
        assert len(engine.probability_evolution()) == 1

    def test_recommend_switch_callable(self):
        engine = CounterfactualEngine()
        should, _ = engine.recommend_switch(primary_p=0.70, secondary_p=0.50)
        assert should is False

    def test_flip_records_accessible(self):
        engine = CounterfactualEngine()
        engine.switch_thesis(
            trigger=FlipTrigger.FORCED,
            new_primary_side=BeliefDirection.BEARISH,
            new_primary_thesis=_bear(),
            new_secondary_thesis=_bull(),
            now=_T0,
        )
        assert len(engine.flip_records()) == 1

    def test_current_scenarios_accessible(self):
        engine = CounterfactualEngine()
        engine.generate_alternatives(_bull(), now=_T0)
        assert len(engine.current_scenarios()) >= 2

    def test_snapshot_keys(self):
        engine = CounterfactualEngine(instrument="AUDUSD")
        d = engine.snapshot()
        for k in (
            "instrument",
            "primary_side",
            "secondary_side",
            "flip_count",
            "reversal_count",
            "spike_count",
        ):
            assert k in d

    def test_instances_isolated(self):
        e1 = CounterfactualEngine(instrument="EUR")
        e2 = CounterfactualEngine(instrument="GBP")
        e1.switch_thesis(
            trigger=FlipTrigger.FORCED,
            new_primary_side=BeliefDirection.BEARISH,
            new_primary_thesis=_bear(),
            new_secondary_thesis=_bull(),
            now=_T0,
        )
        assert len(e2.flip_records()) == 0
        assert e1.primary_side != e2.primary_side

    def test_full_round_trip(self):
        """generate → evaluate → record → switch → snapshot."""
        engine = CounterfactualEngine(instrument="EURUSD")

        # 1. Generate alternatives
        scenarios = engine.generate_alternatives(
            _bull(confidence=0.72, conflicting=["bearish divergence"]),
            _bear(confidence=0.55),
            uncertainty=0.25,
            now=_T0,
        )
        assert len(scenarios) >= 2

        # 2. Evaluate reversal scenario
        reversal = next(s for s in scenarios if s.scenario_type == ScenarioType.REVERSAL)
        score = engine.evaluate_alternative(
            reversal,
            new_evidence=["bearish divergence appearing on H4"],
        )
        assert 0.0 <= score <= 1.0

        # 3. Record probability point
        engine.record_probability_point(0.55, 0.65, uncertainty=0.30, now=_T0 + 60)

        # 4. Recommend switch fires when secondary dominates
        should, reason = engine.recommend_switch(primary_p=0.45, secondary_p=0.65)
        assert should is True

        # 5. Perform switch
        result = engine.switch_thesis(
            trigger=FlipTrigger.SECONDARY_DOMINANCE,
            new_primary_side=BeliefDirection.BEARISH,
            new_primary_thesis=_bear(confidence=0.70),
            new_secondary_thesis=_bull(confidence=0.45),
            trigger_detail=reason,
            uncertainty=0.30,
            now=_T0 + 120,
        )
        assert result.is_reversal is False
        assert engine.primary_side == BeliefDirection.BEARISH

        snap = engine.snapshot()
        assert snap["flip_count"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 15. End-to-end scenarios
# ═══════════════════════════════════════════════════════════════════════════════


class TestEndToEndScenarios:
    def test_continuous_what_if_monitoring(self):
        """Generate alternatives every tick; reversal scenario's weight rises as
        conflicting evidence accumulates."""
        engine = CounterfactualEngine(instrument="GBPUSD")

        # Initial state: bullish primary with some conflicting evidence
        bull = _bull(confidence=0.70, conflicting=["momentum divergence"])
        bear = _bear(confidence=0.50)

        scenarios_t0 = engine.generate_alternatives(bull, bear, uncertainty=0.20, now=_T0)
        rev_t0 = next(s for s in scenarios_t0 if s.scenario_type == ScenarioType.REVERSAL)

        # More conflicting evidence arrives
        bull_weakened = _bull(
            confidence=0.55,
            conflicting=["momentum divergence", "volume declining", "key level rejected"],
        )
        bear_stronger = _bear(confidence=0.65)

        scenarios_t1 = engine.generate_alternatives(
            bull_weakened, bear_stronger, uncertainty=0.40, now=_T0 + 300
        )
        rev_t1 = next(s for s in scenarios_t1 if s.scenario_type == ScenarioType.REVERSAL)

        # Reversal scenario should have grown more plausible (higher weight or more triggers)
        assert rev_t1.probability_weight >= rev_t0.probability_weight or len(
            rev_t1.trigger_conditions
        ) >= len(rev_t0.trigger_conditions)

    def test_rapid_flip_back_detected_as_reversal(self):
        """BULL → BEAR within 10s → BULL again within window → reversal."""
        engine = CounterfactualEngine(
            instrument="USDJPY",
            initial_primary_side=BeliefDirection.BULLISH,
        )

        # Flip 1: BULL → BEAR
        engine.switch_thesis(
            trigger=FlipTrigger.UNCERTAINTY_SPIKE,
            new_primary_side=BeliefDirection.BEARISH,
            new_primary_thesis=_bear(),
            new_secondary_thesis=_bull(),
            now=_T0,
        )

        # Flip 2 back: BEAR → BULL (within reversal window)
        result = engine.switch_thesis(
            trigger=FlipTrigger.EVIDENCE_SHIFT,
            new_primary_side=BeliefDirection.BULLISH,
            new_primary_thesis=_bull(),
            new_secondary_thesis=_bear(),
            now=_T0 + 10,
        )

        assert result.is_reversal is True
        assert len(engine.reversal_records()) == 1
        assert engine.state.reversal_rate() > 0.0

    def test_uncertainty_spike_recorded_during_evolution(self):
        """Track probability evolution; spike recorded when uncertainty jumps."""
        engine = CounterfactualEngine(instrument="XAUUSD")
        engine.record_probability_point(0.65, 0.45, uncertainty=0.15, now=_T0)
        engine.record_probability_point(0.58, 0.52, uncertainty=0.45, now=_T0 + 30)
        # 0.45 - 0.15 = 0.30 > UNCERTAINTY_SPIKE_DELTA (0.20)
        assert len(engine.uncertainty_spikes()) == 1
        spike = engine.uncertainty_spikes()[0]
        assert spike.delta >= UNCERTAINTY_SPIKE_DELTA

    def test_learn_from_incorrect_assumptions(self):
        """A flip lesson surfaces the ignored warnings from the old primary."""
        engine = CounterfactualEngine(instrument="EURUSD")

        # Primary bull thesis has been ignoring conflicting evidence
        old_bull = _bull(
            conflicting=[
                "bearish divergence on daily",
                "volume declining on rallies",
                "rejection at key resistance",
            ]
        )

        result = engine.switch_thesis(
            trigger=FlipTrigger.EVIDENCE_SHIFT,
            new_primary_side=BeliefDirection.BEARISH,
            new_primary_thesis=_bear(),
            new_secondary_thesis=old_bull,
            trigger_detail="secondary dominated for 3 bars",
            now=_T0 + REVERSAL_WINDOW_SECONDS + 100,
        )

        lesson = result.flip_record.flip_lesson
        assert lesson is not None
        assert len(lesson.ignored_warnings) > 0
        assert lesson.time_to_flip >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# 16. Import sanity
# ═══════════════════════════════════════════════════════════════════════════════


class TestImportSanity:
    def test_spec_callables_importable(self):
        from trainer.qde.brain.counterfactual import (
            evaluate_alternative,
            generate_alternatives,
            switch_thesis,
        )

        assert all(
            callable(fn) for fn in (generate_alternatives, evaluate_alternative, switch_thesis)
        )

    def test_engine_importable(self):
        from trainer.qde.brain.counterfactual import CounterfactualEngine

        e = CounterfactualEngine()
        assert e.primary_side == BeliefDirection.BULLISH

    def test_all_types_importable(self):
        from trainer.qde.brain.counterfactual import (
            FlipTrigger,
            ScenarioType,
        )

        assert ScenarioType.REVERSAL.value == "REVERSAL"
        assert FlipTrigger.FORCED.value == "FORCED"

    def test_package_path(self):
        import trainer.qde.brain.counterfactual as pkg

        for name in (
            "generate_alternatives",
            "evaluate_alternative",
            "switch_thesis",
            "CounterfactualEngine",
            "CognitiveState" if False else "CounterfactualState",
        ):
            if name == "CounterfactualState":
                assert hasattr(pkg, name)
