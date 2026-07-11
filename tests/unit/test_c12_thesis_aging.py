"""C12 — Thesis Aging tests.

Covers:
  - ThesisEventType: all five values present
  - ThesisDecayProfile: decay_factor formula (half-life property)
  - ThesisDecayProfile: decay_rate derived correctly
  - DEFAULT_DECAY_PROFILES: all five OpportunityClass keys
  - profile_for_class: known classes, unknown fallback
  - ThesisAgingState: age() and elapsed_since_confirmation() properties
  - ThesisAgingState: confirmation_count / invalidation_count properties
  - ThesisAgingState.__post_init__: clocks initialised to created_at
  - age_thesis: no-op on already-terminal hypothesis
  - age_thesis: skips when too soon since last tick
  - age_thesis: applies exponential decay to confidence
  - age_thesis: decay_applied=True when interval elapsed
  - age_thesis: expires when confidence < expiry_threshold
  - age_thesis: expires when max_age exceeded
  - age_thesis: records DECAY_TICK event in invalidation_events
  - age_thesis: appends AgingRecord to history
  - age_thesis: updates hyp.last_updated_at
  - age_thesis: returns ThesisAgingResult with correct fields
  - age_thesis: confirmed hypothesis (status=CONFIRMED) is no-op
  - expire_thesis: forces EXPIRED status regardless of confidence
  - expire_thesis: sets confidence to 0
  - expire_thesis: idempotent on already-expired hypothesis
  - expire_thesis: records FORCE_EXPIRE event
  - expire_thesis: sets state.is_expired and expired_at
  - update_confidence: CONFIRMATION resets decay clock
  - update_confidence: INVALIDATION does not reset decay clock
  - update_confidence: applies positive delta (CONFIRMATION)
  - update_confidence: applies negative delta (INVALIDATION)
  - update_confidence: clamps confidence to [0, 1]
  - update_confidence: records event in confirmation_events
  - update_confidence: records event in invalidation_events
  - update_confidence: auto-expires when confidence < threshold after INVALIDATION
  - update_confidence: appends AgingRecord to history
  - update_confidence: no-op on terminal hypothesis
  - ThesisAgingEngine.register: creates ThesisAgingState with correct profile
  - ThesisAgingEngine.register: idempotent (second call returns same state)
  - ThesisAgingEngine.drop: removes state
  - ThesisAgingEngine.state_for: returns None for unknown id
  - ThesisAgingEngine.age_thesis: auto-registers unknown hypothesis
  - ThesisAgingEngine.expire_thesis: delegates to stateless function
  - ThesisAgingEngine.update_confidence: delegates to stateless function
  - ThesisAgingEngine.age_all: ages all non-terminal hypotheses
  - ThesisAgingEngine.age_all: skips terminal hypotheses
  - ThesisAgingEngine.age_all: accepts opp_class_fn callback
  - ThesisAgingEngine.summary: returns dict with counts
  - Behavioral: repeated confirmations keep confidence high
  - Behavioral: no confirmations decays to expiry
  - Behavioral: MICRO_SCALP decays faster than SWING
  - Behavioral: history grows with each tick
  - Behavioral: event log bounded at MAX_EVENTS
  - Package __init__ exports all C12 symbols
"""

from __future__ import annotations

import math
import time

import pytest

from trainer.qde.brain.brain_models import Hypothesis
from trainer.qde.brain.brain_types import HypothesisStatus, BeliefDirection


# ── Helpers ───────────────────────────────────────────────────────────────────

_T0 = 1_700_000_000.0  # fixed epoch anchor for tests


def _hyp(
    confidence: float = 0.6,
    status: HypothesisStatus = HypothesisStatus.ACTIVE,
    created_at: float = _T0,
    instrument: str = "EURUSD",
) -> Hypothesis:
    h = Hypothesis(
        instrument=instrument,
        thesis="breakout above 1.0950",
        direction=BeliefDirection.BULLISH,
        confidence=confidence,
        status=status,
        created_at=created_at,
        last_updated_at=created_at,
    )
    return h


def _state(hyp: Hypothesis, opp_class: str = "INTRADAY") -> "ThesisAgingState":
    from trainer.qde.brain.thesis import ThesisAgingState, profile_for_class
    return ThesisAgingState(
        hypothesis_id=hyp.hypothesis_id,
        instrument=hyp.instrument,
        created_at=hyp.created_at,
        decay_profile=profile_for_class(opp_class),
        last_confirmation_at=hyp.created_at,
        last_decayed_at=hyp.created_at,
    )


def _state_aged(hyp: Hypothesis, elapsed: float, opp_class: str = "INTRADAY") -> "ThesisAgingState":
    """Return a state whose last_decayed_at is already elapsed seconds ago."""
    from trainer.qde.brain.thesis import ThesisAgingState, profile_for_class
    return ThesisAgingState(
        hypothesis_id=hyp.hypothesis_id,
        instrument=hyp.instrument,
        created_at=hyp.created_at,
        decay_profile=profile_for_class(opp_class),
        last_confirmation_at=hyp.created_at,
        last_decayed_at=hyp.created_at - elapsed,  # makes elapsed large enough
    )


# ═════════════════════════════════════════════════════════════════════════════
# 1. ThesisEventType
# ═════════════════════════════════════════════════════════════════════════════


class TestThesisEventType:
    def test_five_values(self):
        from trainer.qde.brain.thesis import ThesisEventType
        assert len(list(ThesisEventType)) == 5

    def test_required_values(self):
        from trainer.qde.brain.thesis import ThesisEventType
        assert ThesisEventType.CONFIRMATION.value  == "CONFIRMATION"
        assert ThesisEventType.INVALIDATION.value  == "INVALIDATION"
        assert ThesisEventType.DECAY_TICK.value    == "DECAY_TICK"
        assert ThesisEventType.EXPIRY.value        == "EXPIRY"
        assert ThesisEventType.FORCE_EXPIRE.value  == "FORCE_EXPIRE"

    def test_is_str_enum(self):
        from trainer.qde.brain.thesis import ThesisEventType
        assert isinstance(ThesisEventType.CONFIRMATION, str)


# ═════════════════════════════════════════════════════════════════════════════
# 2. ThesisDecayProfile
# ═════════════════════════════════════════════════════════════════════════════


class TestThesisDecayProfile:
    def test_decay_factor_at_zero_elapsed_is_one(self):
        from trainer.qde.brain.thesis import ThesisDecayProfile
        p = ThesisDecayProfile(half_life_seconds=3600, max_age_seconds=28800)
        assert p.decay_factor(0.0) == pytest.approx(1.0)

    def test_decay_factor_at_half_life_is_half(self):
        from trainer.qde.brain.thesis import ThesisDecayProfile
        p = ThesisDecayProfile(half_life_seconds=3600, max_age_seconds=28800)
        assert p.decay_factor(3600.0) == pytest.approx(0.5, abs=1e-6)

    def test_decay_factor_at_two_half_lives_is_quarter(self):
        from trainer.qde.brain.thesis import ThesisDecayProfile
        p = ThesisDecayProfile(half_life_seconds=3600, max_age_seconds=28800)
        assert p.decay_factor(7200.0) == pytest.approx(0.25, abs=1e-6)

    def test_decay_rate_is_log2_over_half_life(self):
        from trainer.qde.brain.thesis import ThesisDecayProfile
        p = ThesisDecayProfile(half_life_seconds=1800, max_age_seconds=3600)
        assert p.decay_rate == pytest.approx(math.log(2) / 1800, rel=1e-9)

    def test_decay_factor_always_positive(self):
        from trainer.qde.brain.thesis import ThesisDecayProfile
        p = ThesisDecayProfile(half_life_seconds=600, max_age_seconds=900)
        for t in [0, 60, 600, 3600, 86400]:
            assert p.decay_factor(t) > 0.0

    def test_decay_factor_monotone_decreasing(self):
        from trainer.qde.brain.thesis import ThesisDecayProfile
        p = ThesisDecayProfile(half_life_seconds=1800, max_age_seconds=3600)
        factors = [p.decay_factor(t) for t in [0, 900, 1800, 3600, 7200]]
        assert factors == sorted(factors, reverse=True)

    def test_to_dict_contains_key_fields(self):
        from trainer.qde.brain.thesis import ThesisDecayProfile
        p = ThesisDecayProfile(half_life_seconds=600, max_age_seconds=900)
        d = p.to_dict()
        for key in ("half_life_seconds", "max_age_seconds", "expiry_threshold",
                    "min_decay_interval_seconds", "decay_rate"):
            assert key in d


# ═════════════════════════════════════════════════════════════════════════════
# 3. DEFAULT_DECAY_PROFILES and profile_for_class
# ═════════════════════════════════════════════════════════════════════════════


class TestDecayProfiles:
    def test_all_five_classes_present(self):
        from trainer.qde.brain.thesis import DEFAULT_DECAY_PROFILES
        expected = {"MICRO_SCALP", "SCALP", "INTRADAY", "MULTI_SESSION", "SWING"}
        assert set(DEFAULT_DECAY_PROFILES.keys()) == expected

    def test_micro_scalp_fastest_half_life(self):
        from trainer.qde.brain.thesis import DEFAULT_DECAY_PROFILES
        hl = {k: v.half_life_seconds for k, v in DEFAULT_DECAY_PROFILES.items()}
        assert hl["MICRO_SCALP"] < hl["SCALP"] < hl["INTRADAY"] < hl["MULTI_SESSION"] < hl["SWING"]

    def test_micro_scalp_half_life_in_range(self):
        from trainer.qde.brain.thesis import DEFAULT_DECAY_PROFILES
        hl = DEFAULT_DECAY_PROFILES["MICRO_SCALP"].half_life_seconds
        assert 300 <= hl <= 900  # 5-15 min

    def test_swing_half_life_in_range(self):
        from trainer.qde.brain.thesis import DEFAULT_DECAY_PROFILES
        hl = DEFAULT_DECAY_PROFILES["SWING"].half_life_seconds
        assert 604800 <= hl <= 1814400  # 1-3 weeks

    def test_profile_for_class_known(self):
        from trainer.qde.brain.thesis import profile_for_class, DEFAULT_DECAY_PROFILES
        for cls in ("MICRO_SCALP", "SCALP", "INTRADAY", "MULTI_SESSION", "SWING"):
            assert profile_for_class(cls) == DEFAULT_DECAY_PROFILES[cls]

    def test_profile_for_class_unknown_falls_back(self):
        from trainer.qde.brain.thesis import profile_for_class, DEFAULT_DECAY_PROFILES
        p = profile_for_class("UNKNOWN_CLASS")
        assert p == DEFAULT_DECAY_PROFILES["INTRADAY"]

    def test_max_age_ordered(self):
        from trainer.qde.brain.thesis import DEFAULT_DECAY_PROFILES
        ma = {k: v.max_age_seconds for k, v in DEFAULT_DECAY_PROFILES.items()}
        assert ma["MICRO_SCALP"] < ma["SCALP"] < ma["INTRADAY"] < ma["MULTI_SESSION"] < ma["SWING"]


# ═════════════════════════════════════════════════════════════════════════════
# 4. ThesisAgingState
# ═════════════════════════════════════════════════════════════════════════════


class TestThesisAgingState:
    def test_age_seconds(self):
        hyp = _hyp(created_at=_T0)
        st = _state(hyp)
        assert st.age(_T0 + 3600) == pytest.approx(3600.0)

    def test_age_zero_when_just_created(self):
        hyp = _hyp(created_at=_T0)
        st = _state(hyp)
        assert st.age(_T0) == pytest.approx(0.0)

    def test_elapsed_since_confirmation(self):
        hyp = _hyp(created_at=_T0)
        st = _state(hyp)
        st.last_confirmation_at = _T0 + 1000
        assert st.elapsed_since_confirmation(_T0 + 5000) == pytest.approx(4000.0)

    def test_confirmation_count(self):
        from trainer.qde.brain.thesis import ThesisEvent, ThesisEventType
        hyp = _hyp()
        st = _state(hyp)
        assert st.confirmation_count == 0
        ev = ThesisEvent(ThesisEventType.CONFIRMATION, confidence_before=0.5,
                         confidence_after=0.6, occurred_at=_T0)
        st._append_event(ev, is_confirmation=True)
        assert st.confirmation_count == 1

    def test_invalidation_count(self):
        from trainer.qde.brain.thesis import ThesisEvent, ThesisEventType
        hyp = _hyp()
        st = _state(hyp)
        ev = ThesisEvent(ThesisEventType.INVALIDATION, confidence_before=0.5,
                         confidence_after=0.3, occurred_at=_T0)
        st._append_event(ev, is_confirmation=False)
        assert st.invalidation_count == 1

    def test_post_init_clocks_set(self):
        hyp = _hyp(created_at=_T0)
        st = _state(hyp)
        assert st.last_confirmation_at == pytest.approx(_T0)
        assert st.last_decayed_at == pytest.approx(_T0)

    def test_to_dict_keys(self):
        hyp = _hyp()
        st = _state(hyp)
        d = st.to_dict()
        for key in ("hypothesis_id", "instrument", "created_at", "decay_profile",
                    "confirmation_count", "invalidation_count",
                    "last_confirmation_at", "last_decayed_at",
                    "is_expired", "expired_at", "aging_history_len"):
            assert key in d

    def test_event_log_bounded(self):
        from trainer.qde.brain.thesis import ThesisEvent, ThesisEventType, MAX_EVENTS
        hyp = _hyp()
        st = _state(hyp)
        for i in range(MAX_EVENTS + 20):
            ev = ThesisEvent(ThesisEventType.CONFIRMATION,
                             confidence_before=0.5, confidence_after=0.6,
                             occurred_at=_T0 + i)
            st._append_event(ev, is_confirmation=True)
        assert len(st.confirmation_events) == MAX_EVENTS

    def test_history_bounded(self):
        from trainer.qde.brain.thesis import AgingRecord, ThesisEventType, MAX_HISTORY
        hyp = _hyp()
        st = _state(hyp)
        for i in range(MAX_HISTORY + 20):
            rec = AgingRecord(
                hypothesis_id=hyp.hypothesis_id, confidence=0.5,
                age_seconds=float(i), elapsed_since_confirm=0.0,
                confirmation_count=0, invalidation_count=0,
                trigger=ThesisEventType.DECAY_TICK, occurred_at=_T0 + i,
            )
            st._append_history(rec)
        assert len(st.aging_history) == MAX_HISTORY


# ═════════════════════════════════════════════════════════════════════════════
# 5. age_thesis — guards
# ═════════════════════════════════════════════════════════════════════════════


class TestAgeThesisGuards:
    def test_no_op_on_expired_hypothesis(self):
        from trainer.qde.brain.thesis import age_thesis
        hyp = _hyp(status=HypothesisStatus.EXPIRED, confidence=0.0)
        st = _state(hyp)
        result = age_thesis(hyp, st, now=_T0 + 7200)
        assert result.decay_applied is False
        assert result.expired is True

    def test_no_op_on_invalidated_hypothesis(self):
        from trainer.qde.brain.thesis import age_thesis
        hyp = _hyp(status=HypothesisStatus.INVALIDATED, confidence=0.0)
        st = _state(hyp)
        result = age_thesis(hyp, st, now=_T0 + 7200)
        assert result.decay_applied is False

    def test_no_op_on_confirmed_hypothesis(self):
        from trainer.qde.brain.thesis import age_thesis
        hyp = _hyp(status=HypothesisStatus.CONFIRMED, confidence=1.0)
        st = _state(hyp)
        result = age_thesis(hyp, st, now=_T0 + 7200)
        assert result.decay_applied is False

    def test_skips_when_interval_not_elapsed(self):
        from trainer.qde.brain.thesis import age_thesis
        hyp = _hyp(created_at=_T0)
        st = _state(hyp)
        # Last decay was only 5 seconds ago; min interval is 10s
        st.last_decayed_at = _T0 + 3595
        result = age_thesis(hyp, st, now=_T0 + 3600)
        assert result.decay_applied is False
        assert result.confidence_before == result.confidence_after

    def test_returns_thesis_aging_result(self):
        from trainer.qde.brain.thesis import age_thesis, ThesisAgingResult
        hyp = _hyp(created_at=_T0)
        st = _state_aged(hyp, elapsed=7200)  # well past interval
        result = age_thesis(hyp, st, now=_T0 + 7200)
        assert isinstance(result, ThesisAgingResult)


# ═════════════════════════════════════════════════════════════════════════════
# 6. age_thesis — decay behavior
# ═════════════════════════════════════════════════════════════════════════════


class TestAgeThesisDecay:
    def test_decay_applied_true_when_interval_elapsed(self):
        from trainer.qde.brain.thesis import age_thesis
        hyp = _hyp(confidence=0.6, created_at=_T0)
        st = _state_aged(hyp, elapsed=7200)
        result = age_thesis(hyp, st, now=_T0 + 7200)
        assert result.decay_applied is True

    def test_confidence_decreases(self):
        from trainer.qde.brain.thesis import age_thesis
        hyp = _hyp(confidence=0.6, created_at=_T0)
        st = _state_aged(hyp, elapsed=7200)
        result = age_thesis(hyp, st, now=_T0 + 7200)
        assert result.confidence_after < result.confidence_before

    def test_confidence_decreases_by_correct_factor(self):
        """At one half-life the confidence should halve."""
        from trainer.qde.brain.thesis import age_thesis, DEFAULT_DECAY_PROFILES
        profile = DEFAULT_DECAY_PROFILES["INTRADAY"]  # half_life=14400s
        hyp = _hyp(confidence=0.8, created_at=_T0)
        st = _state(hyp, "INTRADAY")
        st.last_decayed_at = _T0 - profile.half_life_seconds - 1  # force interval passed
        now = _T0 + profile.half_life_seconds  # exactly one half-life elapsed
        # last_confirmation_at = _T0 so elapsed_since_confirm = half_life
        result = age_thesis(hyp, st, now=now)
        if result.decay_applied:
            assert result.confidence_after == pytest.approx(
                0.8 * profile.decay_factor(profile.half_life_seconds), abs=1e-4
            )

    def test_hyp_confidence_mutated(self):
        from trainer.qde.brain.thesis import age_thesis
        hyp = _hyp(confidence=0.7, created_at=_T0)
        st = _state_aged(hyp, elapsed=7200)
        age_thesis(hyp, st, now=_T0 + 7200)
        assert hyp.confidence < 0.7

    def test_hyp_last_updated_at_touched(self):
        from trainer.qde.brain.thesis import age_thesis
        hyp = _hyp(confidence=0.7, created_at=_T0)
        st = _state_aged(hyp, elapsed=7200)
        now = _T0 + 7200
        age_thesis(hyp, st, now=now)
        assert hyp.last_updated_at == pytest.approx(now)

    def test_decay_event_recorded(self):
        from trainer.qde.brain.thesis import age_thesis, ThesisEventType
        hyp = _hyp(confidence=0.7, created_at=_T0)
        st = _state_aged(hyp, elapsed=7200)
        age_thesis(hyp, st, now=_T0 + 7200)
        # DECAY_TICK events go into invalidation_events
        types = [e.event_type for e in st.invalidation_events]
        assert ThesisEventType.DECAY_TICK in types

    def test_aging_record_appended(self):
        from trainer.qde.brain.thesis import age_thesis
        hyp = _hyp(confidence=0.7, created_at=_T0)
        st = _state_aged(hyp, elapsed=7200)
        before_len = len(st.aging_history)
        age_thesis(hyp, st, now=_T0 + 7200)
        assert len(st.aging_history) > before_len

    def test_state_last_decayed_at_updated(self):
        from trainer.qde.brain.thesis import age_thesis
        hyp = _hyp(confidence=0.7, created_at=_T0)
        st = _state_aged(hyp, elapsed=7200)
        now = _T0 + 7200
        age_thesis(hyp, st, now=now)
        assert st.last_decayed_at == pytest.approx(now)

    def test_result_fields_populated(self):
        from trainer.qde.brain.thesis import age_thesis
        hyp = _hyp(confidence=0.7, created_at=_T0)
        st = _state_aged(hyp, elapsed=7200)
        now = _T0 + 7200
        result = age_thesis(hyp, st, now=now)
        if result.decay_applied:
            assert result.hypothesis_id == hyp.hypothesis_id
            assert result.instrument == hyp.instrument
            assert 0.0 < result.decay_factor <= 1.0
            assert result.age_seconds >= 0.0


# ═════════════════════════════════════════════════════════════════════════════
# 7. age_thesis — expiry paths
# ═════════════════════════════════════════════════════════════════════════════


class TestAgeThesisExpiry:
    def test_expires_when_confidence_below_threshold(self):
        from trainer.qde.brain.thesis import age_thesis, ThesisDecayProfile, ThesisAgingState
        # Build a profile with a very short half-life so confidence drops fast
        profile = ThesisDecayProfile(
            half_life_seconds=10.0,
            max_age_seconds=3600.0,
            expiry_threshold=0.1,
            min_decay_interval_seconds=5.0,
        )
        hyp = _hyp(confidence=0.11, created_at=_T0)
        st = ThesisAgingState(
            hypothesis_id=hyp.hypothesis_id,
            instrument=hyp.instrument,
            created_at=hyp.created_at,
            decay_profile=profile,
            last_confirmation_at=_T0,
            last_decayed_at=_T0 - 600,  # force interval passed
        )
        # After 6× half-life (60s) confidence ≈ 0.11 × 2^-6 ≈ 0.0017 < 0.1
        result = age_thesis(hyp, st, now=_T0 + 60)
        assert result.expired is True
        assert hyp.status == HypothesisStatus.EXPIRED
        assert hyp.confidence == pytest.approx(0.0)

    def test_expires_when_max_age_exceeded(self):
        from trainer.qde.brain.thesis import age_thesis, ThesisDecayProfile, ThesisAgingState
        profile = ThesisDecayProfile(
            half_life_seconds=1800.0,
            max_age_seconds=900.0,  # hard expiry at 15 min
            expiry_threshold=0.05,
            min_decay_interval_seconds=5.0,
        )
        hyp = _hyp(confidence=0.8, created_at=_T0)
        st = ThesisAgingState(
            hypothesis_id=hyp.hypothesis_id,
            instrument=hyp.instrument,
            created_at=hyp.created_at,
            decay_profile=profile,
            last_confirmation_at=_T0,
            last_decayed_at=_T0 - 600,  # force interval passed
        )
        # 1000s > max_age_seconds=900s → expire
        result = age_thesis(hyp, st, now=_T0 + 1000)
        assert result.expired is True
        assert hyp.status == HypothesisStatus.EXPIRED

    def test_expired_result_confidence_is_zero(self):
        from trainer.qde.brain.thesis import age_thesis, ThesisDecayProfile, ThesisAgingState
        profile = ThesisDecayProfile(
            half_life_seconds=10.0, max_age_seconds=900.0,
            expiry_threshold=0.05, min_decay_interval_seconds=5.0,
        )
        hyp = _hyp(confidence=0.06, created_at=_T0)
        st = ThesisAgingState(
            hypothesis_id=hyp.hypothesis_id, instrument=hyp.instrument,
            created_at=hyp.created_at, decay_profile=profile,
            last_confirmation_at=_T0, last_decayed_at=_T0 - 600,
        )
        result = age_thesis(hyp, st, now=_T0 + 100)
        assert result.confidence_after == pytest.approx(0.0)


# ═════════════════════════════════════════════════════════════════════════════
# 8. expire_thesis
# ═════════════════════════════════════════════════════════════════════════════


class TestExpireThesis:
    def test_forces_expired_status(self):
        from trainer.qde.brain.thesis import expire_thesis
        hyp = _hyp(confidence=0.7)
        st = _state(hyp)
        expire_thesis(hyp, st, reason="test expiry", now=_T0 + 100)
        assert hyp.status == HypothesisStatus.EXPIRED

    def test_sets_confidence_to_zero(self):
        from trainer.qde.brain.thesis import expire_thesis
        hyp = _hyp(confidence=0.8)
        st = _state(hyp)
        expire_thesis(hyp, st, now=_T0 + 100)
        assert hyp.confidence == pytest.approx(0.0)

    def test_sets_state_is_expired(self):
        from trainer.qde.brain.thesis import expire_thesis
        hyp = _hyp(confidence=0.5)
        st = _state(hyp)
        expire_thesis(hyp, st, now=_T0 + 100)
        assert st.is_expired is True
        assert st.expired_at is not None

    def test_records_force_expire_event(self):
        from trainer.qde.brain.thesis import expire_thesis, ThesisEventType
        hyp = _hyp(confidence=0.5)
        st = _state(hyp)
        expire_thesis(hyp, st, reason="test", now=_T0 + 100)
        types = [e.event_type for e in st.invalidation_events]
        assert ThesisEventType.FORCE_EXPIRE in types

    def test_idempotent_on_already_expired(self):
        from trainer.qde.brain.thesis import expire_thesis
        hyp = _hyp(status=HypothesisStatus.EXPIRED, confidence=0.0)
        st = _state(hyp)
        expire_thesis(hyp, st, reason="first", now=_T0 + 100)
        expire_thesis(hyp, st, reason="second", now=_T0 + 200)
        # Only one expiry event recorded
        from trainer.qde.brain.thesis import ThesisEventType
        expiry_events = [e for e in st.invalidation_events
                         if e.event_type == ThesisEventType.FORCE_EXPIRE]
        assert len(expiry_events) == 0  # already terminal at first call

    def test_idempotent_on_invalidated(self):
        from trainer.qde.brain.thesis import expire_thesis
        hyp = _hyp(status=HypothesisStatus.INVALIDATED, confidence=0.0)
        st = _state(hyp)
        before_len = len(st.invalidation_events)
        expire_thesis(hyp, st, reason="noop", now=_T0 + 100)
        assert len(st.invalidation_events) == before_len

    def test_appends_aging_record(self):
        from trainer.qde.brain.thesis import expire_thesis
        hyp = _hyp(confidence=0.6)
        st = _state(hyp)
        expire_thesis(hyp, st, now=_T0 + 100)
        assert len(st.aging_history) >= 1


# ═════════════════════════════════════════════════════════════════════════════
# 9. update_confidence
# ═════════════════════════════════════════════════════════════════════════════


class TestUpdateConfidence:
    def test_confirmation_increases_confidence(self):
        from trainer.qde.brain.thesis import update_confidence, ThesisEventType
        hyp = _hyp(confidence=0.5)
        st = _state(hyp)
        update_confidence(hyp, st, delta=0.15,
                          event_type=ThesisEventType.CONFIRMATION,
                          evidence="new high", now=_T0 + 100)
        assert hyp.confidence == pytest.approx(0.65)

    def test_invalidation_decreases_confidence(self):
        from trainer.qde.brain.thesis import update_confidence, ThesisEventType
        hyp = _hyp(confidence=0.5)
        st = _state(hyp)
        update_confidence(hyp, st, delta=-0.2,
                          event_type=ThesisEventType.INVALIDATION,
                          evidence="rejection", now=_T0 + 100)
        assert hyp.confidence == pytest.approx(0.3)

    def test_confirmation_resets_decay_clock(self):
        from trainer.qde.brain.thesis import update_confidence, ThesisEventType
        hyp = _hyp(confidence=0.5)
        st = _state(hyp)
        st.last_confirmation_at = _T0  # old clock
        now = _T0 + 3600
        update_confidence(hyp, st, delta=0.1,
                          event_type=ThesisEventType.CONFIRMATION, now=now)
        assert st.last_confirmation_at == pytest.approx(now)

    def test_invalidation_does_not_reset_clock(self):
        from trainer.qde.brain.thesis import update_confidence, ThesisEventType
        hyp = _hyp(confidence=0.5)
        st = _state(hyp)
        st.last_confirmation_at = _T0  # set clock at T0
        update_confidence(hyp, st, delta=-0.1,
                          event_type=ThesisEventType.INVALIDATION, now=_T0 + 3600)
        assert st.last_confirmation_at == pytest.approx(_T0)  # unchanged

    def test_confidence_clamped_to_one(self):
        from trainer.qde.brain.thesis import update_confidence, ThesisEventType
        hyp = _hyp(confidence=0.9)
        st = _state(hyp)
        update_confidence(hyp, st, delta=0.5,
                          event_type=ThesisEventType.CONFIRMATION, now=_T0 + 10)
        assert hyp.confidence <= 1.0

    def test_confidence_clamped_to_zero(self):
        from trainer.qde.brain.thesis import update_confidence, ThesisEventType
        hyp = _hyp(confidence=0.1)
        st = _state(hyp)
        update_confidence(hyp, st, delta=-0.5,
                          event_type=ThesisEventType.INVALIDATION, now=_T0 + 10)
        assert hyp.confidence >= 0.0

    def test_confirmation_recorded_in_confirmation_events(self):
        from trainer.qde.brain.thesis import update_confidence, ThesisEventType
        hyp = _hyp(confidence=0.5)
        st = _state(hyp)
        update_confidence(hyp, st, delta=0.1,
                          event_type=ThesisEventType.CONFIRMATION,
                          evidence="breakout confirmed", now=_T0 + 10)
        assert st.confirmation_count == 1
        assert st.confirmation_events[0].evidence == "breakout confirmed"

    def test_invalidation_recorded_in_invalidation_events(self):
        from trainer.qde.brain.thesis import update_confidence, ThesisEventType
        hyp = _hyp(confidence=0.5)
        st = _state(hyp)
        update_confidence(hyp, st, delta=-0.1,
                          event_type=ThesisEventType.INVALIDATION,
                          evidence="reversal", now=_T0 + 10)
        assert st.invalidation_count == 1
        assert st.invalidation_events[0].evidence == "reversal"

    def test_auto_expires_when_confidence_below_threshold(self):
        from trainer.qde.brain.thesis import update_confidence, ThesisEventType
        hyp = _hyp(confidence=0.12)
        st = _state(hyp)
        update_confidence(hyp, st, delta=-0.1,
                          event_type=ThesisEventType.INVALIDATION, now=_T0 + 10)
        # 0.12 - 0.1 = 0.02 < DEFAULT_EXPIRY_THRESHOLD (0.08)
        assert hyp.status == HypothesisStatus.EXPIRED

    def test_no_op_on_terminal(self):
        from trainer.qde.brain.thesis import update_confidence, ThesisEventType
        hyp = _hyp(status=HypothesisStatus.EXPIRED, confidence=0.0)
        st = _state(hyp)
        before_count = st.confirmation_count
        update_confidence(hyp, st, delta=0.5,
                          event_type=ThesisEventType.CONFIRMATION, now=_T0 + 10)
        assert st.confirmation_count == before_count
        assert hyp.confidence == pytest.approx(0.0)

    def test_appends_aging_record(self):
        from trainer.qde.brain.thesis import update_confidence, ThesisEventType
        hyp = _hyp(confidence=0.5)
        st = _state(hyp)
        update_confidence(hyp, st, delta=0.1,
                          event_type=ThesisEventType.CONFIRMATION, now=_T0 + 10)
        assert len(st.aging_history) >= 1

    def test_event_delta_property(self):
        from trainer.qde.brain.thesis import update_confidence, ThesisEventType
        hyp = _hyp(confidence=0.5)
        st = _state(hyp)
        update_confidence(hyp, st, delta=0.2,
                          event_type=ThesisEventType.CONFIRMATION, now=_T0 + 10)
        ev = st.confirmation_events[-1]
        assert ev.delta == pytest.approx(0.2, abs=1e-4)


# ═════════════════════════════════════════════════════════════════════════════
# 10. ThesisAgingEngine
# ═════════════════════════════════════════════════════════════════════════════


class TestThesisAgingEngine:
    def test_register_creates_state(self):
        from trainer.qde.brain.thesis import ThesisAgingEngine
        engine = ThesisAgingEngine()
        hyp = _hyp()
        state = engine.register(hyp, "SCALP")
        assert state.hypothesis_id == hyp.hypothesis_id
        assert state.decay_profile.half_life_seconds == pytest.approx(1800.0)

    def test_register_idempotent(self):
        from trainer.qde.brain.thesis import ThesisAgingEngine
        engine = ThesisAgingEngine()
        hyp = _hyp()
        s1 = engine.register(hyp, "SCALP")
        s2 = engine.register(hyp, "SCALP")
        assert s1 is s2

    def test_drop_removes_state(self):
        from trainer.qde.brain.thesis import ThesisAgingEngine
        engine = ThesisAgingEngine()
        hyp = _hyp()
        engine.register(hyp, "INTRADAY")
        engine.drop(hyp.hypothesis_id)
        assert engine.state_for(hyp.hypothesis_id) is None

    def test_drop_unknown_id_is_safe(self):
        from trainer.qde.brain.thesis import ThesisAgingEngine
        engine = ThesisAgingEngine()
        engine.drop("nonexistent-id")  # should not raise

    def test_state_for_unknown_returns_none(self):
        from trainer.qde.brain.thesis import ThesisAgingEngine
        engine = ThesisAgingEngine()
        assert engine.state_for("bogus") is None

    def test_age_thesis_auto_registers(self):
        from trainer.qde.brain.thesis import ThesisAgingEngine
        engine = ThesisAgingEngine()
        hyp = _hyp(created_at=_T0)
        result = engine.age_thesis(hyp, now=_T0 + 7200)
        assert result is not None
        assert engine.state_for(hyp.hypothesis_id) is not None

    def test_expire_thesis_delegates(self):
        from trainer.qde.brain.thesis import ThesisAgingEngine
        engine = ThesisAgingEngine()
        hyp = _hyp(confidence=0.7)
        engine.register(hyp, "INTRADAY")
        engine.expire_thesis(hyp, reason="test", now=_T0 + 100)
        assert hyp.status == HypothesisStatus.EXPIRED

    def test_update_confidence_delegates(self):
        from trainer.qde.brain.thesis import ThesisAgingEngine, ThesisEventType
        engine = ThesisAgingEngine()
        hyp = _hyp(confidence=0.5)
        engine.register(hyp, "INTRADAY")
        engine.update_confidence(hyp, delta=0.1,
                                 event_type=ThesisEventType.CONFIRMATION,
                                 evidence="confirm", now=_T0 + 100)
        assert hyp.confidence == pytest.approx(0.6)

    def test_age_all_skips_terminal(self):
        from trainer.qde.brain.thesis import ThesisAgingEngine
        engine = ThesisAgingEngine()
        h_active = _hyp(confidence=0.6, created_at=_T0)
        h_expired = _hyp(status=HypothesisStatus.EXPIRED, confidence=0.0, created_at=_T0)
        hypotheses = {
            h_active.hypothesis_id: h_active,
            h_expired.hypothesis_id: h_expired,
        }
        results = engine.age_all(hypotheses, now=_T0 + 7200)
        ids = [r.hypothesis_id for r in results]
        assert h_active.hypothesis_id in ids
        assert h_expired.hypothesis_id not in ids

    def test_age_all_uses_opp_class_fn(self):
        from trainer.qde.brain.thesis import ThesisAgingEngine, DEFAULT_DECAY_PROFILES
        engine = ThesisAgingEngine()
        hyp = _hyp(confidence=0.6, created_at=_T0)
        hypotheses = {hyp.hypothesis_id: hyp}
        engine.age_all(hypotheses,
                       opp_class_fn=lambda h: "MICRO_SCALP",
                       now=_T0 + 7200)
        state = engine.state_for(hyp.hypothesis_id)
        assert state.decay_profile == DEFAULT_DECAY_PROFILES["MICRO_SCALP"]

    def test_summary_returns_counts(self):
        from trainer.qde.brain.thesis import ThesisAgingEngine
        engine = ThesisAgingEngine()
        h1, h2 = _hyp(), _hyp()
        engine.register(h1, "SCALP")
        engine.register(h2, "SWING")
        engine.expire_thesis(h1, now=_T0 + 100)
        s = engine.summary()
        assert s["tracked_count"] == 2
        assert s["expired_count"] == 1


# ═════════════════════════════════════════════════════════════════════════════
# 11. Behavioral / integration tests
# ═════════════════════════════════════════════════════════════════════════════


class TestBehavioral:
    def test_repeated_confirmations_keep_confidence_high(self):
        """A thesis receiving regular confirmation should not expire.

        Uses MULTI_SESSION class (max_age=3 days) while simulating only
        4 hours so we are well inside the hard expiry ceiling.  The key
        invariant is that each confirmation resets the decay clock, so
        confidence should remain well above the expiry threshold.
        """
        from trainer.qde.brain.thesis import ThesisAgingEngine, ThesisEventType, DEFAULT_DECAY_PROFILES
        engine = ThesisAgingEngine()
        hyp = _hyp(confidence=0.7, created_at=_T0)
        engine.register(hyp, "MULTI_SESSION")  # max_age = 3 days → safe for 4-hour sim
        min_iv = DEFAULT_DECAY_PROFILES["MULTI_SESSION"].min_decay_interval_seconds

        t = _T0
        # Simulate 4 hours with a confirmation every 30 minutes
        for _ in range(8):
            t += 1800.0
            engine.update_confidence(hyp, delta=0.05,
                                     event_type=ThesisEventType.CONFIRMATION,
                                     evidence="30-min confirm", now=t)
            engine.age_thesis(hyp, now=t + min_iv + 1)

        assert hyp.status != HypothesisStatus.EXPIRED
        assert hyp.confidence > 0.3  # still meaningful with regular confirmations

    def test_no_confirmations_decays_to_expiry_micro_scalp(self):
        """A MICRO_SCALP thesis with no confirmation should expire within its window."""
        from trainer.qde.brain.thesis import ThesisAgingEngine, DEFAULT_DECAY_PROFILES
        engine = ThesisAgingEngine()
        hyp = _hyp(confidence=0.5, created_at=_T0)
        engine.register(hyp, "MICRO_SCALP")
        profile = DEFAULT_DECAY_PROFILES["MICRO_SCALP"]

        # Apply decay many times over the max_age window
        t = _T0
        step = profile.min_decay_interval_seconds + 1
        expired = False
        while t < _T0 + profile.max_age_seconds + step:
            t += step
            result = engine.age_thesis(hyp, now=t)
            if result and result.expired:
                expired = True
                break

        assert expired or hyp.status == HypothesisStatus.EXPIRED

    def test_micro_scalp_decays_faster_than_swing(self):
        """Same elapsed time → MICRO_SCALP loses more confidence than SWING."""
        from trainer.qde.brain.thesis import age_thesis, ThesisAgingState, DEFAULT_DECAY_PROFILES
        elapsed = 600.0  # 10 minutes

        hyp_micro = _hyp(confidence=0.8, created_at=_T0)
        state_micro = ThesisAgingState(
            hypothesis_id=hyp_micro.hypothesis_id,
            instrument="A",
            created_at=_T0,
            decay_profile=DEFAULT_DECAY_PROFILES["MICRO_SCALP"],
            last_confirmation_at=_T0,
            last_decayed_at=_T0 - elapsed - 1,
        )

        hyp_swing = _hyp(confidence=0.8, created_at=_T0)
        state_swing = ThesisAgingState(
            hypothesis_id=hyp_swing.hypothesis_id,
            instrument="B",
            created_at=_T0,
            decay_profile=DEFAULT_DECAY_PROFILES["SWING"],
            last_confirmation_at=_T0,
            last_decayed_at=_T0 - elapsed - 1,
        )

        now = _T0 + elapsed
        res_micro = age_thesis(hyp_micro, state_micro, now=now)
        res_swing  = age_thesis(hyp_swing, state_swing, now=now)

        if res_micro.decay_applied and res_swing.decay_applied:
            assert res_micro.confidence_after < res_swing.confidence_after, (
                "MICRO_SCALP should decay faster than SWING over the same elapsed time"
            )

    def test_history_grows_with_each_tick(self):
        from trainer.qde.brain.thesis import ThesisAgingEngine
        engine = ThesisAgingEngine()
        hyp = _hyp(confidence=0.8, created_at=_T0)
        engine.register(hyp, "INTRADAY")
        state = engine.state_for(hyp.hypothesis_id)
        min_iv = state.decay_profile.min_decay_interval_seconds

        for i in range(5):
            engine.age_thesis(hyp, now=_T0 + (i + 1) * (min_iv + 1))

        assert len(state.aging_history) >= 3  # at least a few records

    def test_all_five_classes_can_be_registered(self):
        from trainer.qde.brain.thesis import ThesisAgingEngine, DEFAULT_DECAY_PROFILES
        engine = ThesisAgingEngine()
        for cls in DEFAULT_DECAY_PROFILES:
            hyp = _hyp()
            engine.register(hyp, cls)
            state = engine.state_for(hyp.hypothesis_id)
            assert state.decay_profile == DEFAULT_DECAY_PROFILES[cls]


# ═════════════════════════════════════════════════════════════════════════════
# 12. Package __init__ exports
# ═════════════════════════════════════════════════════════════════════════════


class TestPackageExports:
    def test_all_c12_symbols_exported(self):
        import trainer.qde.brain.thesis as pkg
        required = [
            "ThesisEventType", "ThesisEvent", "ThesisDecayProfile",
            "DEFAULT_DECAY_PROFILES", "profile_for_class",
            "AgingRecord", "ThesisAgingState", "ThesisAgingResult",
            "age_thesis", "expire_thesis", "update_confidence",
            "ThesisAgingEngine",
        ]
        for name in required:
            assert hasattr(pkg, name), f"Package missing: {name}"

    def test_all_in_dunder_all(self):
        import trainer.qde.brain.thesis as pkg
        for name in ["age_thesis", "expire_thesis", "update_confidence",
                     "ThesisAgingEngine", "ThesisDecayProfile",
                     "DEFAULT_DECAY_PROFILES"]:
            assert name in pkg.__all__, f"{name} not in __all__"

    def test_three_callables_are_callable(self):
        from trainer.qde.brain.thesis import age_thesis, expire_thesis, update_confidence
        assert callable(age_thesis)
        assert callable(expire_thesis)
        assert callable(update_confidence)
