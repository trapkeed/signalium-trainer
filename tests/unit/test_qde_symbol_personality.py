"""QDE Brain — symbol personality engine tests.

Covers continuous behavioral-profile learning (`behavioral_memory`), the
pure influence functions (`personality_profiles`), and the orchestration
facade (`symbol_personality_engine.SymbolPersonalityEngine`): trust
ramping with low sample counts, EMA-based dimension/session learning,
bounded confidence/risk/threshold/duration adjustments, working-memory
logging on notable shifts, and persistence round-tripping alongside the
distinct, untouched `SymbolPersonality` model.
"""

from __future__ import annotations

import time

from trainer.qde.brain.behavioral_memory import BehavioralMemory, BehavioralObservation
from trainer.qde.brain.brain import QDEBrain, _state_from_dict
from trainer.qde.brain.personality_profiles import (
    MIN_SAMPLES_FOR_FULL_TRUST,
    PersonalityProfile,
    confidence_adjustment,
    expected_holding_duration_seconds,
    risk_appetite_multiplier,
    setup_requirement_overrides,
)
from trainer.qde.brain.setup_lifecycle import DEFAULT_SETUP_THRESHOLDS
from trainer.qde.brain.symbol_personality_engine import SymbolPersonalityEngine

SYM = "EURUSD"


def _persistent_trend_observation() -> BehavioralObservation:
    return BehavioralObservation(
        instrument=SYM,
        trend_run_seconds=4 * 3600.0,
        pullback_depth_ratio=0.15,
        volatility_cluster_hit=False,
        liquidity_sweep_detected=False,
        session="LONDON",
        session_outcome_quality=0.7,
        reversal_detected=False,
    )


def _choppy_reversal_observation() -> BehavioralObservation:
    return BehavioralObservation(
        instrument=SYM,
        trend_run_seconds=60.0,
        pullback_depth_ratio=0.9,
        volatility_cluster_hit=True,
        liquidity_sweep_detected=True,
        session="ASIAN",
        session_outcome_quality=-0.6,
        reversal_detected=True,
    )


# ── PersonalityProfile / trust ───────────────────────────────────────────────


def test_fresh_profile_has_zero_trust_and_neutral_defaults():
    profile = PersonalityProfile(instrument=SYM)
    assert profile.trust() == 0.0
    assert profile.trend_persistence == 0.5
    assert profile.best_session() is None
    assert profile.worst_session() is None


def test_trust_ramps_in_and_caps_at_one():
    profile = PersonalityProfile(instrument=SYM, sample_count=MIN_SAMPLES_FOR_FULL_TRUST // 2)
    assert 0.0 < profile.trust() < 1.0
    profile.sample_count = MIN_SAMPLES_FOR_FULL_TRUST * 5
    assert profile.trust() == 1.0


# ── BehavioralMemory ──────────────────────────────────────────────────────────


def test_behavioral_memory_learns_persistent_trend_toward_one():
    memory = BehavioralMemory()
    profile = PersonalityProfile(instrument=SYM)
    for _ in range(60):
        memory.observe(profile, _persistent_trend_observation())
    assert profile.trend_persistence > 0.8
    assert profile.avg_pullback_depth < 0.3
    assert profile.reversal_tendency < 0.2
    assert profile.sample_count == 60


def test_behavioral_memory_learns_choppy_reversal_prone_symbol():
    memory = BehavioralMemory()
    profile = PersonalityProfile(instrument=SYM)
    for _ in range(60):
        memory.observe(profile, _choppy_reversal_observation())
    assert profile.volatility_clustering > 0.8
    assert profile.liquidity_sweep_frequency > 0.8
    assert profile.reversal_tendency > 0.8


def test_behavioral_memory_updates_session_affinity_independently():
    memory = BehavioralMemory()
    profile = PersonalityProfile(instrument=SYM)
    for _ in range(30):
        memory.observe(
            profile,
            BehavioralObservation(instrument=SYM, session="LONDON", session_outcome_quality=0.8),
        )
    for _ in range(30):
        memory.observe(
            profile,
            BehavioralObservation(instrument=SYM, session="ASIAN", session_outcome_quality=-0.8),
        )
    assert profile.best_session() == "LONDON"
    assert profile.worst_session() == "ASIAN"
    assert profile.session_affinity["LONDON"] > profile.session_affinity["ASIAN"]


def test_behavioral_memory_no_op_observation_does_not_increment_sample_count():
    memory = BehavioralMemory()
    profile = PersonalityProfile(instrument=SYM)
    memory.observe(profile, BehavioralObservation(instrument=SYM))
    assert profile.sample_count == 0


# ── Pure influence functions ─────────────────────────────────────────────────


def test_confidence_adjustment_is_noop_for_untrusted_profile():
    profile = PersonalityProfile(instrument=SYM, trend_persistence=1.0, reversal_tendency=0.0)
    assert confidence_adjustment(profile, 0.5) == 0.5


def test_confidence_adjustment_boosts_trend_persistent_symbol_once_trusted():
    profile = PersonalityProfile(
        instrument=SYM,
        trend_persistence=1.0,
        reversal_tendency=0.0,
        sample_count=MIN_SAMPLES_FOR_FULL_TRUST,
    )
    assert confidence_adjustment(profile, 0.5) > 0.5


def test_confidence_adjustment_penalizes_reversal_prone_symbol_once_trusted():
    profile = PersonalityProfile(
        instrument=SYM,
        trend_persistence=0.0,
        reversal_tendency=1.0,
        sample_count=MIN_SAMPLES_FOR_FULL_TRUST,
    )
    assert confidence_adjustment(profile, 0.5) < 0.5


def test_confidence_adjustment_never_leaves_zero_one_bounds():
    profile = PersonalityProfile(
        instrument=SYM,
        trend_persistence=1.0,
        reversal_tendency=0.0,
        sample_count=MIN_SAMPLES_FOR_FULL_TRUST,
    )
    assert confidence_adjustment(profile, 0.99) <= 1.0
    profile.trend_persistence, profile.reversal_tendency = 0.0, 1.0
    assert confidence_adjustment(profile, 0.01) >= 0.0


def test_risk_appetite_multiplier_reduces_for_choppy_symbol():
    profile = PersonalityProfile(
        instrument=SYM,
        volatility_clustering=0.9,
        liquidity_sweep_frequency=0.9,
        news_sensitivity=0.9,
        trend_persistence=0.1,
        sample_count=MIN_SAMPLES_FOR_FULL_TRUST,
    )
    assert risk_appetite_multiplier(profile) < 1.0


def test_risk_appetite_multiplier_stays_within_bounds():
    profile = PersonalityProfile(
        instrument=SYM,
        volatility_clustering=1.0,
        liquidity_sweep_frequency=1.0,
        news_sensitivity=1.0,
        sample_count=MIN_SAMPLES_FOR_FULL_TRUST,
    )
    assert 0.5 <= risk_appetite_multiplier(profile) <= 1.25


def test_setup_requirement_overrides_noop_below_minimum_trust():
    profile = PersonalityProfile(instrument=SYM, sample_count=0)
    thresholds = setup_requirement_overrides(profile, DEFAULT_SETUP_THRESHOLDS)
    assert thresholds == DEFAULT_SETUP_THRESHOLDS


def test_setup_requirement_overrides_raises_bar_for_choppy_symbol():
    profile = PersonalityProfile(
        instrument=SYM,
        liquidity_sweep_frequency=0.9,
        reversal_tendency=0.9,
        trend_persistence=0.1,
        sample_count=MIN_SAMPLES_FOR_FULL_TRUST,
    )
    thresholds = setup_requirement_overrides(profile, DEFAULT_SETUP_THRESHOLDS)
    assert thresholds.min_confirmation_ticks >= DEFAULT_SETUP_THRESHOLDS.min_confirmation_ticks
    assert thresholds.ready_min_conviction >= DEFAULT_SETUP_THRESHOLDS.ready_min_conviction


def test_setup_requirement_overrides_never_drops_confirmation_below_one():
    profile = PersonalityProfile(
        instrument=SYM,
        liquidity_sweep_frequency=0.0,
        reversal_tendency=0.0,
        trend_persistence=1.0,
        sample_count=MIN_SAMPLES_FOR_FULL_TRUST,
    )
    thresholds = setup_requirement_overrides(profile, DEFAULT_SETUP_THRESHOLDS)
    assert thresholds.min_confirmation_ticks >= 1


def test_expected_holding_duration_scales_within_bounds():
    profile = PersonalityProfile(
        instrument=SYM,
        trend_persistence=1.0,
        avg_pullback_depth=0.0,
        sample_count=MIN_SAMPLES_FOR_FULL_TRUST,
    )
    scaled = expected_holding_duration_seconds(profile, 1000.0)
    assert 1000.0 < scaled <= 2000.0

    profile.trend_persistence, profile.avg_pullback_depth = 0.0, 1.0
    scaled = expected_holding_duration_seconds(profile, 1000.0)
    assert 500.0 <= scaled < 1000.0


# ── SymbolPersonalityEngine orchestration ────────────────────────────────────


def test_engine_observe_updates_and_returns_profile_owned_by_brain_state():
    brain = QDEBrain()
    engine = SymbolPersonalityEngine(brain)

    for _ in range(30):
        engine.observe(SYM, _persistent_trend_observation())

    profile = engine.profile_for(SYM)
    assert profile is brain.get_state(SYM).behavioral_profile
    assert profile.trend_persistence > 0.5
    assert profile.sample_count == 30


def test_engine_influence_seams_delegate_to_pure_functions():
    brain = QDEBrain()
    engine = SymbolPersonalityEngine(brain)
    for _ in range(30):
        engine.observe(SYM, _persistent_trend_observation())

    profile = engine.profile_for(SYM)
    assert engine.adjust_confidence(SYM, 0.5) == confidence_adjustment(profile, 0.5)
    assert engine.risk_appetite_multiplier(SYM) == risk_appetite_multiplier(profile)
    assert engine.setup_thresholds_for(SYM) == setup_requirement_overrides(
        profile, DEFAULT_SETUP_THRESHOLDS
    )
    assert engine.expected_holding_duration(SYM, 500.0) == expected_holding_duration_seconds(
        profile, 500.0
    )


def test_engine_logs_notable_shift_to_working_memory():
    brain = QDEBrain()
    engine = SymbolPersonalityEngine(brain)
    state = brain.get_state(SYM)

    # A single strong, opposite-of-neutral observation on a fresh profile
    # (alpha=0.12 default) produces a delta well above the notable-shift
    # threshold on at least one tracked dimension.
    engine.observe(SYM, _choppy_reversal_observation())

    kinds = [item.kind for item in state.working_memory]
    assert "PERSONALITY_SHIFT" in kinds


def test_engine_does_not_log_shift_for_negligible_change():
    brain = QDEBrain()
    engine = SymbolPersonalityEngine(brain)
    state = brain.get_state(SYM)

    # An observation matching the neutral defaults produces ~0 delta.
    engine.observe(
        SYM,
        BehavioralObservation(
            instrument=SYM,
            trend_run_seconds=0.0,
            pullback_depth_ratio=0.5,
            volatility_cluster_hit=False,
        ),
    )
    kinds = [item.kind for item in state.working_memory]
    assert "PERSONALITY_SHIFT" not in kinds or state.behavioral_profile.sample_count <= 1


def test_engine_scoped_per_symbol():
    brain = QDEBrain()
    engine = SymbolPersonalityEngine(brain)
    for _ in range(30):
        engine.observe("EURUSD", _persistent_trend_observation())
    for _ in range(30):
        engine.observe(
            "GBPUSD",
            BehavioralObservation(instrument="GBPUSD", trend_run_seconds=60.0),
        )

    eur_profile = engine.profile_for("EURUSD")
    gbp_profile = engine.profile_for("GBPUSD")
    assert eur_profile.trend_persistence != gbp_profile.trend_persistence


def test_behavioral_profile_is_distinct_from_symbol_personality():
    brain = QDEBrain()
    state = brain.get_state(SYM)
    assert state.personality is not state.behavioral_profile
    assert not hasattr(state.personality, "trend_persistence")
    assert not hasattr(state.behavioral_profile, "traits")


def test_behavioral_profile_round_trips_through_persistence():
    brain = QDEBrain()
    engine = SymbolPersonalityEngine(brain)
    for _ in range(25):
        engine.observe(SYM, _persistent_trend_observation())

    state = brain.get_state(SYM)
    raw = state.to_dict()
    assert "behavioral_profile" in raw

    restored = _state_from_dict(raw)
    assert restored.behavioral_profile.trend_persistence == round(
        state.behavioral_profile.trend_persistence, 4
    )
    assert restored.behavioral_profile.sample_count == state.behavioral_profile.sample_count
    assert restored.behavioral_profile.session_affinity == {
        k: round(v, 4) for k, v in state.behavioral_profile.session_affinity.items()
    }


def test_state_without_behavioral_profile_field_defaults_gracefully():
    # Simulates a persisted state written before this feature existed.
    raw = {
        "instrument": SYM,
        "working_memory": [],
        "long_term_memory": [],
        "hypotheses": {},
        "opportunities": {},
        "narrative": None,
        "belief": None,
        "personality": {},
        "attention": {},
        "tick_count": 0,
        "created_at": time.time(),
        "last_updated_at": time.time(),
    }
    restored = _state_from_dict(raw)
    assert restored.behavioral_profile.trend_persistence == 0.5
    assert restored.behavioral_profile.sample_count == 0
