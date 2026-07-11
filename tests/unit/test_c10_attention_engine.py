"""C10 — Attention Engine tests.

Covers:
  - AttentionTier classification (tier_from_composite)
  - ResourceBudget lookup per tier (TIER_RESOURCES)
  - Signal enrichment: 4 new factors (volume_anomaly, key_level_proximity,
    spread_change, failed_move)
  - OpportunitySignals backward-compat: old 6-field construction still works
  - AttentionWeights.score() includes 4 new fields when non-zero
  - C10_WEIGHTS computes correct composite contributions
  - AttentionMemory: record, history, tier_transitions, current_tier
  - AttentionMemory: tier-transition events only on tier changes
  - AttentionMemory: evolution_summary
  - AttentionAPI: calculate_attention → AttentionSnapshot
  - AttentionAPI: rank_symbols ordering
  - AttentionAPI: allocate_resources per tier
  - AttentionAPI: attention_history, tier_transitions via memory
  - AttentionAPI: high_attention_symbols
  - AttentionAPI: drop lifecycle
  - brain.__init__ re-exports all C10 symbols
"""

from __future__ import annotations

import time
import math
import sys

import pytest

# ── Sys-modules pre-registration for isolated test runs ───────────────────────
# (pattern from mie-test-isolation.md)
try:
    import loguru  # noqa: F401
except ImportError:
    import logging
    sys.modules.setdefault("loguru", type(sys)("loguru"))
    sys.modules["loguru"].logger = logging.getLogger("loguru")  # type: ignore


# ── Helpers ───────────────────────────────────────────────────────────────────


def _obs(instrument: str = "EURUSD", **kwargs):
    """Build a minimal SymbolObservationInput for testing."""
    from trainer.qde.qde_observer_models import SymbolObservationInput
    from trainer.lifecycle.lifecycle_models import MarketContext

    market = MarketContext(
        momentum=kwargs.pop("momentum", 0.0),
        trend_valid=kwargs.pop("trend_valid", True),
        news_risk=kwargs.pop("news_risk", 0.0),
        session=kwargs.pop("session", "LONDON"),
    )
    return SymbolObservationInput(
        instrument=instrument,
        market=market,
        volatility_state=kwargs.pop("volatility_state", "NORMAL"),
        volatility_quality=kwargs.pop("volatility_quality", 0.5),
        liquidity_state=kwargs.pop("liquidity_state", "NORMAL"),
        liquidity_quality=kwargs.pop("liquidity_quality", 0.5),
        session_quality=kwargs.pop("session_quality", 0.6),
        market_regime=kwargs.pop("market_regime", "RANGE"),
        microstructure_quality=kwargs.pop("microstructure_quality", 0.6),
        **kwargs,
    )


def _make_brain_and_engine(instrument: str = "EURUSD"):
    """Create a minimal QDEBrain and attached AttentionEngine for testing."""
    from trainer.qde.brain.brain import QDEBrain
    from trainer.qde.brain.attention_engine import AttentionEngine

    brain = QDEBrain()
    engine = AttentionEngine(brain=brain)
    # Seed the brain state for the instrument.
    _ = brain.get_state(instrument)
    return brain, engine


# ═════════════════════════════════════════════════════════════════════════════
# 1. AttentionTier classification
# ═════════════════════════════════════════════════════════════════════════════


class TestAttentionTierClassification:
    def test_watch_at_zero(self):
        from trainer.qde.brain.attention import tier_from_composite, AttentionTier
        assert tier_from_composite(0.0) == AttentionTier.WATCH

    def test_watch_below_threshold(self):
        from trainer.qde.brain.attention import tier_from_composite, AttentionTier
        assert tier_from_composite(0.24) == AttentionTier.WATCH

    def test_observe_at_threshold(self):
        from trainer.qde.brain.attention import tier_from_composite, AttentionTier
        assert tier_from_composite(0.25) == AttentionTier.OBSERVE

    def test_observe_mid(self):
        from trainer.qde.brain.attention import tier_from_composite, AttentionTier
        assert tier_from_composite(0.40) == AttentionTier.OBSERVE

    def test_high_attention_at_threshold(self):
        from trainer.qde.brain.attention import tier_from_composite, AttentionTier
        assert tier_from_composite(0.50) == AttentionTier.HIGH_ATTENTION

    def test_high_attention_mid(self):
        from trainer.qde.brain.attention import tier_from_composite, AttentionTier
        assert tier_from_composite(0.65) == AttentionTier.HIGH_ATTENTION

    def test_critical_at_threshold(self):
        from trainer.qde.brain.attention import tier_from_composite, AttentionTier
        assert tier_from_composite(0.75) == AttentionTier.CRITICAL_ATTENTION

    def test_critical_at_one(self):
        from trainer.qde.brain.attention import tier_from_composite, AttentionTier
        assert tier_from_composite(1.0) == AttentionTier.CRITICAL_ATTENTION

    def test_all_tiers_distinct(self):
        from trainer.qde.brain.attention import AttentionTier
        tiers = list(AttentionTier)
        assert len(tiers) == 4

    def test_tier_enum_values(self):
        from trainer.qde.brain.attention import AttentionTier
        assert AttentionTier.WATCH.value == "WATCH"
        assert AttentionTier.OBSERVE.value == "OBSERVE"
        assert AttentionTier.HIGH_ATTENTION.value == "HIGH_ATTENTION"
        assert AttentionTier.CRITICAL_ATTENTION.value == "CRITICAL_ATTENTION"


# ═════════════════════════════════════════════════════════════════════════════
# 2. ResourceBudget per tier
# ═════════════════════════════════════════════════════════════════════════════


class TestResourceBudget:
    def test_watch_has_largest_intervals(self):
        from trainer.qde.brain.attention import TIER_RESOURCES, AttentionTier
        watch = TIER_RESOURCES[AttentionTier.WATCH]
        critical = TIER_RESOURCES[AttentionTier.CRITICAL_ATTENTION]
        assert watch.observation_interval_s > critical.observation_interval_s

    def test_critical_enables_every_tick_observation(self):
        from trainer.qde.brain.attention import TIER_RESOURCES, AttentionTier
        critical = TIER_RESOURCES[AttentionTier.CRITICAL_ATTENTION]
        assert critical.observation_interval_s == 0.0

    def test_critical_enables_every_tick_inference(self):
        from trainer.qde.brain.attention import TIER_RESOURCES, AttentionTier
        critical = TIER_RESOURCES[AttentionTier.CRITICAL_ATTENTION]
        assert critical.inference_interval_s == 0.0

    def test_critical_has_highest_reasoning_rate(self):
        from trainer.qde.brain.attention import TIER_RESOURCES, AttentionTier
        tiers = [
            AttentionTier.WATCH,
            AttentionTier.OBSERVE,
            AttentionTier.HIGH_ATTENTION,
            AttentionTier.CRITICAL_ATTENTION,
        ]
        rates = [TIER_RESOURCES[t].reasoning_cycles_per_hour for t in tiers]
        assert rates == sorted(rates), "reasoning rate should increase with tier"

    def test_resource_budget_to_dict(self):
        from trainer.qde.brain.attention import TIER_RESOURCES, AttentionTier
        d = TIER_RESOURCES[AttentionTier.OBSERVE].to_dict()
        assert "observation_interval_s" in d
        assert "inference_interval_s" in d
        assert "reasoning_cycles_per_hour" in d
        assert "reasoning_interval_s" in d

    def test_reasoning_interval_s_inverse_of_rate(self):
        from trainer.qde.brain.attention import TIER_RESOURCES, AttentionTier
        budget = TIER_RESOURCES[AttentionTier.OBSERVE]
        expected = 3600.0 / budget.reasoning_cycles_per_hour
        assert abs(budget.reasoning_interval_s - expected) < 0.01

    def test_watch_reasoning_interval_is_hour(self):
        from trainer.qde.brain.attention import TIER_RESOURCES, AttentionTier
        budget = TIER_RESOURCES[AttentionTier.WATCH]
        assert budget.reasoning_interval_s == 3600.0

    def test_all_four_tiers_have_budgets(self):
        from trainer.qde.brain.attention import TIER_RESOURCES, AttentionTier
        for tier in AttentionTier:
            assert tier in TIER_RESOURCES


# ═════════════════════════════════════════════════════════════════════════════
# 3. OpportunitySignals backward compat + 4 new fields
# ═════════════════════════════════════════════════════════════════════════════


class TestOpportunitySignalsBackwardCompat:
    def test_old_six_field_construction_still_works(self):
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        s = OpportunitySignals(
            hypothesis_activity=0.5,
            volatility_expansion=0.3,
            liquidity_event=0.2,
            news_impact=0.1,
            unusual_behavior=0.4,
            session_transition=0.0,
        )
        # New fields default to 0.0
        assert s.volume_anomaly == 0.0
        assert s.key_level_proximity == 0.0
        assert s.spread_change == 0.0
        assert s.failed_move == 0.0

    def test_to_dict_includes_four_new_fields(self):
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        s = OpportunitySignals()
        d = s.to_dict()
        for key in ("volume_anomaly", "key_level_proximity", "spread_change", "failed_move"):
            assert key in d

    def test_max_signal_uses_all_ten(self):
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        s = OpportunitySignals(key_level_proximity=0.95)
        assert s.max_signal() == pytest.approx(0.95)

    def test_default_construction_has_all_zeros(self):
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        s = OpportunitySignals()
        assert s.max_signal() == 0.0


# ═════════════════════════════════════════════════════════════════════════════
# 4. Signal enricher — per-factor detection
# ═════════════════════════════════════════════════════════════════════════════


class TestSignalEnricher:
    def test_no_context_preserves_base_signals(self):
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        from trainer.qde.brain.attention import enrich_signals, ExtendedSignalContext
        base = OpportunitySignals(hypothesis_activity=0.8, volatility_expansion=0.4)
        ctx = ExtendedSignalContext()
        enriched = enrich_signals(base, ctx)
        assert enriched.hypothesis_activity == pytest.approx(0.8)
        assert enriched.volatility_expansion == pytest.approx(0.4)

    def test_volume_anomaly_high_ratio(self):
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        from trainer.qde.brain.attention import enrich_signals, ExtendedSignalContext
        ctx = ExtendedSignalContext(volume_ratio=3.0)
        enriched = enrich_signals(OpportunitySignals(), ctx)
        assert enriched.volume_anomaly == pytest.approx(1.0)

    def test_volume_anomaly_normal(self):
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        from trainer.qde.brain.attention import enrich_signals, ExtendedSignalContext
        ctx = ExtendedSignalContext(volume_ratio=1.0)
        enriched = enrich_signals(OpportunitySignals(), ctx)
        assert enriched.volume_anomaly == pytest.approx(0.0)

    def test_volume_anomaly_moderate(self):
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        from trainer.qde.brain.attention import enrich_signals, ExtendedSignalContext
        ctx = ExtendedSignalContext(volume_ratio=1.75)  # between 1.5 and 2.0
        enriched = enrich_signals(OpportunitySignals(), ctx)
        assert 0.5 < enriched.volume_anomaly < 1.0

    def test_volume_anomaly_below_baseline(self):
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        from trainer.qde.brain.attention import enrich_signals, ExtendedSignalContext
        ctx = ExtendedSignalContext(volume_ratio=0.1)  # very thin
        enriched = enrich_signals(OpportunitySignals(), ctx)
        assert enriched.volume_anomaly > 0.0

    def test_key_level_proximity_very_close(self):
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        from trainer.qde.brain.attention import enrich_signals, ExtendedSignalContext
        # Price within 0.05% of level
        ctx = ExtendedSignalContext(
            current_price=1.1000,
            nearest_level_price=1.1001,  # 0.009% away
        )
        enriched = enrich_signals(OpportunitySignals(), ctx)
        assert enriched.key_level_proximity >= 0.9

    def test_key_level_proximity_far(self):
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        from trainer.qde.brain.attention import enrich_signals, ExtendedSignalContext
        ctx = ExtendedSignalContext(
            current_price=1.1000,
            nearest_level_price=1.1500,  # >4% away
        )
        enriched = enrich_signals(OpportunitySignals(), ctx)
        assert enriched.key_level_proximity == pytest.approx(0.0)

    def test_key_level_proximity_no_data(self):
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        from trainer.qde.brain.attention import enrich_signals, ExtendedSignalContext
        ctx = ExtendedSignalContext(current_price=0.0, nearest_level_price=0.0)
        enriched = enrich_signals(OpportunitySignals(), ctx)
        assert enriched.key_level_proximity == pytest.approx(0.0)

    def test_spread_change_spike(self):
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        from trainer.qde.brain.attention import enrich_signals, ExtendedSignalContext
        ctx = ExtendedSignalContext(spread_normalized=0.9, prior_spread_normalized=0.1)
        enriched = enrich_signals(OpportunitySignals(), ctx)
        assert enriched.spread_change == pytest.approx(1.0)

    def test_spread_change_stable(self):
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        from trainer.qde.brain.attention import enrich_signals, ExtendedSignalContext
        ctx = ExtendedSignalContext(spread_normalized=0.5, prior_spread_normalized=0.5)
        enriched = enrich_signals(OpportunitySignals(), ctx)
        assert enriched.spread_change == pytest.approx(0.0)

    def test_spread_change_notable(self):
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        from trainer.qde.brain.attention import enrich_signals, ExtendedSignalContext
        ctx = ExtendedSignalContext(spread_normalized=0.7, prior_spread_normalized=0.5)
        # delta = 0.20 → between 0.15 and 0.30
        enriched = enrich_signals(OpportunitySignals(), ctx)
        assert 0.5 <= enriched.spread_change <= 1.0

    def test_failed_move_with_breakout(self):
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        from trainer.qde.brain.attention import enrich_signals, ExtendedSignalContext
        ctx = ExtendedSignalContext(
            recent_failed_breakout=True,
            failed_move_wick_ratio=2.5,
        )
        enriched = enrich_signals(OpportunitySignals(), ctx)
        assert enriched.failed_move == pytest.approx(1.0)

    def test_failed_move_without_breakout(self):
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        from trainer.qde.brain.attention import enrich_signals, ExtendedSignalContext
        ctx = ExtendedSignalContext(recent_failed_breakout=False, failed_move_wick_ratio=5.0)
        enriched = enrich_signals(OpportunitySignals(), ctx)
        assert enriched.failed_move == pytest.approx(0.0)

    def test_failed_move_mild_breakout(self):
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        from trainer.qde.brain.attention import enrich_signals, ExtendedSignalContext
        ctx = ExtendedSignalContext(recent_failed_breakout=True, failed_move_wick_ratio=0.5)
        enriched = enrich_signals(OpportunitySignals(), ctx)
        assert enriched.failed_move == pytest.approx(0.4)  # mild flat signal

    def test_enrich_preserves_original_six_unchanged(self):
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        from trainer.qde.brain.attention import enrich_signals, ExtendedSignalContext
        base = OpportunitySignals(
            hypothesis_activity=0.7,
            volatility_expansion=0.5,
            liquidity_event=0.3,
            news_impact=0.2,
            unusual_behavior=0.1,
            session_transition=0.6,
        )
        ctx = ExtendedSignalContext(volume_ratio=3.0, recent_failed_breakout=True)
        enriched = enrich_signals(base, ctx)
        assert enriched.hypothesis_activity == pytest.approx(0.7)
        assert enriched.volatility_expansion == pytest.approx(0.5)
        assert enriched.liquidity_event == pytest.approx(0.3)
        assert enriched.news_impact == pytest.approx(0.2)
        assert enriched.unusual_behavior == pytest.approx(0.1)
        assert enriched.session_transition == pytest.approx(0.6)


# ═════════════════════════════════════════════════════════════════════════════
# 5. AttentionWeights with C10 factors
# ═════════════════════════════════════════════════════════════════════════════


class TestAttentionWeights:
    def test_default_weights_ignore_c10_fields(self):
        from trainer.qde.brain.attention_allocator import DEFAULT_WEIGHTS
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        s1 = OpportunitySignals(hypothesis_activity=0.5)
        s2 = OpportunitySignals(
            hypothesis_activity=0.5,
            volume_anomaly=1.0,
            key_level_proximity=1.0,
        )
        # Default weights have 0.0 for new fields → same score
        assert DEFAULT_WEIGHTS.score(s1) == pytest.approx(DEFAULT_WEIGHTS.score(s2))

    def test_c10_weights_include_new_signals(self):
        from trainer.qde.brain.attention_allocator import C10_WEIGHTS
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        base = OpportunitySignals(hypothesis_activity=0.5)
        enriched = OpportunitySignals(
            hypothesis_activity=0.5,
            volume_anomaly=1.0,
            key_level_proximity=1.0,
        )
        assert C10_WEIGHTS.score(enriched) > C10_WEIGHTS.score(base)

    def test_c10_weights_fields_sum_reasonable(self):
        from trainer.qde.brain.attention_allocator import C10_WEIGHTS
        total = (
            C10_WEIGHTS.hypothesis_activity
            + C10_WEIGHTS.volatility_expansion
            + C10_WEIGHTS.liquidity_event
            + C10_WEIGHTS.news_impact
            + C10_WEIGHTS.unusual_behavior
            + C10_WEIGHTS.session_transition
            + C10_WEIGHTS.volume_anomaly
            + C10_WEIGHTS.key_level_proximity
            + C10_WEIGHTS.spread_change
            + C10_WEIGHTS.failed_move
        )
        assert 0.8 <= total <= 1.2, f"C10_WEIGHTS total should be near 1.0, got {total}"

    def test_score_clamped_at_one_in_allocator(self):
        from trainer.qde.brain.attention_allocator import (
            AttentionScoreAllocator,
            C10_WEIGHTS,
        )
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        # All signals at 1.0 should clamp to composite ≤ 1.0
        all_max = OpportunitySignals(
            hypothesis_activity=1.0,
            volatility_expansion=1.0,
            liquidity_event=1.0,
            news_impact=1.0,
            unusual_behavior=1.0,
            session_transition=1.0,
            volume_anomaly=1.0,
            key_level_proximity=1.0,
            spread_change=1.0,
            failed_move=1.0,
        )
        allocator = AttentionScoreAllocator(weights=C10_WEIGHTS)
        score = allocator.update("EURUSD", all_max, now=time.time())
        assert score.composite <= 1.0


# ═════════════════════════════════════════════════════════════════════════════
# 6. AttentionMemory
# ═════════════════════════════════════════════════════════════════════════════


class TestAttentionMemory:
    def _snapshot(self, instrument: str, tier_name: str, composite: float) -> object:
        from trainer.qde.brain.attention import (
            AttentionSnapshot, AttentionTier, TIER_RESOURCES, tier_from_composite
        )
        tier = AttentionTier(tier_name)
        return AttentionSnapshot(
            instrument=instrument,
            tier=tier,
            composite=composite,
            weight=composite,
            signals={},
            resources=TIER_RESOURCES[tier],
        )

    def test_record_returns_none_on_same_tier(self):
        from trainer.qde.brain.attention import AttentionMemory
        mem = AttentionMemory()
        s1 = self._snapshot("EURUSD", "WATCH", 0.1)
        s2 = self._snapshot("EURUSD", "WATCH", 0.12)
        assert mem.record(s1) is not None  # first snapshot always produces a record
        assert mem.record(s2) is None      # same tier → no transition

    def test_record_returns_event_on_tier_change(self):
        from trainer.qde.brain.attention import AttentionMemory
        mem = AttentionMemory()
        s1 = self._snapshot("EURUSD", "WATCH", 0.1)
        s2 = self._snapshot("EURUSD", "OBSERVE", 0.30)
        mem.record(s1)
        event = mem.record(s2)
        assert event is not None
        assert event.from_tier.value == "WATCH"
        assert event.to_tier.value == "OBSERVE"

    def test_first_record_has_none_from_tier(self):
        from trainer.qde.brain.attention import AttentionMemory
        mem = AttentionMemory()
        s = self._snapshot("GBPUSD", "HIGH_ATTENTION", 0.6)
        event = mem.record(s)
        assert event is not None
        assert event.from_tier is None

    def test_history_returns_n_most_recent(self):
        from trainer.qde.brain.attention import AttentionMemory
        mem = AttentionMemory()
        for i in range(10):
            mem.record(self._snapshot("EURUSD", "WATCH", 0.05))
        hist = mem.history("EURUSD", n=5)
        assert len(hist) == 5

    def test_history_empty_for_unknown_instrument(self):
        from trainer.qde.brain.attention import AttentionMemory
        mem = AttentionMemory()
        assert mem.history("UNKNOWN") == []

    def test_current_tier_tracks_latest(self):
        from trainer.qde.brain.attention import AttentionMemory, AttentionTier
        mem = AttentionMemory()
        mem.record(self._snapshot("EURUSD", "WATCH", 0.1))
        mem.record(self._snapshot("EURUSD", "OBSERVE", 0.3))
        mem.record(self._snapshot("EURUSD", "HIGH_ATTENTION", 0.6))
        assert mem.current_tier("EURUSD") == AttentionTier.HIGH_ATTENTION

    def test_current_tier_none_for_unseen(self):
        from trainer.qde.brain.attention import AttentionMemory
        mem = AttentionMemory()
        assert mem.current_tier("EURUSD") is None

    def test_tier_transitions_returns_changes_only(self):
        from trainer.qde.brain.attention import AttentionMemory
        mem = AttentionMemory()
        # 3 WATCH snapshots → only 1 transition (initial)
        for _ in range(3):
            mem.record(self._snapshot("EURUSD", "WATCH", 0.1))
        # 1 OBSERVE snapshot → second transition
        mem.record(self._snapshot("EURUSD", "OBSERVE", 0.3))
        transitions = mem.tier_transitions("EURUSD")
        assert len(transitions) == 2  # initial + WATCH→OBSERVE

    def test_evolution_summary_tracked(self):
        from trainer.qde.brain.attention import AttentionMemory
        mem = AttentionMemory()
        mem.record(self._snapshot("EURUSD", "WATCH", 0.1))
        summary = mem.evolution_summary("EURUSD")
        assert summary["tracked"] is True
        assert summary["current_tier"] == "WATCH"
        assert summary["snapshots_stored"] == 1

    def test_evolution_summary_untracked(self):
        from trainer.qde.brain.attention import AttentionMemory
        mem = AttentionMemory()
        summary = mem.evolution_summary("UNKNOWN")
        assert summary["tracked"] is False

    def test_drop_clears_state(self):
        from trainer.qde.brain.attention import AttentionMemory
        mem = AttentionMemory()
        mem.record(self._snapshot("EURUSD", "WATCH", 0.1))
        mem.drop("EURUSD")
        assert mem.current_tier("EURUSD") is None
        assert mem.history("EURUSD") == []

    def test_bounded_snapshot_depth(self):
        from trainer.qde.brain.attention import AttentionMemory
        mem = AttentionMemory(max_snapshots=5)
        for _ in range(10):
            mem.record(self._snapshot("EURUSD", "WATCH", 0.1))
        assert len(mem.history("EURUSD")) == 5

    def test_time_in_current_tier(self):
        from trainer.qde.brain.attention import AttentionMemory
        mem = AttentionMemory()
        t0 = time.time() - 60.0
        snap = self._snapshot("EURUSD", "WATCH", 0.1)
        snap.captured_at = t0  # type: ignore[attr-defined]  # dataclass not frozen
        mem.record(snap)
        elapsed = mem.time_in_current_tier("EURUSD")
        assert elapsed >= 50.0   # at least ~60s have passed

    def test_dominant_signal_in_transition(self):
        from trainer.qde.brain.attention import AttentionMemory, TIER_RESOURCES, AttentionTier
        from trainer.qde.brain.attention import AttentionSnapshot
        mem = AttentionMemory()
        tier = AttentionTier.HIGH_ATTENTION
        snap = AttentionSnapshot(
            instrument="XAUUSD",
            tier=tier,
            composite=0.65,
            weight=0.5,
            signals={"volume_anomaly": 0.9, "hypothesis_activity": 0.2},
            resources=TIER_RESOURCES[tier],
        )
        event = mem.record(snap)
        assert event is not None
        assert event.dominant_signal == "volume_anomaly"


# ═════════════════════════════════════════════════════════════════════════════
# 7. AttentionAPI — integration-level tests
# ═════════════════════════════════════════════════════════════════════════════


class TestAttentionAPI:
    def _setup(self, instrument: str = "EURUSD"):
        from trainer.qde.brain.attention import AttentionAPI
        brain, engine = _make_brain_and_engine(instrument)
        api = AttentionAPI(engine=engine)
        return api, engine, brain

    def test_calculate_attention_returns_snapshot(self):
        from trainer.qde.brain.attention import AttentionSnapshot
        api, _, _ = self._setup()
        obs = _obs("EURUSD")
        snapshot = api.calculate_attention(obs)
        assert isinstance(snapshot, AttentionSnapshot)
        assert snapshot.instrument == "EURUSD"
        assert 0.0 <= snapshot.composite <= 1.0
        assert snapshot.tier is not None

    def test_calculate_attention_tier_assigned(self):
        from trainer.qde.brain.attention import AttentionTier
        api, _, _ = self._setup()
        snapshot = api.calculate_attention(_obs("EURUSD"))
        assert snapshot.tier in list(AttentionTier)

    def test_calculate_attention_resources_non_null(self):
        from trainer.qde.brain.attention import ResourceBudget
        api, _, _ = self._setup()
        snap = api.calculate_attention(_obs("EURUSD"))
        assert isinstance(snap.resources, ResourceBudget)

    def test_calculate_attention_with_extended_context(self):
        from trainer.qde.brain.attention import ExtendedSignalContext
        api, _, _ = self._setup()
        ctx = ExtendedSignalContext(
            volume_ratio=3.0,
            spread_normalized=0.9,
            prior_spread_normalized=0.1,
            nearest_level_price=1.1001,
            current_price=1.1000,
            recent_failed_breakout=True,
            failed_move_wick_ratio=2.5,
        )
        snap = api.calculate_attention(_obs("EURUSD"), extended_context=ctx)
        # Signals dict should include the C10 factors
        assert "volume_anomaly" in snap.signals
        assert "key_level_proximity" in snap.signals
        assert "spread_change" in snap.signals
        assert "failed_move" in snap.signals

    def test_rank_symbols_returns_list(self):
        api, _, _ = self._setup()
        api.calculate_attention(_obs("EURUSD"))
        api.calculate_attention(_obs("GBPUSD"))
        ranked = api.rank_symbols()
        assert len(ranked) == 2

    def test_rank_symbols_ordered_by_weight(self):
        api, _, _ = self._setup()
        api.calculate_attention(_obs("EURUSD"))
        api.calculate_attention(_obs("GBPUSD"))
        api.rebalance()
        ranked = api.rank_symbols()
        weights = [r.weight for r in ranked]
        assert weights == sorted(weights, reverse=True)

    def test_rank_symbols_rank_field_sequential(self):
        api, _, _ = self._setup()
        api.calculate_attention(_obs("EURUSD"))
        api.calculate_attention(_obs("GBPUSD"))
        ranked = api.rank_symbols()
        assert [r.rank for r in ranked] == list(range(1, len(ranked) + 1))

    def test_rank_symbols_empty_before_observations(self):
        api, _, _ = self._setup()
        assert api.rank_symbols() == []

    def test_allocate_resources_known_symbol(self):
        from trainer.qde.brain.attention import ResourceBudget
        api, _, _ = self._setup()
        api.calculate_attention(_obs("EURUSD"))
        budget = api.allocate_resources("EURUSD")
        assert isinstance(budget, ResourceBudget)

    def test_allocate_resources_unknown_symbol_returns_watch(self):
        from trainer.qde.brain.attention import ResourceBudget, AttentionTier, TIER_RESOURCES
        api, _, _ = self._setup()
        budget = api.allocate_resources("UNKNOWN")
        assert budget == TIER_RESOURCES[AttentionTier.WATCH]

    def test_allocate_resources_critical_has_zero_observation_interval(self):
        from trainer.qde.brain.attention import (
            AttentionAPI, AttentionTier, TIER_RESOURCES, ExtendedSignalContext
        )
        api, _, _ = self._setup()
        # Force critical tier by injecting all-max signals
        from trainer.qde.brain.attention_allocator import AttentionScoreAllocator, C10_WEIGHTS
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        all_max = OpportunitySignals(
            hypothesis_activity=1.0, volatility_expansion=1.0,
            liquidity_event=1.0, news_impact=1.0,
            unusual_behavior=1.0, session_transition=1.0,
            volume_anomaly=1.0, key_level_proximity=1.0,
            spread_change=1.0, failed_move=1.0,
        )
        # Directly update allocator to force CRITICAL composite
        api._engine._allocator.update("EURUSD", all_max, now=time.time())
        api._engine._allocator.reallocate()
        budget = api.allocate_resources("EURUSD")
        assert budget == TIER_RESOURCES[AttentionTier.CRITICAL_ATTENTION]

    def test_attention_history_stored_in_memory(self):
        api, _, _ = self._setup()
        for _ in range(5):
            api.calculate_attention(_obs("EURUSD"))
        hist = api.attention_history("EURUSD", n=3)
        assert len(hist) == 3

    def test_tier_transitions_accessible_via_api(self):
        api, _, _ = self._setup()
        api.calculate_attention(_obs("EURUSD"))
        transitions = api.tier_transitions("EURUSD")
        assert len(transitions) >= 1   # at least the initial recording

    def test_current_tier_returns_tier(self):
        from trainer.qde.brain.attention import AttentionTier
        api, _, _ = self._setup()
        api.calculate_attention(_obs("EURUSD"))
        tier = api.current_tier("EURUSD")
        assert tier in list(AttentionTier)

    def test_current_tier_watch_for_unseen(self):
        from trainer.qde.brain.attention import AttentionTier
        api, _, _ = self._setup()
        assert api.current_tier("UNKNOWN") == AttentionTier.WATCH

    def test_high_attention_symbols_respects_tier_threshold(self):
        from trainer.qde.brain.attention import AttentionTier, ExtendedSignalContext
        api, _, _ = self._setup("EURUSD")
        # EURUSD with no signals → low weight
        api.calculate_attention(_obs("EURUSD"))
        # Should not appear in HIGH_ATTENTION unless it earned it
        high = api.high_attention_symbols(min_tier=AttentionTier.CRITICAL_ATTENTION)
        assert "EURUSD" not in high  # new symbol won't instantly be CRITICAL

    def test_drop_removes_symbol(self):
        api, _, _ = self._setup()
        api.calculate_attention(_obs("EURUSD"))
        api.drop("EURUSD")
        assert api.rank_symbols() == []
        assert api.attention_history("EURUSD") == []

    def test_evolution_summary_via_api(self):
        api, _, _ = self._setup()
        api.calculate_attention(_obs("EURUSD"))
        summary = api.evolution_summary("EURUSD")
        assert summary["tracked"] is True

    def test_diagnostics_returns_structure(self):
        api, _, _ = self._setup()
        api.calculate_attention(_obs("EURUSD"))
        diag = api.diagnostics()
        assert "tracked_count" in diag
        assert "tier_counts" in diag
        assert "symbols" in diag


# ═════════════════════════════════════════════════════════════════════════════
# 8. SignalEnricher (stateful wrapper)
# ═════════════════════════════════════════════════════════════════════════════


class TestSignalEnricherStateful:
    def test_enrich_uses_cached_context(self):
        from trainer.qde.brain.attention import SignalEnricher, ExtendedSignalContext
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        enricher = SignalEnricher()
        ctx = ExtendedSignalContext(volume_ratio=3.0)
        enricher.update_context("EURUSD", ctx)
        base = OpportunitySignals()
        enriched = enricher.enrich("EURUSD", base)
        assert enriched.volume_anomaly == pytest.approx(1.0)

    def test_enrich_no_context_returns_base_unchanged(self):
        from trainer.qde.brain.attention import SignalEnricher
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        enricher = SignalEnricher()
        base = OpportunitySignals(hypothesis_activity=0.5)
        result = enricher.enrich("EURUSD", base)
        assert result.hypothesis_activity == pytest.approx(0.5)
        assert result.volume_anomaly == pytest.approx(0.0)

    def test_remove_clears_context(self):
        from trainer.qde.brain.attention import SignalEnricher, ExtendedSignalContext
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        enricher = SignalEnricher()
        enricher.update_context("EURUSD", ExtendedSignalContext(volume_ratio=3.0))
        enricher.remove("EURUSD")
        result = enricher.enrich("EURUSD", OpportunitySignals())
        assert result.volume_anomaly == pytest.approx(0.0)


# ═════════════════════════════════════════════════════════════════════════════
# 9. Brain __init__ re-exports
# ═════════════════════════════════════════════════════════════════════════════


class TestBrainExports:
    def test_all_c10_symbols_exported(self):
        import trainer.qde.brain as brain_pkg

        required = [
            "AttentionTier",
            "ResourceBudget",
            "TIER_RESOURCES",
            "TIER_THRESHOLDS",
            "tier_from_composite",
            "AttentionSnapshot",
            "AttentionEvolutionRecord",
            "AttentionMemory",
            "ExtendedSignalContext",
            "enrich_signals",
            "SignalEnricher",
            "AttentionAPI",
            "RankedSymbol",
            "allocate_resources",
            "calculate_attention",
            "rank_symbols",
            "C10_WEIGHTS",
        ]
        for name in required:
            assert hasattr(brain_pkg, name), f"brain package missing export: {name}"

    def test_c10_weights_exported_from_allocator(self):
        from trainer.qde.brain.attention_allocator import C10_WEIGHTS
        assert C10_WEIGHTS is not None

    def test_opportunity_signals_has_new_fields(self):
        from trainer.qde.brain.opportunity_scanner import OpportunitySignals
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(OpportunitySignals)}
        for name in ("volume_anomaly", "key_level_proximity", "spread_change", "failed_move"):
            assert name in field_names


# ═════════════════════════════════════════════════════════════════════════════
# 10. Module-level convenience wrappers
# ═════════════════════════════════════════════════════════════════════════════


class TestModuleLevelWrappers:
    def test_calculate_attention_wrapper(self):
        from trainer.qde.brain.attention import calculate_attention, AttentionAPI
        brain, engine = _make_brain_and_engine("EURUSD")
        api = AttentionAPI(engine=engine)
        snap = calculate_attention(api, _obs("EURUSD"))
        assert snap.instrument == "EURUSD"

    def test_rank_symbols_wrapper(self):
        from trainer.qde.brain.attention import rank_symbols, AttentionAPI
        brain, engine = _make_brain_and_engine("EURUSD")
        api = AttentionAPI(engine=engine)
        api.calculate_attention(_obs("EURUSD"))
        ranked = rank_symbols(api)
        assert len(ranked) >= 1

    def test_allocate_resources_wrapper(self):
        from trainer.qde.brain.attention import allocate_resources, AttentionAPI, ResourceBudget
        brain, engine = _make_brain_and_engine("EURUSD")
        api = AttentionAPI(engine=engine)
        api.calculate_attention(_obs("EURUSD"))
        budget = allocate_resources(api, "EURUSD")
        assert isinstance(budget, ResourceBudget)
