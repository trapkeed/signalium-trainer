"""C11 — Opportunity Engine tests.

Covers:
  - OpportunityClass enum values and ordering
  - classify_opportunity: all five classes by duration
  - classify_opportunity: fallback by reward when duration=0
  - OpportunityInput backward compat: old callers still work
  - OpportunityInput new fields default to neutral values
  - score_opportunity: returns OpportunityScore with all eight factors
  - score_opportunity: opportunity_score in [0, 1]
  - score_opportunity: prefers small highly-probable gains over large uncertain ones
  - score_opportunity: spread cost reduces score
  - score_opportunity: slippage reduces score
  - score_opportunity: poor liquidity reduces score
  - score_opportunity: concentrating portfolio reduces score
  - score_opportunity: diversifying portfolio increases score
  - score_opportunity: zero conviction → score = 0
  - score_opportunity: all neutral inputs → finite non-zero score
  - score_opportunity: does NOT favour large expected returns (log utility)
  - score_opportunity: does NOT favour large position size (pos_size_modifier absent)
  - score_opportunity: faster duration → higher duration_score
  - rank_opportunities: returns list sorted by opportunity_score descending
  - rank_opportunities: assigns 1-based sequential ranks
  - rank_opportunities: empty input → empty output
  - rank_opportunities: actionable_only filter
  - rank_opportunities: best opportunity correctly identified
  - classify_opportunity × score_opportunity: each class produces reasonable score
  - OpportunityScore.to_dict includes all required keys
  - __init__.py exports all C11 symbols
"""

from __future__ import annotations

import pytest

from trainer.qde.brain.opportunity.opportunity_cost_engine import OpportunityInput

# ── Helpers ───────────────────────────────────────────────────────────────────


def _inp(
    instrument: str = "EURUSD",
    conviction: float = 0.7,
    confidence: float = 0.8,
    expected_reward_atr: float = 1.5,
    expected_mae_atr: float = 0.6,
    expected_duration_seconds: float = 3600.0,  # 1 hour → INTRADAY
    spread_cost_atr: float = 0.05,
    slippage_estimate_atr: float = 0.03,
    liquidity_quality: float = 0.75,
    portfolio_interaction: float = 0.0,
    setup_state: str = "READY",
    **kwargs,
) -> OpportunityInput:
    return OpportunityInput(
        instrument=instrument,
        conviction=conviction,
        confidence=confidence,
        expected_reward_atr=expected_reward_atr,
        expected_mae_atr=expected_mae_atr,
        expected_duration_seconds=expected_duration_seconds,
        spread_cost_atr=spread_cost_atr,
        slippage_estimate_atr=slippage_estimate_atr,
        liquidity_quality=liquidity_quality,
        portfolio_interaction=portfolio_interaction,
        setup_state=setup_state,
        **kwargs,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 1. OpportunityClass enum
# ═════════════════════════════════════════════════════════════════════════════


class TestOpportunityClass:
    def test_all_five_values_present(self):
        from trainer.qde.brain.opportunity import OpportunityClass

        assert len(list(OpportunityClass)) == 5

    def test_enum_string_values(self):
        from trainer.qde.brain.opportunity import OpportunityClass

        assert OpportunityClass.MICRO_SCALP.value == "MICRO_SCALP"
        assert OpportunityClass.SCALP.value == "SCALP"
        assert OpportunityClass.INTRADAY.value == "INTRADAY"
        assert OpportunityClass.MULTI_SESSION.value == "MULTI_SESSION"
        assert OpportunityClass.SWING.value == "SWING"

    def test_enum_is_str_subclass(self):
        from trainer.qde.brain.opportunity import OpportunityClass

        assert isinstance(OpportunityClass.SCALP, str)


# ═════════════════════════════════════════════════════════════════════════════
# 2. classify_opportunity
# ═════════════════════════════════════════════════════════════════════════════


class TestClassifyOpportunity:
    def test_micro_scalp_below_5min(self):
        from trainer.qde.brain.opportunity import OpportunityClass, classify_opportunity

        assert (
            classify_opportunity(_inp(expected_duration_seconds=60)) == OpportunityClass.MICRO_SCALP
        )

    def test_micro_scalp_at_boundary(self):
        from trainer.qde.brain.opportunity import OpportunityClass, classify_opportunity

        assert (
            classify_opportunity(_inp(expected_duration_seconds=299))
            == OpportunityClass.MICRO_SCALP
        )

    def test_scalp_5min_to_2hr(self):
        from trainer.qde.brain.opportunity import OpportunityClass, classify_opportunity

        assert classify_opportunity(_inp(expected_duration_seconds=300)) == OpportunityClass.SCALP
        assert classify_opportunity(_inp(expected_duration_seconds=1800)) == OpportunityClass.SCALP
        assert classify_opportunity(_inp(expected_duration_seconds=7199)) == OpportunityClass.SCALP

    def test_intraday_2hr_to_8hr(self):
        from trainer.qde.brain.opportunity import OpportunityClass, classify_opportunity

        assert (
            classify_opportunity(_inp(expected_duration_seconds=7200)) == OpportunityClass.INTRADAY
        )
        assert (
            classify_opportunity(_inp(expected_duration_seconds=14400)) == OpportunityClass.INTRADAY
        )
        assert (
            classify_opportunity(_inp(expected_duration_seconds=28799)) == OpportunityClass.INTRADAY
        )

    def test_multi_session_8hr_to_3days(self):
        from trainer.qde.brain.opportunity import OpportunityClass, classify_opportunity

        assert (
            classify_opportunity(_inp(expected_duration_seconds=28800))
            == OpportunityClass.MULTI_SESSION
        )
        assert (
            classify_opportunity(_inp(expected_duration_seconds=86400))
            == OpportunityClass.MULTI_SESSION
        )
        assert (
            classify_opportunity(_inp(expected_duration_seconds=259199))
            == OpportunityClass.MULTI_SESSION
        )

    def test_swing_above_3days(self):
        from trainer.qde.brain.opportunity import OpportunityClass, classify_opportunity

        assert (
            classify_opportunity(_inp(expected_duration_seconds=259200)) == OpportunityClass.SWING
        )
        assert (
            classify_opportunity(_inp(expected_duration_seconds=604800)) == OpportunityClass.SWING
        )

    def test_fallback_by_reward_when_duration_zero_micro(self):
        from trainer.qde.brain.opportunity import OpportunityClass, classify_opportunity

        inp = _inp(expected_duration_seconds=0.0, expected_reward_atr=0.1)
        assert classify_opportunity(inp) == OpportunityClass.MICRO_SCALP

    def test_fallback_by_reward_when_duration_zero_scalp(self):
        from trainer.qde.brain.opportunity import OpportunityClass, classify_opportunity

        inp = _inp(expected_duration_seconds=0.0, expected_reward_atr=0.5)
        assert classify_opportunity(inp) == OpportunityClass.SCALP

    def test_fallback_by_reward_when_duration_zero_swing(self):
        from trainer.qde.brain.opportunity import OpportunityClass, classify_opportunity

        inp = _inp(expected_duration_seconds=0.0, expected_reward_atr=5.0)
        assert classify_opportunity(inp) == OpportunityClass.SWING

    def test_all_five_classes_reachable(self):
        from trainer.qde.brain.opportunity import OpportunityClass, classify_opportunity

        durations = [60, 900, 14400, 86400, 604800]
        results = {classify_opportunity(_inp(expected_duration_seconds=d)) for d in durations}
        assert results == set(OpportunityClass)


# ═════════════════════════════════════════════════════════════════════════════
# 3. OpportunityInput backward compat
# ═════════════════════════════════════════════════════════════════════════════


class TestOpportunityInputBackwardCompat:
    def test_old_fields_still_work(self):
        inp = OpportunityInput(
            instrument="EURUSD",
            conviction=0.7,
            confidence=0.8,
            expected_reward_atr=1.5,
        )
        assert inp.instrument == "EURUSD"

    def test_new_fields_default_neutral(self):
        inp = OpportunityInput(instrument="EURUSD")
        assert inp.spread_cost_atr == pytest.approx(0.0)
        assert inp.slippage_estimate_atr == pytest.approx(0.0)
        assert inp.liquidity_quality == pytest.approx(0.5)
        assert inp.portfolio_interaction == pytest.approx(0.0)

    def test_to_dict_includes_new_fields(self):
        inp = _inp()
        d = inp.to_dict()
        for key in (
            "spread_cost_atr",
            "slippage_estimate_atr",
            "liquidity_quality",
            "portfolio_interaction",
        ):
            assert key in d, f"to_dict missing {key}"


# ═════════════════════════════════════════════════════════════════════════════
# 4. score_opportunity — structure and bounds
# ═════════════════════════════════════════════════════════════════════════════


class TestScoreOpportunityStructure:
    def test_returns_opportunity_score(self):
        from trainer.qde.brain.opportunity import OpportunityScore, score_opportunity

        s = score_opportunity(_inp())
        assert isinstance(s, OpportunityScore)

    def test_opportunity_score_in_unit_interval(self):
        from trainer.qde.brain.opportunity import score_opportunity

        s = score_opportunity(_inp())
        assert 0.0 <= s.opportunity_score <= 1.0

    def test_all_eight_factors_present(self):
        from trainer.qde.brain.opportunity import score_opportunity

        s = score_opportunity(_inp())
        for attr in (
            "prob_success",
            "expected_return_score",
            "expected_loss_score",
            "spread_cost_score",
            "slippage_score",
            "duration_score",
            "portfolio_interaction_score",
            "liquidity_quality_score",
        ):
            assert hasattr(s, attr), f"OpportunityScore missing {attr}"

    def test_all_eight_factors_in_unit_interval(self):
        from trainer.qde.brain.opportunity import score_opportunity

        s = score_opportunity(_inp())
        for attr in (
            "prob_success",
            "expected_return_score",
            "expected_loss_score",
            "spread_cost_score",
            "slippage_score",
            "duration_score",
            "portfolio_interaction_score",
            "liquidity_quality_score",
        ):
            val = getattr(s, attr)
            assert 0.0 <= val <= 1.0, f"{attr}={val} outside [0,1]"

    def test_opportunity_class_assigned(self):
        from trainer.qde.brain.opportunity import OpportunityClass, score_opportunity

        s = score_opportunity(_inp())
        assert s.opportunity_class in list(OpportunityClass)

    def test_instrument_passthrough(self):
        from trainer.qde.brain.opportunity import score_opportunity

        s = score_opportunity(_inp(instrument="GBPUSD"))
        assert s.instrument == "GBPUSD"

    def test_rank_is_zero_for_single_call(self):
        from trainer.qde.brain.opportunity import score_opportunity

        s = score_opportunity(_inp())
        assert s.rank == 0

    def test_to_dict_has_all_required_keys(self):
        from trainer.qde.brain.opportunity import score_opportunity

        d = score_opportunity(_inp()).to_dict()
        required = [
            "rank",
            "instrument",
            "opportunity_id",
            "setup_state",
            "direction",
            "opportunity_class",
            "opportunity_score",
            "prob_success",
            "expected_return_score",
            "expected_loss_score",
            "spread_cost_score",
            "slippage_score",
            "duration_score",
            "portfolio_interaction_score",
            "liquidity_quality_score",
        ]
        for key in required:
            assert key in d, f"to_dict missing {key}"


# ═════════════════════════════════════════════════════════════════════════════
# 5. score_opportunity — behavioral properties
# ═════════════════════════════════════════════════════════════════════════════


class TestScoreOpportunityBehavior:
    def test_zero_conviction_yields_zero_score(self):
        from trainer.qde.brain.opportunity import score_opportunity

        s = score_opportunity(_inp(conviction=0.0, confidence=0.0))
        assert s.opportunity_score == pytest.approx(0.0)

    def test_zero_conviction_zero_score_regardless_of_large_reward(self):
        """Score must be zero even with a huge expected return when p=0."""
        from trainer.qde.brain.opportunity import score_opportunity

        s = score_opportunity(_inp(conviction=0.0, confidence=0.0, expected_reward_atr=10.0))
        assert s.opportunity_score == pytest.approx(0.0)

    def test_high_probability_small_gain_beats_low_probability_large_gain(self):
        """Core C11 requirement: small but highly probable > large uncertain."""
        from trainer.qde.brain.opportunity import score_opportunity

        # Small certain: p=0.9, 0.5 ATR
        certain = score_opportunity(
            _inp(
                conviction=0.95,
                confidence=0.95,
                expected_reward_atr=0.5,
                expected_mae_atr=0.2,
                spread_cost_atr=0.0,
                slippage_estimate_atr=0.0,
                liquidity_quality=0.9,
            )
        )
        # Large uncertain: p=0.5, 4.0 ATR
        uncertain = score_opportunity(
            _inp(
                conviction=0.7,
                confidence=0.7,
                expected_reward_atr=4.0,
                expected_mae_atr=3.0,
                spread_cost_atr=0.0,
                slippage_estimate_atr=0.0,
                liquidity_quality=0.9,
            )
        )
        assert certain.opportunity_score > uncertain.opportunity_score, (
            f"Certain small gain ({certain.opportunity_score:.4f}) should beat "
            f"uncertain large gain ({uncertain.opportunity_score:.4f})"
        )

    def test_doubling_expected_return_does_not_double_score(self):
        """Log utility: sub-linear in return size."""
        from trainer.qde.brain.opportunity import score_opportunity

        base = score_opportunity(
            _inp(expected_reward_atr=1.0, spread_cost_atr=0.0, slippage_estimate_atr=0.0)
        )
        doubled = score_opportunity(
            _inp(expected_reward_atr=2.0, spread_cost_atr=0.0, slippage_estimate_atr=0.0)
        )
        assert (
            doubled.opportunity_score < 2.0 * base.opportunity_score
        ), "Score should grow sub-linearly with return size (concave utility)"

    def test_position_size_modifier_absent_from_score(self):
        """Score must NOT favour large position sizes."""
        from trainer.qde.brain.opportunity import score_opportunity

        # Score should be identical regardless of proj_position_size_modifier
        s1 = score_opportunity(_inp(proj_position_size_modifier=0.5))
        s2 = score_opportunity(_inp(proj_position_size_modifier=2.0))
        assert s1.opportunity_score == pytest.approx(
            s2.opportunity_score
        ), "opportunity_score must not change with proj_position_size_modifier"

    def test_spread_cost_reduces_score(self):
        from trainer.qde.brain.opportunity import score_opportunity

        no_spread = score_opportunity(_inp(spread_cost_atr=0.0))
        high_spread = score_opportunity(_inp(spread_cost_atr=1.0))
        assert no_spread.opportunity_score > high_spread.opportunity_score

    def test_spread_cost_above_return_gives_low_spread_score(self):
        from trainer.qde.brain.opportunity import score_opportunity

        s = score_opportunity(_inp(expected_reward_atr=0.5, spread_cost_atr=0.6))
        assert s.spread_cost_score == pytest.approx(0.0)

    def test_slippage_reduces_score(self):
        from trainer.qde.brain.opportunity import score_opportunity

        no_slip = score_opportunity(_inp(slippage_estimate_atr=0.0))
        high_slip = score_opportunity(_inp(slippage_estimate_atr=1.0))
        assert no_slip.opportunity_score > high_slip.opportunity_score

    def test_poor_liquidity_reduces_score(self):
        from trainer.qde.brain.opportunity import score_opportunity

        good_liq = score_opportunity(_inp(liquidity_quality=0.9))
        poor_liq = score_opportunity(_inp(liquidity_quality=0.1))
        assert good_liq.opportunity_score > poor_liq.opportunity_score

    def test_diversifying_portfolio_increases_score(self):
        from trainer.qde.brain.opportunity import score_opportunity

        diversify = score_opportunity(_inp(portfolio_interaction=1.0))
        concentrate = score_opportunity(_inp(portfolio_interaction=-1.0))
        assert diversify.opportunity_score > concentrate.opportunity_score

    def test_portfolio_interaction_score_maps_minus1_to_0(self):
        from trainer.qde.brain.opportunity import score_opportunity

        s = score_opportunity(_inp(portfolio_interaction=-1.0))
        assert s.portfolio_interaction_score == pytest.approx(0.0)

    def test_portfolio_interaction_score_maps_plus1_to_1(self):
        from trainer.qde.brain.opportunity import score_opportunity

        s = score_opportunity(_inp(portfolio_interaction=1.0))
        assert s.portfolio_interaction_score == pytest.approx(1.0)

    def test_faster_duration_higher_duration_score(self):
        from trainer.qde.brain.opportunity import score_opportunity

        fast = score_opportunity(_inp(expected_duration_seconds=300))  # 5 min
        slow = score_opportunity(_inp(expected_duration_seconds=86400))  # 24 hr
        assert fast.duration_score > slow.duration_score

    def test_prob_success_equals_conviction_times_confidence(self):
        from trainer.qde.brain.opportunity import score_opportunity

        s = score_opportunity(_inp(conviction=0.8, confidence=0.75))
        assert s.prob_success == pytest.approx(0.8 * 0.75, abs=1e-4)

    def test_prob_success_clamped_to_one(self):
        from trainer.qde.brain.opportunity import score_opportunity

        s = score_opportunity(_inp(conviction=1.0, confidence=1.0))
        assert s.prob_success <= 1.0

    def test_higher_conviction_gives_higher_score(self):
        from trainer.qde.brain.opportunity import score_opportunity

        low_conv = score_opportunity(_inp(conviction=0.3, confidence=0.4))
        high_conv = score_opportunity(_inp(conviction=0.9, confidence=0.95))
        assert high_conv.opportunity_score > low_conv.opportunity_score

    def test_all_neutral_inputs_finite_nonzero_score(self):
        """Default-valued input should still produce a meaningful score."""
        from trainer.qde.brain.opportunity import score_opportunity

        s = score_opportunity(
            _inp(
                conviction=0.6,
                confidence=0.7,
                expected_reward_atr=1.0,
                expected_mae_atr=0.5,
                spread_cost_atr=0.0,
                slippage_estimate_atr=0.0,
                liquidity_quality=0.5,
                portfolio_interaction=0.0,
            )
        )
        assert 0.0 < s.opportunity_score < 1.0
        assert not any(
            v != v  # NaN check
            for v in [s.opportunity_score, s.prob_success, s.expected_return_score]
        )

    def test_score_monotone_in_confidence(self):
        """Increasing confidence should never decrease opportunity_score."""
        from trainer.qde.brain.opportunity import score_opportunity

        scores = [
            score_opportunity(_inp(confidence=c, conviction=0.7)).opportunity_score
            for c in [0.2, 0.4, 0.6, 0.8, 1.0]
        ]
        assert scores == sorted(scores), "Score should be non-decreasing in confidence"

    def test_expected_loss_score_lower_mae_is_better(self):
        from trainer.qde.brain.opportunity import score_opportunity

        small_loss = score_opportunity(_inp(expected_mae_atr=0.2))
        large_loss = score_opportunity(_inp(expected_mae_atr=3.0))
        assert small_loss.expected_loss_score > large_loss.expected_loss_score

    def test_opportunity_class_attached_to_score(self):
        from trainer.qde.brain.opportunity import OpportunityClass, score_opportunity

        fast = score_opportunity(_inp(expected_duration_seconds=60))
        assert fast.opportunity_class == OpportunityClass.MICRO_SCALP
        slow = score_opportunity(_inp(expected_duration_seconds=604800))
        assert slow.opportunity_class == OpportunityClass.SWING

    def test_is_actionable_passthrough_ready(self):
        from trainer.qde.brain.opportunity import score_opportunity

        s = score_opportunity(_inp(setup_state="READY"))
        assert s.is_actionable is True

    def test_is_actionable_passthrough_watching(self):
        from trainer.qde.brain.opportunity import score_opportunity

        s = score_opportunity(_inp(setup_state="WATCHING"))
        assert s.is_actionable is False


# ═════════════════════════════════════════════════════════════════════════════
# 6. rank_opportunities
# ═════════════════════════════════════════════════════════════════════════════


class TestRankOpportunities:
    def _candidates(self):
        """Five candidates with clearly differentiated expected utility."""
        return [
            # Best: high p, reasonable return, fast, good liquidity
            _inp(
                "EURUSD",
                conviction=0.92,
                confidence=0.90,
                expected_reward_atr=0.8,
                expected_mae_atr=0.3,
                expected_duration_seconds=900,
                spread_cost_atr=0.02,
                slippage_estimate_atr=0.01,
                liquidity_quality=0.9,
                portfolio_interaction=0.5,
                setup_state="READY",
            ),
            # Good: decent p, moderate return
            _inp(
                "GBPUSD",
                conviction=0.75,
                confidence=0.80,
                expected_reward_atr=1.5,
                expected_mae_atr=0.7,
                expected_duration_seconds=7200,
                spread_cost_atr=0.05,
                slippage_estimate_atr=0.03,
                liquidity_quality=0.7,
                portfolio_interaction=0.0,
                setup_state="READY",
            ),
            # OK: lower probability
            _inp(
                "USDJPY",
                conviction=0.55,
                confidence=0.60,
                expected_reward_atr=2.0,
                expected_mae_atr=1.5,
                expected_duration_seconds=28800,
                spread_cost_atr=0.1,
                slippage_estimate_atr=0.05,
                liquidity_quality=0.6,
                portfolio_interaction=0.0,
                setup_state="CONFIRMING",
            ),
            # Weak: very high friction
            _inp(
                "XAUUSD",
                conviction=0.65,
                confidence=0.70,
                expected_reward_atr=1.0,
                expected_mae_atr=0.8,
                expected_duration_seconds=3600,
                spread_cost_atr=0.8,
                slippage_estimate_atr=0.4,
                liquidity_quality=0.4,
                portfolio_interaction=-0.5,
                setup_state="WATCHING",
            ),
            # Pass: zero conviction
            _inp(
                "AUDUSD",
                conviction=0.0,
                confidence=0.0,
                expected_reward_atr=5.0,
                expected_mae_atr=0.1,
                expected_duration_seconds=60,
                spread_cost_atr=0.0,
                slippage_estimate_atr=0.0,
                liquidity_quality=1.0,
                portfolio_interaction=1.0,
                setup_state="WATCHING",
            ),
        ]

    def test_returns_list(self):
        from trainer.qde.brain.opportunity import rank_opportunities

        result = rank_opportunities(self._candidates())
        assert isinstance(result, list)

    def test_sorted_descending(self):
        from trainer.qde.brain.opportunity import rank_opportunities

        result = rank_opportunities(self._candidates())
        scores = [r.opportunity_score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_sequential_one_indexed(self):
        from trainer.qde.brain.opportunity import rank_opportunities

        result = rank_opportunities(self._candidates())
        assert [r.rank for r in result] == list(range(1, len(result) + 1))

    def test_rank_1_is_best_score(self):
        from trainer.qde.brain.opportunity import rank_opportunities

        result = rank_opportunities(self._candidates())
        assert result[0].rank == 1
        assert result[0].opportunity_score == max(r.opportunity_score for r in result)

    def test_eurusd_wins_high_probability_small_gain(self):
        """EURUSD (high p, small gain, fast) should rank first."""
        from trainer.qde.brain.opportunity import rank_opportunities

        result = rank_opportunities(self._candidates())
        assert result[0].instrument == "EURUSD"

    def test_zero_conviction_candidate_ranks_last(self):
        """AUDUSD with zero conviction should rank last despite large reward."""
        from trainer.qde.brain.opportunity import rank_opportunities

        result = rank_opportunities(self._candidates())
        assert result[-1].instrument == "AUDUSD"
        assert result[-1].opportunity_score == pytest.approx(0.0)

    def test_empty_input_returns_empty(self):
        from trainer.qde.brain.opportunity import rank_opportunities

        assert rank_opportunities([]) == []

    def test_single_candidate_rank_1(self):
        from trainer.qde.brain.opportunity import rank_opportunities

        result = rank_opportunities([_inp()])
        assert len(result) == 1
        assert result[0].rank == 1

    def test_actionable_only_filters_non_ready(self):
        from trainer.qde.brain.opportunity import rank_opportunities

        candidates = [
            _inp("EURUSD", setup_state="READY"),
            _inp("GBPUSD", setup_state="WATCHING"),
            _inp("USDJPY", setup_state="CONFIRMING"),
        ]
        result = rank_opportunities(candidates, actionable_only=True)
        for r in result:
            assert r.is_actionable is True

    def test_actionable_only_false_includes_all(self):
        from trainer.qde.brain.opportunity import rank_opportunities

        candidates = [
            _inp("EURUSD", setup_state="READY"),
            _inp("GBPUSD", setup_state="WATCHING"),
        ]
        result = rank_opportunities(candidates, actionable_only=False)
        assert len(result) == 2

    def test_rank_field_set_zero_by_score_opportunity(self):
        from trainer.qde.brain.opportunity import score_opportunity

        s = score_opportunity(_inp())
        assert s.rank == 0

    def test_rank_field_set_by_rank_opportunities(self):
        from trainer.qde.brain.opportunity import rank_opportunities

        results = rank_opportunities([_inp("A"), _inp("B")])
        for r in results:
            assert r.rank > 0

    def test_multiple_symbols_all_scored(self):
        from trainer.qde.brain.opportunity import rank_opportunities

        candidates = [_inp(f"SYM{i}") for i in range(10)]
        result = rank_opportunities(candidates)
        assert len(result) == 10
        assert {r.instrument for r in result} == {f"SYM{i}" for i in range(10)}


# ═════════════════════════════════════════════════════════════════════════════
# 7. Cross-class scoring sanity
# ═════════════════════════════════════════════════════════════════════════════


class TestCrossClassScoring:
    """Each opportunity class should produce a finite, valid score."""

    CLASSES = [
        ("MICRO_SCALP", 60, 0.3),
        ("SCALP", 1800, 0.8),
        ("INTRADAY", 14400, 1.5),
        ("MULTI_SESSION", 86400, 2.5),
        ("SWING", 604800, 4.0),
    ]

    @pytest.mark.parametrize("cls_name,duration_s,reward_atr", CLASSES)
    def test_each_class_scores_valid(self, cls_name, duration_s, reward_atr):
        from trainer.qde.brain.opportunity import OpportunityClass, score_opportunity

        inp = _inp(
            expected_duration_seconds=duration_s,
            expected_reward_atr=reward_atr,
            expected_mae_atr=reward_atr * 0.4,
            conviction=0.75,
            confidence=0.80,
        )
        s = score_opportunity(inp)
        assert 0.0 <= s.opportunity_score <= 1.0
        assert s.opportunity_class == OpportunityClass(cls_name)
        # No NaNs
        import math

        assert math.isfinite(s.opportunity_score)
        assert math.isfinite(s.prob_success)


# ═════════════════════════════════════════════════════════════════════════════
# 8. Package __init__ exports
# ═════════════════════════════════════════════════════════════════════════════


class TestPackageExports:
    def test_all_c11_symbols_in_package(self):
        import trainer.qde.brain.opportunity as pkg

        required = [
            "OpportunityClass",
            "OpportunityScore",
            "classify_opportunity",
            "score_opportunity",
            "rank_opportunities",
        ]
        for name in required:
            assert hasattr(pkg, name), f"Package missing export: {name}"

    def test_all_in_dunder_all(self):
        import trainer.qde.brain.opportunity as pkg

        for name in [
            "OpportunityClass",
            "OpportunityScore",
            "classify_opportunity",
            "score_opportunity",
            "rank_opportunities",
        ]:
            assert name in pkg.__all__, f"{name} not in __all__"

    def test_opportunity_input_new_fields_exported(self):
        import dataclasses

        from trainer.qde.brain.opportunity import OpportunityInput

        fields = {f.name for f in dataclasses.fields(OpportunityInput)}
        for name in (
            "spread_cost_atr",
            "slippage_estimate_atr",
            "liquidity_quality",
            "portfolio_interaction",
        ):
            assert name in fields, f"OpportunityInput missing field {name}"
