---
name: Opportunity package design
description: trainer/qde/brain/opportunity/ — cross-symbol ranking, capital efficiency, opportunity cost, tier allocation.
---

## Rule
`trainer/qde/brain/opportunity/` aggregates per-symbol signals (conviction, uncertainty, projection, attention) across all active opportunities in a single pass. It never recomputes those signals — it reads them from `OpportunityInput` fields. It does not duplicate `AttentionScoreAllocator` (per-symbol attention) or `ExpectedMoveEngine` (per-symbol projection).

## Input interface
`OpportunityInput` — flat dataclass populated by callers from existing per-symbol outputs:
- `conviction`, `confidence`, `conviction_profile` ← ConvictionScore
- `uncertainty`, `uncertainty_level` ← UncertaintyScore
- `expected_reward_atr`, `expected_mae_atr`, `expected_duration_seconds`, `reward_sufficiency_score`, `reward_risk_ratio`, `proj_opportunity_rank_signal`, `proj_position_size_modifier`, `proj_risk_appetite_modifier` ← ExpectedProjection
- `attention_weight` ← AttentionScoreAllocator.reallocate()
- `setup_state` ← SetupState.value (string)

## `OpportunityCostEngine` (stateless, O(N) pass)
- `capital_efficiency = expected_reward_atr / max(duration_hours, 0.25)` — ATR/hour
- `opportunity_cost_score = (best_eff - this_eff) / best_eff` — 0=best candidate, 1=worst
- Negative reward_atr → negative efficiency → cost=1.0 (clamped)
- Handles single candidate (cost=0) and empty list correctly

## `OpportunityRanker` composite score formula
Four components normalized within current candidate set:
- conviction_joint (30%) = conviction × confidence
- ev_score (25%) = percentile_rank of (reward_atr × sufficiency)
- uncertainty_discount (25%) = 1 - uncertainty
- capital_efficiency_score (20%) = percentile_rank of capital_efficiency
- Attention boost: mild multiplicative (1.0 + 0.10 × attn_weight)
- Opportunity cost penalty: −0.15 × opportunity_cost_score (subtractive)

## `CapitalPriorityEngine` tier thresholds
- TIER_1: net_score ≥ 0.55, actionable, within max_concurrent, within capital budget
- TIER_2: net_score ≥ 0.38, actionable, within budget
- TIER_3: net_score ≥ 0.20 (monitor only, suggested_size_modifier=0)
- PASS: below floor, not actionable, or budget exhausted
- WATCHING/ARMED → `is_actionable=False` → always suggested_size_modifier=0 regardless of score

## Tier size modifiers
- TIER_1: proj_position_size_modifier × 1.0 (full projection suggestion)
- TIER_2: proj_position_size_modifier × 0.6
- TIER_3: 0 (monitor only)

## Attention distribution
- TIER_1 group shares 60% of budget proportional to net_score
- TIER_2 group shares 30%
- TIER_3 group shares 10%
- Total ≤ 1.0

## Integration point
`CapitalPriorityEngine.context_for_inference(instrument, candidates)` → 8 `opp_*` keys:
`opp_tier`, `opp_rank`, `opp_net_score`, `opp_suggested_size_mod`, `opp_attention_allocation`,
`opp_is_actionable`, `opp_total_candidates`, `opp_concurrent_budget_pct`
