---
name: C11 Opportunity Engine design
description: Expected-utility scoring, opportunity classification, and ranking added to trainer/qde/brain/opportunity/. Three new public callables; OpportunityInput extended with 4 new fields.
---

## What C11 adds to trainer/qde/brain/opportunity/

New file: `opportunity_score.py` — three public callables:

- `classify_opportunity(inp) → OpportunityClass`
- `score_opportunity(inp) → OpportunityScore`
- `rank_opportunities(candidates, actionable_only=False) → list[OpportunityScore]`

New enum: `OpportunityClass` (MICRO_SCALP / SCALP / INTRADAY / MULTI_SESSION / SWING) — structural duration classes, not trader identities.

## OpportunityInput extended (backward-compatible)

Four new fields added to `OpportunityInput` in `opportunity_cost_engine.py`, all with defaults:

- `spread_cost_atr: float = 0.0`
- `slippage_estimate_atr: float = 0.0`
- `liquidity_quality: float = 0.5`
- `portfolio_interaction: float = 0.0`  ← −1 concentrating, +1 diversifying

Old callers not supplying these fields continue to work unchanged.

## classify_opportunity — duration thresholds

- MICRO_SCALP: < 300 s (< 5 min)
- SCALP: 300–7200 s (5 min – 2 hr)
- INTRADAY: 7200–28800 s (2–8 hr)
- MULTI_SESSION: 28800–259200 s (8 hr – 3 days)
- SWING: ≥ 259200 s

Fallback when `expected_duration_seconds == 0`: classify by `expected_reward_atr` as proxy.

## Expected utility formula (core C11 requirement)

C11 requires:
1. Does not favour large profits → **log utility** (concave): `log1p(net_gain / REF_ATR)` grows sub-linearly
2. Does not favour large position sizes → `proj_position_size_modifier` is absent from the formula entirely
3. Prefers small but highly probable gains → `prob_success = conviction × confidence` is the **direct linear multiplier**; a high-probability setup with small gain always beats a low-probability setup with large gain

Formula: `opportunity_score = prob_success × geometric_product(7 factor scores)`

Geometric product uses exponent weights (sum = 1.0):
- expected_return 0.25, expected_loss 0.20, spread 0.12, slippage 0.12, duration 0.10, portfolio 0.08, liquidity 0.13

Geometric weighting means any single factor near zero pulls the whole score down.

## Relationship to existing infrastructure

`OpportunityRanker`, `OpportunityCostEngine`, `CapitalPriorityEngine` are **unchanged** — they do relative cross-symbol ranking. `score_opportunity` / `rank_opportunities` provide a complementary **absolute** expected-utility score that does not depend on the field of candidates.

## How to apply

```python
from trainer.qde.brain.opportunity import (
    OpportunityInput, OpportunityClass,
    classify_opportunity, score_opportunity, rank_opportunities,
)

inp = OpportunityInput(
    instrument="EURUSD",
    conviction=0.9, confidence=0.85,
    expected_reward_atr=0.8, expected_mae_atr=0.3,
    expected_duration_seconds=900,
    spread_cost_atr=0.02, slippage_estimate_atr=0.01,
    liquidity_quality=0.9, portfolio_interaction=0.5,
)
opp_class = classify_opportunity(inp)      # → SCALP
score     = score_opportunity(inp)         # → OpportunityScore
ranked    = rank_opportunities([inp, ...]) # → list[OpportunityScore], sorted desc
```
