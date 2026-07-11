---
name: Exposure package design
description: trainer/qde/brain/exposure/ — hidden portfolio concentration across currency, sector, correlation, and regional dimensions.
---

## Rule
`trainer/qde/brain/exposure/` detects concentration risk that is invisible to per-symbol engines. Reuses `InstrumentInfo.currency_exposures()` as the authoritative source for currency legs — never re-derives it. Does not duplicate `PortfolioScorer` (different inputs/purpose). No `QDEDecision` import anywhere.

## Currency direction bug (fixed)
`currency_exposures()` returns `{base: +1.0, quote: -1.0}` assuming a long position.  
**Must use `sign * norm_w` (signed weight), not `sign * abs(norm_w)`.**  
Short USDCHF (norm_w = -0.25) → USD gets `1.0 * (-0.25) = -0.25` (selling USD ✓). Using `abs(norm_w)` would flip it to +0.25 (wrong).

## PositionExposure input
`PositionExposure(instrument, direction, weight)` — `weight` is always non-negative (0-1 fraction of portfolio notional). Direction (+1/-1) signs all exposure contributions via `signed_weight() = direction * weight`.

## ExposureClusterEngine (stateless)
Four dimensions via HHI (same formula as PortfolioScorer but applied to net exposure vectors, not position weights):
1. **Currency** — `InstrumentInfo.currency_exposures()` × signed norm_w, summed across positions
2. **Sector** — `InstrumentInfo.asset_class.value`, signed norm_w
3. **Structural correlation** — from `InstrumentInfo.tags`: priority order `safe_haven → risk_on → commodity_currency → commodity → crypto → forex_major → forex_cross → uncorrelated`
4. **Regional** — `InstrumentInfo.region.value`, signed norm_w

Composite = 0.35×currency_hhi + 0.20×sector_hhi + 0.30×correlation_hhi + 0.15×regional_hhi

`marginal(candidate, positions)` → (before, after) cluster pair; delta = marginal concentration cost.

## TradeSaturationEngine structural formula
```python
structural = 0.65 * before.composite_concentration + 0.35 * max(0.0, d_composite)
```
- before-state (0.65 weight): "how concentrated is the portfolio right now?"
- marginal delta (0.35 weight): "does this trade make it materially worse?"
- First trade into empty portfolio: before=0, delta may be large → structural ~0.29 (low) ✓
- Trade on already-concentrated portfolio: before>0 → structural proportionally higher ✓

## No hardcoded limits
`saturation_score = structural + hist_adjustment * structural` where:
- `hist_adjustment = (0.5 - hist_quality) * 2 * hist_conf`
- Bad historical outcomes at similar concentration → hist_quality < 0.5 → positive adjustment (boost saturation)
- Good historical outcomes → hist_quality > 0.5 → negative adjustment (reduce saturation)
- No history → hist_conf=0 → structural governs

## Historical lookup
L2 distance over 5-dim concentration vector `[currency_hhi, sector_hhi, corr_conc, regional_hhi, n_positions_norm]`.  
`similarity = 1 / (1 + L2_dist / 0.5)`. Top-K=20 weighted by similarity × recency (tau=30 days).

## Integration signals (all monotone functions of saturation_score)
- `size_modifier = 1.0 - 0.5 × saturation` → [0.5, 1.0]
- `conviction_modifier = -0.3 × saturation` → [-0.3, 0]
- `rank_penalty = 0.25 × saturation` → [0, 0.25]
- `risk_appetite_modifier = -0.3 × saturation` → [-0.3, 0]

## context_for_inference keys (14 × exp_*)
`exp_currency_hhi`, `exp_sector_hhi`, `exp_correlation_conc`, `exp_regional_hhi`, `exp_composite_conc`, `exp_dominant_currency`, `exp_marginal_currency_hhi`, `exp_marginal_composite`, `exp_saturation_score`, `exp_size_modifier`, `exp_conviction_modifier`, `exp_rank_penalty`, `exp_historical_quality`, `exp_history_confidence`

## Learning loop
1. `build_observation(candidate, positions)` at trade entry — captures before-state + marginals
2. Store the `ExposureObservation`
3. After trade closes: `ExposureOutcome.compute_quality(portfolio_return, max_drawdown)` — normalized to portfolio's own vol, no hardcoded return target
4. `record_outcome(observation, outcome)` — writes to `ExposureMemory` (2000-record bounded FIFO)
5. Memory persists via `memory_snapshot()` / `load_memory(snapshot)`
