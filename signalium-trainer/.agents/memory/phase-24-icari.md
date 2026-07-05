---
name: Phase 24 ICARI package
description: Design rules and gotchas for the trainer/icari/ Institutional Capital Allocation & Risk Intelligence system.
---

## The Package

`trainer/icari/` — 15 modules (+ __init__), stdlib + loguru only.
Institutional Capital Allocation & Risk Intelligence Platform (ICARI).
Never executes trades. Never modifies forecasting models. Never connects brokers.

## Module Load Order (for spec_from_file_location tests)

```
icari_types → icari_models → allocation_engine → risk_budget_engine →
position_sizing_engine → dynamic_sizer → capital_efficiency →
portfolio_risk_engine → drawdown_manager → correlation_engine →
scenario_engine → allocation_optimizer → capital_health →
icari_database → icari_dashboard → icari_api
```

## Critical Gotchas

### AllocationEngine factor weights
`_WEIGHTS` dict in `allocation_engine.py` has an `assert abs(sum - 1.0) < 1e-9` at module load.
Any modification must keep weights exactly summing to 1.0.

### DynamicSizer constructor parameters
Parameters are `drawdown_soft_limit` and `drawdown_hard_limit` (NOT `soft_dd_limit` / `hard_dd_limit`).
Tests must use the full names.

### Low-confidence denial tests
The AllocationEngine composite score includes `expected_value` (normalised) and `expected_return`
(normalised) which can dominate over small forecast/research scores.
To force a denial in tests, set `expected_value=-0.01` and `expected_return=-0.02` so both
normalised factors evaluate to 0.0.

### AllocationOptimizer max_weight enforcement
The iterative weight projection (clamp → normalise → repeat) diverges when only 1 non-zero
instrument remains — normalization always restores it to 1.0.
**Fix:** pass `max_weight` into `_grid_weights()` so candidates are generated with the cap
already applied (alpha sweep stops at cap, dominant share set to min(cap, 0.60)).
Do NOT rely on post-hoc projection alone for the 2-instrument case.

### allocation_usd rounding
`AllocationResult.allocation_usd` and `allocation_frac` are rounded independently.
Test with `delta=2.0` not `places=1`.

## Enums

- SizingMethod (10): FIXED_FRACTIONAL, FIXED_RISK, ATR_ADJUSTED, VOLATILITY_ADJUSTED, EXPECTED_VALUE, CONFIDENCE_ADJUSTED, RISK_BUDGET_ADJUSTED, KELLY, FRACTIONAL_KELLY, CUSTOM
- RiskBudgetType (12): DAILY, WEEKLY, MONTHLY, PORTFOLIO, INSTRUMENT, CURRENCY, COMMODITY, CRYPTO, INDEX, STRATEGY, CORRELATED, OPEN_TRADE
- DrawdownMode (6): NORMAL, SOFT_LIMIT, HARD_LIMIT, PRESERVATION, RECOVERY, EMERGENCY_LOCKDOWN
- ScenarioType (11): FLASH_CRASH, LIQUIDITY_CRISIS, SPREAD_EXPANSION, HIGH_VOLATILITY, CENTRAL_BANK, RISK_ON, RISK_OFF, STRONG_USD, WEAK_USD, COMMODITY_SHOCK, CRYPTO_CRASH
- OptimizationObjective (8): MAX_SHARPE, MAX_SORTINO, MIN_DRAWDOWN, MAX_DIVERSIFICATION, RISK_PARITY, BALANCED, CAPITAL_PRESERVATION, EXPECTED_UTILITY
- AllocationRecommendation (8): INCREASE_POSITION, REDUCE_POSITION, DELAY_ENTRY, SKIP_TRADE, REDUCE_PORTFOLIO_RISK, INCREASE_DIVERSIFICATION, REDUCE_CORRELATED, ACTIVATE_PRESERVATION
- AllocationStatus (4): APPROVED, REDUCED, DENIED, PENDING
- CapitalHealthGrade (5): EXCELLENT (≥0.85), GOOD (≥0.70), FAIR (≥0.55), POOR (≥0.40), CRITICAL (<0.40)

## Key Design Decisions

### Allocation Engine
- 11 weighted factors (sum=1.0); composite < 0.30 → DENIED
- Regime multipliers: TRENDING=1.10, NORMAL=1.00, RANGING=0.85, VOLATILE=0.65
- Max allocation per instrument: 25% of capital
- Exposure penalty kicks in above 50% portfolio exposure
- Emergency lockdown (DrawdownMode) blocks all allocations

### Portfolio Risk (historical simulation)
- VaR via sorted returns percentile (no parametric assumption)
- CVaR = mean of tail returns below 5th percentile
- Diversification Ratio = weighted_vol_sum / port_vol
- Risk Concentration = Herfindahl index of risk contributions

### Optimization (grid search, pure Python)
- _grid_weights generates candidates with `max_weight` cap baked in
- Iterative projection applied as secondary safety net, but primary enforcement is in candidate generation
- 8 objectives all use the same candidate set

### Drawdown Modes
- Risk multipliers: NORMAL=1.0, SOFT=0.75, HARD=0.50, PRESERVATION=0.25, RECOVERY=0.60, LOCKDOWN=0.0
- DrawdownManager tracks high-water mark automatically; LOCKDOWN blocks IcariAPI.allocate()

### Capital Health Weights (sum=1.0)
efficiency=0.20, risk=0.20, allocation=0.15, diversification=0.15, stability=0.15, utilisation=0.15

## Database (8 tables)
capital_allocation, risk_budgets, allocation_history, portfolio_risk,
scenario_results, capital_health, optimization_results, drawdown_events.
SQLite :memory: uses persistent connection (same pattern as all prior phases).

## Dashboard Views (7)
capital_allocation, risk_dashboard, exposure_dashboard, drawdown_dashboard,
capital_health, optimization, scenario_analysis.

## Test Count
- Phase 24 ICARI: 222 tests
- Total across all phases: 1140 tests (all green)
