---
name: Phase 21 execution package
description: Design rules and gotchas for the trainer/execution/ engine.
---

## The Package

`trainer/execution/` — 16 modules, stdlib + loguru only, zero model changes.
Pure analytics recommendation layer — does NOT place trades, does NOT connect to brokers.

## Module Load Order (for spec_from_file_location tests)

```
execution_types → entry_zones → entry_quality
→ stop_loss_engine → take_profit_engine → risk_engine
→ position_sizer → trade_manager → execution_timing
→ order_strategy → trade_scorer → execution_simulator
→ execution_planner → execution_database
→ execution_dashboard → execution_api
```

## Critical Geometry Rule: Entry Zone Invalidation

Invalidation must be placed **relative to primary entry**, extended further in the pullback direction — NOT relative to current price:

```python
# CORRECT
sign = -1.0 if is_buy else 1.0          # BUY: zones below price
pri  = price + sign * PRI_MULT * atr    # primary pullback entry
inval = pri + sign * INVAL_MULT * atr   # further in same direction

# WRONG — places invalidation on opposite side of price
inval = price - sign * INVAL_MULT * atr
```

**Why:** For BUY trades, zones are pullback entries below current price. Invalidation must be further below the primary entry (deeper into the pullback), not above current price.

## Key Design Rule: SQLite :memory: Persistence

`ExecutionDatabase` uses `self._persistent_conn` for `:memory:` databases, same pattern as Phase 20's PortfolioDatabase. A new `sqlite3.connect(":memory:")` always creates a fresh empty database.

## Scoring Weights (TradeScorer)

```
forecast_quality 20% + entry_quality 20% + risk 15% + liquidity 10%
+ spread 10% + volatility 10% + reward 10% + portfolio_impact 5% = 100%
```

## Entry Quality Grades

EXCELLENT ≥ 0.85, GOOD ≥ 0.70, ACCEPTABLE ≥ 0.55, POOR ≥ 0.40, REJECT < 0.40

## Management Action Logic

- news_risk > 0.60 → EXIT_EARLY (primary)
- confidence < 0.55 → PARTIAL_CLOSE
- confidence > 0.75 AND momentum > 0.20 (aligned) → SCALE_IN
- otherwise → HOLD

## Timing Recommendation Logic

- spread > max_spread OR news_risk > 0.75 → AVOID
- volatility extreme (>0.035) → AVOID
- composite score ≥ 0.65 → EXECUTE_NOW
- composite score ≥ 0.45 → WAIT
- else → AVOID

## Order Type Logic

- AVOID timing → CONDITIONAL
- EXECUTE_NOW + confidence ≥ 0.70 + spread OK → MARKET
- EXECUTE_NOW + lower confidence → LIMIT
- WAIT → LIMIT (24h GTC)
- high momentum (>0.60) → consider STOP (breakout)
