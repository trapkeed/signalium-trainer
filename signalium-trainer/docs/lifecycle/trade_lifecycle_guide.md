# Trade Lifecycle Guide

## Overview

The Trade Lifecycle Intelligence Engine (Phase 21.5) treats every trade as a
living object. Instead of a simple BUY or SELL, each trade evolves through a
defined set of states from idea generation to final closure.

## Lifecycle States

### Forward Path

| State | Description |
|---|---|
| IDEA | Market identified as interesting |
| WATCHLIST | Potential setup detected |
| PREPARATION | Conditions approaching execution |
| ENTRY_READY | All required conditions satisfied |
| PENDING_ENTRY | Waiting for optimal execution |
| ACTIVE | Position opened |
| POSITION_BUILDING | Optional scaling into position |
| PROTECTED | Stop moved to breakeven |
| PARTIAL_EXIT | Partial profit secured |
| RUNNER | Remaining position allowed to run |
| TRAILING | Dynamic trailing stop active |
| EXIT_READY | Exit conditions developing |
| CLOSING | Trade being finalised |
| CLOSED | Trade completed |

### Alternative Outcomes

| State | Description |
|---|---|
| INVALIDATED | Setup no longer valid |
| CANCELLED | Trade aborted before entry |
| EXPIRED | Entry window timed out |
| STOPPED | Stop loss triggered |
| FORCED_EXIT | External condition forced closure |
| EMERGENCY_EXIT | Critical risk event |

## Creating a Trade Lifecycle

```python
from trainer.lifecycle import TradeLifecycle, MarketContext

tl = TradeLifecycle.new(
    instrument = "EURUSD",
    direction  = "BUY",
    confidence = 0.65,
    context    = MarketContext(price=1.2500, atr=0.001, volatility=0.012),
)
print(tl.trade_id, tl.current_state.value)  # IDEA
```

## Transitioning States

```python
tl.transition_to(
    new_state  = LifecycleState.WATCHLIST,
    reason     = "Setup pattern detected on H4 timeframe",
    confidence = 0.68,
)
```

## Lifecycle Progress

A typical successful trade follows this path:

```
IDEA → WATCHLIST → PREPARATION → ENTRY_READY → PENDING_ENTRY
     → ACTIVE → PROTECTED → PARTIAL_EXIT → RUNNER → TRAILING
     → EXIT_READY → CLOSING → CLOSED
```

A stopped trade:

```
IDEA → WATCHLIST → PREPARATION → ENTRY_READY → PENDING_ENTRY
     → ACTIVE → STOPPED → CLOSED
```

## Key Properties

- `is_terminal` — True once CLOSED, INVALIDATED, CANCELLED, or EXPIRED
- `is_position_open` — True from ACTIVE through CLOSING
- `age_hours` — Total age since creation
- `state_count` — Number of state transitions
- `critical_events` — Events with severity = CRITICAL

## Recommendations

The lifecycle provides four recommendation types:

```python
# Suggest next state
rec = tl.recommend_transition(context=ctx)

# Position sizing / management
plan = tl.position_management(context=ctx, metrics=metrics)

# Confidence assessment
lc = tl.lifecycle_confidence(context=ctx)

# Institutional reasoning
report = tl.reasoning(context=ctx, prev_conf=0.65)
```

## Important Notes

- The engine is **recommendation-only** — it never places trades
- All transitions are validated by the StateMachine before execution
- Complete state history is maintained throughout the trade's life
- Events are emitted on every state change and detected condition change
