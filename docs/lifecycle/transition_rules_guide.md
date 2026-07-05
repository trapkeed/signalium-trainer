# Transition Rules Guide

## The Transition Engine

The `TransitionEngine` evaluates all valid transitions from the current state
and recommends the best next state based on:

1. Emergency conditions (highest priority)
2. Invalidation (confidence collapse, trend break)
3. Forward progression (ranked by expected_benefit × confidence)
4. Stay (if no conditions are met)

## Priority Order

```
1. EMERGENCY_EXIT  (news_risk > 0.80 OR volatility > 0.035)
2. INVALIDATED     (confidence < 0.40 OR trend_valid = False in pre-entry)
3. Best positive transition (max expected_benefit × confidence)
4. Stay in current state
```

## Complete Transition Map

### Pre-Entry States

```
IDEA
  ├── → WATCHLIST      (confidence ≥ 0.50, setup detected)
  ├── → INVALIDATED    (setup failed)
  └── → CANCELLED      (manual abort)

WATCHLIST
  ├── → PREPARATION    (confidence ≥ 0.55, trend valid)
  ├── → INVALIDATED    (trend broken)
  ├── → CANCELLED      (manual abort)
  └── → EXPIRED        (48h timeout)

PREPARATION
  ├── → ENTRY_READY    (confidence ≥ 0.65, news_risk < 0.40, spread_ok)
  ├── → WATCHLIST      (conditions deteriorated — step back)
  ├── → INVALIDATED    (setup invalid)
  ├── → CANCELLED      (manual abort)
  └── → EXPIRED        (24h timeout)

ENTRY_READY
  ├── → PENDING_ENTRY  (confidence ≥ 0.65, session_ok, spread_ok)
  ├── → PREPARATION    (step back — conditions changed)
  ├── → INVALIDATED    (setup failed)
  └── → CANCELLED      (manual abort)

PENDING_ENTRY
  ├── → ACTIVE         (fill confirmed — notional)
  ├── → ENTRY_READY    (order not filled — step back)
  ├── → CANCELLED      (order cancelled)
  └── → EXPIRED        (4h timeout)
```

### Position States

```
ACTIVE
  ├── → POSITION_BUILDING (confidence ≥ 0.75, momentum aligned)
  ├── → PROTECTED         (price beyond TP1 — breakeven hit)
  ├── → PARTIAL_EXIT      (TP1 reached — 50% closed)
  ├── → EXIT_READY        (exit signal)
  ├── → CLOSING           (direct close)
  ├── → STOPPED           (stop loss hit)
  ├── → FORCED_EXIT       (external closure)
  └── → EMERGENCY_EXIT    (critical event)

POSITION_BUILDING
  ├── → ACTIVE            (scale-in complete or aborted)
  ├── → PROTECTED         (breakeven triggered during build)
  ├── → STOPPED           (stop hit during build)
  └── → EMERGENCY_EXIT    (critical event)

PROTECTED
  ├── → PARTIAL_EXIT  (TP2 approaching)
  ├── → RUNNER        (TP2 hit — runner active)
  ├── → EXIT_READY    (exit signal)
  └── → STOPPED       (stop hit — scratch)

PARTIAL_EXIT
  ├── → RUNNER        (TP2 hit — runner active)
  ├── → TRAILING      (trail activated on remainder)
  └── → CLOSING       (direct close remaining)

RUNNER
  ├── → TRAILING      (trail activated)
  ├── → EXIT_READY    (target hit)
  └── → CLOSING       (direct close)

TRAILING
  ├── → EXIT_READY    (trail hit OR target reached)
  ├── → CLOSING       (direct close)
  └── → STOPPED       (trail stop triggered)
```

### Resolution States

```
EXIT_READY  → CLOSING
CLOSING     → CLOSED
STOPPED     → CLOSED
FORCED_EXIT → CLOSED
EMERGENCY_EXIT → CLOSED
```

## Using the TransitionEngine

```python
from trainer.lifecycle import TransitionEngine, LifecycleState, MarketContext

engine = TransitionEngine()
ctx    = MarketContext(price=1.2500, atr=0.001, news_risk=0.15)

rec = engine.evaluate(
    current_state = LifecycleState.PREPARATION,
    confidence    = 0.68,
    context       = ctx,
)

print(rec.recommended)    # LifecycleState.ENTRY_READY
print(rec.confidence)     # 0.742
print(rec.reason)         # "Entry ready when all pre-trade conditions satisfied"
print(rec.should_stay)    # False

# Get all candidates
for candidate in rec.all_candidates:
    print(candidate.to_state.value, candidate.allowed, candidate.confidence)
```

## Auto-Transition via API

```python
from trainer.lifecycle import LifecycleAPI

api = LifecycleAPI()
r   = api.auto_transition(trade_id)
# Returns: {"action": "TRANSITIONED" | "STAY", "new_state": ..., "reason": ...}
```

## Timeout Handling

Each state has a configured timeout (see `conf/lifecycle/lifecycle.yaml`).
If a trade remains in a state beyond its timeout without transitioning,
the lifecycle engine will recommend EXPIRED or CANCELLED depending on the
pre-entry vs position phase.
