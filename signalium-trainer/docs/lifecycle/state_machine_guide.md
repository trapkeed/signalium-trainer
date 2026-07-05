# State Machine Guide

## Overview

The `StateMachine` class implements a deterministic finite state machine (FSM)
for trade lifecycle management. Every possible state transition has:

- **Entry Conditions** — what must be true to enter the new state
- **Exit Conditions** — conditions that signal readiness to leave the current state
- **Validation Rules** — constraints checked at transition time
- **Timeout Rules** — maximum time to wait in a state
- **Confidence** — base certainty when conditions are met
- **Expected Benefit** — upside from the transition (0–1)
- **Risk Rating** — LOW, MEDIUM, HIGH, or CRITICAL

## Transition Rules

All rules are declared in `state_machine.py` as `TransitionRule` objects
and stored in the `_RULES` registry. A fast `(from_state, to_state)` lookup
dictionary (`_RULE_LOOKUP`) enables O(1) rule retrieval.

### Example Rule

```
PREPARATION → ENTRY_READY

Entry Conditions:
  - Price near entry zone
  - All conditions confirmed

Exit Conditions:
  - Spread acceptable
  - Liquidity adequate
  - No news risk

Validation Rules:
  - confidence >= 0.65
  - news_risk < 0.40
  - spread_ok
  - trend_valid

Timeout: 24 hours
Base Confidence: 0.70
Expected Benefit: 0.60
Risk Rating: MEDIUM
```

## Usage

```python
from trainer.lifecycle import StateMachine, LifecycleState, MarketContext

sm  = StateMachine()
ctx = MarketContext(price=1.2500, atr=0.001, news_risk=0.20, trend_valid=True)

result = sm.validate_transition(
    from_state = LifecycleState.PREPARATION,
    to_state   = LifecycleState.ENTRY_READY,
    context    = ctx,
    confidence = 0.72,
)

print(result.allowed)           # True / False
print(result.conditions_met)    # List of satisfied conditions
print(result.conditions_failed) # List of failed conditions
print(result.reason)            # Human-readable explanation
```

## Querying Rules

```python
# All rules leaving ACTIVE state
rules = sm.rules_for_state(LifecycleState.ACTIVE)

# Specific rule
rule = sm.get_rule(LifecycleState.ACTIVE, LifecycleState.PROTECTED)
print(rule.timeout_hours)  # 48.0

# All rules
all_rules = sm.all_rules()
```

## Context Validation

When a `MarketContext` is provided, the state machine checks:

| Rule | Passes if |
|---|---|
| `news_risk < 0.40` | `context.news_risk < 0.40` |
| `news_risk < 0.30` | `context.news_risk < 0.30` |
| `spread_ok` | `spread < atr × 0.30` |
| `trend_valid` | `context.trend_valid == True` |
| `session_ok` | `session != "CLOSED"` |
| `momentum_aligned` | `abs(momentum) > 0.20` |

## Risk Ratings

| Rating | Meaning |
|---|---|
| LOW | Standard transition, well-defined path |
| MEDIUM | Some uncertainty — monitor closely |
| HIGH | Significant risk — stop-loss or exit paths |
| CRITICAL | Emergency conditions — immediate action required |

## Adding New Rules

Add a `TransitionRule` to the `_RULES` list in `state_machine.py`:

```python
TransitionRule(
    LifecycleState.MY_FROM, LifecycleState.MY_TO,
    entry_conditions  = ["Condition A", "Condition B"],
    exit_conditions   = ["Exit trigger X"],
    validation_rules  = ["confidence >= 0.65", "spread_ok"],
    timeout_hours     = 12.0,
    base_confidence   = 0.70,
    expected_benefit  = 0.60,
    risk_rating       = "MEDIUM",
    description       = "Explanation of this transition",
)
```

The `_RULE_LOOKUP` dict is built automatically at module load time.
