# Counterfactual Analysis Guide — Phase 21.6

## Overview

The Counterfactual Engine evaluates alternative scenarios for every completed decision. It answers questions like "What if BUY had become WAIT?" and "What if TP was 50% larger?" — allowing Signalium to continuously learn from alternatives not taken.

## Why Counterfactuals?

Counterfactual analysis is essential for:
- Identifying systematic decision improvements
- Understanding where TP/SL sizing needs adjustment
- Detecting overconfidence patterns
- Comparing aggressive vs conservative execution approaches

## Scenarios Generated

| Scenario | Alternative Action | Description |
|---|---|---|
| WAIT instead of BUY/SELL | WAIT | No position, PnL = 0 |
| NO_TRADE taken | NO_TRADE | Complete abstention |
| TP set at 1.5× | LARGER_TP | Higher target, lower probability |
| SL set at 0.7× | TIGHTER_SL | Smaller stop, higher frequency |
| Delayed entry | DELAYED_ENTRY | Wait for pullback before entry |
| Earlier trailing stop | EARLY_TRAIL | Lock in gains at 50–80% of move |

## Counterfactual Result Structure

```
CounterfactualResult
├── scenario           # Human-readable description
├── alternative_action # Action label (WAIT, LARGER_TP, etc.)
├── estimated_pnl      # Estimated outcome PnL (price units)
├── estimated_pnl_pct  # Estimated outcome PnL (%)
├── probability        # P(this outcome under alternative)
├── better_than_actual # True if alternative was better
├── delta_pnl          # alternative_pnl - actual_pnl
└── notes              # Additional context
```

## Usage

### Analyse a single decision

```python
from trainer.memory import MemoryAPI, DecisionType, OutcomeType

api = MemoryAPI(db_path="decisions.db")

# Record decision
record = api.record_decision(
    DecisionType.BUY, "EURUSD", snapshot, reasoning
)

# Record outcome (required before counterfactual analysis)
api.record_outcome(
    record.decision_id,
    OutcomeType.HIT_SL,
    actual_pnl     = -0.00060,
    hit_sl         = True,
    holding_time_h = 3.5,
)

# Analyse counterfactuals
scenarios = api.analyse_counterfactuals(record.decision_id)
for cf in scenarios:
    print(f"{cf['alternative_action']}: est_pnl={cf['estimated_pnl']:.5f}, "
          f"better={cf['better_than_actual']}")
```

### Find best alternative

```python
from trainer.memory import CounterfactualEngine

engine = CounterfactualEngine()
best = engine.best_alternative(record)
print(f"Best alternative: {best.alternative_action} "
      f"(estimated PnL: {best.estimated_pnl:.5f})")
```

### Count how many alternatives were better

```python
count = engine.better_alternatives_count(record)
print(f"{count} alternative scenarios outperformed the actual decision")
```

## Counterfactual Explorer Dashboard

```python
dashboard = api.get_dashboard("counterfactual_explorer")
# Returns:
# {
#   "view": "COUNTERFACTUAL_EXPLORER",
#   "total": 120,
#   "total_scenarios": 480,
#   "scenario_summary": {
#     "WAIT":       {"count": 120, "better_rate": 0.35, "avg_delta_pnl": -0.0001},
#     "LARGER_TP":  {"count": 120, "better_rate": 0.40, "avg_delta_pnl": +0.0002},
#     "TIGHTER_SL": {"count": 120, "better_rate": 0.55, "avg_delta_pnl": +0.0003},
#     ...
#   }
# }
```

## Interpretation

**`better_rate` for TIGHTER_SL > 0.50** → Stops are consistently too wide.
Consider reducing stop sizes by 0.30× ATR.

**`better_rate` for LARGER_TP > 0.50** → Targets are consistently too tight.
Consider extending take profits by 0.50× ATR.

**`better_rate` for WAIT > 0.40 on entry decisions** → Entry timing may be premature.
Consider adding a pullback confirmation before entry.

**`better_rate` for NO_TRADE > 0.50** → Filters need tightening.
Too many marginal decisions are reducing overall quality.

## Probability Model

Each counterfactual uses a simplified probability model:

| Scenario | Probability Logic |
|---|---|
| WAIT | 0.50 — equal chance market improves or worsens |
| LARGER_TP | Original TP_prob × 0.65 — 35% penalty for harder target |
| TIGHTER_SL | Original TP_prob × 0.80 — slightly higher stop-out risk |
| DELAYED_ENTRY | 0.40 — pullback may or may not materialise |
| EARLY_TRAIL (TP hit) | 0.75 — likely captures large fraction of move |
| EARLY_TRAIL (SL hit) | 0.60 — likely prevents full SL loss |

## Configuration

```yaml
counterfactual_engine:
  larger_tp_factor:  1.50
  tight_sl_factor:   0.70
  wait_probability:  0.50
```
