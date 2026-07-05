# IRSE Weight Configuration Guide

## Overview

All IRSE category weights are configurable without modifying code.

Weights are loaded from:
1. `conf/research/scoring.yaml` — default weights and named presets
2. `ScoringWeights.from_dict(d)` — runtime dict override
3. `IRSEApi.set_weights(d)` — API-level update (persisted to DB)
4. `db.save_weight_configuration(name, weights)` — named preset storage

---

## Validation Rules

- All weights must be **non-negative**
- Weights must **sum to 1.0** (tolerance: 1e-6)
- Weights are validated on creation and on any update call

```python
from trainer.research.weights import ScoringWeights

w = ScoringWeights(
    validation_quality=0.30,
    trading_performance=0.30,
    robustness=0.15,
    calibration=0.08,
    generalization=0.08,
    efficiency=0.04,
    explainability=0.03,
    operational_readiness=0.02,
)
w.validate()       # raises ValueError if invalid
w_norm = w.normalise()  # returns copy normalised to sum=1.0
```

---

## Built-in Presets

```yaml
# conf/research/scoring.yaml

weight_presets:
  validation_heavy:
    validation_quality: 0.40
    trading_performance: 0.25
    # ...

  trading_heavy:
    validation_quality: 0.20
    trading_performance: 0.40
    # ...

  efficiency_focused:
    efficiency: 0.20
    # ...
```

Load a preset at runtime:

```python
import yaml
config = yaml.safe_load(open("conf/research/scoring.yaml"))
preset = config["weight_presets"]["validation_heavy"]
weights = ScoringWeights.from_yaml_section(preset)
scorer.update_weights(weights)
```

---

## API Weight Management

```python
# Get current weights
resp = api.get_weights()

# Set new weights
resp = api.set_weights({
    "validation_quality":    0.35,
    "trading_performance":   0.30,
    "robustness":            0.15,
    "calibration":           0.07,
    "generalization":        0.07,
    "efficiency":            0.02,
    "explainability":        0.02,
    "operational_readiness": 0.02,
})

# Save a named preset
resp = api.save_weight_preset(
    "q4_2026_config",
    weights_dict,
    description="Q4 2026 production weighting"
)

# List all presets
resp = api.list_weight_presets()
```

---

## Sensitivity Analysis

Before changing weights, use sensitivity analysis to understand the
impact of each weight on the composite score:

```python
scorer = ResearchScorer()
inputs = my_experiment_inputs

# Returns: {category: score_change} for +5% weight on each category
sensitivity = scorer.sensitivity_analysis(inputs, delta=0.05)

for cat, impact in sorted(sensitivity.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {cat:<24} impact: {impact:+.3f}")
```

---

## Design Guidance

| Goal                             | Adjustment                    |
|----------------------------------|-------------------------------|
| Stricter live performance filter | Increase `trading_performance`|
| Prioritise regime robustness     | Increase `robustness`         |
| Latency-critical deployment      | Increase `efficiency`         |
| Regulatory/audit requirements    | Increase `explainability` + `operational_readiness` |
| Research mode (no deployment)    | Reduce `operational_readiness`, increase `validation_quality` |

---

## Impact of Weight Changes

Weight changes affect **all existing scores in the leaderboard** if you
re-score experiments with new weights.  Scores stored in the database are
frozen at the weights used at scoring time — `weights_used` is stored per
score record.

To re-score all experiments with new weights:

```python
scorer.update_weights(new_weights)
for inputs in all_experiment_inputs:
    api.score_experiment(inputs)  # overwrites existing record
```
