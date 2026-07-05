# Hyperparameter Optimization Guide

## Overview

Phase 16 supports four hyperparameter optimization strategies, configurable entirely through YAML. The search space covers all 21 configurable parameters defined in the Phase 16 specification.

---

## Supported Parameters

| Parameter           | Type        | Range/Choices                           | Log Scale |
|---------------------|-------------|------------------------------------------|-----------|
| `learning_rate`     | Float       | [1e-5, 1e-2]                            | Yes       |
| `batch_size`        | Int         | [16, 256] step 16                       | No        |
| `optimizer`         | Categorical | adam, adamw, sgd, rmsprop               | —         |
| `weight_decay`      | Float       | [1e-6, 1e-2]                            | Yes       |
| `dropout`           | Float       | [0.0, 0.5] step 0.05                   | No        |
| `hidden_size`       | Int         | [64, 512] step 64                       | No        |
| `embedding_size`    | Int         | [32, 256] step 32                       | No        |
| `num_layers`        | Int         | [1, 6]                                  | No        |
| `sequence_length`   | Int         | [20, 200] step 20                       | No        |
| `backbone`          | Categorical | lstm, gru, transformer, cnn, hybrids    | —         |
| `fusion_strategy`   | Categorical | concat, attention, mean, max            | —         |
| `gradient_clipping` | Float       | [0.1, 5.0]                             | No        |
| `warmup_steps`      | Int         | [0, 1000] step 100                      | No        |
| `scheduler`         | Categorical | cosine, linear, constant, onecycle      | —         |
| `activation`        | Categorical | relu, gelu, silu, tanh                  | —         |
| `normalization`     | Categorical | layer_norm, batch_norm, group_norm      | —         |
| `attention_heads`   | Int (cond.) | [2, 16] step 2 (transformer only)      | No        |
| `momentum`          | Float (cond.)| [0.8, 0.99] (SGD only)                | No        |

---

## Defining a Search Space

```python
from trainer.automl import (
    SearchSpace, FloatParameter, IntParameter,
    CategoricalParameter, ConditionalParameter,
)

space = SearchSpace("my_space")
space.add(FloatParameter("learning_rate", low=1e-5, high=1e-2, log=True))
space.add(IntParameter("batch_size", low=32, high=128, step=32))
space.add(CategoricalParameter("optimizer", choices=["adam", "adamw"]))
space.add(CategoricalParameter("backbone", choices=["lstm", "transformer_encoder"]))

# Conditional: attention_heads only when backbone == transformer
space.add(ConditionalParameter(
    name="attention_heads",
    parent="backbone",
    parent_value="transformer_encoder",
    parameter=IntParameter("attention_heads", low=2, high=8, step=2),
    default=4,
))

# Sample one configuration
config = space.sample()

# Enumerate grid (3 values per float/int param)
grid = space.grid(max_per_param=3)
```

---

## Strategy Guide

### Random Search

Best for: initial exploration, large spaces, limited compute.

```yaml
automl:
  strategy: random
  n_trials: 50
  seed: 42
```

Guarantees: every trial is independent. Parallelizes trivially.

---

### Bayesian Optimization

Best for: expensive evaluations where each trial counts.

```yaml
automl:
  strategy: bayesian
  n_trials: 30
  strategy_config:
    n_initial_random: 8   # random warm-up before GP kicks in
    n_candidates: 100     # candidates evaluated per acquisition step
    exploration: 0.01     # EI exploration factor xi
```

Uses a Gaussian Process surrogate with Expected Improvement (EI) acquisition. Pure Python — no external dependencies.

---

### Grid Search

Best for: small spaces (≤ 3 parameters, ≤ 5 values each).

```yaml
automl:
  strategy: grid
  strategy_config:
    max_per_param: 4  # values per float/int parameter
```

Grid is pre-enumerated. All categorical choices are always included.

---

### Manual Search

Best for: reproducing specific configurations, ablation studies.

```python
from trainer.automl import ManualStrategy, SearchSpace

strategy = ManualStrategy(space, configs=[
    {"learning_rate": 0.001, "backbone": "lstm"},
    {"learning_rate": 0.0001, "backbone": "transformer_encoder"},
])
```

---

## Pruning

Prune poor-performing trials early to save compute.

### Median Pruner (recommended)

```yaml
pruner: median
pruner_config:
  n_startup_trials: 5    # don't prune until 5 trials complete
  n_warmup_steps: 2      # don't prune within first 2 steps of a trial
  percentile: 50.0       # prune if below median
```

### Successive Halving

```yaml
pruner: successive_halving
pruner_config:
  min_resource: 1
  reduction_factor: 3    # keep top 1/3 at each rung
```

### Disable Pruning

```yaml
pruner: none
```

---

## YAML Search Space Override

Override individual parameters without changing code:

```yaml
automl:
  custom_search_space:
    name: narrow_sweep
    parameters:
      - type: float
        name: learning_rate
        low: 0.0005
        high: 0.005
        log: true
      - type: categorical
        name: backbone
        choices: [lstm, gru]
    conditionals: []
```

---

## Objectives

```python
from trainer.automl import build_research_scorer, ObjectiveScorer, Objective

# Default research scorer (5 objectives, weighted)
scorer = build_research_scorer()

# Custom scorer
scorer = ObjectiveScorer([
    Objective("sharpe_ratio", "maximize", weight=0.5),
    Objective("max_drawdown", "minimize", weight=0.3),
    Objective("profit_factor", "maximize", weight=0.2),
])

# Score a single metrics dict
score = scorer.composite_score(metrics)

# Pareto front
front_indices = scorer.pareto_front(metrics_list)
```

Available standard objectives: `profit_factor`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `expected_calibration_error`, `validation_score`, `latency_ms`, `memory_mb`, `win_rate`, `accuracy`, `f1_score`, `composite_research_score`, `calmar_ratio`, `omega_ratio`.
