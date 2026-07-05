# Expert Routing Guide

## Overview

The **ExpertRouter** intelligently selects which experts contribute to a given prediction based on current market context. This ensures that the most relevant experts have the highest influence for each specific prediction.

---

## Routing Logic

```
Context: {regime, instrument, timeframe}
           ↓
1. Filter by instrument (if expert.instruments is set)
2. Filter by timeframe  (if expert.timeframes is set)
3. Compute weights via DynamicWeightEngine
4. Apply min_weight_threshold (optional)
5. Select top-N experts (optional)
6. Normalize weights → sum to 1.0
           ↓
RoutingDecision: {selected_expert_ids, weights, routing_reason}
```

---

## Regime → Expert Affinity

Built-in routing priorities by regime:

| Regime          | Preferred Architecture         | Why                            |
|-----------------|--------------------------------|--------------------------------|
| `trending_up`   | Transformer, Hybrid LSTM-Trans | Strong sequential pattern      |
| `trending_down` | Transformer, Hybrid LSTM-Trans | Directional persistence        |
| `ranging`       | LSTM, GRU                     | Mean-reversion cycles          |
| `volatile`      | Temporal CNN, Hybrid GRU-Trans | Fast local feature extraction  |
| `high_volatility` | Temporal CNN               | Volatility spike detection     |
| `low_volatility`  | LSTM                        | Slow, stable pattern capture   |
| `breakout`      | Transformer, TCN              | Pattern-break detection        |
| `pullback`      | LSTM, GRU                    | Retracement recognition        |
| `crisis`        | Temporal CNN                  | Tail-risk regime               |

These are applied as a multiplicative boost (`regime_boost`, default 2.0×) to experts whose architecture or specialization matches the detected regime.

---

## Usage

```python
from trainer.ensemble import ExpertRouter, RouterConfig, DynamicWeightEngine

# Default routing: all active experts, all filters enabled
router = ExpertRouter(registry, DynamicWeightEngine())

decision = router.route(context={
    "regime": "trending_up",
    "instrument": "EURUSD",
    "timeframe": "H1",
})
print(f"Selected {decision.n_experts_selected} experts")
print(f"Reason: {decision.routing_reason}")
for eid, w in decision.weights.items():
    print(f"  {eid}: weight={w:.3f}")
```

---

## Configuration

```python
from trainer.ensemble import RouterConfig

config = RouterConfig(
    min_experts=2,          # always select at least 2 experts
    max_experts=5,          # never more than 5 experts per prediction
    min_weight_threshold=0.05,  # exclude experts with < 5% weight
    use_specialization_filter=True,
    use_instrument_filter=True,
    use_timeframe_filter=True,
    fallback_to_all=True,   # if filters leave 0 experts, use all
)
```

---

## Dynamic Weight Factors

The `DynamicWeightEngine` computes each expert's weight from 7 factors:

| Factor                    | Default Weight | Description                           |
|---------------------------|---------------|---------------------------------------|
| `recent_performance`      | 0.35          | Rolling accuracy + Sharpe             |
| `regime_match`            | 0.25          | Architecture/specialization vs regime |
| `instrument_match`        | 0.15          | Expert instrument specialization      |
| `timeframe_match`         | 0.10          | Expert timeframe specialization       |
| `calibration`             | 0.10          | Calibration quality (1 - ECE)         |
| `uncertainty`             | 0.05          | Penalty for high uncertainty          |
| `history`                 | 0.05          | Lifetime accuracy                     |

Weights update every `weight_adjust_interval` predictions (default: every 50).

---

## Weight Evolution

Weights adapt as market conditions change:

```python
# Get weight evolution chart for all experts
resp = engine.api.get_weight_evolution()
chart = resp["data"]["chart"]

# Or access raw history
history = engine.weight_engine.all_weight_evolution()
for expert_id, weights in history.items():
    print(f"{expert_id}: {len(weights)} weight snapshots")
```

---

## Custom Weight Config

```python
from trainer.ensemble import DynamicWeightEngine, WeightFactorConfig

custom_cfg = WeightFactorConfig(
    recent_performance_factor=0.50,   # emphasize recent performance
    regime_match_factor=0.30,
    instrument_match_factor=0.10,
    timeframe_match_factor=0.05,
    calibration_factor=0.03,
    uncertainty_factor=0.01,
    history_factor=0.01,
    regime_boost=3.0,                 # sharper regime routing
)
engine.weight_engine = DynamicWeightEngine(config=custom_cfg)
```

---

## Routing History

```python
# All routing decisions
resp = engine.api.get_routing_history(regime="trending_up", limit=50)
history = resp["data"]["history"]
for decision in history:
    print(f"{decision['decided_at']} → {decision['n_experts_selected']} experts")

# Routing statistics
stats = engine.router.routing_stats()
print(f"Total routes: {stats['total_routes']}")
print(f"By regime: {stats['regime_distribution']}")
print(f"Avg experts: {stats['avg_experts_selected']:.1f}")
```
