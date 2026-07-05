# Ensemble Meta Learning Guide

## Overview

The **EnsembleMetaModel** is a learned combiner that improves upon simple weighted averaging by conditioning its combination on market context (regime, instrument, timeframe) and expert performance history.

---

## Architecture

```
Inputs:
  Expert Predictions (buy/sell/no_trade per expert)
  Per-expert confidence and uncertainty
  Expert weights from DynamicWeightEngine
  Market Regime (one-hot encoded)
  [Optional: instrument, timeframe]

         ↓

  MetaFeatureBuilder → feature vector

         ↓

  LinearMetaLayer (logistic regression, online SGD)
    - 3 output logits: buy, sell, no_trade
    - Softmax → final probabilities
    - L2 regularization

         ↓

  MetaPrediction:
    buy_prob, sell_prob, no_trade_prob
    confidence, uncertainty
    institutional_forecast (grade A/B/C/D)
```

---

## When the Meta Model Activates

The meta model has a **warm-up phase**:
- During warm-up (n_samples < `meta_min_samples`): falls back to weighted averaging.
- After warm-up: uses the learned linear layer.

```yaml
ensemble:
  meta_min_samples: 50    # samples before meta model activates
  meta_learning_rate: 0.01
```

---

## Training the Meta Model

### Online Learning (Automatic)

Each call to `engine.record_outcome()` provides a labeled example for meta model training:

```python
bundle = engine.predict(features, context)
# ... time passes, ground truth becomes known ...
engine.update_meta_model(
    expert_predictions=bundle.expert_predictions,
    weights=bundle.routing_decision.weights,
    true_class="buy",
    context=bundle.context,
)
```

### Batch Training from History

```python
# Build examples from historical data
examples = [
    {
        "predictions": [p.to_dict() for p in expert_predictions],
        "weights": routing_decision.weights,
        "context": {"regime": "trending_up", "instrument": "EURUSD"},
        "true_class": "buy",
    }
    for expert_predictions, routing_decision, true_class in historical_data
]

# Fit with multiple epochs
losses = engine.meta_model.fit(examples, epochs=5)
print(f"Final avg loss: {sum(losses[-100:]) / 100:.4f}")
```

---

## Meta Prediction Format

```python
bundle = engine.predict(features, context)
meta = bundle.meta_prediction

print(meta.buy_prob)               # float [0, 1]
print(meta.sell_prob)              # float [0, 1]
print(meta.no_trade_prob)          # float [0, 1]
print(meta.confidence)             # float [0, 1]
print(meta.uncertainty)            # float [0, 1]
print(meta.agreement_score)        # inter-expert agreement [0, 1]
print(meta.n_training_samples)     # number of training examples used
print(meta.institutional_forecast) # "[A] LONG | trending_up | conf=0.85"

# Institutional grade:
#   A = conf >= 0.80 AND agree >= 0.75
#   B = conf >= 0.65 AND agree >= 0.60
#   C = conf >= 0.50
#   D = low confidence
```

---

## Persistence

The meta model state is saved to the database:

```python
meta_dict = engine.meta_model.to_dict()
engine.db.save_meta_model(meta_dict)

# Load latest state
meta_dict = engine.db.get_latest_meta_model()
engine.meta_model = EnsembleMetaModel.from_dict(meta_dict)
```

---

## Meta Feature Vector

The `MetaFeatureBuilder` constructs a flat feature vector per prediction:

| Feature Group               | Size | Description                               |
|-----------------------------|------|-------------------------------------------|
| Average expert probs        | 8    | Mean buy/sell/nt, conf, unc, std, n       |
| Weighted per-expert probs   | 4×N  | w×buy, w×sell, w×nt, w×conf per expert   |
| Regime one-hot              | 10   | One-hot over 10 regime categories         |

Feature vector grows with more experts — the layer auto-resizes.

---

## Performance Monitoring

```python
# Check training progress
print(f"Is trained: {engine.meta_model.is_trained}")
print(f"Training samples: {engine.meta_model._n_training_samples}")
print(f"Recent loss: {engine.meta_model.training_loss:.4f}")

# Loss history (last 100 examples)
history = engine.meta_model._training_loss_history[-100:]
```
