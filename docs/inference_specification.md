# Inference Specification

**Phase**: 14.5 | **Model**: QuantModelV1 | **Output Version**: 2.0.0

---

## Inference Contract

A single call to `adapter.predict(x)` returns a dict containing all of the
following fields. Fields for disabled heads are absent from the dict.

### Always Present

| Field              | Type          | Shape      | Range     | Description                                      |
|--------------------|---------------|------------|-----------|--------------------------------------------------|
| `buy_prob`         | FloatTensor   | (batch,)   | [0, 1]    | P(BUY) — softmax probability                     |
| `sell_prob`        | FloatTensor   | (batch,)   | [0, 1]    | P(SELL) — softmax probability                    |
| `no_trade_prob`    | FloatTensor   | (batch,)   | [0, 1]    | P(NO TRADE) — softmax probability                |
| `probabilities`    | FloatTensor   | (batch, 3) | [0, 1]    | Full softmax distribution [BUY, SELL, NO_TRADE]  |
| `logits`           | FloatTensor   | (batch, 3) | ℝ         | Raw direction logits (use for CE loss)           |
| `predicted_class`  | LongTensor    | (batch,)   | {0, 1, 2} | Argmax class (0=BUY, 1=SELL, 2=NO_TRADE)         |
| `confidence`       | FloatTensor   | (batch, 1) | [0, 1]    | Calibrated model certainty                       |

### Optional (when head is enabled)

| Field                    | Type        | Shape      | Range  | Description                          |
|--------------------------|-------------|------------|--------|--------------------------------------|
| `expected_move`          | FloatTensor | (batch, 1) | ℝ      | Predicted move (signed, in config units) |
| `stop_loss`              | FloatTensor | (batch, 1) | [0, ∞) | Recommended SL distance from entry  |
| `take_profit`            | FloatTensor | (batch, 1) | [0, ∞) | Recommended TP distance from entry  |
| `holding_time`           | FloatTensor | (batch, 1) | [0, ∞) | Expected trade duration              |
| `tp_probability`         | FloatTensor | (batch, 1) | [0, 1] | P(TP reached before SL)             |
| `prediction_timestamp`   | str         | —          | —      | UTC ISO-8601 timestamp of prediction |
| `model_version`          | str         | —          | —      | Semantic version, e.g. "1.1.0"      |

---

## Consuming the Inference Output

### Python (eager mode)

```python
from trainer.framework.inference import InferenceAdapterFactory

adapter = InferenceAdapterFactory.from_model(model)
result = adapter.predict(x)   # x: (batch, seq_len, feature_dim)

# Direction
buy_prob      = result["buy_prob"]         # FloatTensor (batch,)
sell_prob     = result["sell_prob"]
no_trade_prob = result["no_trade_prob"]
predicted     = result["predicted_class"]  # LongTensor (batch,)

# Quantitative outputs
if "expected_move" in result:
    move = result["expected_move"]    # (batch, 1)
if "stop_loss" in result:
    sl   = result["stop_loss"]        # (batch, 1) ≥ 0
if "take_profit" in result:
    tp   = result["take_profit"]      # (batch, 1) ≥ 0
if "holding_time" in result:
    ht   = result["holding_time"]     # (batch, 1) ≥ 0
if "tp_probability" in result:
    tp_p = result["tp_probability"]   # (batch, 1) ∈ [0, 1]
```

### Python (TorchScript .pt file)

```python
adapter = InferenceAdapterFactory.from_artifact(Path("model.torchscript.pt"))
result = adapter.predict(x)
# Returns direction + confidence only (TorchScript export strips extra heads)
```

### Confidence Gating

```python
# Suppress low-confidence signals by mapping them to NO_TRADE
adapter = InferenceAdapterFactory.from_model(model, confidence_threshold=0.6)
result = adapter.predict(x)
# predicted_class is NO_TRADE wherever confidence < 0.6
```

---

## Class Index Convention

| Index | Label      |
|-------|------------|
| 0     | BUY        |
| 1     | SELL       |
| 2     | NO_TRADE   |

This convention is **fixed across all QuantModelV1 versions** and must not
be changed without a major version bump.

---

## Unit Convention

Output units for quantitative heads are set at training time via the YAML
config and stored in the export manifest. Always check the manifest before
consuming regression outputs:

```json
{
  "stop_loss_unit": "pips",
  "take_profit_unit": "pips",
  "holding_time_unit": "minutes"
}
```

---

## Error Conditions

| Condition                         | Behaviour                             |
|-----------------------------------|---------------------------------------|
| Head disabled, key accessed       | Key absent from result dict           |
| confidence < threshold            | predicted_class replaced with NO_TRADE |
| NaN / Inf in input features       | Undefined (caller must sanitise input) |
| Batch size 0                      | Returns empty tensors                 |
| TorchScript adapter + multi-task  | Only direction + confidence returned  |

---

## Versioning

The inference schema version is embedded in the export manifest under
`output_version`. Breaking changes (field removal, type changes) require a
major version bump. Field additions are non-breaking and increment only the
minor version.

| Version | Changes                                             |
|---------|-----------------------------------------------------|
| 1.0.0   | Original — direction + confidence only              |
| 2.0.0   | Phase 14.5 — added quantitative prediction heads    |
