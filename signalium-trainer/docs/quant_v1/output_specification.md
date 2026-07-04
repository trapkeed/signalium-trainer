# QuantModelV1 — Output Specification

## `QuantModelOutput` Fields

Every inference call returns a `QuantModelOutput` dataclass:

```python
@dataclass
class QuantModelOutput:
    logits:          torch.Tensor  # (batch, 3)   — raw, for loss computation
    probabilities:   torch.Tensor  # (batch, 3)   — softmax, sums to 1.0
    confidence:      torch.Tensor  # (batch, 1)   — ∈ [0, 1]
    predicted_class: torch.Tensor  # (batch,)     — integer 0, 1, or 2
    buy_prob:        torch.Tensor  # (batch,)     — P(BUY)
    sell_prob:       torch.Tensor  # (batch,)     — P(SELL)
    no_trade_prob:   torch.Tensor  # (batch,)     — P(NO TRADE)
```

---

## Class Index Map

| Index | Label | Meaning |
|---|---|---|
| 0 | `BUY` | Model predicts bullish opportunity |
| 1 | `SELL` | Model predicts bearish opportunity |
| 2 | `NO_TRADE` | Model predicts no high-quality opportunity |

---

## Probability Distribution

`probabilities[:, 0]` + `probabilities[:, 1]` + `probabilities[:, 2]` = 1.0 (guaranteed by Softmax).

```python
out = model.predict(x)
print(f"BUY:      {out.buy_prob[0]:.3f}")
print(f"SELL:     {out.sell_prob[0]:.3f}")
print(f"NO TRADE: {out.no_trade_prob[0]:.3f}")
print(f"Signal:   {out_labels[out.predicted_class[0]]}")
print(f"Confidence: {out.confidence[0].item():.3f}")
```

---

## Confidence Score

The confidence score is independent of the predicted class probability. It is produced
by a separate `ConfidenceHead` trained jointly with the classifier.

- A **high confidence + high BUY probability** → strong BUY signal
- A **high probability but low confidence** → model is uncertain despite the prediction
- Downstream systems should gate actions on a minimum confidence threshold

---

## Using in Loss Computation

The training pipeline uses `logits` (not `probabilities`) for loss:

```python
logits = model(x)            # returns logits via forward()
loss = criterion(logits, y)  # CrossEntropyLoss expects raw logits
```

For full structured output:
```python
out = model.predict(x)       # returns QuantModelOutput
```

---

## ONNX Export Output

The ONNX model's single output corresponds to `logits` (shape `[batch, 3]`).
Apply softmax post-inference to obtain probabilities:

```python
import numpy as np
logits = onnx_session.run(None, {"input": x_np})[0]  # (batch, 3)
probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
```
