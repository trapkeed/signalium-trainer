# Multi-Task Quant Model Architecture Guide

**Phase**: 14.5  
**Version**: 1.1.0  
**Model**: QuantModelV1 Multi-Task

---

## Overview

Phase 14.5 extends QuantModelV1 from a single direction classifier into a
**multi-task quantitative prediction model**. A single forward pass now
produces seven distinct predictions that share the same learned market
representation — with zero additional backbone passes.

The shared encoder and backbone are **completely unchanged**. All new
capability is added through independent prediction heads that branch off the
same context vector.

---

## Architecture Diagram

```
Input (batch, seq_len, input_dim)
    │
    ▼
InputEncoder
    LayerNorm → Linear(input_dim → hidden_dim) → GELU → Dropout
    │
    ▼
SequenceEncoder (LSTM / GRU backbone)
    │
    context vector (batch, backbone_output_dim)
    │
    ├──► ClassificationHead  → direction logits (batch, 3)         [ALWAYS ON]
    │        Linear → LayerNorm → GELU → Dropout → Linear
    │        (Softmax applied in OutputLayer)
    │
    ├──► ConfidenceHead       → confidence (batch, 1) ∈ [0,1]      [ALWAYS ON]
    │        Linear → GELU → Dropout → Linear → Sigmoid
    │
    ├──► ExpectedMoveHead     → move size (batch, 1)               [optional]
    │        Linear → GELU → Dropout → Linear
    │        (unrestricted output: positive=up, negative=down)
    │
    ├──► StopLossHead         → SL distance (batch, 1) ≥ 0         [optional]
    │        Linear → GELU → Dropout → Linear → Softplus
    │
    ├──► TakeProfitHead       → TP distance (batch, 1) ≥ 0         [optional]
    │        Linear → GELU → Dropout → Linear → Softplus
    │
    ├──► HoldingTimeHead      → duration (batch, 1) ≥ 0            [optional]
    │        Linear → GELU → Dropout → Linear → Softplus
    │
    └──► TPProbabilityHead    → P(TP before SL) (batch, 1) ∈ [0,1] [optional]
             Linear → GELU → Dropout → Linear → Sigmoid
```

---

## Design Principles

### 1. Shared Representation, No Extra Backbone Passes
All heads consume the **same context vector** produced by the backbone. There
is exactly one encoder pass and one backbone pass per forward call — the cost
of adding six heads is proportional only to the size of those small head
networks.

### 2. Strict Backward Compatibility
- `forward(x)` still returns **only the direction logits** (batch, num_classes).
  Existing training loops using CrossEntropyLoss work without modification.
- `predict(x)` returns `MultiTaskQuantModelOutput`, a strict superset of the
  original `QuantModelOutput`. Callers can call `.to_base_output()` to get the
  original type.
- All new heads are **disabled by default**. Existing configs produce identical
  behaviour.

### 3. Opt-In by Configuration
Enable any head by setting `enabled: true` in the YAML config:

```yaml
stop_loss_head:
  enabled: true
  unit: pips
```

Disabled heads add **zero parameters** and **zero compute** to the model.

### 4. Non-Negativity Guarantees
Stop Loss, Take Profit, and Holding Time distances are physically constrained
to be non-negative. These heads use a **Softplus** activation on the final
layer, which maps ℝ → (0, ∞) smoothly and has well-defined gradients
everywhere (unlike ReLU, which has zero gradients for all negative inputs).

### 5. Calibrated Probabilities
The TP Probability and Confidence heads both use **Sigmoid** activations.
The Confidence head is trained with MSE against a target confidence value
derived from historical win-rate at similar model confidence levels.

---

## GPU Memory Optimisation

- **Single shared context vector**: no duplication of intermediate activations.
- **Independent head parameters**: each head is a small 2-layer MLP (~64 hidden
  units). Total additional parameter budget across all 5 new heads is typically
  < 1% of the total model parameter count.
- **Selective head enabling**: disable any unused head to free both parameter
  memory and gradient computation.
- **Mixed precision compatible**: all new heads are fully AMP-compatible.
  The Softplus activation is numerically stable under float16.

---

## Inference Speed

- **Zero extra backbone calls**: all heads share one context vector.
- **Parallel head execution**: PyTorch executes the independent head networks
  in the same forward graph — the scheduler can overlap them on GPU.
- **Expected overhead vs. base model**: < 2 ms per batch on CUDA, primarily
  from the six small Linear layers.

---

## Parameter Budget (example config)

With `hidden_size=256`, each head's extra parameter count:

| Head              | Hidden | Params (approx) |
|-------------------|--------|-----------------|
| ExpectedMoveHead  | 64     | ~17 K           |
| StopLossHead      | 64     | ~17 K           |
| TakeProfitHead    | 64     | ~17 K           |
| HoldingTimeHead   | 64     | ~17 K           |
| TPProbabilityHead | 64     | ~17 K           |
| **Total new**     |        | **~85 K**        |

Compared to the base model backbone (~2-4 M params), the new heads add < 5%.

---

## Data Flow Summary

```
                     Training                        Inference
                     ────────                        ─────────
x →  encoder → backbone → context
                            │
                  forward_multi_task(x)          predict(x)
                            │                        │
                 dict[str, Tensor]        MultiTaskQuantModelOutput
                            │                        │
                 MultiTaskLoss.forward()         .to_dict()
                            │                        │
                  MultiTaskLossOutput         inference dict
```

---

## File Structure

```
trainer/models/quant_v1/
    config.py          — All config dataclasses (new + original)
    head.py            — All prediction heads (new + original)
    model.py           — QuantModelV1 with multi-task methods
    output.py          — QuantModelOutput + MultiTaskQuantModelOutput

trainer/losses/
    multi_task.py      — MultiTaskLoss + MultiTaskLossOutput

trainer/training/
    engine.py          — Original single-task engine (unchanged)
    multi_task_engine.py — MultiTaskTrainingEngine

trainer/datasets/
    multi_task.py      — MultiTaskParquetDataset + collate_multi_task

trainer/metrics/
    multi_task.py      — MultiTaskMetrics + MultiTaskMetricsTracker

conf/model/
    quant_v1.yaml            — Original config (all new heads disabled)
    quant_v1_multitask.yaml  — Phase 14.5 config (all heads enabled)
```
