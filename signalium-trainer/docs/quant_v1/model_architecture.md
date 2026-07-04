# QuantModelV1 — Model Architecture

**Version:** 1.0.0  
**Type:** Supervised Deep Learning Classification  
**Task:** Predict BUY / SELL / NO TRADE trading signals

---

## Overview

QuantModelV1 is Signalium's first proprietary trading signal classifier. It takes a
sliding window of engineered market features and outputs a probability distribution over
three classes: **BUY**, **SELL**, and **NO TRADE**, along with a scalar **confidence score**.

The primary objective is **high-quality decision making**, not trading frequency.
The NO TRADE class is a first-class output — the model is explicitly allowed and encouraged
to decline to signal.

---

## Architecture Diagram

```
Input
(batch, seq_len, input_dim)
        │
        ▼
┌─────────────────────────────────────┐
│         InputEncoder                │
│   LayerNorm → Linear → GELU → Drop │
│   (input_dim → hidden_dim)          │
└──────────────┬──────────────────────┘
               │ (batch, seq_len, hidden_dim)
               ▼
┌─────────────────────────────────────┐
│       SequenceEncoder (backbone)    │
│    LSTM / GRU / [future: Transformer│
│      Temporal CNN, Hybrid]          │
│    → context vector                 │
└──────┬────────────────────┬─────────┘
       │ (batch, hidden_size)│
       ▼                     ▼
┌────────────────┐  ┌──────────────────┐
│ Classification │  │  Confidence Head │
│     Head       │  │                  │
│ Linear → LN    │  │ Linear → GELU    │
│ → GELU → Drop  │  │ → Drop → Linear  │
│ → Linear       │  │ → Sigmoid        │
│ (→ num_classes)│  │ (→ 1 scalar)     │
└───────┬────────┘  └────────┬─────────┘
        │ logits              │ confidence
        ▼                     ▼
┌─────────────────────────────────────┐
│           OutputLayer               │
│   Softmax(logits) → probabilities   │
│   argmax → predicted_class          │
│   Assemble QuantModelOutput         │
└─────────────────────────────────────┘
        │
        ▼
   QuantModelOutput
   ├── buy_prob        P(BUY)
   ├── sell_prob       P(SELL)
   ├── no_trade_prob   P(NO TRADE)
   ├── confidence      ∈ [0, 1]
   ├── predicted_class argmax index
   ├── probabilities   full softmax
   └── logits          raw (for loss)
```

---

## Component Summary

| Component | Module | Role |
|---|---|---|
| `FeatureNormalizer` | `encoder.py` | LayerNorm on raw input |
| `InputEncoder` | `encoder.py` | Project input_dim → hidden_dim |
| `SequenceEncoder` | `backbone.py` | Abstract temporal encoder interface |
| `LSTMBackbone` | `backbone.py` | Initial concrete backbone |
| `GRUBackbone` | `backbone.py` | Drop-in GRU alternative |
| `ClassificationHead` | `head.py` | Context → class logits |
| `ConfidenceHead` | `head.py` | Context → confidence [0,1] |
| `OutputLayer` | `output.py` | Assemble QuantModelOutput |

---

## Backbone Swappability

The backbone implements the `SequenceEncoder` abstract interface:
```python
class SequenceEncoder(ABC):
    @property
    @abstractmethod
    def output_dim(self) -> int: ...

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...  # (batch, seq) → (batch, dim)
```

Switch backbones with a single config change:
```yaml
backbone:
  type: gru  # was: lstm
```

No changes to `QuantModelV1`, `TrainingEngine`, or any other infrastructure.

---

## Weight Initialisation

| Layer | Init Strategy |
|---|---|
| `nn.Linear` | Xavier uniform (weights), zeros (bias) |
| LSTM `weight_ih` | Xavier uniform |
| LSTM `weight_hh` | Orthogonal |
| LSTM forget gate bias | 1.0 (improves gradient flow) |
| GRU weights | Orthogonal |
| LayerNorm | ones (weight), zeros (bias) |
