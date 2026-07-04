# QuantModelV1 — Developer Guide

## Package Layout

```
trainer/
├── models/
│   └── quant_v1/
│       ├── __init__.py      Exports + public API
│       ├── config.py        QuantV1Config dataclasses (all hyper-params)
│       ├── encoder.py       InputEncoder + FeatureNormalizer
│       ├── backbone.py      SequenceEncoder ABC + LSTMBackbone + GRUBackbone
│       ├── head.py          ClassificationHead + ConfidenceHead
│       ├── output.py        QuantModelOutput + OutputLayer
│       └── model.py         QuantModelV1 (registered as "quant_v1")
├── losses/
│   ├── __init__.py
│   ├── focal.py             FocalLoss
│   ├── weighted_ce.py       WeightedCrossEntropyLoss
│   └── factory.py           build_loss(config) → nn.Module
└── sampling/
    ├── __init__.py
    └── strategy.py          compute_class_weights, make_weighted_sampler,
                              oversample_dataset, undersample_dataset,
                              BalancingStrategyBuilder
```

---

## Adding a New Backbone

1. Add the new type to `BackboneType` in `config.py`:
   ```python
   class BackboneType(str, Enum):
       LSTM = "lstm"
       GRU = "gru"
       TRANSFORMER = "transformer"   # add this
   ```

2. Implement `SequenceEncoder` in `backbone.py`:
   ```python
   class TransformerBackbone(SequenceEncoder):
       def __init__(self, input_dim: int, config: BackboneConfig) -> None: ...
       @property
       def output_dim(self) -> int: ...
       def forward(self, x: torch.Tensor) -> torch.Tensor: ...
   ```

3. Add a branch in `build_backbone()`:
   ```python
   if config.type == BackboneType.TRANSFORMER:
       return TransformerBackbone(input_dim=input_dim, config=config)
   ```

4. Update `conf/model/quant_v1.yaml`:
   ```yaml
   backbone:
     type: transformer
   ```

**The training pipeline, model registration, and export pipeline require zero changes.**

---

## Adding a New Loss Function

1. Add the type to `LossType` in `config.py`.
2. Implement `nn.Module` in `trainer/losses/my_loss.py`.
3. Add a branch in `trainer/losses/factory.py:build_loss()`.
4. Add unit tests in `tests/unit/test_losses.py`.

---

## Using QuantModelV1 Programmatically

```python
from trainer.config.schema import TrainerConfig
from trainer.models.quant_v1 import QuantModelV1, QuantV1Config

# Default config
model = QuantModelV1(trainer_config=TrainerConfig())

# Custom config
from trainer.models.quant_v1.config import BackboneConfig, BackboneType
cfg = QuantV1Config()
cfg.backbone.type = BackboneType.GRU
cfg.backbone.hidden_size = 512
cfg.backbone.num_layers = 3
model = QuantModelV1(TrainerConfig(), cfg)

# Forward pass (returns logits for training)
logits = model(x)  # x: (batch, seq_len, input_dim)

# Full inference (returns QuantModelOutput)
out = model.predict(x)
print(out.buy_prob, out.sell_prob, out.no_trade_prob)
print(out.confidence)
print(out.predicted_class)  # 0=BUY, 1=SELL, 2=NO_TRADE
```

---

## Architecture Summary

```python
model = QuantModelV1(TrainerConfig())
print(model.get_architecture_summary())
# {
#   "architecture": "QuantModelV1-LSTM",
#   "version": "1.0.0",
#   "input_dim": 128,
#   "encoder_hidden_dim": 256,
#   "backbone_type": "lstm",
#   "backbone_hidden_size": 256,
#   "backbone_num_layers": 2,
#   "backbone_bidirectional": False,
#   "window_length": 60,
#   "num_classes": 3,
#   "label_names": ["BUY", "SELL", "NO_TRADE"],
#   "total": 847363,
#   "trainable": 847363
# }
```

---

## Testing

Run only the QuantModelV1 tests:
```bash
pytest tests/ -k "quant_v1 or loss or sampling" -v
```

Run with gradient checks:
```bash
pytest tests/unit/test_quant_v1_model.py::TestGradientFlow -v
```

Run export tests:
```bash
pytest tests/integration/test_quant_v1_export.py -v
```
