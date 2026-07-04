# Signalium Model Framework — Architecture Report

**Project:** Signalium Trainer v0.2.0  
**Phase:** Framework Refactor  
**Generated:** 2024-07-04  
**Status:** COMPLETE

---

## 1. Mission

Refactor the existing QuantModelV1 implementation into a reusable, modular Model Framework
so that multiple deep learning architectures can be built, trained, evaluated, and deployed
without modifying the training pipeline.

**Constraints honoured:**
- ✅ No working functionality removed
- ✅ Training pipeline (`trainer/training/engine.py`) unchanged
- ✅ QuantModelV1 LSTM model still functions identically
- ✅ No GRU, Transformer, or TCN implementations added — framework only
- ✅ Zero breaking changes to the existing CLI, exporters, checkpoints, or evaluation

---

## 2. Framework Package Map

```
trainer/framework/
├── __init__.py                     Public API — one import for all framework symbols
├── abstractions/
│   ├── __init__.py
│   ├── encoder.py                  BaseFeatureEncoder (ABC)
│   ├── sequence_encoder.py         BaseSequenceEncoder (ABC)
│   ├── head.py                     BaseClassificationHead (ABC) + BaseConfidenceHead (ABC)
│   ├── loss.py                     BaseLoss (ABC extends nn.Module)
│   ├── metrics.py                  BaseMetrics (ABC)
│   ├── exporter.py                 BaseFrameworkExporter (ABC)
│   └── inference.py                BaseInferenceAdapter (ABC)
├── contracts/
│   ├── __init__.py
│   ├── model_input.py              ModelInput (dataclass, validated)
│   ├── model_output.py             ModelOutput (dataclass, standardised)
│   └── model_spec.py               ModelSpec + FeatureSchema + OutputSchema
├── factory.py                      ModelFactory (config-driven instantiation)
├── metrics.py                      MetricsEngine (concrete, model-agnostic)
└── inference.py                    PyTorchInferenceAdapter + TorchScriptInferenceAdapter
                                    + InferenceAdapterFactory
```

---

## 3. Abstract Base Classes

All ABCs live in `trainer/framework/abstractions/`. They define the minimum
interface that components must implement to be used by the training pipeline.

| ABC | Contract | Implemented By |
|---|---|---|
| `BaseFeatureEncoder` | `output_dim → int`, `forward(x) → Tensor` | `InputEncoder` |
| `BaseSequenceEncoder` | `output_dim → int`, `forward(x) → Tensor` | `LSTMBackbone`, `GRUBackbone` |
| `BaseClassificationHead` | `forward(context) → logits` | `ClassificationHead` |
| `BaseConfidenceHead` | `forward(context) → confidence` | `ConfidenceHead` |
| `BaseLoss` | `forward(logits, targets) → loss` | `FocalLoss`, `WeightedCrossEntropyLoss` |
| `BaseMetrics` | `update`, `compute`, `reset` | `MetricsEngine` |
| `BaseFrameworkExporter` | `export(dir) → paths`, `validate(path)` | _(future)_ |
| `BaseInferenceAdapter` | `load(path)`, `predict(x)`, `is_loaded` | `PyTorchInferenceAdapter`, `TorchScriptInferenceAdapter` |

---

## 4. Input / Output Contracts

### `ModelInput`
```python
@dataclass
class ModelInput:
    features: Tensor          # (batch, seq_len, feature_dim) — required
    mask: Optional[Tensor]    # (batch, seq_len) bool — future Transformer support
    metadata: Optional[dict]  # per-sample metadata — not used in forward pass
    auxiliary: Optional[dict] # extra model-specific tensors — future use
```

Validation (`ModelInput.validate()`) enforces:
- `features` must be 3-dimensional
- `features` must be float dtype
- `mask` shape must match `features[:2]` if provided

### `ModelOutput`
```python
@dataclass
class ModelOutput:
    logits:          Tensor  # (batch, num_classes) — for loss computation
    probabilities:   Tensor  # (batch, num_classes) — softmax, Σ=1
    confidence:      Tensor  # (batch, 1) — ∈ [0, 1]
    predicted_class: Tensor  # (batch,) — 0=BUY, 1=SELL, 2=NO_TRADE
    auxiliary:       Optional[dict]
```

Convenience properties: `buy_prob`, `sell_prob`, `no_trade_prob`, `predicted_label`.
`gate_by_confidence(threshold)` replaces low-confidence predictions with NO_TRADE.

---

## 5. Model Registry — Enhanced

`ModelRegistry` now supports rich metadata via `ModelSpec` alongside the existing
class registration:

```python
# Before (still works)
@ModelRegistry.register("my_model")
class MyModel(BaseQuantModel): ...

# After (with spec)
@ModelRegistry.register("my_model", spec=ModelSpec(
    name="my_model", version="1.0.0", architecture="MyArch",
    input_schema=FeatureSchema(feature_dim=128, seq_len=60),
    output_schema=OutputSchema(num_classes=3),
))
class MyModel(BaseQuantModel): ...
```

New registry methods:
| Method | Description |
|---|---|
| `get_spec(name)` | Retrieve the ModelSpec for a model |
| `has_spec(name)` | True if a spec is registered |
| `list_specs()` | All specs as a dict |
| `summary()` | Table of all models with name/class/version/architecture |

QuantModelV1 spec registered via `trainer/models/quant_v1/registration.py`:
- `name`: `"quant_v1"`, `version`: `"1.0.0"`, `architecture`: `"LSTM-2L-256H"`
- Input: 128 features × 60 timesteps, 11 feature groups
- Output: 3 classes (BUY/SELL/NO_TRADE) + confidence

---

## 6. Model Factory

`ModelFactory` provides configuration-driven model instantiation:

```python
# By name
model = ModelFactory.create("quant_v1", trainer_config)

# With model config
model = ModelFactory.create("quant_v1", trainer_config, model_config=QuantV1Config())

# From YAML-parsed dict
model = ModelFactory.from_dict({
    "name": "quant_v1",
    "backbone": {"type": "gru", "hidden_size": 512},
}, trainer_config)
```

Adding a new model requires only:
1. Implement `BaseQuantModel` and `@ModelRegistry.register("new_name")`
2. Add a `_build_<new_name>_config()` branch in `ModelFactory`

---

## 7. MetricsEngine

`MetricsEngine` is the concrete, model-agnostic metrics accumulator
implementing `BaseMetrics`:

| Metric | Key |
|---|---|
| Cross-entropy loss | `loss` |
| Overall accuracy | `accuracy` |
| Macro precision | `precision_macro` |
| Macro recall | `recall_macro` |
| Macro F1 | `f1_macro` |
| Weighted F1 | `f1_weighted` |
| Per-class accuracy | `accuracy_{label}` |

Zero external dependencies — pure PyTorch. For ROC-AUC, ECE, and confusion
matrix, the existing `trainer.metrics.EvaluationMetrics` is the full suite.

---

## 8. InferenceAdapter

Two concrete adapters behind the `BaseInferenceAdapter` interface:

| Adapter | Backend | Load |
|---|---|---|
| `PyTorchInferenceAdapter` | Live `BaseQuantModel` | Passed at construction |
| `TorchScriptInferenceAdapter` | `.pt` TorchScript file | `load(path)` |

`InferenceAdapterFactory` dispatches by type:
```python
adapter = InferenceAdapterFactory.from_model(model, confidence_threshold=0.6)
adapter = InferenceAdapterFactory.from_artifact(Path("exports/model.torchscript.pt"))
result = adapter.predict(x)  # dict with probabilities, confidence, predicted_class
```

---

## 9. Migration — QuantModelV1 Component Inheritance

The following components were updated with a single-line inheritance change
(no functional modifications):

| Component | Before | After |
|---|---|---|
| `InputEncoder` | `nn.Module` | `BaseFeatureEncoder` |
| `SequenceEncoder` | `nn.Module, ABC` | `BaseSequenceEncoder` |
| `LSTMBackbone` | `SequenceEncoder` | `SequenceEncoder` → `BaseSequenceEncoder` |
| `GRUBackbone` | `SequenceEncoder` | `SequenceEncoder` → `BaseSequenceEncoder` |
| `ClassificationHead` | `nn.Module` | `BaseClassificationHead` |
| `ConfidenceHead` | `nn.Module` | `BaseConfidenceHead` |
| `FocalLoss` | `nn.Module` | `BaseLoss` |
| `WeightedCrossEntropyLoss` | `nn.Module` | `BaseLoss` |

---

## 10. Tests

| File | Tests | Category |
|---|---|---|
| `test_framework.py` | 35 | Unit — ABCs, contracts, factory, registry, metrics |
| `test_framework_integration.py` | 20 | Integration — full pipeline end-to-end |
| **Total new** | **55** | |
| **Phase 2 carried over** | **61** | |
| **Grand total** | **116+** | |

---

## 11. What Remains Unchanged

| Component | Status |
|---|---|
| `trainer/training/engine.py` | ✅ Not touched |
| `trainer/checkpoints/` | ✅ Not touched |
| `trainer/evaluation/` | ✅ Not touched |
| `trainer/api/` | ✅ Not touched |
| `trainer/dataloaders/` | ✅ Not touched |
| `trainer/datasets/` | ✅ Not touched |
| `trainer/cli/` | ✅ Not touched |
| `trainer/exporters/` | ✅ Not touched |
| `trainer/logging/` | ✅ Not touched |
| All existing tests | ✅ Not touched |
| `Dockerfile`, CI | ✅ Not touched |

---

## 12. Adding a New Architecture (Future)

```python
# 1. Implement components using framework ABCs
class TransformerBackbone(BaseSequenceEncoder):
    @property
    def output_dim(self) -> int: ...
    def forward(self, x): ...

# 2. Assemble the model
@ModelRegistry.register("quant_transformer", spec=ModelSpec(
    name="quant_transformer", version="1.0.0", architecture="Transformer-8H-4L",
    input_schema=FeatureSchema(feature_dim=128, seq_len=120),
    output_schema=OutputSchema(num_classes=3),
))
class QuantTransformer(BaseQuantModel): ...

# 3. Add config builder in ModelFactory._build_quant_transformer_config()

# 4. Train without touching TrainingEngine
model = ModelFactory.create("quant_transformer", trainer_config)
```

The training pipeline, evaluation, export, and CLI require zero changes.
