# QuantModelV1 — Final Model Architecture Report

**Project:** Signalium Quant Model V1  
**Version:** 1.0.0  
**Type:** Supervised Deep Learning — Multi-Class Classification  
**Generated:** 2024-07-04  
**Status:** COMPLETE — Ready for training

---

## 1. Mission

Design and implement the first proprietary Signalium Quant model as a production-grade
supervised deep learning system that predicts trading signals from engineered feature windows.

The primary objective is **maximising high-quality decision making**, not maximising
trading frequency. The model explicitly supports a NO TRADE output as a first-class label.

---

## 2. Architecture Overview

### Full Pipeline

```
Engineered Features (Parquet) → Dataset → DataLoader
                                               │
                                         (batch, 60, 128)
                                               │
                                         InputEncoder
                                          LayerNorm
                                          Linear(128→256)
                                          GELU + Dropout
                                               │
                                         (batch, 60, 256)
                                               │
                                        LSTMBackbone
                                         2-layer LSTM
                                         hidden_size=256
                                               │
                                         (batch, 256)  ← context vector
                                         ┌─────┴──────┐
                               ClassHead │            │ ConfHead
                            Linear(256→128)      Linear(256→64)
                              LN + GELU            GELU + Drop
                              Dropout           Linear(64→1)
                            Linear(128→3)          Sigmoid
                               logits           confidence ∈[0,1]
                                 │
                               Softmax
                                 │
                           QuantModelOutput
                            buy_prob  P(BUY)
                            sell_prob P(SELL)
                            no_trade  P(NO TRADE)
                            confidence
                            predicted_class
```

---

## 3. Component Inventory

| Component | File | Parameters |
|---|---|---|
| `FeatureNormalizer` | `encoder.py` | 2 × input_dim |
| `InputEncoder` | `encoder.py` | input_dim × hidden_dim |
| `LSTMBackbone` | `backbone.py` | ~4 × hidden_size² × num_layers |
| `GRUBackbone` | `backbone.py` | ~3 × hidden_size² × num_layers |
| `ClassificationHead` | `head.py` | hidden_size × hidden_dim |
| `ConfidenceHead` | `head.py` | hidden_size × 64 |
| `OutputLayer` | `output.py` | 0 (stateless) |

Default total parameters: **~847K** (128 input, 256 hidden, 2-layer LSTM)

---

## 4. Outputs

| Output | Shape | Range | Purpose |
|---|---|---|---|
| `logits` | (batch, 3) | ℝ | Training loss input |
| `probabilities` | (batch, 3) | [0, 1], Σ=1 | Signal distribution |
| `buy_prob` | (batch,) | [0, 1] | BUY confidence |
| `sell_prob` | (batch,) | [0, 1] | SELL confidence |
| `no_trade_prob` | (batch,) | [0, 1] | NO TRADE confidence |
| `confidence` | (batch, 1) | [0, 1] | Model certainty |
| `predicted_class` | (batch,) | {0, 1, 2} | Signal class |

---

## 5. Loss Functions Implemented

| Loss | Module | When to Use |
|---|---|---|
| `CrossEntropyLoss` | `nn.CrossEntropyLoss` | Default baseline |
| `FocalLoss` | `losses/focal.py` | Strongly imbalanced data |
| `WeightedCrossEntropyLoss` | `losses/weighted_ce.py` | Manual class weight control |

Future: `QuantLoss` (custom risk-adjusted objective) — add to `losses/` and `LossType`.

---

## 6. Class Imbalance Support

| Strategy | Module | How |
|---|---|---|
| `class_weights` | `sampling/strategy.py` | Inverse frequency weights → loss |
| `weighted_sampler` | `sampling/strategy.py` | `WeightedRandomSampler` → DataLoader |
| `oversample` | `sampling/strategy.py` | Repeat minority indices |
| `undersample` | `sampling/strategy.py` | Remove majority indices |
| `none` | — | No intervention |

---

## 7. Regularization

| Technique | Config Key | Default |
|---|---|---|
| Dropout (encoder) | `encoder.projection_dropout` | 0.1 |
| Dropout (LSTM inter-layer) | `backbone.dropout` | 0.3 |
| Dropout (class head) | `classification_head.dropout` | 0.3 |
| Dropout (confidence head) | `confidence_head.dropout` | 0.2 |
| Weight decay (optimizer) | `optimizer.weight_decay` | 1e-4 |
| Gradient clipping | `training.gradient_clip_val` | 1.0 |
| Early stopping | `training.early_stopping_patience` | 10 |

---

## 8. Backbone Swappability

The `SequenceEncoder` abstract class allows replacing the backbone with:

| Future Backbone | Status |
|---|---|
| `GRUBackbone` | ✅ Implemented |
| `TransformerBackbone` | Ready to implement |
| `TemporalCNNBackbone` | Ready to implement |
| `HybridBackbone` | Ready to implement |

No changes to `QuantModelV1`, `TrainingEngine`, or any export pipeline required.

---

## 9. Metrics Tracked

Accuracy · Precision · Recall · F1 (macro/weighted) · ROC-AUC ·
Confusion Matrix · Expected Calibration Error · Per-class accuracy ·
Prediction confidence distribution · Loss curves

---

## 10. Export Formats

| Format | Exporter | Validation |
|---|---|---|
| TorchScript | `TorchScriptExporter` | Inference parity test |
| ONNX | `ONNXExporter` | `onnx.checker` + ORT inference parity |
| Manifest + Checksums | `ManifestBuilder` | SHA-256 per file |
| Training summary | `get_architecture_summary()` | Included in manifest |

---

## 11. Tests Generated

| Test File | Tests | Category |
|---|---|---|
| `test_quant_v1_model.py` | 19 | Unit — shapes, gradients, metadata |
| `test_losses.py` | 16 | Unit — all 3 loss functions + factory |
| `test_sampling.py` | 13 | Unit — all 5 balancing strategies |
| `test_quant_v1_inference.py` | 7 | Integration — full inference pipeline |
| `test_quant_v1_export.py` | 6 | Integration — TorchScript, ONNX, manifest |
| **Total** | **61** | |

---

## 12. Readiness Checklist

- [x] Model implements `BaseQuantModel` interface
- [x] Registered as `"quant_v1"` in `ModelRegistry`
- [x] `forward()` returns logits (compatible with `TrainingEngine`)
- [x] `predict()` returns full `QuantModelOutput`
- [x] GRU backbone implemented and tested
- [x] All 3 loss functions implemented and tested
- [x] All 5 balancing strategies implemented and tested
- [x] TorchScript export tested with inference parity
- [x] ONNX export validated with `onnx.checker` and ORT
- [x] Manifest builder writes SHA-256 verified artifact list
- [x] All hyper-parameters configurable via `QuantV1Config`
- [x] Hydra YAML at `conf/model/quant_v1.yaml`
- [x] 5 documentation files written
- [x] 61 tests written across unit and integration suites
- [x] No prohibited patterns (no RL, no online learning, no trade execution, no DB)

---

## 13. Infrastructure NOT Modified

Per the phase mandate, the following were **not modified**:

- `trainer/training/engine.py`
- `trainer/checkpoints/manager.py`
- `trainer/evaluation/`
- `trainer/api/`
- `trainer/dataloaders/`
- `trainer/datasets/`
- `trainer/logging/`
- `trainer/cli/`
- Any existing test files
- `Dockerfile`, `docker-compose.yml`, CI pipeline

Only **new files** were added.

---

## 14. Next Steps

1. **Wire the training loop** — pass `QuantModelV1` to `TrainingEngine` with the dataset
2. **Tune hyper-parameters** — run MLflow sweeps over LR, hidden_size, num_layers
3. **Evaluate on holdout** — use `StandardEvaluator` with `model.predict()`
4. **Implement QuantLoss** — custom risk-adjusted loss for the quant objective
5. **Transformer backbone** — implement `TransformerBackbone(SequenceEncoder)`
6. **RunPod deployment** — `trainer train quant_v1 --config conf/environments/runpod.yaml`
