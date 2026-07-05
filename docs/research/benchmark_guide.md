# Benchmark Guide

## Purpose

The Benchmark Runner evaluates multiple models under strictly identical conditions
so results are directly comparable. No manual book-keeping required.

## BenchmarkConditions

`BenchmarkConditions` is the contract that fixes every experimental variable:

```python
from trainer.research import BenchmarkConditions

conditions = BenchmarkConditions(
    random_seed=42,          # Applied to Python, NumPy, and PyTorch
    dataset_version="2.1.0",
    feature_version="1.0.0",
    label_version="1.0.0",
    train_ratio=0.70,
    val_ratio=0.15,
    max_epochs=100,
    batch_size=256,
    window_length=60,
    device="cuda",
)
conditions.validate()  # Raises ValueError if constraints are violated
```

## Registering Models

```python
from trainer.research import BenchmarkRunner

runner = BenchmarkRunner(conditions=conditions)

runner.register_model(
    name="quant_v1_lstm",
    model_factory=lambda: ModelFactory.create("quant_v1", trainer_config),
    hyperparameters={"hidden_size": 256, "dropout": 0.2},
)
runner.register_model(
    name="quant_v1_gru",
    model_factory=lambda: ModelFactory.create("quant_v1", trainer_config, gru_config),
    hyperparameters={"hidden_size": 256, "dropout": 0.2},
)
```

## Running the Benchmark

```python
result = runner.run(
    train_data=(x_train, y_train),
    val_data=(x_val, y_val),
    learning_rate=1e-3,
)
```

## Inspecting Results

```python
print(f"Benchmark: {result.benchmark_id}")
print(f"Total time: {result.total_duration_seconds:.1f}s")

for run in result.completed_runs():
    print(f"{run.model_name}: acc={run.metrics['accuracy']:.4f}, "
          f"f1={run.metrics['f1_macro']:.4f}, "
          f"lat={run.inference_latency_ms:.2f}ms, "
          f"size={run.model_size_mb:.2f}MB")
```

## Metrics Tracked

| Metric | Key | Notes |
|---|---|---|
| Validation Loss | `loss` | Cross-entropy |
| Training Loss | `train_loss` | Per epoch |
| Accuracy | `accuracy` | Overall |
| Precision (macro) | `precision_macro` | Per-class avg |
| Recall (macro) | `recall_macro` | Per-class avg |
| F1 (macro) | `f1_macro` | Per-class avg |
| F1 (weighted) | `f1_weighted` | Class-frequency weighted |
| Inference Latency | `inference_latency_ms` | Median over 50 runs |
| Model Size | `model_size_mb` | Parameter bytes |
| GPU Memory | `gpu_memory_mb` | Peak allocated |
| Training Time | `training_time_seconds` | Wall-clock |

## Adding a New Model Family

```python
runner.register_model(
    name="quant_transformer",
    model_factory=lambda: ModelFactory.create("quant_transformer", trainer_config),
)
```

No other changes needed. The runner handles seeding, training, and evaluation
automatically.
