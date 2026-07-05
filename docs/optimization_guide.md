# Optimization Guide — Phase 14.9B

## GPU Memory Optimization

### 1. Gradient Checkpointing

Reduce peak activation memory by ~40–60% for deep Transformer backbones:

```yaml
backbone:
  type: transformer
  gradient_checkpointing: true
```

**Trade-off:** ~33% more compute per step; no accuracy impact.

**When to use:**
- Sequence length > 120 timesteps
- Transformer backbone with > 4 layers
- OOM errors during training

### 2. Mixed Precision (AMP)

Enabled by default. Uses `bfloat16` / `float16` for forward passes and
`float32` for weight updates via GradScaler:

```yaml
training:
  mixed_precision: true
```

Memory reduction: ~50% for activations and gradients.

### 3. Gradient Accumulation

Simulate large batches with less VRAM:

```yaml
training:
  accumulate_grad_batches: 8  # Effective batch = dataloader.batch_size × 8
```

```yaml
dataloader:
  batch_size: 32  # Physical batch
  # Effective batch: 32 × 8 = 256
```

**When to use:**
- Batch size required by scheduler > available VRAM
- Multi-task training with many large prediction heads

### 4. DataLoader Performance

```yaml
dataloader:
  num_workers: 8
  pin_memory: true
  prefetch_factor: 4
  persistent_workers: true
  drop_last: true
```

- `pin_memory: true` — enables DMA transfer (significant speedup on CUDA)
- `persistent_workers: true` — avoids worker spawn overhead each epoch
- `prefetch_factor: 4` — keeps GPU fed during data-loading stalls
- `drop_last: true` — stable batch sizes; important for gradient accumulation

---

## Training Speed Optimization

### TF32 (Ampere+ GPUs)

```yaml
gpu:
  allow_tf32: true
```

Enables NVIDIA TF32 format for matrix multiplications — same range as float32
but 19-bit mantissa (~3× faster, with negligible accuracy impact).

### CuDNN Benchmark

```yaml
gpu:
  cudnn_benchmark: true    # Auto-select fastest convolution algorithm
  cudnn_deterministic: false
```

**Trade-off:** `cudnn_benchmark` requires a warm-up phase each run (first few
batches slower). Disable `cudnn_deterministic` for maximum speed.

### Backbone Selection for Speed

| Backbone | Relative Speed | Notes |
|---|---|---|
| `temporal_cnn` | ⚡⚡⚡ Fastest | Parallelised convolutions |
| `gru` | ⚡⚡ | Lighter than LSTM |
| `lstm` | ⚡ | Standard RNN baseline |
| `transformer` | ⚡ (GPU) | Parallelises well with long sequences |
| `hybrid_lstm_transformer` | Moderate | LSTM limits parallelism |

---

## Optimizer Selection

### For Adam-family (default)

```yaml
optimizer:
  type: adamw
  learning_rate: 3.0e-4
  weight_decay: 1.0e-4
  betas: [0.9, 0.999]
  eps: 1.0e-8
```

**AdamW** is the recommended default — decoupled weight decay avoids the
interaction bug in vanilla Adam with L2 regularisation.

### For large learning rates + cosine

```yaml
optimizer:
  type: adamw
  learning_rate: 1.0e-3
scheduler:
  type: cosine
  warmup_epochs: 10
  T_max: 90
  eta_min: 1.0e-7
```

### For SGD + OneCycle (fastest convergence on large datasets)

```yaml
optimizer:
  type: sgd
  learning_rate: 0.1
  momentum: 0.9
  weight_decay: 1.0e-4
scheduler:
  type: one_cycle
  max_lr: 0.1
  total_steps: 50000
  pct_start: 0.3
```

### For sparse / irregular features

```yaml
optimizer:
  type: adagrad
  learning_rate: 0.01
  weight_decay: 1.0e-4
```

---

## Scheduler Selection

### Cosine + Warmup (recommended)

Stable, gradual LR reduction with cold-start protection:

```yaml
scheduler:
  type: cosine
  warmup_epochs: 10
  T_max: 90
  eta_min: 1.0e-7
```

### Cosine Warm Restarts (for long runs)

Periodic restarts to escape local minima:

```yaml
scheduler:
  type: cosine_warm_restarts
  T_0: 20
  T_mult: 2
  eta_min: 1.0e-7
```

### ReduceOnPlateau (adaptive)

Reduce LR automatically when validation loss stalls:

```yaml
scheduler:
  type: reduce_on_plateau
  patience: 5
  factor: 0.5
```

---

## Profiling & Benchmarking

Use `TrainingProfiler` to identify bottlenecks before long runs:

```python
from trainer.training.profiler import TrainingProfiler
import torch

profiler = TrainingProfiler(model, device=torch.device("cuda"))

# Model size
size = profiler.profile_model_size()
print(f"Parameters: {size.total_params:,} | ~{size.estimated_mb:.1f} MB")

# Training speed
speed = profiler.profile_training_speed(train_loader, n_batches=100)
print(f"Throughput: {speed.samples_per_sec:.0f} samples/sec")
print(f"Epoch estimate: {speed.estimated_epoch_minutes:.1f} min")

# Inference latency
infer = profiler.profile_inference_speed(
    input_shape=(60, 128),  # (seq_len, features)
    n_runs=500,
)
print(f"Inference p99: {infer.latency_p99_ms:.2f} ms")

# Full benchmark
report = profiler.full_benchmark(train_loader, input_shape=(60, 128))
print(report.summary())
```

---

## Reproducibility Checklist

For fully reproducible experiments:

1. Pin `dataset_version`, `feature_version`, `label_version` in config
2. Set `training.seed` explicitly
3. Set `gpu.cudnn_deterministic: true` (slower but deterministic)
4. Enable `checkpoint.save_rng_state: true` for exact resume
5. Commit all code before the run (avoid dirty working tree)
6. Use `ReproducibilityManager.verify()` before resuming from checkpoint
