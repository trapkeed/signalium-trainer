# Training Guide — Phase 14.9B

## Overview

The Signalium Trainer provides an institutional-grade training infrastructure
with interchangeable backbones, adaptive optimizers, warmup scheduling,
gradient accumulation, and complete reproducibility tracking.

---

## Quick Start

### 1. Minimal training run

```python
from trainer.config.schema import TrainerConfig
from trainer.training.engine import TrainingEngine
from trainer.models.quant_v1 import build_quant_model_v1
import torch.nn as nn

config = TrainerConfig()
model = build_quant_model_v1(config)
engine = TrainingEngine(
    model=model,
    config=config,
    criterion=nn.CrossEntropyLoss(),
)
results = engine.fit(train_loader, val_loader)
print(results["best_metrics"])
```

### 2. Resume from checkpoint

```python
from pathlib import Path

results = engine.fit(
    train_loader,
    val_loader,
    resume_checkpoint=Path("checkpoints/best.pt"),
)
```

---

## Backbone Selection

Select the backbone architecture in your YAML config:

```yaml
backbone:
  type: transformer        # lstm | gru | transformer | temporal_cnn | hybrid_lstm_transformer
  hidden_size: 256
  num_layers: 4
  num_heads: 8             # Transformer / Hybrid only
  ff_dim: 1024             # Transformer / Hybrid only
  dropout: 0.1
  window_length: 60
  max_seq_len: 60
  pooling: last            # last | mean (Transformer / Hybrid)
  gradient_checkpointing: false
```

### Backbone summary

| Backbone | Config type | Strengths |
|---|---|---|
| `lstm` | LSTM | Sequential patterns, fast training |
| `gru` | GRU | Lighter LSTM, similar performance |
| `transformer` | TRANSFORMER | Long-range dependencies, parallelisable |
| `temporal_cnn` | TEMPORAL_CNN | Dilated receptive field, very fast |
| `hybrid_lstm_transformer` | HYBRID_LSTM_TRANSFORMER | Best of LSTM + Transformer |

### TCN configuration

```yaml
backbone:
  type: temporal_cnn
  hidden_size: 256
  tcn_channels: [64, 128, 256]  # One layer per entry; dilation = 2^i
  tcn_kernel_size: 3
  dropout: 0.1
```

### Hybrid LSTM-Transformer

```yaml
backbone:
  type: hybrid_lstm_transformer
  hidden_size: 256
  rnn_layers: 2         # LSTM layers
  transformer_layers: 2 # Transformer encoder layers after LSTM
  num_heads: 8
  ff_dim: 1024
```

---

## Optimizers

```yaml
optimizer:
  type: adamw     # adam | adamw | sgd | rmsprop | adagrad | adamax
  learning_rate: 3.0e-4
  weight_decay: 1.0e-4
  betas: [0.9, 0.999]
  eps: 1.0e-8
```

| Type | Notes |
|---|---|
| `adamw` | Recommended default — decoupled weight decay |
| `adam` | Classic adaptive optimizer |
| `sgd` | With momentum; best with warmup + cosine |
| `rmsprop` | Good for RNN-heavy architectures |
| `adagrad` | Per-parameter LR; good for sparse data |
| `adamax` | Infinity-norm variant of Adam |

---

## LR Schedulers

```yaml
scheduler:
  type: cosine           # cosine | step | reduce_on_plateau | exponential
                         # | cosine_warm_restarts | one_cycle | linear | none
  warmup_epochs: 10      # Linear warmup before main scheduler
  T_max: 190
  eta_min: 1.0e-7
```

### Warmup scheduler

When `warmup_epochs > 0`, the engine wraps the configured scheduler with a
`WarmupScheduler` that linearly interpolates LR from `0.01 × base_lr` to
`base_lr` over the first N epochs, then hands off to the main scheduler.

```yaml
scheduler:
  type: cosine
  warmup_epochs: 10
  T_max: 190
  eta_min: 1.0e-7
```

### Cosine Warm Restarts

```yaml
scheduler:
  type: cosine_warm_restarts
  T_0: 20      # First restart period
  T_mult: 2    # Period multiplier after each restart
  eta_min: 1.0e-7
```

### One Cycle Policy

```yaml
scheduler:
  type: one_cycle
  max_lr: 1.0e-2
  total_steps: 10000
  pct_start: 0.3
  div_factor: 25.0
  final_div_factor: 10000.0
```

---

## Gradient Accumulation

Simulate large batch sizes on memory-constrained hardware:

```yaml
training:
  accumulate_grad_batches: 4   # Effective batch = batch_size × 4
```

The engine automatically divides the loss by `accumulate_grad_batches` and
only calls `optimizer.step()` at the end of each accumulation window, keeping
gradient magnitudes invariant to the step count.

---

## Gradient Checkpointing

Trade computation for GPU memory. Only supported by Transformer and Hybrid
backbones:

```yaml
backbone:
  gradient_checkpointing: true
```

Each Transformer encoder block's forward pass is recomputed during
the backward pass, reducing peak activation memory at the cost of ~33% more
compute per step.

---

## Checkpointing

```yaml
checkpoint:
  checkpoint_dir: checkpoints/run_001
  keep_top_k: 5                 # Keep 5 best per-epoch files
  monitor_metric: val_loss
  monitor_mode: min             # min | max
  save_optimizer_state: true
  save_scheduler_state: true
  save_rng_state: true          # Required for exact resume
  save_training_config: true
```

Each saved checkpoint contains:
- `model_state_dict` — model weights
- `optimizer_state_dict` — optimizer moments
- `scheduler_state_dict` — scheduler state
- `rng_state` — Python / NumPy / PyTorch / CUDA RNG state
- `metrics` — full epoch metrics dict
- `config` — serialised training configuration
- `epoch`, `timestamp`, `version`

Two special symlinks are always maintained:
- `best.pt` → checkpoint with best monitored metric
- `latest.pt` → most recently saved checkpoint

---

## Reproducibility

```yaml
reproducibility:
  enabled: true
  fingerprint_dir: checkpoints/fingerprints
  dataset_version: "v3.2.1"
  feature_version: "v2.0.0"
  label_version: "v1.4.0"
  verify_on_resume: true
```

Use `ReproducibilityManager` to capture and verify run conditions:

```python
from trainer.training.reproducibility import ReproducibilityManager

mgr = ReproducibilityManager(config)
fp = mgr.capture(
    dataset_version="v3.2.1",
    feature_version="v2.0.0",
    label_version="v1.4.0",
)
mgr.save(fp, Path("checkpoints/fingerprints/run_001.json"))

# On resume:
passed, issues = mgr.verify(Path("checkpoints/fingerprints/run_001.json"))
if not passed:
    for issue in issues:
        print("WARNING:", issue)
```

The fingerprint captures: git commit, git dirty state, dataset/feature/label
versions, config SHA-256 hash, seed, Python / PyTorch / NumPy versions, CUDA
version, hostname, and timestamp.

---

## Institutional YAML

Use `conf/training/institutional.yaml` for production runs:

```bash
python -m trainer.cli.main train \
    --config-path conf \
    --config-name config \
    +training=institutional
```
