# Signalium Trainer — Configuration Guide

## Overview

Configuration uses [Hydra](https://hydra.cc/) with typed Python dataclasses as the schema.
The root config is at `conf/config.yaml`. Environment-specific overrides live in
`conf/environments/`.

## Environments

| Environment   | Use case                           |
|---------------|------------------------------------|
| `development` | Local development with verbose logs |
| `staging`     | Pre-production API and MLflow      |
| `production`  | Production API, JSON logs          |
| `runpod`      | RunPod GPU pods (optimised paths)  |

Select an environment:

```bash
trainer train my_model environment=runpod
```

## Key Configuration Sections

### `api`
```yaml
api:
  base_url: https://api.signalium.io
  timeout: 60              # seconds
  max_retries: 5
  retry_backoff_factor: 2.0
```

### `data`
```yaml
data:
  dataset_dir: data/datasets
  validate_checksums: true
  fail_on_validation_error: true  # abort training if validation fails
```

### `dataloader`
```yaml
dataloader:
  batch_size: 32
  num_workers: 4
  pin_memory: true
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15
```

### `training`
```yaml
training:
  epochs: 100
  seed: 42
  mixed_precision: true
  gradient_clip_val: 1.0
  early_stopping_patience: 10
```

### `optimizer`
```yaml
optimizer:
  type: adamw       # adam | adamw | sgd | rmsprop
  learning_rate: 0.001
  weight_decay: 0.0001
```

### `scheduler`
```yaml
scheduler:
  type: cosine      # cosine | step | reduce_on_plateau | exponential | none
  T_max: 100
  eta_min: 1.0e-6
```

### `checkpoint`
```yaml
checkpoint:
  checkpoint_dir: checkpoints
  save_every_n_epochs: 1
  keep_top_k: 3
  monitor_metric: val_loss
  monitor_mode: min
```

### `mlflow`
```yaml
mlflow:
  enabled: true
  tracking_uri: mlruns
  experiment_name: signalium-trainer
  log_artifacts: true
```

## CLI Overrides

Any config key can be overridden on the CLI:

```bash
trainer train my_model training.epochs=50 optimizer.learning_rate=0.01
```
