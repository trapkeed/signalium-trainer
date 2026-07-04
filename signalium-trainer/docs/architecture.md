# Signalium Trainer — Architecture Guide

## Overview

The Signalium Trainer is a modular, GPU-ready deep learning training framework built for
reproducibility, extensibility, and clean separation of concerns. It runs on RunPod GPU
instances and communicates exclusively with the Signalium REST API — it never connects
directly to PostgreSQL.

## Design Principles

- **SOLID principles** — every module has a single responsibility.
- **Dependency injection** — no hidden globals; configs and dependencies flow through constructors.
- **Abstract interfaces** — concrete models inherit from `BaseQuantModel`; concrete trainers from `BaseTrainer`.
- **API-only data access** — all dataset and model I/O goes through `SignaliumClient`.
- **Reproducibility first** — seeds, deterministic ops, config snapshots, and MLflow tracking are built in.

## Package Map

```
trainer/
├── api/              Signalium REST API client (auth, download, upload, retry)
├── checkpoints/      Checkpoint save / load / rotation (latest.pt, best.pt, epoch_NNNN.pt)
├── cli/              Typer CLI — one subcommand module per command
├── config/           Hydra/dataclass config schemas — single source of truth
├── dataloaders/      DataLoaderFactory — reproducible, pinned-memory, prefetching
├── datasets/         BaseDataset, ParquetDataset, StreamingParquetDataset, validation
├── evaluation/       StandardEvaluator, EvaluationReporter — full metric suite
├── exporters/        TorchScriptExporter, ONNXExporter, ManifestBuilder
├── logging/          Loguru setup — JSON/text, file rotation, structured sinks
├── metrics/          MetricsTracker (epoch history), EvaluationMetrics (TorchMetrics + sklearn)
├── models/           Abstract interfaces: BaseQuantModel, BaseTrainer, BaseEvaluator, BaseExporter
├── registry/         ModelRegistry — decorator-based model registration
├── training/         TrainingEngine, EarlyStopping, MLflowTracker, seed utilities
└── utils/            device resolution, GPU info, git commit hash
```

## Data Flow

```
Signalium API
     │
     │  manifest + presigned URLs
     ▼
SignaliumClient.download_dataset_file()
     │
     ▼
DatasetValidator (checksums, schema, splits)
     │
     ▼
ParquetDataset / StreamingParquetDataset
     │
     ▼
DataLoaderFactory → train / val / test DataLoaders
     │
     ▼
TrainingEngine.fit()
  ├── optimiser + scheduler + AMP scaler
  ├── CheckpointManager (latest, best, epoch rotation)
  ├── EarlyStopping
  └── MLflowTracker (params, metrics, artifacts)
     │
     ▼
StandardEvaluator → EvaluationReporter (report.json, summary.txt)
     │
     ▼
TorchScriptExporter + ONNXExporter + ManifestBuilder
     │
     ▼
SignaliumClient.upload_model_file() / upload_manifest() / upload_metrics()
     │
     ▼
Signalium API
```

## Extending the Framework

### Adding a New Model

```python
from trainer.models.base import BaseQuantModel
from trainer.registry import ModelRegistry
import torch

@ModelRegistry.register("my_classifier")
class MyClassifier(BaseQuantModel):
    def __init__(self, config):
        super().__init__(config)
        self.net = torch.nn.Linear(config.data.feature_dim, config.training.num_classes)

    @property
    def feature_dim(self): return self._config.data.feature_dim

    @property
    def num_classes(self): return 2

    def forward(self, x): return self.net(x)
```

Then run: `trainer train my_classifier --dataset-dir data/datasets`

### Adding a New CLI Command

Create `trainer/cli/commands/my_command.py` with a `typer.Typer()` app, then register it in
`trainer/cli/commands/__init__.py` and `trainer/cli/main.py`.
