# Signalium Trainer

A modular, GPU-ready deep learning training framework for [Signalium](https://signalium.io).

Runs on RunPod GPU instances. Communicates exclusively with the Signalium REST API.
**Never connects to PostgreSQL directly.**

---

## Quick Start

```bash
pip install -e ".[dev]"

# Check your environment
trainer doctor

# Download a dataset
trainer download <dataset_id> --output-dir data/datasets

# Validate it
trainer validate data/datasets

# Train (after implementing a concrete model)
trainer train my_model --dataset-dir data/datasets --epochs 100

# Evaluate
trainer evaluate checkpoints/best.pt --dataset-dir data/datasets

# Export to TorchScript + ONNX
trainer export checkpoints/best.pt --output-dir exports

# Upload to Signalium
trainer upload exports --model-id my_model --version 1.0.0
```

## CLI Commands

| Command              | Description                                     |
|---------------------|-------------------------------------------------|
| `trainer download`  | Download a dataset export from Signalium        |
| `trainer validate`  | Validate checksums, schema, and splits          |
| `trainer train`     | Start a training run                            |
| `trainer evaluate`  | Evaluate a checkpoint on the test split         |
| `trainer export`    | Export to TorchScript and/or ONNX               |
| `trainer upload`    | Upload model artifacts to Signalium             |
| `trainer resume`    | Resume training from a checkpoint              |
| `trainer benchmark` | Benchmark DataLoader and model throughput       |
| `trainer clean`     | Remove checkpoints, exports, logs, cache        |
| `trainer doctor`    | Run environment diagnostics                     |
| `trainer version`   | Print framework version                         |

## Tech Stack

Python 3.12 · PyTorch · TorchMetrics · NumPy · Pandas · PyArrow · scikit-learn ·
MLflow · Hydra · FastAPI · Pydantic · Loguru · Rich · ONNX · Docker

## Documentation

| Guide | Location |
|-------|----------|
| Architecture | `docs/architecture.md` |
| Developer Guide | `docs/developer_guide.md` |
| RunPod Deployment | `docs/runpod_deployment.md` |
| Configuration | `docs/configuration_guide.md` |
| Training Framework | `docs/training_framework_guide.md` |
| Contributing | `docs/contributing.md` |

## Environment Variables

| Variable              | Description                        |
|-----------------------|------------------------------------|
| `SIGNALIUM_API_KEY`   | Signalium API key                  |
| `SIGNALIUM_TOKEN`     | Signalium Bearer token (alternative)|
| `MLFLOW_TRACKING_URI` | MLflow server URI                  |

## License

MIT — see [LICENSE](LICENSE).
