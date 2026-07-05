# RunPod Deployment Guide

## Overview

The Signalium Trainer runs on RunPod GPU instances using the Docker image built from
`Dockerfile`. All data flows through the Signalium REST API — no direct DB access.

## Prerequisites

- RunPod account with GPU pod access
- Docker Hub or private registry access for the trainer image
- `SIGNALIUM_API_KEY` or `SIGNALIUM_TOKEN` secret
- MLflow tracking server URL (optional — defaults to local `mlruns/`)

## Building the Image

```bash
docker build -t signalium-trainer:latest .
docker push your-registry/signalium-trainer:latest
```

## RunPod Pod Configuration

Recommended pod template:
- **GPU**: NVIDIA A100 40GB or H100 80GB
- **CPU**: 16+ vCPUs
- **RAM**: 64 GB+
- **Disk**: 200 GB persistent volume at `/workspace`

## Environment Variables

| Variable              | Required | Description                        |
|-----------------------|----------|------------------------------------|
| `SIGNALIUM_API_KEY`   | Yes*     | API key for Signalium auth         |
| `SIGNALIUM_TOKEN`     | Yes*     | Bearer token (alternative to key)  |
| `MLFLOW_TRACKING_URI` | No       | MLflow server URL                  |
| `PYTHONUNBUFFERED`    | No       | Set to `1` for real-time logs      |

*One of `SIGNALIUM_API_KEY` or `SIGNALIUM_TOKEN` is required.

## Running a Training Job

```bash
# Download dataset
trainer download <dataset_id> --output-dir /workspace/data/datasets

# Validate
trainer validate /workspace/data/datasets

# Train
trainer train my_model \
  --dataset-dir /workspace/data/datasets \
  --epochs 200

# Evaluate
trainer evaluate checkpoints/best.pt \
  --dataset-dir /workspace/data/datasets

# Export
trainer export checkpoints/best.pt \
  --output-dir /workspace/exports

# Upload
trainer upload /workspace/exports \
  --model-id my_model \
  --version 1.0.0
```

## Volumes

Mount `/workspace` as a persistent network volume to preserve:
- `/workspace/data/` — datasets and cache
- `/workspace/checkpoints/` — training checkpoints
- `/workspace/exports/` — exported model artifacts
- `/workspace/logs/` — training logs
- `/workspace/mlruns/` — MLflow run data

## Health Check

The container's health check runs: `python -c "import torch; import trainer; print('OK')"`

Monitor via: `docker inspect --format='{{.State.Health.Status}}' <container_id>`
