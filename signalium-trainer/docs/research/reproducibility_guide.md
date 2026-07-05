# Reproducibility Guide

## Goal

Signalium Quant must be capable of reproducing **any historical experiment** with identical inputs and identical configuration. This guide explains how the Phase 19.1 reproducibility system achieves this.

---

## What is Captured

### Automatic Capture (via `EnvironmentSnapshot.capture()`)

| Category | Field | Source |
|----------|-------|--------|
| Repository | Git commit hash | `git rev-parse HEAD` |
| Repository | Git branch | `git rev-parse --abbrev-ref HEAD` |
| Repository | Repository version | `git describe --tags --always` |
| Runtime | Python version | `sys.version` |
| Runtime | PyTorch version | `torch.__version__` |
| Runtime | CUDA version | `torch.version.cuda` |
| Runtime | Random seed | Caller-provided |
| OS | OS name | `platform.system()` |
| OS | OS version | `platform.version()` |
| Hardware | CPU model | `/proc/cpuinfo` |
| Hardware | RAM GB | `/proc/meminfo` |
| Hardware | GPU model | `nvidia-smi` |
| Container | Docker image | `IMAGE_TAG` env var |
| Environment | Safe env vars | Allow-listed prefixes |

### Manual Attachment (four snapshot types)

| Snapshot | Key Contents |
|----------|-------------|
| `DatasetSnapshot` | Version, hash, instruments, timeframes, sample count, quality score |
| `FeatureSnapshot` | Feature names, parameters, importance, drift status |
| `ModelSnapshot` | Architecture, backbone, optimizer, LR, checkpoint hash |
| `ValidationSnapshot` | Walk-forward results, calibration, trading metrics |

---

## Reproducibility Score

The score is computed automatically from what is captured:

| Component | Weight | Satisfied when |
|-----------|--------|----------------|
| Git commit hash | 25% | `git_commit_hash` non-empty |
| Python version | 15% | `python_version` non-empty |
| PyTorch version | 15% | `pytorch_version` non-empty |
| Random seed | 20% | `random_seed` is set |
| OS captured | 10% | `os_name` non-empty |
| CPU/GPU captured | 5+5% | `cpu_model` and `gpu_model`/`ram_gb` present |
| Env vars captured | 5% | `env_vars` non-empty |
| + Dataset attached | +5% | `dataset_snapshot` present |
| + Feature attached | +5% | `feature_snapshot` present |
| + Checkpoint hash | +10% | `model_snapshot.checkpoint_hash` non-empty |
| + Validation attached | +5% | `validation_snapshot` present |

**Maximum score: 100% (Grade A)**

---

## Achieving Full Reproducibility

### Step 1 — Commit your code before training

```bash
git add -A && git commit -m "experiment: transformer_4h_v2"
```

This ensures `git_commit_hash` points to an exact, recoverable state.

### Step 2 — Set a random seed

```python
import torch, random, numpy as np

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

api.register_experiment("exp_001", random_seed=SEED, capture_environment=True)
```

### Step 3 — Attach all four snapshots

```python
# After loading data
api.attach_dataset_snapshot(node_id, {
    "dataset_version": "v2.3.1",
    "dataset_hash": hashlib.sha256(data_bytes).hexdigest(),
    ...
})

# After computing features
api.attach_feature_snapshot(node_id, {
    "feature_names": feature_pipeline.feature_names,
    "feature_set_hash": compute_feature_hash(feature_pipeline),
    ...
})

# After creating model
api.attach_model_snapshot(node_id, {
    "architecture": model.architecture,
    "checkpoint_hash": hash_checkpoint(checkpoint_path),
    ...
})

# After validation
api.attach_validation_snapshot(node_id, {
    "walk_forward_accuracy": results.accuracy,
    "sharpe_ratio": results.sharpe,
    ...
})
```

### Step 4 — Save the checkpoint artifact

Store the checkpoint at a deterministic path and record its hash in `ModelSnapshot.checkpoint_hash`. The checkpoint + the `EnvironmentSnapshot` together allow full experiment reproduction.

---

## Reproducing a Historical Experiment

```python
# Load the lineage record
resp = api.get_experiment(node_id)
node = resp["data"]

# Read the environment snapshot
env = node["environment"]
print(f"Reproduce at git commit: {env['git_commit_hash']}")
print(f"Random seed:             {env['random_seed']}")
print(f"PyTorch version:         {env['pytorch_version']}")

# Read the dataset snapshot
ds = node["dataset_snapshot"]
print(f"Dataset version:  {ds['dataset_version']}")
print(f"Dataset hash:     {ds['dataset_hash']}")

# Load the model checkpoint
ms = node["model_snapshot"]
print(f"Checkpoint hash:  {ms['checkpoint_hash']}")
```

Then:
1. `git checkout <git_commit_hash>`
2. Install `pytorch=<pytorch_version>`
3. Load dataset at `dataset_version`
4. Set `random_seed`
5. Train with configuration from `model_snapshot`

---

## Environment Variable Filtering

Non-sensitive environment variables are captured automatically. The allow-list includes:

- `LANG`, `LC_*`, `TZ`, `PATH`, `PYTHONPATH`
- `CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`
- `SIGNALIUM_*`, `TRAINER_*`, `TORCH_*`, `NCCL_*`

Variables matching `SECRET`, `TOKEN`, `PASSWORD`, `KEY`, `CREDENTIAL`, `AUTH`, `PRIVATE`, or `CERT` are **never** captured.
