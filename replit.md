# Signalium Trainer

A modular, GPU-ready deep learning training framework for [Signalium](https://signalium.io). Designed to run on RunPod GPU instances; communicates exclusively with the Signalium REST API — never connects to PostgreSQL directly.

## Run & Operate

```bash
# Install Python dependencies (one-time)
pip install -e ".[dev]"

# Check environment
trainer doctor

# Install Node.js dependencies (one-time, from repo root)
pnpm install

# Run the trainer API server (from repo root)
pnpm --filter @workspace/trainer-api run dev

# Run the test suite
pytest tests/
```

**Note:** GPU-dependent training runs (`trainer train`) require a RunPod instance. The CLI, API server, and tests all work on Replit without a GPU.

## Required Environment Variables

| Variable              | Description                                      |
|-----------------------|--------------------------------------------------|
| `SIGNALIUM_API_KEY`   | Signalium API key (for API communication)        |
| `SIGNALIUM_TOKEN`     | Signalium Bearer token (alternative to API key)  |
| `MLFLOW_TRACKING_URI` | MLflow server URI (optional, for experiment tracking) |
| `SESSION_SECRET`      | Session secret (already configured in Replit secrets) |
| `ARTIFACT_SIGNING_PRIVATE_KEY` | Ed25519 private key (PEM) used to sign exported artifacts/manifests. Set as a Replit secret; without it, exports are unsigned and promotion to SHADOW/CANDIDATE/ACTIVE is blocked. |
| `ARTIFACT_SIGNING_PUBLIC_KEY`  | Ed25519 public key (PEM) used to verify signatures at promotion time. Not sensitive, but env-based so verifiers don't need the private key. |

## Stack

- **Python 3.12** · PyTorch · TorchMetrics · NumPy · Pandas · PyArrow · scikit-learn
- **API:** FastAPI + Uvicorn + Pydantic
- **Config:** Hydra + OmegaConf (`config/config.yaml`)
- **Experiment tracking:** MLflow
- **Export:** ONNX + TorchScript
- **CLI:** Rich + Loguru
- **Trainer API app:** TypeScript (Node.js) in `apps/trainer-api/`

## Where things live

| Path | Purpose |
|------|---------|
| `trainer/` | Core Python ML framework package |
| `trainer/cli/` | CLI entry points (`trainer` command) |
| `trainer/api/` | FastAPI application |
| `trainer/models/` | Model definitions |
| `trainer/training/` | Training loop framework |
| `trainer/qde/` | Quantitative decision engine |
| `trainer/research_lab/` | Research Lab persistence (SQLite at `data/research_lab.db`) |
| `apps/trainer-api/` | TypeScript trainer API service |
| `config/` | Hydra YAML configs |
| `tests/` | pytest test suite (`tests/unit/`, `tests/integration/`) |
| `docs/` | Architecture, developer guide, deployment docs |
| `deploy/` | PM2, monitoring, and backup scripts |

## Architecture decisions

- **No direct DB access:** All external data flows through the Signalium REST API.
- **Hydra config:** All hyperparameters and paths come from `config/config.yaml` and environment-specific overrides in `config/environments/`.
- **Research Lab** uses a local SQLite database (`data/research_lab.db`) via `RESEARCH_LAB_DB_PATH` env var; injectable via `ResearchLabAPI(db=...)`.
- **QDE (Quantitative Decision Engine):** Pure output-heads model; a single translator function maps heads → execution vocabulary. Capabilities never import `QDEDecision` directly.
- **Registry champion/terminal ordering:** Re-promoting a terminal-state entry must exit terminal state before granting champion status.

## Gotchas

- `torch` requires a real `.so` shared library; on Replit the torch extension may be missing. `trainer.registry` uses lazy imports to avoid forcing torch on unrelated code paths.
- `mypy` CI runs only on `trainer/`, not `tests/`.
- 21 pre-existing failures in `test_icari.py` (related to `AllocationRequest.ensemble_confidence`) are unrelated to other work — do not chase them.
- The `trainer/ensemble/` directory does not exist in the repo history; it was removed before the graft root commit.

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._
