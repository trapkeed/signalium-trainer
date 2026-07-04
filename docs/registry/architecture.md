# AI Model Registry — Architecture

## Overview

The Signalium Model Registry is the authoritative source for every AI model used by the platform.
It lives inside the pnpm monorepo (`artifacts/api-server`) and is accessed exclusively via REST API.

The Signalium Trainer (`signalium-trainer/`) writes to the Registry by calling its API after
every training run. The Registry never talks to the trainer directly.

## System Architecture

```
┌─────────────────────┐        REST API         ┌──────────────────────────┐
│  signalium-trainer  │ ───POST /api/registry──► │  artifacts/api-server    │
│  (RunPod / Python)  │                          │  (Express 5 + Drizzle)   │
└─────────────────────┘                          │                          │
                                                 │  ┌──────────────────┐   │
                                                 │  │  Model Registry  │   │
                                                 │  │  Services        │   │
                                                 │  └────────┬─────────┘   │
                                                 │           │             │
                                                 │  ┌────────▼─────────┐   │
                                                 │  │  PostgreSQL      │   │
                                                 │  │  (Drizzle ORM)   │   │
                                                 │  └──────────────────┘   │
                                                 │                          │
                                                 │  ┌──────────────────┐   │
                                                 │  │  Local Storage   │   │
                                                 │  │  ./data/artifacts│   │
                                                 │  └──────────────────┘   │
                                                 └──────────────────────────┘
```

## Database Schema — 10 Tables

| Table | Purpose |
|---|---|
| `models` | Base model registry with soft delete |
| `model_versions` | Versioned model releases with lifecycle status |
| `model_artifacts` | Uploaded files (TorchScript, ONNX, manifests, etc.) |
| `model_metrics` | Training/eval/inference metric key-value store |
| `model_validation` | Per-version validation checklist and report |
| `model_aliases` | Named pointers (champion, production, challenger) |
| `model_tags` | Key-value tags per model |
| `deployment_history` | Deployment event records per environment |
| `rollback_history` | Rollback event records with reason and operator |
| `model_audit_log` | Immutable append-only event audit trail |

## Service Layer

```
src/services/
├── registryService.ts   — Model and version CRUD
├── artifactService.ts   — Binary upload, storage, checksum
├── validationService.ts — Automated validation checklist
├── promotionService.ts  — Lifecycle state machine + rollback
├── searchService.ts     — Multi-filter registry search
├── metricsService.ts    — Metric recording and retrieval
└── auditService.ts      — Audit log writes
```

## Background Workers

```
src/workers/
├── queue.ts              — In-memory task queue with retry
├── checksumWorker.ts     — Post-upload SHA-256 verification
├── validationWorker.ts   — Async model validation
└── cleanupWorker.ts      — Stale corrupted artifact cleanup (daily)
```

## Security Layers

1. **API Key Authentication** — `X-API-Key` header required for all endpoints
2. **Admin Key Separation** — mutating operations require `REGISTRY_ADMIN_API_KEY`
3. **Rate Limiting** — 200 req/min per IP, in-memory sliding window
4. **Replay Protection** — `X-Request-Timestamp` + `X-Request-Id` deduplication
5. **Checksum Verification** — SHA-256 verified on every artifact upload
6. **Audit Logging** — every state transition written to `model_audit_log`
7. **Soft Delete** — models are never hard-deleted from the DB

## Storage

Artifacts are stored on the local filesystem at `ARTIFACT_STORAGE_PATH` (default: `./data/artifacts/<versionId>/<filename>`).
The storage layer is pluggable — replace `artifactService.ts` with S3/GCS when needed.
