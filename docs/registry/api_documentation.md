# AI Model Registry API Documentation

Base path: `/api/registry`

## Authentication

All endpoints require the `X-API-Key` header.

| Key | Access Level |
|---|---|
| `REGISTRY_API_KEY` | Read-only access |
| `REGISTRY_ADMIN_API_KEY` | Full read+write access |

If neither env var is set, the server runs in open mode (development only).

## Endpoints

### Models

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/models` | read | List all models |
| POST | `/models` | admin | Register a new model |
| GET | `/models/:modelId` | read | Get model by ID |
| DELETE | `/models/:modelId` | admin | Soft-delete a model |
| GET | `/models/:modelId/tags` | read | List tags |
| POST | `/models/:modelId/tags` | admin | Add a tag |
| DELETE | `/models/:modelId/tags/:tagId` | admin | Remove a tag |

### Versions

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/models/:modelId/versions` | read | List versions |
| POST | `/models/:modelId/versions` | admin | Register version |
| GET | `/models/:modelId/versions/:versionId` | read | Get version |
| GET | `/models/:modelId/versions/:versionId/metrics` | read | Get metrics |
| POST | `/models/:modelId/versions/:versionId/metrics` | admin | Record metrics |
| GET | `/models/:modelId/versions/:versionId/validation` | read | Get validation result |

### Artifacts

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/models/:modelId/versions/:versionId/artifacts` | read | List artifacts |
| POST | `/models/:modelId/versions/:versionId/artifacts` | admin | Upload artifact (binary) |
| GET | `/models/:modelId/versions/:versionId/artifacts/download/:artifactId` | read | Download artifact |

Upload headers:
- `X-Artifact-Type`: `torchscript` \| `onnx` \| `manifest` \| `metadata` \| `training_report` \| `evaluation_report` \| `logs` \| `config` \| `other`
- `X-Artifact-Filename`: filename with extension
- `Content-Type: application/octet-stream`

### Lifecycle

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/models/:modelId/versions/:versionId/validate` | admin | Run validation |
| POST | `/models/:modelId/versions/:versionId/promote` | admin | Promote status |
| POST | `/models/:modelId/versions/:versionId/reject` | admin | Reject version |
| POST | `/models/:modelId/versions/:versionId/archive` | admin | Archive version |
| POST | `/models/:modelId/rollback` | admin | Rollback production |
| GET | `/models/:modelId/aliases` | read | List aliases |
| GET | `/models/:modelId/aliases/:aliasName` | read | Get alias |
| GET | `/models/:modelId/deployments` | read | Deployment history |
| GET | `/models/:modelId/rollbacks` | read | Rollback history |

### Search

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/search` | read | Multi-filter search |
| GET | `/stats` | read | Registry statistics |

Search query parameters:
`architecture`, `status`, `datasetVersion`, `featureVersion`, `labelVersion`, `experimentId`, `gitCommit`, `trainerVersion`, `mlflowRunId`, `createdAfter`, `createdBefore`, `page`, `limit`

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `REGISTRY_API_KEY` | No | Read-access API key |
| `REGISTRY_ADMIN_API_KEY` | No | Admin API key |
| `ARTIFACT_STORAGE_PATH` | No | Artifact filesystem root (default: `./data/artifacts`) |
