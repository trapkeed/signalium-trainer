# Signalium Platform

Production-grade AI Model Registry for the Signalium trading signal platform — lifecycle management, versioning, validation, promotion, rollback, search, and audit logging.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `pnpm --filter @workspace/api-server test` — run registry unit tests (73 tests)
- Required env: `DATABASE_URL` — Postgres connection string

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM (10 tables)
- Validation: Zod (`zod`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)
- Testing: Vitest

## Where things live

```
lib/db/src/schema/          — 10 Drizzle tables (models, versions, artifacts, metrics,
                              validation, aliases, tags, deployments, rollbacks, audit_log)
artifacts/api-server/src/
  routes/registry/          — REST API routes (/api/registry/*)
  services/                 — Business logic (registry, artifacts, validation, promotion,
                              search, metrics, audit)
  middlewares/              — Auth (apiKey), rate limiting, replay protection
  workers/                  — In-memory task queue + checksum/validation/cleanup workers
  tests/                    — 5 test files, 73 tests
docs/registry/              — Architecture, lifecycle guide, API docs, DB schema reference
signalium-trainer/          — Python ML trainer (DO NOT MODIFY)
```

## Architecture Decisions

- **API key auth split**: `REGISTRY_API_KEY` (read), `REGISTRY_ADMIN_API_KEY` (admin writes). Both empty = open dev mode.
- **Artifact storage**: Local filesystem at `ARTIFACT_STORAGE_PATH` (default `./data/artifacts/<versionId>/<filename>`). Pluggable — replace `artifactService.ts` for S3/GCS.
- **No automatic promotions**: Every `champion` or `production` transition requires an explicit admin API call. Enforced by the state machine in `promotionService.ts`.
- **Immutable audit log**: `model_audit_log` records every state change, actor, and IP. Never modified, only appended.
- **In-memory workers**: No Redis/BullMQ. Workers use a simple in-memory queue with 3 retries + exponential backoff. Fine for current load; swap for a proper queue at scale.

## Product

The Model Registry is the authoritative source for all AI models on the platform:
- Register and version models coming from the Python trainer
- Upload and verify binary artifacts (TorchScript, ONNX, manifests, metadata)
- Run automated validation checklists before promotion
- Promote models through the lifecycle: `uploading → uploaded → validated → candidate → challenger → champion → production`
- Rollback production models with full audit trail
- Search/filter the registry by architecture, dataset version, feature version, git commit, experiment ID, and more
- Named aliases (champion, production, challenger) point to specific versions

## User Preferences

- The Python trainer (`signalium-trainer/`) must never be modified.
- Keep the OpenAPI spec `info.title` as `Api` — changing it breaks generated filenames.

## Gotchas

- Always run `pnpm run typecheck:libs` before leaf artifact checks after changing `lib/db`.
- The `@workspace/db` exports from source (not compiled declarations) — run `typecheck:libs` to rebuild.
- Artifact upload uses raw HTTP body streaming — `express.json()` is bypassed for `application/octet-stream`.
- `req.params` values in Express 5 routes require `String(req.params.X)` casting.

## Pointers

- See `docs/registry/architecture.md` for system diagram and service map.
- See `docs/registry/lifecycle_guide.md` for state machine transitions and promotion CLI examples.
- See `docs/registry/api_documentation.md` for full endpoint reference.
- See `docs/registry/database_schema.md` for column-level DB reference.
- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.
