---
name: Model Registry architecture
description: Phase 12 AI Model Registry — layout, routing quirks, and non-obvious decisions
---

# Model Registry — Durable Architecture Notes

## State machine
`isValidTransition` in `promotionService.ts` is the single source of truth. The full happy path:
`uploading → uploaded → validating → validated → candidate → challenger → champion → production → retired → archived`
Any non-terminal state (except `uploading`) can go to `rejected`. Terminal states: `archived`, `rejected`.

## Routing gotcha — stats vs search
The `/stats` route is registered **directly** on the registry router (not mounted as a sub-router via `registry.use("/stats", ...)`) because mounting a sub-router at `/stats` causes `router.get("/")` to capture `GET /stats`, returning search results instead of stats. Any future stats-like endpoints should follow this pattern.

## req.params cast
Express 5 types `req.params[key]` as `string | string[]`. Always cast with `String(req.params.X)` in route handlers.

## Artifact upload
Artifacts are streamed via `req.pipe(writeStream)` — no `express.raw()` needed because `express.json()` only fires on `Content-Type: application/json`. Binary uploads bypass it automatically.

## DB lib rebuild
After changing `lib/db/src/schema/`, run `pnpm run typecheck:libs` before checking leaf packages. The db package exports from source (not compiled declarations).

**Why:** The `@workspace/db` package.json uses `"./src/index.ts"` as its export, so TypeScript needs `tsc --build` to regenerate declarations when cross-package types change.

## OpenAPI codegen — naming conflicts
Inline requestBody schemas in the OpenAPI spec generate type names in **both** `api.ts` (Zod) and `types/` dir (TS). This produces `export *` conflicts. Always use `$ref: "#/components/schemas/NamedSchema"` for request bodies instead of inline objects.

## Worker queue
In-memory, no Redis. `queue.ts` handles retries (3 max, exponential backoff). `checksumWorker.ts` and `validationWorker.ts` register their handlers on module import — both are imported in `app.ts` to ensure registration before any requests arrive.

## Artifact workflow name
The running API server workflow is named `artifacts/api-server: API Server` (managed by the artifact scaffold). Use `restartWorkflow({ workflowName: "artifacts/api-server: API Server" })` to restart it. There is no separate "Start application" workflow for this project.
