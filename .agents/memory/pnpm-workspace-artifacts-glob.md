---
name: pnpm-workspace artifacts glob
description: pnpm-workspace.yaml must list artifacts/* for artifacts placed under artifacts/ to be resolvable via pnpm --filter.
---

The `createArtifact` tool can register and place a new package under `artifacts/<slug>/` even when `pnpm-workspace.yaml`'s `packages:` list only has `apps/*`, `packages/*`, `scripts` (no `artifacts/*` glob). In that case `pnpm --filter @workspace/<name> run dev` fails with a silent-looking `No projects matched the filters` and the workflow exits immediately (status FINISHED, not a crash) — easy to misread as a app-code problem.

**Why:** pnpm workspace membership is purely glob-driven; a package existing on disk with a valid `package.json` is invisible to `pnpm --filter` if its directory isn't covered by any glob in `packages:`.

**How to apply:** After creating a new artifact (or any package) in a directory prefix not already covered, grep `pnpm-workspace.yaml`'s `packages:` list first before debugging workflow logs. Add the missing glob (e.g. `artifacts/*`) and rerun `pnpm install`, then restart the workflow.
