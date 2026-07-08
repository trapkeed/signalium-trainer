# Phase S6 Report — Research Lab Becomes Mission Control

## Goal

Make the Research Lab the SOLE authority allowed to: approve deployment, roll back deployment, promote models, archive models, and retire models — while remaining fully admin-only (no public/customer/API access), and without collapsing the process isolation between `trainer.qde` and `trainer.research_lab`.

## What Changed

### 1. QDE Lifecycle — Archive/Retire (new terminal stages)

- Added an `ARCHIVED` stage to the `QDEStage` state machine (`trainer/qde/qde_models.py`), plus an `archived_at` timestamp field.
- Added `QDERegistry.archive()` (`trainer/qde/qde_registry.py`): moves a version to `ARCHIVED`, refuses to archive a version that is currently `ACTIVE` (it must be promoted-over or rolled back from first).
- `RETIRED` already existed as a stage; retiring an archived version was already legal — this phase wires it into the same authorization gate as everything else.
- Added `QDELifecycleAPI.archive_version` / `retire_version` (`trainer/qde/qde_api.py`) and corresponding `POST /qde/versions/:id/archive` / `retire` endpoints on `qde_server.py`.

### 2. Research Lab — Lifecycle Authorization (the authority primitive)

- New `LifecycleAuthorization` model (`trainer/research_lab/rl_models.py`): tracks `action` (`PROMOTE`/`ROLLBACK`/`ARCHIVE`/`RETIRE`), `version_id`, and a one-way status machine `PENDING → APPROVED → CONSUMED` (or `→ REJECTED`).
- New `lifecycle_authorizations` table + CRUD in `rl_database.py`.
- `research_lab_api.py`:
  - `approve_deployment` and `approve_rollback` now **auto-issue** an APPROVED authorization (`PROMOTE`/`ROLLBACK` respectively) as a side effect of their existing gate/approval logic — no new admin action required for the common path.
  - New `authorize_lifecycle_action` (ARCHIVE/RETIRE only — explicitly rejects PROMOTE/ROLLBACK, which must go through the gated flows above).
  - New `consume_lifecycle_authorization`: atomically finds the newest APPROVED, unconsumed authorization for `(version_id, action)` and marks it CONSUMED in the same operation (no reuse, no race window).
- New FastAPI routes on `research_lab_server.py`: `/research/lifecycle/authorize`, `/research/lifecycle/consume`, `/research/lifecycle/authorizations[/:id]` — all behind the existing `_require_admin` dependency.

### 3. Express Gateway — Structural Enforcement

Because `trainer.qde` and `trainer.research_lab` must never import each other, the "Research Lab is sole authority" rule could not be enforced as a Python-to-Python call. It is enforced at the one layer that legitimately talks to both services:

- New middleware `requireResearchLabAuthorization(action, extractVersionId)` (`apps/trainer-api/src/middlewares/require_rl_authorization.ts`): calls Research Lab's `/research/lifecycle/consume` using the shared `RESEARCH_LAB_ADMIN_TOKEN`, and blocks with 403 if no matching authorization is found.
- Applied (with `adminAuth`) to all four QDE lifecycle-mutation routes in `apps/trainer-api/src/routes/qde.ts`: `promotion/approve`, `promotion/auto`, `rollback/:targetVersionId`, `rollback`, plus the two new `versions/:id/archive` and `/retire` routes.
- Rollback authorizations are keyed by the *current active* version (what `approve_rollback` records), so the middleware resolves that from `/qde/active` rather than trusting the request body.
- New `/admin/research/lifecycle/*` proxy routes added to `apps/trainer-api/src/routes/research_lab.ts`, inside the existing `/admin/research` namespace (no new admin-auth surface).

## Verification Performed

- `python -m py_compile` on all touched Python files — clean.
- `tsc --noEmit` on `apps/trainer-api` — clean except one pre-existing, unrelated error (`packages/api-zod` dist not built, present before this phase).
- Workflow restart: `apps/trainer-api` builds and serves successfully.
- Functional (in-process, no HTTP) tests of the new Python logic:
  - Archive → Retire happy path succeeds.
  - Archiving an `ACTIVE` version is correctly rejected.
  - `authorize_lifecycle_action` → `consume_lifecycle_authorization` happy path succeeds exactly once; a second consume attempt on the same authorization correctly fails.
  - Direct `authorize_lifecycle_action` calls for `PROMOTE`/`ROLLBACK` are correctly rejected (must go through `approve_deployment`/`approve_rollback`).
  - Consuming with no prior authorization correctly fails.
- Full HTTP round-trip through both Python services + Express gateway was **not** exercised in this dev environment, because `RESEARCH_LAB_ADMIN_TOKEN` is not configured here (pre-existing condition — the admin surface is fail-closed with or without S6). The Python-level tests above exercise the exact same code paths the HTTP routes call into.

## Documentation Deliverables

- `docs/research_lab/permissions_matrix.md` (new) — authoritative lifecycle-action-by-action permissions matrix.
- `docs/research_lab/architecture_diagram.md` (new) — updated system diagram showing the gateway-mediated authority relationship and why the import boundary matters.
- `docs/research_lab/deployment_flow.md` (updated) — documents the auto-issued `PROMOTE` authorization and the new archive/retire flow.
- `docs/research_lab/rollback_flow.md` (updated) — documents the auto-issued `ROLLBACK` authorization and adds S6 API references.
- `docs/research_lab/access_model.md` (updated) — adds the S6 sole-authority responsibilities to the "what admins can do" table; confirms no new public surface was introduced.

## Non-Goals / Explicitly Out of Scope

- No change to the retail/customer-facing surface — Research Lab remains admin-only with zero public interfaces, exactly as established in earlier phases.
- No change to the deployment gate thresholds or the two-admin rollback-approval workflow — S6 adds an authorization *record* around those existing decisions, it does not change the decisions themselves.
- Setting `RESEARCH_LAB_ADMIN_TOKEN` for this dev environment was out of scope for this phase (it is an operational/deployment concern, not a code change) — the gateway already fails closed without it, consistent with pre-S6 behavior.
