# Research Lab — Permissions Matrix (Phase S6)

## Claim

> **Research Lab is the SOLE authority permitted to approve deployment, roll back deployment, promote models, archive models, and retire models. It remains admin-only: no public, customer, or unauthenticated API access exists anywhere in this system.**

This matrix is the authoritative reference for who/what can perform every QDE version lifecycle action, and through which gate.

---

## 1. Lifecycle Actions Matrix

| Action | Who may request it | Who may authorize it | Where authorization is recorded | Enforcement point |
|---|---|---|---|---|
| **Approve deployment** (promote to ACTIVE) | Any admin, via `promotion/approve` or `promotion/auto` | **Research Lab only** — auto-issued by `approve_deployment` after all 3 deployment gates pass | `lifecycle_authorizations` table, action=`PROMOTE` | `requireResearchLabAuthorization("PROMOTE")` in `apps/trainer-api/src/routes/qde.ts`, consumed against `research_lab_server.py` |
| **Roll back deployment** | Any admin, via `/qde/rollback` or `/qde/rollback/:targetVersionId` | **Research Lab only** — auto-issued by `approve_rollback` after the two-admin rollback-request workflow | `lifecycle_authorizations` table, action=`ROLLBACK` | `requireResearchLabAuthorization("ROLLBACK")` in `qde.ts`, keyed by the current ACTIVE version resolved from `/qde/active` |
| **Promote a model** | *(same action as "approve deployment" — promotion IS the deployment approval)* | **Research Lab only** | same as above | same as above |
| **Archive a model** | Any admin, via `/qde/versions/:versionId/archive` | **Research Lab only** — via `POST /admin/research/lifecycle/authorize` (`action=ARCHIVE`) | `lifecycle_authorizations` table, action=`ARCHIVE` | `requireResearchLabAuthorization("ARCHIVE")` in `qde.ts` |
| **Retire a model** | Any admin, via `/qde/versions/:versionId/retire` | **Research Lab only** — via `POST /admin/research/lifecycle/authorize` (`action=RETIRE`) | `lifecycle_authorizations` table, action=`RETIRE` | `requireResearchLabAuthorization("RETIRE")` in `qde.ts` |

**No lifecycle mutation route in `qde.ts` can execute without first consuming a matching, unconsumed, APPROVED `LifecycleAuthorization` issued by Research Lab.** This is verified at the Express gateway layer (see §3), not merely by convention.

---

## 2. Authorization Lifecycle

Every `LifecycleAuthorization` moves through a strict, one-way state machine:

```
PENDING ──approve──▶ APPROVED ──consume──▶ CONSUMED
   │
   └──reject──▶ REJECTED
```

- **PENDING**: created but not yet approved (not currently reachable via any route — `PROMOTE`/`ROLLBACK` are auto-created as `APPROVED`; `ARCHIVE`/`RETIRE` are created as `APPROVED` directly by `authorize_lifecycle_action`).
- **APPROVED**: eligible to be consumed exactly once.
- **CONSUMED**: terminal. Cannot be reused — a second mutation attempt against the same authorization is rejected with 403.
- **REJECTED**: terminal. Never eligible for consumption.

This guarantees **one authorization → exactly one mutation**. An admin cannot approve a promotion once and then use that same sign-off to promote a different version later, or promote-then-rollback-then-promote-again on a single approval.

---

## 3. Enforcement Is Structural, Not Conventional

| Layer | What it guarantees |
|---|---|
| `trainer/qde/qde_api.py` (`archive_version`/`retire_version`) | Business rule: an ACTIVE version cannot be archived directly (must be promoted-over or rolled-back-from first). |
| `trainer/research_lab/research_lab_api.py` (`authorize_lifecycle_action`) | Rejects any attempt to directly authorize `PROMOTE`/`ROLLBACK` — those must go through the gated `approve_deployment`/`approve_rollback` workflows, which already enforce the 3 deployment gates / two-admin rollback approval. |
| `trainer/research_lab/research_lab_api.py` (`consume_lifecycle_authorization`) | Atomically finds the most recent APPROVED, unconsumed authorization for `(version_id, action)` and marks it CONSUMED in the same operation — no TOCTOU window where two callers could both succeed off the same authorization. |
| `apps/trainer-api/src/middlewares/require_rl_authorization.ts` | Gateway-level enforcement: every lifecycle mutation route calls this middleware, which makes a server-to-server call to Research Lab's `/research/lifecycle/consume` and blocks (403) unless Research Lab approves. `trainer.qde` never imports `trainer.research_lab` (or vice versa) — the authority relationship is enforced entirely at the Express layer, preserving process isolation between the two Python services. |
| `apps/trainer-api/src/routes/qde.ts` | Applies `adminAuth` + `requireResearchLabAuthorization` to all four mutation routes: `promotion/approve`, `promotion/auto`, `rollback/:targetVersionId`, `rollback`, `versions/:versionId/archive`, `versions/:versionId/retire`. |

**Verification:**

```bash
# Confirm every lifecycle mutation route requires both adminAuth and RL authorization
grep -n "requireResearchLabAuthorization" apps/trainer-api/src/routes/qde.ts
# Expected: 6 matches — promotion/approve, promotion/auto, rollback/:targetVersionId, rollback, archive, retire

# Confirm authorize_lifecycle_action rejects PROMOTE/ROLLBACK (only Research Lab's gated
# deployment/rollback approval flows may issue those)
grep -n "must go through approve_deployment" trainer/research_lab/research_lab_api.py

# Confirm trainer.qde and trainer.research_lab never import each other
grep -rn "^from trainer.research_lab\|^import trainer.research_lab" trainer/qde/
grep -rn "^from trainer.qde\|^import trainer.qde" trainer/research_lab/
# Expected: zero results for both
```

---

## 4. Admin-Only Access Is Unchanged

Everything in `docs/research_lab/access_model.md` continues to hold under S6: Research Lab still has zero public interfaces, still runs on an internal-only port (8002), still requires the same fail-closed, two-layer admin token check (`adminAuth` at the Express gateway + `_require_admin` at the FastAPI layer). S6 adds **new authority**, not new exposure — `LifecycleAuthorization` records and the four `/admin/research/lifecycle/*` routes sit entirely inside the existing `/admin/research/*` namespace, behind the same `router.use("/admin/research", adminAuth)` guard as every other Research Lab route.

| Question | Answer |
|---|---|
| Can a retail user call any lifecycle-authorization route? | No — same `adminAuth` gate as all other Research Lab routes. |
| Can a retail user call `/qde/versions/:id/archive` directly? | No — `adminAuth` + `requireResearchLabAuthorization` both apply. |
| Can an admin bypass Research Lab and promote/rollback/archive/retire directly against `qde_server.py`? | No — `qde_server.py` itself has no auth of its own; the ONLY path to it that carries lifecycle authority is through the Express gateway, and the gateway refuses to forward the mutation without a consumed Research Lab authorization. |
| Does Research Lab gain any new public surface? | No — 4 new routes, all under the existing `/admin/research/*` prefix, all behind the existing `adminAuth` middleware. |

---

## 5. Full Lifecycle Action → Route Reference

| Action | Admin-facing route (Express, port 8080) | Upstream (QDE, port 8001) | Authorization source (Research Lab, port 8002) |
|---|---|---|---|
| Evaluate deployment gates | `POST /api/admin/research/experiments/:id/gates` | — | n/a (gate evaluation, not authorization) |
| **Approve deployment / Promote** | `POST /api/admin/research/experiments/:id/approve` (issues authorization) → `POST /api/qde/versions/:id/promotion/approve` (consumes it) | `POST /qde/versions/:id/promotion/approve` | `POST /research/lifecycle/consume` (action=PROMOTE) |
| Auto-promote | `POST /api/qde/versions/:id/promotion/auto` | `POST /qde/versions/:id/promotion/auto` | `POST /research/lifecycle/consume` (action=PROMOTE) |
| Request rollback | `POST /api/admin/research/rollbacks` | — | n/a |
| **Approve rollback** | `POST /api/admin/research/rollbacks/:id/approve` (issues authorization) → `POST /api/qde/rollback` or `/api/qde/rollback/:targetVersionId` (consumes it) | `POST /qde/rollback[/:targetVersionId]` | `POST /research/lifecycle/consume` (action=ROLLBACK) |
| **Archive a model** | `POST /api/admin/research/lifecycle/authorize` (action=ARCHIVE, issues authorization) → `POST /api/qde/versions/:id/archive` (consumes it) | `POST /qde/versions/:id/archive` | `POST /research/lifecycle/consume` (action=ARCHIVE) |
| **Retire a model** | `POST /api/admin/research/lifecycle/authorize` (action=RETIRE, issues authorization) → `POST /api/qde/versions/:id/retire` (consumes it) | `POST /qde/versions/:id/retire` | `POST /research/lifecycle/consume` (action=RETIRE) |
| List/inspect authorizations | `GET /api/admin/research/lifecycle/authorizations[/:id]` | — | `GET /research/lifecycle/authorizations[/:id]` |
