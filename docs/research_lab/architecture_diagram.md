# Research Lab — Architecture Diagram (Phase S6)

## Overview

This diagram supersedes the process/authentication diagram in `access_model.md` §1 by adding the S6 authority relationship: Research Lab is now mission control for every QDE version lifecycle mutation, while remaining fully admin-only and process-isolated from `trainer.qde`.

---

## 1. System Diagram

```
                                   Internet
                                      │
                                      ▼
                    ┌──────────────────────────────────────┐
                    │   Express API — port 8080 (public)    │
                    │   (the ONLY externally exposed port)  │
                    └──────────────────────────────────────┘
                           │                        │
                           │                        │
             ┌─────────────┘                        └─────────────┐
             ▼                                                     ▼
  ┌───────────────────────────┐                       ┌───────────────────────────┐
  │  /api/qde/*                │                       │  /api/admin/research/*    │
  │  routes/qde.ts             │                       │  routes/research_lab.ts   │
  │                             │                       │  router.use(adminAuth)    │
  │  Reads/pipeline/A-B: open   │                       │  applied to ENTIRE prefix │
  │  Lifecycle mutations        │◀──────consume────────│                            │
  │  (promote/rollback/         │   RESEARCH_LAB_       │  Lifecycle authority:      │
  │   archive/retire):          │   ADMIN_TOKEN         │  - approve_deployment      │
  │  adminAuth +                │   (server-to-server)  │  - approve_rollback        │
  │  requireResearchLabAuth-    │                       │  - authorize_lifecycle_    │
  │  orization middleware       │                       │    action (ARCHIVE/RETIRE) │
  └───────────────┬─────────────┘                       └───────────────┬───────────┘
                  │                                                      │
                  │ proxyPost/proxyGet                                   │ proxyPost/proxyGet
                  │ (QDE_API_URL)                                        │ (RESEARCH_LAB_API_URL)
                  ▼                                                      ▼
  ┌───────────────────────────┐                       ┌───────────────────────────┐
  │  QDE Python Server         │                       │  Research Lab Python       │
  │  qde_server.py — port 8001 │                       │  Server                    │
  │  (internal only)            │                       │  research_lab_server.py    │
  │                             │                       │  — port 8002 (internal     │
  │  QDELifecycleAPI            │                       │  only)                     │
  │   - register_version        │                       │  _require_admin on every   │
  │   - promote / rollback      │                       │  route (defence-in-depth)  │
  │   - archive / retire  (S6)  │                       │                            │
  │  QDERegistry                │                       │  ResearchLabAPI            │
  │   - stage machine incl.     │                       │   - datasets/features/     │
  │     ARCHIVED (S6)           │                       │     experiments/backtests/ │
  │                             │                       │     shadow-tests/failures/ │
  │  SQLite: qde.db              │                       │     regime/comparisons     │
  └───────────────────────────┘                       │   - LifecycleAuthorization  │
                                                        │     issue/consume (S6)     │
                                                        │  SQLite: research_lab.db   │
                                                        └───────────────────────────┘

  NO import edge exists between trainer.qde and trainer.research_lab in either
  direction. The authority relationship above is enforced ENTIRELY at the
  Express gateway layer (routes/qde.ts + require_rl_authorization.ts), which
  is the only component allowed to talk to both Python services.
```

---

## 2. Why the Import Boundary Matters

`trainer.qde` and `trainer.research_lab` are independently deployable Python services. Neither module imports the other:

```bash
grep -rn "^from trainer.research_lab\|^import trainer.research_lab" trainer/qde/    # zero results
grep -rn "^from trainer.qde\|^import trainer.qde" trainer/research_lab/            # zero results
```

This means Research Lab's new lifecycle-mission-control authority (S6) cannot be implemented as a direct Python function call from `qde_api.py` into `research_lab_api.py` — that would create the forbidden import edge. Instead, authority is enforced as an **HTTP contract mediated by the Express gateway**:

1. Research Lab issues/holds a `LifecycleAuthorization` (its own SQLite table, its own process).
2. The Express gateway (`apps/trainer-api`), which already talks to both Python services, is the single component responsible for checking that a matching authorization exists and consuming it, before it will proxy a mutating request through to QDE.
3. `qde_server.py` itself remains unaware that Research Lab exists — it just serves the lifecycle mutation once asked, exactly as it always has.

This keeps the S3-era process isolation guarantee intact while adding a strictly enforced cross-service authority relationship.

---

## 3. Request Flow — Promote (illustrative of all 4 lifecycle actions)

```
Admin
  │  1. POST /api/admin/research/experiments/:id/gates        (evaluate)
  │  2. POST /api/admin/research/experiments/:id/approve      (approve — Research Lab
  │                                                              auto-issues an APPROVED
  │                                                              PROMOTE LifecycleAuthorization)
  ▼
Express :8080 → Research Lab :8002 (adminAuth, then _require_admin)
  │
  │  3. POST /api/qde/versions/:id/promotion/approve
  ▼
Express :8080 (adminAuth)
  │
  │  requireResearchLabAuthorization("PROMOTE", versionId)
  │  ──▶ POST http://localhost:8002/research/lifecycle/consume
  │        { version_id, action: "PROMOTE" }
  │  ◀── { success: true, status: "CONSUMED" }   (or 403 if none found)
  ▼
Express proxies to QDE :8001
  │
  │  POST /qde/versions/:id/promotion/approve
  ▼
QDE Python Server — version becomes ACTIVE
```

Archive and retire follow the identical shape, substituting `POST /api/admin/research/lifecycle/authorize` (action=`ARCHIVE`|`RETIRE`) for step 2, since those two actions have no separate gate-evaluation workflow.

---

## 4. Component Responsibility Summary

| Component | Responsibility | Cannot do |
|---|---|---|
| Express gateway (`apps/trainer-api`) | Public entrypoint; admin auth; cross-service authority enforcement | Business logic for either QDE or Research Lab |
| `qde_server.py` / `trainer.qde` | Version registry, stage machine, mechanical promote/rollback/archive/retire | Decide *whether* a mutation is authorized — it trusts whatever the gateway forwards |
| `research_lab_server.py` / `trainer.research_lab` | Experiment tracking, deployment gates, failure analysis, and now: sole authority to approve/authorize all 4 lifecycle actions | Directly mutate QDE state — it has no network path to `qde_server.py` and no import of `trainer.qde` |
