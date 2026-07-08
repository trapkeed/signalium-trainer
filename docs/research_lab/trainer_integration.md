# Research Lab — Trainer Integration Architecture

## Overview

The Research Lab is the continuous improvement subsystem of the Signalium Trainer. It lives entirely inside the trainer — it has no external dependencies, no user-facing UI, and no public entry points.

The trainer is a headless system. It exposes internal admin APIs consumed by the Signalium Admin Portal. Retail users never interact with the trainer or the Research Lab directly.

---

## Trainer Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          TRAINER                                    │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Express API (port 8080)                    │  │
│  │                                                              │  │
│  │  /api/health         — public health check                   │  │
│  │  /api/qde/*          — QDE lifecycle (internal admin)        │  │
│  │  /api/admin/research/* — Research Lab (admin token required) │  │
│  │                                                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│          │ proxy                          │ proxy                   │
│          ▼                               ▼                         │
│  ┌─────────────────┐          ┌─────────────────────────┐         │
│  │ QDE Server      │          │ Research Lab Server     │         │
│  │ (port 8001)     │          │ (port 8002)              │         │
│  │ FastAPI         │          │ FastAPI                  │         │
│  │                 │          │ NO docs, NO CORS        │         │
│  │ QDE lifecycle:  │          │                         │         │
│  │ version mgmt,   │          │ 10 responsibilities:    │         │
│  │ promotion,      │          │ datasets, features,     │         │
│  │ rollback, A/B   │          │ experiments, training,  │         │
│  └─────────────────┘          │ validation, backtesting, │        │
│                               │ shadow testing,          │        │
│  ┌─────────────────┐          │ promotion, failure,      │        │
│  │ QDE Intelligence│          │ deployment approval      │        │
│  │ (in-process)    │          └─────────────────────────┘        │
│  │                 │                    │                          │
│  │ 11 capabilities │                    ▼                          │
│  │ → QDEDecision   │          ┌─────────────────────────┐         │
│  └─────────────────┘          │   ResearchLabAPI        │         │
│                               │   (in-process Python)   │         │
│                               │                         │         │
│                               │   • in-memory stores    │         │
│                               │   • DB adapter ready    │         │
│                               └─────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
         ▲
         │ HTTPS, admin token required
         │
┌────────────────────────┐
│  Signalium Admin Portal │   ← Only authorised consumer of Research Lab
└────────────────────────┘
```

---

## Research Lab Responsibilities

The Research Lab fulfils ten responsibilities. All are admin-only. None are accessible to retail users.

| # | Responsibility | What It Does |
|---|---|---|
| 1 | Dataset Creation | Register versioned datasets for training and evaluation |
| 2 | Feature Discovery | Track, rank, accept, and reject candidate features |
| 3 | Experiment Tracking | Full lifecycle: create → start → complete/fail → archive |
| 4 | Training Runs | Initiate and monitor training jobs via experiment state machine |
| 5 | Validation | Evaluate deployment gates against configurable criteria |
| 6 | Backtesting | Register historical test runs and mirror results to experiments |
| 7 | Shadow Testing | Observe challenger QDE against champion without executing challenger decisions |
| 8 | Version Promotion | Approve deployment after gates pass; record promotion history |
| 9 | Failure Analysis | Record, investigate, and resolve failure events |
| 10 | Deployment Approval | Gate-controlled sign-off before any version goes active |

---

## Shadow Testing — Detail

Shadow testing is the Research Lab's mechanism for validating a new QDE version in real market conditions without real risk.

```
Live Market Snapshot
  │
  ├──► Champion QDE ──► QDEDecision ──► EXECUTED (live trading)
  │
  └──► Challenger QDE ──► QDEDecision ──► NOT EXECUTED (observed only)
                                │
                                ▼
                    ShadowDecisionRecord
                    (stored in Research Lab)
                                │
                                ▼
                    ShadowTestRun.conclude()
                    (compute agreement rate,
                     per-regime analysis,
                     promotion recommendation)
```

**The challenger's decisions are never executed.** The Research Lab records each decision pair (champion vs. challenger), computes agreement rates per instrument and regime, and produces one of four recommendations:

| Recommendation | Condition |
|---|---|
| `PROMOTE` | ≥85% agreement AND challenger is more confident ≥50% of the time AND minimum decisions reached |
| `REJECT` | <60% agreement |
| `EXTEND` | Minimum decisions not yet reached, or agreement between 60–85% |
| `INCONCLUSIVE` | No decisions recorded |

---

## Data Flow — Experiment Lifecycle

```
Admin Portal
  │
  │  POST /api/admin/research/experiments
  ▼
create_experiment() → ExperimentRecord{PENDING}
  │
  │  POST /api/admin/research/experiments/:id/start
  ▼
start_experiment() → ExperimentRecord{RUNNING}
  │
  │  [training runs outside the trainer; results posted back]
  │
  │  POST /api/admin/research/backtests
  ▼
register_backtest() → metrics mirrored to ExperimentRecord
  │
  │  POST /api/admin/research/experiments/:id/complete
  ▼
complete_experiment() → ExperimentRecord{COMPLETED}
  │
  │  POST /api/admin/research/experiments/:id/gates
  ▼
evaluate_deployment_gates() → DeploymentGateResult{passed/failed}
  │
  │  [if gates pass, and shadow test passes]
  │
  │  POST /api/admin/research/experiments/:id/approve
  ▼
approve_deployment() → ExperimentRecord{ACTIVE}
```

---

## File Map

```
trainer/research_lab/
  __init__.py               — package exports (ResearchLabAPI)
  rl_models.py              — all data models (ExperimentRecord, ShadowTestRun, ...)
  research_lab_api.py       — ResearchLabAPI class (10 responsibilities)
  research_lab_server.py    — FastAPI server (port 8002, no docs, no CORS)

apps/trainer-api/src/
  middlewares/admin_auth.ts — fail-closed token middleware
  routes/research_lab.ts    — Express proxy (all under /api/admin/research/*)
  routes/index.ts           — registers researchLabRouter

docs/research_lab/
  access_model.md           — formal access model with proofs
  trainer_integration.md    — this file
  experiment_flow.md        — experiment state machine
  deployment_flow.md        — deployment gate flow
  validation_flow.md        — validation procedures
  research_flow.md          — general research workflow
  rollback_flow.md          — rollback procedures
```

---

## Security Properties

| Property | Implementation |
|---|---|
| No retail access | `adminAuth` middleware on all `/api/admin/research/*` routes |
| Fail-closed | Requests rejected when `RESEARCH_LAB_ADMIN_TOKEN` is not set |
| Defence-in-depth | Second token check inside the Python server itself |
| No public API browser | `docs_url=None`, `redoc_url=None`, `openapi_url=None` |
| No CORS on internal server | `CORSMiddleware` removed from port 8002 |
| No frontend | Trainer is headless — no HTML, CSS, or JavaScript served |
| Internal port | Port 8002 is never mapped externally — only accessible by Express proxy |
| Constant-time token comparison | Express middleware uses bitwise XOR comparison to prevent timing attacks |

---

## Adding a New Research Lab Responsibility

1. Add the data model to `trainer/research_lab/rl_models.py`
2. Add the method(s) to `ResearchLabAPI` in `trainer/research_lab/research_lab_api.py`
3. Add request models and endpoint(s) to `trainer/research_lab/research_lab_server.py` — every endpoint must include `_: None = Depends(_require_admin)`
4. Add proxy route(s) to `apps/trainer-api/src/routes/research_lab.ts` — all routes must be under `/admin/research/`
5. Update `docs/research_lab/trainer_integration.md` with the new responsibility
6. Update `docs/research_lab/access_model.md` if the access model changes

**The new route must never be placed outside `/api/admin/research/`.** Any route outside this namespace would bypass the admin auth middleware.
