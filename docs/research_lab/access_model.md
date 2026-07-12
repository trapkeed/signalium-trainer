# Research Lab — Access Model

## Claim

> **The Research Lab exists entirely inside the trainer. Only administrators may access it.**

This document proves the claim through five independent mechanisms: process isolation, authentication layers, route namespacing, zero public interfaces, and the access call graph.

---

## 1. Process Isolation

The Research Lab runs as a Python FastAPI server on an internal port (default: 8002) inside the trainer container.

```
Internet
  │
  ▼
[Express API — port 8080]          ← Only port exposed externally
  │   adminAuth middleware
  │   (token required, fail-closed)
  ▼
[Research Lab Python server — port 8002]   ← Internal only, never exposed
  │   _require_admin (defence-in-depth)
  │   (second token check)
  ▼
[ResearchLabAPI — in-process Python class]  ← No HTTP exposure
```

Port 8002 is never mapped to the outside world. The only path to the Research Lab is through the Express proxy, which enforces admin authentication first.

---

## 2. Authentication Layers

Access requires passing **two independent authentication checks**:

| Layer | Component | Mechanism | Failure Behaviour |
|---|---|---|---|
| 1 (outer) | Express `adminAuth` middleware | `Authorization: Bearer <platform-jwt>` verified against `SIGNALIUM_JWKS_URL` (ES256, `iss=signalium`, `aud=signalium-trainer`) | 401/403, request rejected |
| 2 (inner) | FastAPI `_require_admin` dependency | Bearer token vs. `RESEARCH_LAB_ADMIN_TOKEN` | 403, request rejected |

Layer 1 is **fail-closed**: the platform JWT must be present, unexpired, and signed by the platform's JWKS endpoint. There is no static-token fallback and no `X-Admin-Token` path.
Layer 2 is **fail-closed**: if `RESEARCH_LAB_ADMIN_TOKEN` is not set in the environment, all internal trainer→Research Lab calls are rejected with 503.

The token check uses constant-time comparison at the Express layer to prevent timing side-channel attacks.

---

## 3. Route Namespacing

Every Research Lab route in the Express API is prefixed with `/api/admin/research/`:

```
/api/admin/research/healthz
/api/admin/research/summary
/api/admin/research/datasets
/api/admin/research/features
/api/admin/research/experiments
/api/admin/research/shadow-tests
/api/admin/research/backtests
/api/admin/research/searches
/api/admin/research/benchmarks
/api/admin/research/failures
/api/admin/research/regime
/api/admin/research/comparisons
/api/admin/research/deployments
```

The `adminAuth` middleware is applied to the entire `/admin/research` namespace via `router.use("/admin/research", adminAuth)` — not to individual routes. There is no route under `/api/admin/research/*` that bypasses auth.

Retail user routes (if any exist in other routers) are never prefixed with `/admin/`.

---

## 4. Zero Public Interfaces

The Research Lab has no public interfaces of any kind:

| Interface Type | Status | Why |
|---|---|---|
| API documentation browser (Swagger) | **REMOVED** | `docs_url=None` in FastAPI |
| ReDoc documentation UI | **REMOVED** | `redoc_url=None` in FastAPI |
| OpenAPI schema endpoint | **REMOVED** | `openapi_url=None` in FastAPI |
| CORS headers allowing browser access | **REMOVED** | No `CORSMiddleware` on port 8002 |
| Public health endpoint (unauthenticated) | **DOES NOT EXIST** | All endpoints require admin token |
| Frontend UI or dashboard | **DOES NOT EXIST** | Trainer is headless — no frontend |
| Retail-accessible routes | **DOES NOT EXIST** | All routes require admin token |

The Research Lab server on port 8002 returns 403 for every request without a valid token, including requests to the root path.

---

## 5. The Access Call Graph

```
Signalium Admin Portal
  │  HTTP request with Authorization: Bearer <admin-token>
  ▼
Express API — /api/admin/research/*
  │  adminAuth() — validates token (fail-closed)
  │  If token invalid: return 401/403, no upstream request made
  ▼
Research Lab Python Server — port 8002, /research/*
  │  _require_admin() — second token validation (defence-in-depth)
  │  If token invalid: return 403
  ▼
ResearchLabAPI — in-process Python methods
  │  Business logic: datasets, experiments, shadow tests, etc.
  ▼
ApiResponse → JSON → Express → Admin Portal
```

No other call path exists. Retail users cannot reach the Research Lab by any route.

---

## 6. What Administrators Can Do

Administrators with a valid `RESEARCH_LAB_ADMIN_TOKEN` can access all Research Lab responsibilities:

| Responsibility | Route Prefix |
|---|---|
| Dataset Creation | `POST /api/admin/research/datasets` |
| Feature Discovery | `POST /api/admin/research/features` |
| Experiment Tracking | `/api/admin/research/experiments` |
| Training Runs | `POST /api/admin/research/experiments/:id/start` |
| Validation | `POST /api/admin/research/experiments/:id/gates` |
| Backtesting | `POST /api/admin/research/backtests` |
| Shadow Testing | `/api/admin/research/shadow-tests` |
| Version Promotion | `POST /api/admin/research/experiments/:id/approve` |
| Failure Analysis | `/api/admin/research/failures` |
| Deployment Approval | `POST /api/admin/research/experiments/:id/approve` |
| **(S6) Rollback Approval** | `POST /api/admin/research/rollbacks/:id/approve` |
| **(S6) Archive Authorization** | `POST /api/admin/research/lifecycle/authorize` (action=`ARCHIVE`) |
| **(S6) Retire Authorization** | `POST /api/admin/research/lifecycle/authorize` (action=`RETIRE`) |

### 6a. Phase S6 — Sole Lifecycle Authority

As of Phase S6, Research Lab is not just *a* participant in deployment/rollback decisions — it is the **sole authority** permitted to approve deployment, roll back deployment, promote, archive, and retire QDE versions. This is enforced structurally at the Express gateway (see `docs/research_lab/permissions_matrix.md` §3), not merely by convention: every QDE lifecycle-mutation route requires a matching, unconsumed, APPROVED `LifecycleAuthorization` record that only Research Lab can issue. Full detail: `docs/research_lab/permissions_matrix.md`. Updated component diagram: `docs/research_lab/architecture_diagram.md`.

---

## 7. What Retail Users Cannot Do

Retail users have zero access to any Research Lab functionality:

- They cannot list experiments.
- They cannot view training metrics.
- They cannot read shadow test results.
- They cannot approve or reject deployments.
- They cannot view failure events.
- They cannot access any `/api/admin/*` route.

This is enforced by the Express `adminAuth` middleware, not by the absence of documentation or UI. The restriction is in code, not in convention.

---

## Verification

```bash
# Verify adminAuth is applied to ALL research routes (not individual routes)
grep -n "router.use.*adminAuth" apps/trainer-api/src/routes/research_lab.ts
# Expected: one result at the top — router.use("/admin/research", adminAuth)

# Verify no unauthenticated research routes exist
grep -n "router\.(get\|post\|put\|delete\|patch)" apps/trainer-api/src/routes/research_lab.ts | grep -v "admin/research"
# Expected: zero results

# Verify FastAPI docs are disabled
grep -n "docs_url\|redoc_url\|openapi_url" trainer/research_lab/research_lab_server.py
# Expected: all three set to None

# Verify no CORS middleware on the Python server
grep -n "CORSMiddleware" trainer/research_lab/research_lab_server.py
# Expected: zero results
```
