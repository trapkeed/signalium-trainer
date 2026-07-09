---
name: Layer new SSO identity behind existing narrow dependency
description: How to add a full identity/role system to a FastAPI service with ~90 existing endpoints without touching most call sites, plus the audit/error-path pitfalls that come with it.
---

When adding real identity (JWT-based SSO with roles/capabilities) on top of a server where most endpoints already depend on a thin `Depends(_require_admin)` returning just a string:

- Keep `_require_admin` as a **thin wrapper** around a new `_require_signalium_identity` dependency (which does full JWT validation, session establishment, sets `request.state.identity`) that just returns `identity.admin_id`. FastAPI dedupes dependencies per request, so both resolve from one validation. This means ~90 pre-existing endpoints need zero changes.
- Add new capability checks via a `require_capability(cap)` dependency **factory** wired through `dependencies=[Depends(require_capability(X))]` on route decorators — this doesn't touch function signatures at all, so it's a clean, low-risk way to retrofit authorization onto many existing routes.
- For endpoints that serve multiple resource "types" through one generic route (e.g. one `/approvals/{id}/approve` for every gate type), don't gate the whole route with one static capability — check the required capability *dynamically inside the handler* based on the resource's own type field.
- **Gotcha — creation vs. resolution asymmetry**: if a workflow has a "propose" step and a separate "approve" step, only gate the approve/reject action by capability. Gating creation the same way blocks the proposer (who is deliberately a different, lower-privileged role than the approver) from ever opening the request.

**Why:** this pattern was used for a 7-role/12-capability SSO retrofit onto a Research Lab FastAPI server; the alternative (updating every endpoint's dependency) would have been much higher-risk and touched ~90+ call sites.

## Audit-logging pitfalls this pattern creates
- An HTTP middleware that logs "success" whenever `request.state.identity` is set is WRONG — identity being set only means *authentication* succeeded; a downstream capability check can still 403. Branch on the final `response.status_code` (>=400 → REJECTED, <400 → SUCCESS), not on whether identity exists.
- Wrap **every** audit-log call (`log_success`, `log_rejected`, session establishment) in try/except that only logs-and-continues. Audit/session persistence failures must never mask or replace the actual auth/authz decision (401/403) with an opaque 500.
- Wrap `await call_next(request)` itself in try/except inside the middleware: if a route handler raises unhandled, no response object exists yet — without a guard, that failure path silently produces zero audit trail. Catch it, write a REJECTED row if identity is available, then bare `raise` to preserve FastAPI/Starlette's normal 500 handling and original traceback.
- In the identity-validation dependency, don't catch only your own custom error hierarchy (e.g. `SSOError` subclasses) — add a broad `except Exception` around the validator call that maps *any* unexpected internal failure (bad key config, parsing bug) to a deterministic 401, never letting it escape as a raw 500.
