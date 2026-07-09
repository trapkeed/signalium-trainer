---
name: Research Lab admin auth contract
description: 401 vs 403 semantics for the Research Lab FastAPI admin bearer-token dependency, and its single-shared-token identity limitation
---

Missing `Authorization` header → 401 Unauthorized; a present-but-wrong bearer token → 403 Forbidden. `_require_admin` in `research_lab_server.py` returns the authenticated admin's identity string (not `None`) so callers can attribute actions.

**Why:** Many existing tests conflated "no token" and "wrong token" as both 403; fixing the missing-token case to 401 required updating every pre-existing `assert status_code == 403` test that sent no `Authorization` header at all (health monitor, champion-challenger, deployment manager, approval gates, pipeline controller, artifact registry). Tests that send a deliberately wrong token correctly keep expecting 403 — check which case a test exercises before "fixing" it.

**How to apply:** When adding a new admin-protected endpoint, use `Depends(_require_admin)` and capture the returned identity (`admin_id: str = Depends(_require_admin)`) to record as the actor in any audit/approval record — prefer `req.body_field or admin_id` so an explicit body value can still override.

**Known limitation:** `RESEARCH_LAB_ADMIN_TOKEN` is one shared secret mapped to one static `RESEARCH_LAB_ADMIN_ID` label — every caller with the token is indistinguishable in audit records. A real per-admin identity model (multiple tokens → distinct identities) does not exist yet.
