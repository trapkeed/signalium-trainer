---
name: Visibility vs Capability dependency split
description: Pattern for gating read-only internal dashboards separately from mutating research/deployment actions in the research_lab_server SSO layer.
---

The research lab server's SSO layer (`sso.py`) has two independent per-role
grant sets: `Capability` (mutating actions — promote, approve, retrain, etc.)
and `Visibility` (read-only dashboards/reports). Each has its own FastAPI
dependency factory in `research_lab_server.py`:

- `require_capability(capability)` — checks `identity.has_capability(...)`.
- `require_visibility(visibility)` — checks `identity.has_visibility(...)`.

Both depend on the same cached `_require_signalium_identity`, so adding a
new Visibility-gated dashboard costs one dependency factory call per route
and does not touch the Capability table at all.

**Why:** some roles (e.g. QuantResearcher, MLResearcher, DeploymentApprover)
should be able to *view* internal health/observability dashboards without
gaining the ability to *trigger* promotions, retraining, or approvals. A
single combined permission set would force either over-granting mutation
rights or under-granting visibility.

**How to apply:** when adding a new read-only aggregator/dashboard
endpoint, add a `Visibility.<NAME>` enum value + `ROLE_VISIBILITY` wiring in
`sso.py`, then gate the endpoint with
`Depends(require_visibility(Visibility.<NAME>))` — do not reuse or extend
`Capability` for this purpose, and remember to verify the excluded role
(e.g. Auditor) actually gets 403, not just that included roles get 200.
