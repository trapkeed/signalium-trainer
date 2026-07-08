# Research Lab — Deployment Flow

## Overview

No QDE version may be deployed to production unless it passes mandatory Research Lab deployment gates and receives explicit administrator approval. This document describes the full deployment decision workflow from experiment completion through live activation.

**Phase S6 update:** Research Lab is now the SOLE authority permitted to approve a deployment. `POST /api/admin/research/experiments/:id/approve` no longer just records the decision — it also auto-issues an APPROVED `LifecycleAuthorization` (action=`PROMOTE`) for the associated QDE version. The subsequent `POST /api/qde/versions/:id/promotion/approve` call cannot succeed unless it consumes that exact authorization. See `docs/research_lab/permissions_matrix.md` for the full authority model.

---

## Deployment Gates

Three gates must all pass. Failure at any gate blocks deployment.

### Gate 1 — Decision Quality Improvement

The challenger experiment's `decision_quality` must be greater than or equal to the baseline (incumbent) experiment's `decision_quality`.

```
challenger.decision_quality >= baseline.decision_quality
AND challenger.decision_quality > 0.0
```

**Rationale:** The QDE's primary function is to make high-quality trade decisions. Any regression in decision quality is unacceptable.

### Gate 2 — Risk Adjusted Return Improvement

The challenger's `max(sharpe_ratio, sortino_ratio)` must not regress versus the baseline.

```
max(challenger.sharpe, challenger.sortino) >= max(baseline.sharpe, baseline.sortino)
AND max(challenger.sharpe, challenger.sortino) > 0.0
```

**Rationale:** A new QDE version must deliver at least as much return per unit of risk as the version it replaces.

### Gate 3 — Maximum Drawdown Stability

The challenger's `max_drawdown` must not increase by more than the configured tolerance (default 2%).

```
challenger.max_drawdown <= baseline.max_drawdown + 0.02
```

**Rationale:** Drawdown directly impacts capital preservation. Material drawdown increases require a full review cycle, not deployment.

---

## Deployment Workflow

```
Experiment COMPLETED (with full metrics)
          │
          │  POST /admin/research/experiments/:id/gates
          │  { "baseline_experiment_id": "exp-..." }
          ▼
   Gate Evaluation
          │
          ├── [FAIL] Any gate fails
          │     Response: { "passed": false, "failed_gates": [...] }
          │     → Researcher reviews, fixes, re-runs experiment
          │
          └── [PASS] All three gates pass
                │
                │  POST /admin/research/experiments/:id/approve
                │  { "gate_id": "gate-...", "approver": "quant-lead" }
                ▼
         Deployment APPROVED
                │
                │  Register version in QDE registry
                │  POST /api/qde/versions
                │
                │  Run QDE pipeline
                │  POST /api/qde/versions/:id/pipeline/start
                │    (advance through all 6 stages)
                │
                │  Promote to ACTIVE
                │  POST /api/qde/versions/:id/promotion/approve
                ▼
           QDE ACTIVE
```

---

## Evaluate Gates

```
POST /api/admin/research/experiments/:experimentId/gates
Authorization: Bearer <token>

{
  "baseline_experiment_id": "exp-b3e1a2c9",
  "max_drawdown_tolerance": 0.02
}
```

Response:
```json
{
  "gate_id": "gate-a4f2c1d0",
  "experiment_id": "exp-9b8f45ad",
  "qde_version": "QDE v13",
  "passed": true,
  "gate_results": [
    { "gate": "Decision Quality Improvement", "passed": true, "value": 0.681, "baseline": 0.661, "delta": 0.02 },
    { "gate": "Risk Adjusted Return Improvement", "passed": true, "value": 1.74, "baseline": 1.50, "delta": 0.24 },
    { "gate": "Maximum Drawdown Stability", "passed": true, "value": 0.087, "baseline": 0.110, "delta": -0.023 }
  ],
  "recommendation": "APPROVED: All deployment gates passed."
}
```

---

## Approve Deployment

```
POST /api/admin/research/experiments/:experimentId/approve
Authorization: Bearer <token>

{
  "gate_id": "gate-a4f2c1d0",
  "approver": "quant-lead",
  "notes": "Reviewed shadow metrics, regime analysis, and comparison report. Approved."
}
```

**Rules:**
- `gate_id` must reference a gate evaluation that passed (`passed: true`).
- `approver` is recorded in the audit trail.
- Attempting to approve without a passing gate returns `400 Bad Request`.
- **(S6)** This call also auto-issues an APPROVED `LifecycleAuthorization` (action=`PROMOTE`, version_id=the gate's `qde_version`). This authorization is single-use — it is consumed atomically the first time `POST /api/qde/versions/:id/promotion/approve` (or `/promotion/auto`) succeeds, and cannot be reused for a later promotion.

---

## Reject Deployment

```
POST /api/admin/research/experiments/:experimentId/reject
Authorization: Bearer <token>

{
  "rejector": "risk-team",
  "reason": "Regime analysis shows poor performance in high-volatility conditions. Requires retraining."
}
```

---

## Deployment History

All approval and rejection decisions are recorded permanently:

```
GET /api/admin/research/deployments
Authorization: Bearer <token>
```

Each record includes: `action`, `experiment_id`, `gate_id`, `approver`/`rejector`, `notes`, `timestamp`.

---

## First Deployment (No Baseline)

When there is no incumbent QDE version, gates apply absolute floors only:

- `decision_quality > 0.0`
- `max(sharpe, sortino) > 0.0`
- `max_drawdown` is evaluated without a relative comparison

Omit `baseline_experiment_id` from the gate evaluation request.

---

## Archiving and Retiring a Model (Phase S6)

Once a version has been superseded (promoted-over) or rolled back from ACTIVE, it can be archived and eventually retired. Both actions require Research Lab authorization — there is no gate evaluation step for these (they are administrative cleanup, not a deployment decision), but the SOLE-authority rule still applies.

```
POST /api/admin/research/lifecycle/authorize
Authorization: Bearer <token>

{
  "action": "ARCHIVE",
  "version_id": "qde-v12-...",
  "approver": "quant-lead",
  "reason": "Superseded by v13; retained for lineage but no longer eligible for rollback target"
}
```

```
POST /api/qde/versions/:versionId/archive
Authorization: Bearer <token>

{
  "archiver": "quant-lead",
  "reason": "Superseded by v13"
}
```

**Rules:**
- An ACTIVE version cannot be archived directly — promote a replacement or roll back first.
- `authorize_lifecycle_action` refuses `PROMOTE`/`ROLLBACK` actions (those must go through `approve_deployment`/`approve_rollback` above, which enforce the deployment gates / two-admin rollback approval respectively).
- The archive/retire call must consume the exact authorization issued for that `(version_id, action)` pair — see `docs/research_lab/permissions_matrix.md` §2 for the authorization state machine.
- Retiring follows the identical pattern with `"action": "RETIRE"` and `POST /api/qde/versions/:versionId/retire`.
