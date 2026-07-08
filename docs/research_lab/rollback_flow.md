# Research Lab — Rollback Flow

## Overview

When a live QDE version underperforms or causes unexpected losses, the Research Lab provides the investigation and remediation workflow that supports a rollback. The QDE Lifecycle system handles the mechanical rollback; the Research Lab handles the investigation, failure recording, and path to re-deployment.

**Phase S6 update:** Research Lab is now the SOLE authority permitted to approve a rollback. `POST /api/admin/research/rollbacks/:id/approve` auto-issues an APPROVED `LifecycleAuthorization` (action=`ROLLBACK`, keyed by the CURRENT active version) for the QDE gateway to consume. `POST /api/qde/rollback` / `POST /api/qde/rollback/:targetVersionId` cannot execute without consuming that authorization — see `docs/research_lab/permissions_matrix.md`.

---

## Rollback Trigger Investigation

When a rollback is triggered (via the QDE API), the Research Lab workflow begins immediately in parallel:

```
QDE API:          POST /api/qde/rollback                ← Immediate execution
Research Lab:     POST /api/admin/research/failures      ← Investigation begins
```

These run concurrently. The rollback does not wait for the investigation.

---

## Step 1 — Record the Failure Event

As soon as a rollback is initiated, record the failure event in the Research Lab:

```
POST /api/admin/research/failures
Authorization: Bearer <token>

{
  "category": "UNEXPECTED_DRAWDOWN",
  "severity": "HIGH",
  "qde_version": "QDE v13",
  "description": "Max drawdown exceeded 18% within 48 hours of deployment; circuit breaker triggered",
  "instrument": "EURUSD",
  "regime": "HIGH_VOLATILITY",
  "drawdown_magnitude": 0.183,
  "tags": ["circuit-breaker", "high-volatility"],
  "metadata": {
    "rollback_id": "rb-a4f2c1d0",
    "reverted_to": "QDE v12"
  }
}
```

### Failure Categories

| Category | When to Use |
|---|---|
| `LARGE_LOSS` | Single trade or session P&L breach |
| `UNEXPECTED_DRAWDOWN` | Drawdown exceeds modelled limits |
| `REGIME_FAILURE` | Version performs poorly in a specific market regime |
| `EXECUTION_FAILURE` | Decision engine produces errors, timeouts, or invalid outputs |
| `MARKET_CONDITION_FAILURE` | Conditions outside the training distribution caused failure |

### Failure Severities

| Severity | Definition |
|---|---|
| `LOW` | Observable degradation; no immediate capital risk |
| `MEDIUM` | Meaningful underperformance requiring investigation |
| `HIGH` | Significant capital impact; rollback likely required |
| `CRITICAL` | Immediate rollback mandatory; all trading halted |

---

## Step 2 — Investigate Root Cause

```
POST /api/admin/research/failures/:failureId/investigate
Authorization: Bearer <token>

{
  "root_cause": "The model had not seen high-volatility regime transitions in training data prior to 2024. The EURUSD volatility spike on 2026-07-05 fell outside the training distribution, leading to systematic over-sizing.",
  "notes": "Regime coverage gap: HIGH_VOLATILITY transitions accounted for only 3.2% of training data."
}
```

This moves the failure to `INVESTIGATING` status and records the root cause.

---

## Step 3 — Run Post-Mortem Regime Analysis

```
POST /api/admin/research/regime
Authorization: Bearer <token>

{
  "qde_version": "QDE v13",
  "regime_results": {
    "TRENDING_BULL":   { "sharpe_ratio": 2.1, "max_drawdown": 0.06 },
    "HIGH_VOLATILITY": { "sharpe_ratio": -0.4, "max_drawdown": 0.183 },
    ...
  },
  "notes": "Post-mortem regime analysis after rollback"
}
```

The `worst_regime` field and `overall_regime_stability` score identify where the failure concentrated.

---

## Step 4 — Address the Root Cause

| Root Cause Type | Research Action |
|---|---|
| Training data gap | Register a new dataset with better regime coverage |
| Feature deficiency | Discover and accept regime-specific features |
| Hyperparameter mis-tuning | Re-run parameter search with tighter constraints |
| Architecture limitation | Create a new experiment with updated architecture |

```
POST /api/admin/research/datasets         ← New dataset with better coverage
POST /api/admin/research/features         ← New regime features
POST /api/admin/research/searches         ← Refined parameter search
POST /api/admin/research/experiments      ← New experiment
```

---

## Step 5 — Validate the Fix

Before re-deploying, run the full validation stack against the new experiment:

```
1. POST /api/admin/research/backtests                              ← Verify backtest metrics
2. POST /api/admin/research/regime                                 ← Confirm regime coverage improved
3. POST /api/admin/research/experiments/:id/validate-retraining   ← Compare vs. rolled-back version
4. POST /api/admin/research/experiments/:id/gates                 ← Pass all 3 gates
5. POST /api/admin/research/experiments/:id/approve               ← Admin sign-off
```

---

## Step 6 — Resolve the Failure Record

After the fix is deployed and validated:

```
POST /api/admin/research/failures/:failureId/resolve
Authorization: Bearer <token>

{
  "resolution": "Root cause was training regime imbalance. Dataset expanded to include 2018-2019 high-volatility periods. New QDE v14 deployed with 6× more HIGH_VOLATILITY training samples. Regime stability improved from 0.41 to 0.79.",
  "resolver": "dr-chen"
}
```

---

## Rollback Audit Trail

All rollback records are maintained in the QDE system:

```
GET /api/qde/rollback/history
```

All failure records are maintained in the Research Lab:

```
GET /api/admin/research/failures?status=RESOLVED
GET /api/admin/research/failures/summary
```

The combination provides a complete post-mortem audit trail: what failed, why, what was fixed, and how long it took.

---

## Policy Rules

1. **Every rollback must have a corresponding failure record** — no rollback may go uninvestigated.
2. **CRITICAL failures must be investigated before re-deployment** — check `open_critical` in the failure summary.
3. **Root cause must be documented** — the `root_cause` field must be non-empty before a failure can be resolved.
4. **Re-deployment requires fresh gate evaluation** — a gate evaluation from before the failure may not be reused.
5. **Regression testing** — the new experiment must be validated against the same conditions that caused the failure (same regime, same period if possible).
6. **(S6) Research Lab is the sole authority for the rollback itself** — `POST /api/qde/rollback[/:targetVersionId]` will return `403` unless it can consume an APPROVED `LifecycleAuthorization` (action=`ROLLBACK`) issued by `POST /api/admin/research/rollbacks/:id/approve`. The rolled-back-from version may subsequently be archived (also Research-Lab-authorized) once it is no longer the rollback target for a future re-deployment.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/admin/research/failures` | `POST` | Record a failure event |
| `/api/admin/research/failures/:id/investigate` | `POST` | Record root cause |
| `/api/admin/research/failures/:id/resolve` | `POST` | Close the failure record |
| `/api/admin/research/failures` | `GET` | List failures (filterable) |
| `/api/admin/research/failures/summary` | `GET` | Summary with open criticals |
| `/api/qde/rollback/:targetVersionId` | `POST` | Execute rollback (QDE API; requires consumed Research Lab `ROLLBACK` authorization) |
| `/api/qde/rollback/history` | `GET` | Rollback audit trail (QDE API) |
| `/api/admin/research/rollbacks` | `POST` | Request a rollback (two-admin workflow) |
| `/api/admin/research/rollbacks/:id/approve` | `POST` | Approve rollback request — auto-issues `ROLLBACK` `LifecycleAuthorization` (S6) |
| `/api/admin/research/lifecycle/authorize` | `POST` | Authorize `ARCHIVE`/`RETIRE` for a version (S6) |
| `/api/admin/research/lifecycle/authorizations` | `GET` | List lifecycle authorizations, filterable by version/action/status (S6) |
| `/api/qde/versions/:versionId/archive` | `POST` | Archive a version (QDE API; requires consumed Research Lab `ARCHIVE` authorization) (S6) |
| `/api/qde/versions/:versionId/retire` | `POST` | Retire a version (QDE API; requires consumed Research Lab `RETIRE` authorization) (S6) |
