# QDE Rollback Procedures

## Overview

The QDE Rollback system supports **immediate rollback** to any previously ACTIVE or RETIRED version. Rollback is an emergency operation designed to be executed in seconds, without requiring a new training pipeline run.

---

## Rollback Guarantees

| Property | Guarantee |
|---|---|
| **Speed** | Rollback is instantaneous — no retraining required |
| **Atomicity** | The from-version is marked ROLLED_BACK and the target becomes ACTIVE in a single operation |
| **Audit trail** | Every rollback is recorded with timestamp, actor, reason, and notes |
| **Eligibility** | Any RETIRED, DEPLOYMENT_CANDIDATE, SHADOW_TESTING, or ACTIVE version is a valid rollback target |
| **Irreversibility** | Rollback records are immutable once written |

---

## Rollback Trigger Conditions

Initiate a rollback when any of the following are observed after a new version is deployed:

- **Sharpe ratio degradation** — ACTIVE version Sharpe drops more than 20% within 24 hours
- **Drawdown breach** — Maximum drawdown exceeds the configured absolute limit
- **Decision quality collapse** — Entry/exit precision drops below the minimum threshold
- **Unexpected position sizing** — Position sizes deviate from expected capital allocation
- **System instability** — The version produces errors, timeouts, or crashes
- **Regulatory flag** — Compliance or risk team mandates immediate withdrawal

---

## Rollback Operations

### 1. Rollback to a Specific Version

Roll back to any eligible version by its `version_id`.

```
POST /api/qde/rollback/:targetVersionId
{
  "reason": "Sharpe ratio dropped 35% in first 6 hours",
  "initiated_by": "risk-team",
  "notes": "Emergency rollback — reverting to v2 pending investigation"
}
```

**What happens:**
1. The current ACTIVE version is marked `ROLLED_BACK`.
2. The target version's stage is updated to `ACTIVE`.
3. A `QDERollbackRecord` is written with full audit metadata.

### 2. Rollback to Previous Version

Automatically selects the most recent RETIRED version (highest version number below current).

```
POST /api/qde/rollback
{
  "reason": "Automated circuit breaker triggered",
  "initiated_by": "system"
}
```

**Use this when:** You need to rollback immediately and the specific target version does not matter — you just want the last known good version.

---

## Eligible Rollback Targets

Before rolling back, query the eligible targets to understand your options:

```
GET /api/qde/rollback/targets
```

Response lists all versions in `RETIRED`, `DEPLOYMENT_CANDIDATE`, `SHADOW_TESTING`, or `ACTIVE` stages, sorted by version number.

---

## Rollback Audit Trail

All rollback operations are logged permanently:

```
GET /api/qde/rollback/history
```

Each record contains:

| Field | Description |
|---|---|
| `rollback_id` | Unique identifier for this rollback |
| `from_version_id` | The version that was deactivated |
| `to_version_id` | The version that was restored |
| `reason` | Human-readable reason |
| `initiated_by` | Actor who triggered the rollback (user ID or "system") |
| `rolled_back_at` | Unix timestamp |
| `rolled_back_at_iso` | ISO-8601 UTC timestamp |
| `notes` | Optional additional context |

---

## Post-Rollback Actions

After a rollback:

1. **Investigate the failed version** — review its metrics, A/B test results, and stage timestamps.
2. **Do not re-promote without a root cause analysis** — document what went wrong before re-entering the pipeline.
3. **Consider a side-by-side test** — once a fix is ready, run a side-by-side evaluation against the current active version.
4. **Update promotion criteria** — if the rollback revealed a gap in the promotion rules, tighten the relevant dimension threshold.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/qde/rollback/:targetVersionId` | `POST` | Rollback to specific version |
| `/api/qde/rollback` | `POST` | Rollback to previous version |
| `/api/qde/rollback/history` | `GET` | Full rollback audit trail |
| `/api/qde/rollback/targets` | `GET` | List eligible rollback targets |
