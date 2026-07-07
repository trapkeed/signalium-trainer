# QDE Version Lifecycle

## Overview

Each release of the Quant Decision Engine (QDE) is an immutable, versioned artifact that moves through a defined set of lifecycle stages. No version can skip a stage or move backwards through the pipeline — all progression is forward-only except through an explicit **Rollback** operation.

---

## Version Identification

| Field | Format | Example |
|---|---|---|
| `version_id` | `qde-v{N}-{uuid8}` | `qde-v3-a4f2c1d0` |
| `version_number` | Integer, monotonically increasing | `3` |
| `label` | Human-readable name | `QDE v3` |
| `parent_version_id` | ID of the version this was derived from | `qde-v2-b3e1a2c9` |

---

## Lifecycle Stages

```
REGISTERED
    │
    ▼
TRAINING  ◄──── Pipeline starts here
    │
    ▼
VALIDATION
    │
    ▼
BACKTESTING
    │
    ▼
SHADOW_TESTING
    │
    ▼
DEPLOYMENT_CANDIDATE
    │
    ├──► ACTIVE         (promoted)
    │
    └──► FAILED         (promotion denied / pipeline error)

ACTIVE
    │
    ├──► RETIRED        (replaced by a newer version)
    │
    └──► ROLLED_BACK    (emergency rollback)
```

### Stage Descriptions

| Stage | Description |
|---|---|
| `REGISTERED` | Version created and recorded in the registry. No training has started. |
| `TRAINING` | Historical data ingestion and model training in progress. |
| `VALIDATION` | Walk-forward validation across multiple market regimes. |
| `BACKTESTING` | Full historical backtest on out-of-sample data. |
| `SHADOW_TESTING` | Live shadow deployment — decisions logged but not executed. |
| `DEPLOYMENT_CANDIDATE` | Pipeline complete. Awaiting promotion evaluation. |
| `ACTIVE` | Current champion. All live decisions are routed to this version. |
| `RETIRED` | Superseded by a newer ACTIVE version. Eligible as rollback target. |
| `ROLLED_BACK` | Forcibly deactivated by a rollback operation. |
| `FAILED` | Pipeline or promotion failure. Can restart from REGISTERED. |

---

## Versioning Rules

1. **Monotonic version numbers** — version numbers are assigned sequentially and are never reused.
2. **Lineage tracking** — every version records its `parent_version_id`, creating a full ancestry chain accessible via the lineage API.
3. **Single ACTIVE version** — only one version may be ACTIVE at any time. Activating a new version automatically retires the previous one.
4. **Immutable metrics** — once metrics are recorded for a version, they are not overwritten; new metrics are merged in.
5. **Timestamp audit trail** — every stage transition is timestamped and stored in `stage_timestamps`.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/qde/versions` | `POST` | Register a new version |
| `/api/qde/versions` | `GET` | List all versions (filter by `?stage=`) |
| `/api/qde/versions/:id` | `GET` | Get a specific version |
| `/api/qde/versions/:id/lineage` | `GET` | Full ancestry chain |
| `/api/qde/active` | `GET` | Current active version |
| `/api/qde/summary` | `GET` | Registry statistics |
