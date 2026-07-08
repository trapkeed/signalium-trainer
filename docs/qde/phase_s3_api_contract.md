# Phase S3 API Contract — `/qde/position-management/*`

**Date:** 2026-07-08
**Modules:** `trainer/qde/qde_server.py` (FastAPI upstream), `apps/trainer-api/src/routes/qde.ts` (Express proxy).
**Related:** [`phase_s3_inference_contract.md`](phase_s3_inference_contract.md), [`phase_s3_telemetry_contract.md`](phase_s3_telemetry_contract.md).

## Endpoints

Follows the existing proxy pattern in this codebase exactly: the Express route (`apps/trainer-api`) is a thin `fetch`-based proxy with no business logic, and the FastAPI route (`trainer/qde/qde_server.py`) does the actual work by delegating to a pure trainer module (`QDEPositionManager`) — the same shape as every other `/qde/*` endpoint (registry, pipeline, promotion, rollback, A/B).

### `POST /qde/position-management/review`

Runs one review cycle for one open position and returns a single management decision. Stateless — no registry, no database, nothing persisted server-side. The caller (the live trading loop) owns cadence, history, and persistence of decisions if it needs them.

**Request body:**

```json
{
  "instrument": "EURUSD",
  "position": { "lots": 1.0, "unrealised_pct": 0.012, "...": "any PositionMetrics field" },
  "market": { "news_risk": 0.1, "...": "any MarketContext field" },
  "heads": {
    "direction": "LONG",
    "confidence": 0.62,
    "reduce_signal": 0.1,
    "scale_signal": 0.0,
    "exit_horizon": 0.55,
    "management_confidence": 0.58,
    "risk_signal": 0.25,
    "...": "any remaining QDEOutputHeads field"
  },
  "free_margin_pct": 0.8,
  "seconds_since_last_review": 30.0
}
```

`position`, `market`, and `heads` are passed through as loose dicts (not a fixed field allowlist) and unpacked into `PositionMetrics(**position)`, `MarketContext(**market)`, and `QDEOutputHeads(instrument=..., direction=..., **heads)` respectively — any field omitted takes that dataclass's existing default (e.g. `market.news_risk` defaults to `0.0`, `heads.confidence` defaults to `0.0`). An unknown/misspelled field name raises `HTTP 400` (a `TypeError` from the dataclass constructor is caught and surfaced as a client error, not a 500) rather than being silently dropped.

**Response (200):**

```json
{
  "success": true,
  "decision": {
    "instrument": "EURUSD",
    "action": "TRAIL",
    "confidence": 0.62,
    "reasoning": "unrealised_pct=1.20% positive, management_confidence=0.58 — trailing the stop.",
    "management_confidence": 0.58,
    "risk_signal": 0.25,
    "model_version_id": "",
    "schema_version": "s2_v1_10head",
    "decided_at": 1783500000.123
  }
}
```

**Response (400):** invalid `direction` value or unknown field in `position`/`market`/`heads` → `{"detail": "<error message>"}` (standard FastAPI `HTTPException` shape, matching every other endpoint's error path in this server).

### `GET /qde/position-management/interval`

Returns the server's configured review cadence, so a caller doesn't need to hardcode `30` on its side:

```json
{ "success": true, "review_interval_seconds": 30.0 }
```

## Why no request/response models tied to the registry (`QDELifecycleAPI`)

Every other endpoint in `qde_server.py` delegates to `QDELifecycleAPI`, which persists state (registry entries, pipeline runs, rollback history) via `_respond()`, a helper that unwraps a `success`/`error` response object. `review_position_management()` does not use `_respond()` because `QDEPositionManager.review()` isn't a `QDELifecycleAPI` method — it is a pure function with no persisted state, so it returns a plain dict (`{"success": True, "decision": ...}`) built directly at the route rather than going through the registry's response-object convention. This keeps the "reasoning is pure/stateless" property established in `phase_s3_inference_contract.md` visible all the way out to the HTTP boundary — the route adds nothing beyond input validation and error translation.

## What still needs live wiring (out of scope for this contract)

This endpoint makes `QDEPositionManager.review()` callable over HTTP; it does not itself run a 30-second loop, fetch broker/market data, or persist decisions. Per `phase_s3_telemetry_contract.md`, that remains the responsibility of whatever service owns the live trading loop: poll each open position on the configured interval, assemble `PositionTelemetry` (calling the active `QDEDecisionModel` for fresh heads), POST to this endpoint (or call `QDEPositionManager` in-process if colocated), and route the resulting `action` to `TradeManager`/execution per the mapping table in `phase_s3_lifecycle_contract.md`.

## Test evidence

Smoke-tested via FastAPI `TestClient` against `trainer.qde.qde_server.app`: `POST /qde/position-management/review` returns `200` with a well-formed `TRAIL` decision for a representative in-profit/positive-management-confidence telemetry payload, and `GET /qde/position-management/interval` returns `{"success": true, "review_interval_seconds": 30.0}`. Underlying decision logic covered by `tests/unit/test_qde_position_manager.py` (13 tests, all passing).
