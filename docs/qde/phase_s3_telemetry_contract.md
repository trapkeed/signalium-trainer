# Phase S3 Telemetry Contract — `PositionTelemetry`

**Date:** 2026-07-08
**Module:** `trainer/qde/qde_position_manager.py`
**Related:** [`phase_s2_schema_contract.md`](phase_s2_schema_contract.md), `trainer/lifecycle/lifecycle_models.py`.

## Purpose

Phase S3 turns QDE into a live position manager: something must feed it a fresh read of "what does this open position look like right now" on every review cycle. `PositionTelemetry` is that input contract — the S3 analogue of `QDEInferenceInput` (the entry-decision input contract from Phase S2).

## Shape

```python
@dataclass
class PositionTelemetry:
    instrument: str
    position: PositionMetrics       # trainer.lifecycle.lifecycle_models — reused, not redefined
    market: MarketContext           # trainer.lifecycle.lifecycle_models — reused, not redefined
    heads: QDEOutputHeads           # trainer.qde.qde_outputs — this cycle's 10-head read
    free_margin_pct: float = 1.0
    seconds_since_last_review: float = 30.0
```

Design decisions:

- **Reuses `PositionMetrics`/`MarketContext` rather than inventing new position/market types.** These are already the live-position telemetry shapes used by `PositionMonitor` (`trainer/lifecycle/position_monitor.py`). S3 does not need a second, competing definition of "what a position looks like" — it needs a second *reasoner* over the same telemetry.
- **`heads` carries the model's read for this cycle**, produced the same way an entry-time `QDEOutputHeads` is produced (see `phase_s2_inference_contract.md`) — just re-run against the position's current state rather than a flat/no-position state. This is what makes S3's loop "QDE reasoning," not a second parallel decision system: it is the same model output type, sampled repeatedly against evolving telemetry instead of once at entry.
- **`free_margin_pct`** is carried explicitly because `ADD` decisions must never be considered when there is no room to add — this mirrors the `has_position`/`free_margin_pct` hard constraints already used by `derive_decision_fields()` in Phase S2.
- **`seconds_since_last_review`** exists so cadence enforcement (`due_for_review()`) is a property of the telemetry, not a side-channel timestamp lookup — callers computing telemetry already know when they last reviewed this instrument.

## Who produces `PositionTelemetry`

The calling service (the live trading loop, wherever it lives — e.g. a background worker in `apps/trainer-api` or the trainer's own live-execution harness) is responsible for:

1. Reading the current `PositionMetrics` (broker/account state) and `MarketContext` (live market data) for each open position.
2. Running the active `QDEDecisionModel` implementation against that state to obtain a fresh `QDEOutputHeads` (the model is not aware it's being used for management vs. entry — heads production is identical either way).
3. Constructing `PositionTelemetry` and passing it to `QDEPositionManager.review()`.

This module does not fetch broker data or subscribe to market feeds itself — consistent with the rest of `trainer/`, which stays a pure reasoning layer with no I/O.

## Test evidence

`tests/unit/test_qde_position_manager.py` (13 tests) exercises `PositionTelemetry` construction and `to_dict()` serialization across all branches of `QDEPositionManager.review()`.
