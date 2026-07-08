# Phase S3 Lifecycle Contract — Position Management Actions vs. `LifecycleState`/`ManagementAction`

**Date:** 2026-07-08
**Modules:** `trainer/qde/qde_position_manager.py`, `trainer/lifecycle/lifecycle_states.py`, `trainer/execution/execution_types.py`.
**Related:** [`phase_s3_inference_contract.md`](phase_s3_inference_contract.md), [`phase_s3_api_contract.md`](phase_s3_api_contract.md).

## Why a new `PositionManagementAction` enum instead of reusing an existing one

Two management-action vocabularies already exist in the codebase:

- `trainer.lifecycle.position_monitor.ManagementRecommendation` — `HOLD | SCALE_IN | SCALE_OUT | REDUCE_EXPOSURE | MOVE_STOP | MOVE_TARGET | EXIT_IMMEDIATELY | WAIT`, produced by the pre-existing, `LifecycleState`-driven `PositionMonitor`.
- `trainer.execution.execution_types.ManagementAction` — `MOVE_TO_BREAKEVEN | TRAIL_STOP | PARTIAL_CLOSE | SCALE_IN | SCALE_OUT | EXIT_EARLY | HOLD`, the execution-facing vocabulary `derive_decision_fields()` already emits into `QDEDecision.management_actions`.

Phase S3's spec calls for a *specific* seven-action vocabulary: `HOLD | REDUCE | ADD | CLOSE | MOVE_TO_BREAKEVEN | TRAIL | LOCK_PROFIT`. This doesn't line up 1:1 with either existing enum (`LOCK_PROFIT` and a plain `CLOSE` exist in neither; `ADD`/`REDUCE` are named differently from `SCALE_IN`/`SCALE_OUT`/`PARTIAL_CLOSE`), so introducing `PositionManagementAction` as its own enum in `qde_position_manager.py` was the correct call rather than overloading an existing type with values it wasn't designed for. Mapping table for any integration code that needs to bridge vocabularies:

| `PositionManagementAction` (S3) | Nearest `ManagementAction` (execution) | Nearest `ManagementRecommendation` (lifecycle) |
|---|---|---|
| `HOLD` | `HOLD` | `HOLD` |
| `REDUCE` | `PARTIAL_CLOSE` / `SCALE_OUT` | `REDUCE_EXPOSURE` / `SCALE_OUT` |
| `ADD` | `SCALE_IN` | `SCALE_IN` |
| `CLOSE` | `EXIT_EARLY` | `EXIT_IMMEDIATELY` |
| `MOVE_TO_BREAKEVEN` | `MOVE_TO_BREAKEVEN` | `MOVE_STOP` |
| `TRAIL` | `TRAIL_STOP` | `MOVE_STOP` |
| `LOCK_PROFIT` | *(no direct equivalent — new concept)* | *(no direct equivalent — new concept)* |

This mapping is documentation only — `QDEPositionManager` does not perform this translation itself. If/when the live execution loop needs to hand a `PositionManagementDecision` to `TradeManager` for concrete price-level translation (analogous to how `derive_decision_fields()`'s output feeds `ExecutionPlanner`), a small adapter function should live next to that integration point, following the same "one pure translation function" pattern Phase S2 established with `derive_decision_fields()` — not duplicated ad hoc at each call site.

## Relationship to `LifecycleState`

`PositionTelemetry`/`QDEPositionManager` deliberately do **not** take a `LifecycleState` as input. `PositionMonitor.evaluate()` branches primarily on lifecycle state (`ACTIVE`, `PROTECTED`, `RUNNER`, etc.) with signal/metrics as secondary adjustments. `QDEPositionManager.review()` inverts this: it branches primarily on the QDE signal heads (`reduce_signal`, `scale_signal`, `exit_horizon`, `management_confidence`) with position P&L as the secondary input, and does not consult lifecycle state at all. This is intentional — the Phase S3 goal is "QDE decides," and `LifecycleState` is bookkeeping about *where a trade is in its lifecycle* (an implementation-tracking concern), not a decision-making signal. A future integration may still use `LifecycleState` transitions (`TradeLifecycle`) to *record* the outcome of a `PositionManagementDecision` (e.g. a `CLOSE` decision triggering a lifecycle transition to a closed state) — that is a valid and expected use of `TradeLifecycle`, distinct from using it as a decision input.

## Coexistence, not replacement

`PositionMonitor` is not deleted or deprecated by this phase. Per the Phase S1 principle (advisory/observation systems never become decision-makers, but also are never silently removed if they still provide value), `PositionMonitor`'s lifecycle-state-driven recommendations can continue running as a cross-check/advisory layer alongside `QDEPositionManager`'s signal-driven decisions, exactly as Shadow Observation runs alongside the live QDE model. Any divergence between the two (e.g. `PositionMonitor` says `WAIT` while `QDEPositionManager` says `REDUCE`) is a monitoring signal for model-quality review, not a conflict that needs resolving in code.

## Test evidence

`tests/unit/test_qde_position_manager.py::test_all_seven_actions_are_reachable` confirms the enum matches the Phase S3 spec's seven named actions exactly; the other 12 tests in that file confirm each action is reachable from a distinct, realistic telemetry state.
