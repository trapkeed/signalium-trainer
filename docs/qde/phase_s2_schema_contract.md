# Phase S2 Schema Contract — QDE 10 Output Heads

**Date:** 2026-07-08
**Module:** `trainer/qde/qde_outputs.py`
**Schema version:** `s2_v1_10head` (`OUTPUT_SCHEMA_VERSION`)
**Related:** [`phase_s2_output_contract.md`](phase_s2_output_contract.md), [`qde_output_v1.proto`](qde_output_v1.proto), [`phase_s2_inference_contract.md`](phase_s2_inference_contract.md), [`phase_s2_deployment_contract.md`](phase_s2_deployment_contract.md).

## Objective

Replace QDE's BUY/SELL/NO_TRADE classification with 10 output heads. No head encodes a named trading strategy — every head is a general-purpose signal that any strategy-agnostic model can learn to produce, and no runtime consumer branches on a strategy name or id.

## The 10 heads

| # | Head | Type | Range | Meaning |
|---|---|---|---|---|
| 1 | `direction` | categorical (`Direction`: `LONG`/`SHORT`/`FLAT`) | — | Which way the instrument is expected to move. Replaces the old free-text `"BUY"/"SELL"/"NO_TRADE"` string with a closed enum. |
| 2 | `confidence` | float | `[0, 1]` | Overall conviction in `direction`. |
| 3 | `size_signal` | float | `[0, 1]` | How large a position to take, 0 = none, 1 = maximum allowed. |
| 4 | `risk_signal` | float | `[0, 1]` | How much risk/caution this call carries. Advisory — combined with, never overriding, the Phase S1 capital circuit-breakers (drawdown manager, broker constraints). |
| 5 | `entry_urgency` | float | `[0, 1]` | How urgently to act now vs. wait for confirmation. |
| 6 | `exit_horizon` | float | `[0, 1]` | Expected holding-duration signal, 0 = very short-term, 1 = very long-term. |
| 7 | `scale_signal` | float | `[0, 1]` | Propensity to add to an existing position. |
| 8 | `reduce_signal` | float | `[0, 1]` | Propensity to reduce/exit an existing position. |
| 9 | `management_confidence` | float | `[0, 1]` | Confidence in the ongoing position-management read — feeds Phase S3's live management loop. |
| 10 | `attention_focus` | float | `[0, 1]` | How much monitoring attention this instrument/position needs right now. |

Every head is independently interpretable and independently serializable — none require knowledge of any other head to have meaning, and none encode a bucketed/named strategy classification.

## Validation

`QDEOutputHeads.validate()` returns a list of human-readable errors (empty = valid) by checking each of the 9 float heads is within `[0.0, 1.0]`. `direction` is validated by the type system (it is an `Enum`, not a free string).

## Backward compatibility

`QDEDecision` (`trainer/qde/qde_inference.py`) — the execution-facing type `ExecutionPlanner` already consumes — gained an optional `heads: QDEOutputHeads | None` field. Existing fields (`entry_action`, `exit_action`, `scale_action`, `position_size_lots`, `risk_allocation_pct`, `management_actions`, `confidence`) are unchanged in shape, and are now *derived* from the 10 heads via `derive_decision_fields()` rather than computed ad hoc. `ExecutionPlanner` requires no changes.

## What did not need to change (strategy-name audit)

Grepped `trainer/qde/`, `trainer/execution/`, and `trainer/lifecycle/` for named-strategy branching (e.g. `"trend_following"`, `"mean_reversion"`, `strategy_id`, `StrategyType`). None found in QDE's runtime decision path — `ExecutionForecast.direction` already carries only the generic `"BUY"/"SELL"` strings (mapped to the closed `Direction` enum above), and `derive_decision_fields()` branches only on the 10 signal heads. The `FusionStrategy`/`BalancingStrategy`/`SearchStrategy` enums that do exist in the codebase are model-architecture and training-pipeline configuration (fusion layer choice, class-balancing method, hyperparameter search method) — not runtime trading-decision logic — and are out of scope for this phase.

## Test evidence

`tests/unit/test_qde_outputs.py` (7 tests), `tests/unit/test_qde_inference.py` (60, unchanged, still passing against the heads-based `HeuristicQDEDecisionModel`), `tests/unit/test_qde_rollback.py` (5 tests, new). Combined: 73 passed.
