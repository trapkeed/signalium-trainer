# Phase S2 Output Contract — Consumers of the 10-Head Signal

**Date:** 2026-07-08
**Related:** [`phase_s2_schema_contract.md`](phase_s2_schema_contract.md), [`qde_output_v1.proto`](qde_output_v1.proto).

## Purpose

Define exactly which downstream systems read which heads, and in what form, now that QDE's output is `QDEOutputHeads` (10 signals) rather than a single `BUY/SELL/NO_TRADE` classification.

## Consumers

| Consumer | Heads consumed | Form |
|---|---|---|
| `ExecutionPlanner` (`trainer/execution/execution_planner.py`) | None directly — consumes `QDEDecision`, which is derived from the heads by `derive_decision_fields()`. | `QDEDecision.entry_action` / `.exit_action` / `.scale_action` / `.position_size_lots` / `.risk_allocation_pct` / `.management_actions`. |
| Phase S1 safety systems (Drawdown, Broker, Exposure) | None — unchanged. They read `QDEDecision`-level fields (position size, direction) exactly as before; the heads are an implementation detail of how those fields were derived. | Same as before S2. |
| Shadow Observation (`qde_shadow.py`) | `direction`, `confidence` (via `QDEDecision`), plus optionally the full `heads.to_dict()` for richer offline champion/challenger comparison once wired up. | Logged for offline comparison only — never fed back into a live decision. |
| Memory / decision logging (`trainer/memory/`) | `heads.to_dict()` recommended as an additional logged field alongside the existing `QDEDecision.to_dict()`, so historical decisions retain the full signal vector for later analysis (not required for S2 to function, but recommended for S3's position-management loop, which will want historical head trajectories). | JSON blob. |
| Phase S3 live position-management loop (future) | `reduce_signal`, `scale_signal`, `management_confidence`, `attention_focus`, `exit_horizon`, `risk_signal` — these five heads are specifically designed to carry the signal S3's 30-second reasoning loop needs (see `docs/qde/phase_s2_deployment_contract.md` §Forward compatibility). | Direct read of `QDEOutputHeads`, not translated through `QDEDecision`. |

## What is intentionally NOT in the output

- No strategy name, strategy id, or strategy confidence-by-strategy breakdown. A single instrument/decision produces exactly one `QDEOutputHeads` instance — there is no per-strategy voting or blending surface to output.
- No raw price levels (entry/stop/take-profit) — those remain inputs to QDE (`QDEInferenceInput.entry_price_hint` etc.), computed by supporting execution-mechanics engines, not outputs of the decision itself.

## Versioning

`QDEOutputHeads.schema_version` (default `OUTPUT_SCHEMA_VERSION = "s2_v1_10head"`) travels with every output instance. Any future change to the number, meaning, or range of heads must bump this constant. `QDEVersion.output_schema_version` (added in this phase — see `phase_s2_deployment_contract.md`) records which schema version a given trained artifact targets, so Rollback can detect a mismatch before swapping versions.
