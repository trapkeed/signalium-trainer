# Phase S1 Migration Report — Safety Systems as Pure Infra

**Date:** 2026-07-08
**Related:** [`decision_engines_context_providers.md`](decision_engines_context_providers.md) (prior conversion of Lifecycle/Portfolio/Risk engines to context providers — this report continues that pattern for the 5 systems explicitly designated as capital-protection infra ahead of the "One Brain" migration).

## Objective

Preserve five safety systems — **Drawdown Protection, Broker Validation, Exposure Protection, Shadow Observation, Deployment Rollback** — as pure safety infrastructure during the migration to a single QDE decision-making brain. None of the five may become, or continue acting as, a decision maker (no direction/size/target/hold-duration mutation, no vote, no veto over strategy content). Each system was audited against an explicit forbidden/allowed action list and, where violations were found, converted to advisory-only behavior.

## Forbidden / Allowed actions per system

| System | Forbidden | Allowed |
|---|---|---|
| Drawdown Protection | Setting direction, stop loss, take profit, targets, confidence, hold duration | Capital circuit-breaker actions: reducing risk multiplier, gating `can_trade`, disabling scaling/pyramiding, pausing new trades in lockdown |
| Broker Validation | Changing trade direction, confidence, expected return | Clipping/normalizing order size to broker constraints (lot step, min/max lot, margin) |
| Exposure Protection | Mutating a forecast's confidence or expected return, silently vetoing a trade | Surfacing advisory exposure/correlation warnings and suggested risk multipliers |
| Shadow Observation | Feeding shadow-model output into the live decision path (ensembling, voting, blending) | Logging shadow predictions for offline comparison against the active model |
| Deployment Rollback | Deciding *when* to roll back based on live trade content | Executing an operator/monitor-triggered version swap and recording the audit trail |

## Findings and changes by system

### 1. Drawdown Protection — `trainer/icari/drawdown_manager.py`
**Status before:** Compliant on paper (only touched `risk_multiplier` and `can_trade`), but had no explicit circuit-breaker vocabulary for scaling/pyramiding, so it was easy for future code to bolt a "disable pyramiding" side effect onto it in an undocumented way.
**Change:** Made the capital-circuit-breaker role explicit:
- Added `scaling_disabled` / `pyramiding_disabled` properties, `disable_scaling()` / `enable_scaling()` / `disable_pyramiding()` / `enable_pyramiding()` manual controls.
- Added `_AUTO_RESTRICT_MODES` (`HARD_LIMIT`, `PRESERVATION`, `EMERGENCY_LOCKDOWN`) that auto-disable scaling and pyramiding; manual overrides persist independently of mode and are not cleared by mode recovery.
- `update()` and `stats()` now report `scaling_disabled`, `pyramiding_disabled`, and `new_trades_paused` so QDE can read circuit-breaker state without reaching into private attributes.
- Docstring now states plainly: QDE remains the sole decision maker; this class only ever restricts capital-at-risk, never chooses what to trade.
**Test evidence:** `tests/unit/test_icari.py::TestDrawdownManager` — 13 new tests covering auto vs. manual disable, persistence across mode recovery, updated dict/stats shape, and a guardrail test asserting the public API exposes no direction/target/confidence mutation methods. Full file: 212 passed / 21 pre-existing failures unrelated to this change (see Appendix).

### 2. Broker Validation — `trainer/execution/execution_types.py` (`BrokerConstraints`), `trainer/execution/position_sizer.py`
**Status:** Compliant, no change required. `BrokerConstraints.validate_lot` only clips/rounds order size to exchange lot rules; lot-size adjustment is on the allowed list and never touches direction, confidence, or expected return.

### 3. Exposure Protection — `trainer/portfolio/exposure_engine.py`, `trainer/portfolio/conflict_resolver.py`
**Status before:** `exposure_engine.py` was already a pure monitor. `conflict_resolver.py` violated the spec — `_apply_reduce` rewrote `ModelForecast.confidence` and `expected_return` in place when it detected an instrument conflict, i.e. it was silently making a sizing/conviction decision on QDE's behalf.
**Change (user-selected Option C):** Kept conflict *detection* but removed all forecast mutation.
- `ConflictReport` gained two advisory fields: `instrument_risk_multipliers` and `portfolio_risk_multiplier` (defaulted via `__post_init__`).
- `adjusted_forecasts` is now always equal to `clean_forecasts` — the resolver never rewrites confidence or expected_return.
- Docstring and `to_dict()` updated to state the pure-monitor/advisor role explicitly.
**Test evidence:** `tests/unit/test_portfolio.py` — replaced `test_reduce_resolution_lowers_confidence` (which asserted the old mutation behavior) with 4 new tests asserting non-mutation and correct multiplier fields. `pytest tests/unit/test_portfolio.py --noconftest -o addopts="" -q` → 132 passed.

### 4. Shadow Observation — `trainer/qde/qde_shadow.py`
**Status:** Compliant, no change required. Confirmed no ensembling, voting, or canary-blend logic anywhere in the shadow path — shadow predictions are logged for offline comparison only and never touch the live decision.

### 5. Deployment Rollback — `trainer/qde/qde_rollback.py`, `trainer/qde/qde_models.py`
**Status:** Compliant for the *decision* boundary — rollback is always operator/monitor-triggered, never trade-content-triggered, and only swaps the active model/stage plus writes an audit record.
**Known gap (intentionally deferred to Phase S2):** `QDEVersion` does not yet carry `feature_contract` / `output_schema` fields, so a rollback cannot yet verify that the target version's I/O contract matches what the live pipeline expects. Phase S2 defines the real 10-output-head schema; adding the contract fields now would mean re-doing this work once that schema exists. Tracked so it is not silently lost — see [Appendix](#appendix-deferred-items).

## Appendix: deferred items

- **Rollback schema/contract validation** (`QDEVersion.feature_contract`, `QDEVersion.output_schema`) — deferred to Phase S2, to be implemented against the new 10-head output schema rather than the current BUY/SELL/NO_TRADE shape.
- **21 pre-existing test failures** in `tests/unit/test_icari.py` (`TestAllocationEngine`, part of `TestPositionSizingEngine`, `TestIcariAPI`) — `AllocationRequest` object has no attribute `ensemble_confidence`, raised from `trainer/icari/position_sizing_engine.py:146`. Verified via `git diff --stat` that none of the S1 changes touch `position_sizing_engine.py`, `icari_models.py`, or `allocation_engine.py` — this is a pre-existing bug unrelated to the S1 safety-systems work and out of scope for this phase.
