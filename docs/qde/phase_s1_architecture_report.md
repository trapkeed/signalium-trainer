# Phase S1 Architecture Report — Safety Systems as Pure Infra

**Date:** 2026-07-08
**Related:** [`phase_s1_migration_report.md`](phase_s1_migration_report.md), [`phase_s1_dependency_report.md`](phase_s1_dependency_report.md), [`decision_engines_context_providers.md`](decision_engines_context_providers.md), [`unified_qde_architecture.md`](unified_qde_architecture.md).

## Architectural principle: "One Brain, guarded by pure infra"

The migration target is a single decision-making authority — QDE ("One Brain") — surrounded by safety systems that can **restrict capital-at-risk or halt execution**, but can never **choose direction, size, targets, confidence, or timing** on QDE's behalf. This report describes the boundary as it now stands after the Phase S1 audit and fixes, and the shape each system will need to plug into as QDE's output evolves in S2 (10 output heads) and S3 (live position-management loop).

```
                    ┌─────────────────────────────────────────┐
                    │              QDE ("One Brain")            │
                    │   sole decision maker: direction, size,    │
                    │   targets, confidence, timing               │
                    └───────────────┬─────────────────────────┘
                                     │  reads advisory context from ▼
        ┌───────────┬───────────────┼───────────────┬───────────────┐
        ▼           ▼               ▼               ▼               ▼
   Drawdown     Broker          Exposure         Shadow         Rollback
   Protection   Validation      Protection       Observation    (operator-
   (circuit     (order size     (conflict/       (offline       triggered
   breaker)     clipping)       correlation      comparison     version
                                advisory)         only)          swap)
```

## Boundary contract per system

| System | What it may return to QDE / the execution layer | What it must never do |
|---|---|---|
| Drawdown Protection | `risk_multiplier`, `can_trade`, `scaling_disabled`, `pyramiding_disabled`, `new_trades_paused` | Choose or veto a specific instrument/direction; only ever scales capital-at-risk down, never up, never targets |
| Broker Validation | Clipped/rounded order size | Alter direction, confidence, or expected return |
| Exposure Protection | `ConflictReport` with `instrument_risk_multipliers`, `portfolio_risk_multiplier`, warnings | Mutate `ModelForecast.confidence` / `expected_return` (this was the one violation found and fixed in S1) |
| Shadow Observation | Nothing consumed by the live path — logs only | Feed into an ensemble, vote, or canary blend with the live decision |
| Deployment Rollback | Swapped active version + audit record | Trigger itself based on live trade content; only operator/monitor-triggered |

## What changed structurally in S1

1. **Removed one hidden decision back-channel.** `conflict_resolver.py`'s in-place mutation of `ModelForecast` fields was the only place among the 5 systems where a "safety" system was silently functioning as a decision maker. It has been converted to advisory-only output (`instrument_risk_multipliers` / `portfolio_risk_multiplier`), matching the pattern already used by `exposure_engine.py`.
2. **Made the Drawdown circuit-breaker vocabulary explicit and inspectable.** Previously "not scaling into more risk during a drawdown" was an implicit expectation; now it is explicit, queryable state (`scaling_disabled`, `pyramiding_disabled`, `new_trades_paused`) with both automatic (mode-driven) and manual controls, and a guardrail test that fails if the class's public API ever grows a direction/target/confidence-mutating method.
3. **Confirmed 3 of 5 systems needed no change** (Broker Validation, Shadow Observation, and the previously-converted `exposure_engine.py` half of Exposure Protection) — they already conformed to the "context provider, not decision maker" pattern established in the prior Lifecycle/Portfolio/Risk conversion.
4. **Identified, but deliberately did not close, one forward-looking gap:** `QDEVersion` has no `feature_contract` / `output_schema` fields yet, so Rollback cannot yet validate I/O-contract compatibility across a version swap. This is intentionally sequenced *after* S2 defines what that contract actually looks like (the 10 output heads), to avoid building the validation twice.

## Forward-compatibility with S2 (10 output heads) and S3 (live position management)

None of the 5 safety systems' contracts depend on QDE's output being `BUY/SELL/NO_TRADE`. Each one's inbound contract is either:
- a scalar/boolean read (`can_trade`, `risk_multiplier`) — unaffected by output-head redesign, or
- an advisory multiplier/warning object consumed *before* QDE finalizes a decision — unaffected by whether QDE's decision is a single action or a 10-head vector, or
- entirely decoupled from decision content (Shadow logging, Rollback version swap).

This means Phase S2 (output-head redefinition) and Phase S3 (30-second live position-management loop) can proceed without modifying any of the 5 safety systems' external contracts, with the single documented exception of adding `feature_contract` / `output_schema` to `QDEVersion` once the S2 schema exists. That work item is the designated entry point connecting this report to the Phase S2 architecture work.

## Test evidence

- `pytest tests/unit/test_portfolio.py --noconftest -o addopts="" -q` → 132 passed (Exposure Protection fix).
- `python -m pytest tests/unit/test_icari.py --noconftest -o addopts="" -q` → 212 passed, 21 pre-existing failures unrelated to S1 (see migration report Appendix; confirmed via `git diff --stat` that the failing tests' modules were untouched by this phase).
