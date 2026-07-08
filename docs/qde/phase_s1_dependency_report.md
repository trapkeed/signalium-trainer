# Phase S1 Dependency Report — Safety Systems

**Date:** 2026-07-08
**Related:** [`phase_s1_migration_report.md`](phase_s1_migration_report.md), [`qde-research-lab-dependency-direction.md`](../../.agents/memory/qde-research-lab-dependency-direction.md) (one-way `trainer.qde → trainer.research_lab` import rule, still in force).

## Purpose

Map who calls into, and who is called by, each of the 5 safety systems, so the "One Brain" migration (S2/S3) knows exactly which call sites must be preserved, which are safe to leave untouched, and which direction data flows across the safety boundary.

## Dependency direction rule for all 5 systems

Every safety system in this report is a **leaf consumer of state, not a producer of decisions**: QDE (and, downstream of QDE, the execution layer) calls *into* these systems to read advisory context or to have order parameters clipped to hard limits; none of the five ever calls back into QDE, the strategy layer, or the model registry to change what QDE decided. This is the same "context provider, not decision maker" direction already established for Lifecycle/Portfolio/Risk in the prior conversion (`decision_engines_context_providers.md`).

## 1. Drawdown Protection

- **Module:** `trainer/icari/drawdown_manager.py`
- **Depended on by:** ICARI capital-allocation API (`trainer/icari/icari_api.py`), position sizing (indirectly, via `risk_multiplier` read by callers that scale order size), and any QDE-side pre-trade check that reads `can_trade()` / `scaling_disabled` / `pyramiding_disabled` / `new_trades_paused`.
- **Depends on:** Nothing outside its own module — it is a pure state machine over `(capital, peak, drawdown)`. No imports from `trainer.qde` or `trainer.execution`.
- **Direction:** One-way inbound (callers read state). No outbound calls into decision-making code.

## 2. Broker Validation

- **Modules:** `trainer/execution/execution_types.py` (`BrokerConstraints`), `trainer/execution/position_sizer.py`
- **Depended on by:** Order construction/execution path — any code that finalizes an order size before sending it to a broker adapter calls `BrokerConstraints.validate_lot`.
- **Depends on:** Static broker metadata (lot step, min/max lot, margin requirements) — no dependency on QDE, portfolio, or drawdown modules.
- **Direction:** One-way inbound (execution layer calls in to clip a size). Never reads or writes forecast/decision fields.

## 3. Exposure Protection

- **Modules:** `trainer/portfolio/exposure_engine.py`, `trainer/portfolio/conflict_resolver.py`
- **Depended on by:** Portfolio-level pre-trade advisory step; QDE (or an orchestrator upstream of QDE) reads `ConflictReport` (now including `instrument_risk_multipliers` / `portfolio_risk_multiplier`) as advisory input before finalizing a decision.
- **Depends on:** `ModelForecast` objects (read-only, as of the S1 fix — previously `conflict_resolver.py` mutated these, which was the violation corrected in this phase). No dependency on drawdown, broker, shadow, or rollback modules.
- **Direction:** One-way inbound after the S1 fix. Before the fix, this system had a hidden outbound write into forecast objects (confidence/expected_return), which functioned as an undocumented decision-making back-channel — that back-channel has been removed.

## 4. Shadow Observation

- **Module:** `trainer/qde/qde_shadow.py`
- **Depended on by:** Offline model-comparison tooling / reporting; not depended on by anything in the live trading path.
- **Depends on:** The active and shadow model's raw prediction output (read-only logging), and `trainer.research_lab` for any offline analysis storage, consistent with the existing one-way `trainer.qde → trainer.research_lab` rule.
- **Direction:** No outbound calls into the live decision path at all — this is the strictest isolation of the five, by design (a shadow model must never be able to influence the live decision, directly or via ensembling).

## 5. Deployment Rollback

- **Modules:** `trainer/qde/qde_rollback.py`, `trainer/qde/qde_models.py` (`QDEVersion`)
- **Depended on by:** Operator/monitoring tooling that triggers a rollback (human- or alert-driven, never trade-content-driven).
- **Depends on:** The model registry / version store (`QDEVersion` records) to read/write `active`/`stage`/audit fields. Does not depend on, or get triggered by, live forecast content.
- **Direction:** One-way inbound (operator calls in to swap versions and gets an audit record back). No outbound influence on in-flight trading decisions beyond the version swap itself.

## Cross-system dependencies (safety systems on each other)

None. All 5 systems are independent of one another — no system reads another system's internal state directly. Each is called separately by the layer above it (ICARI capital API, execution layer, portfolio advisory step, or operator tooling). This independence is intentional: it means Phase S2/S3 can change QDE's output shape or decision loop without needing to touch more than one safety system's call site at a time, and a bug or regression in one safety system cannot silently propagate into another.

## Migration implication for S2/S3

Because all 5 systems are pure leaf consumers with a single inbound-only direction, converting QDE's output to the new 10-head schema (S2) and then to a live position-management loop (S3) should only ever require **updating the call sites that read from these systems**, not the systems themselves — except for the one known gap noted in the migration report: `QDEVersion` will need `feature_contract` / `output_schema` fields added once the S2 schema exists, so Deployment Rollback can validate compatibility before swapping versions.
