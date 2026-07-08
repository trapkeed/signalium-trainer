# Live QDE Inference Integration

Extends `docs/qde/unified_qde_architecture.md`: this note describes the
concrete wiring that lets an active QDE model drive the live execution path
so `ExecutionPlanner` is a pure *executor* of one unified decision, not a
computer of independent entry/exit/scale/size/management sub-decisions.

This is the live-inference counterpart to the decision-engines-to-
context-providers conversion (`docs/qde/decision_engines_context_providers.md`):
that work made Risk/Exposure/Lifecycle/Portfolio engines pure information
providers instead of independent hard-blockers; this work makes the QDE the
one place their (and the execution sub-engines') output gets turned into an
actual decision. It also relates directly to
`docs/qde/hardcoded_behavior_audit.md` — every entry/exit/scale/size/
management rule that used to live inside `ExecutionPlanner`'s sub-engines
now flows through a single, auditable decision call instead of being
re-derived ad hoc in multiple places.

## What was added

- **Registry active-model lookup** — `QDERegistry.get_active_model()`
  (`trainer/qde/qde_registry.py`) is the single place execution code asks
  "what model decides right now?". It resolves the champion QDE version via
  `UnifiedModelRegistry` (new `get_active_native(family)` helper) and returns
  its registered `QDEDecisionModel`, falling back to
  `HeuristicQDEDecisionModel` when no trained artifact has been registered
  yet for that version. `QDERegistry.register_decision_model(version_id,
  model)` attaches a real model once one exists.

- **Five snapshot input types** (`trainer/qde/qde_inference.py`), aggregated
  into `QDEInferenceInput`:
  - `MarketContext` (reused) — price/vol/session context.
  - `PortfolioMetrics` (reused, `default_portfolio_snapshot()` for
    single-instrument calls) — portfolio-level risk/diversification state.
  - `PositionMetrics` (reused) — the existing open position, if any.
  - `QDEAccountSnapshot` (new) — balance/equity/margin, a hard-constraint
    surface per architecture §6, not a trading rule.
  - `QDEMemorySnapshot` (new, via `.from_records()`) — a compact summary of
    recent decision history for the instrument, built from
    `DecisionMemory.filter(...)`.

- **One unified decision output** — `QDEDecision`: entry/exit/scale action,
  position size, risk allocation, and the ordered list of trade-management
  actions, all decided together and carried a single `confidence` +
  `rationale`. This is the only object `ExecutionPlanner` executes.

- **`ExecutionPlanner.plan()` rewired** to:
  1. Run the supporting mechanics engines (zones, stop/tp, timing) —
     these remain, but purely as *context* for, and to *carry out*, the
     decision; they no longer gate execution independently.
  2. Build a `QDEInferenceInput` from the forecast plus optional
     `position` / `portfolio` / `account` / `decision_memory` arguments.
  3. Resolve the active model (`_resolve_decision_model()`) — either an
     injected `decision_model` (tests, or a caller-resolved model) or the
     registry's `get_active_model()` — and call `.decide()`.
  4. Execute the decision via `PositionSizer.from_decision()` and
     `TradeManager.plan_from_decision()`, which apply broker/risk validation
     (the one allowed hard safety rail) but do not re-derive size or
     management independently.
  `ExecutionPlan.is_executable()` now derives directly from
  `decision.is_entry` / `decision.scale_action`, and `ExecutionPlan.decision`
  exposes the full decision for downstream consumers.

## Import-cycle note

`trainer.qde` depends on `trainer.execution.execution_types`, and
`trainer.execution.execution_planner` depends on `trainer.qde` — a genuine
package cycle. It's broken by importing the QDE pieces **lazily** inside
`execution_planner.py` (constructor / method bodies, `TYPE_CHECKING` for
annotations) rather than at module scope, so either package can be imported
first without re-entering the other mid-initialization.

Separately, `trainer/registry/__init__.py` now exposes `ModelRegistry`
lazily (via module `__getattr__`) rather than importing it eagerly, since
`ModelRegistry` pulls in `BaseQuantModel` → `torch`. Without this,
importing `trainer.registry.unified_registry` (needed by the QDE registry,
no torch dependency of its own) would force the torch import chain to run
as a side effect of Python always executing a package's `__init__.py`
before handing back any of its submodules.

## Current limitation

`HeuristicQDEDecisionModel` is an interim, explicitly-documented baseline —
it exists so the "one model, one decision" path is invocable end-to-end
today. It is not a permanent trading-rule engine; see
`docs/qde/hardcoded_behavior_audit.md`. Replacing it with a trained artifact
only requires implementing `QDEDecisionModel.decide()` and calling
`QDERegistry.register_decision_model(version_id, model)` — no change to
`ExecutionPlanner` or its callers.

## Tests

- `tests/unit/test_qde_inference.py` — snapshot types, `QDEDecision`,
  `HeuristicQDEDecisionModel` decision paths (reject/wait/enter/exit/scale),
  `QDERegistry.get_active_model()` resolution and fallback,
  `UnifiedModelRegistry.get_active_native()`, and the
  `PositionSizer.from_decision()` / `TradeManager.plan_from_decision()`
  executors. Uses normal package imports (verified free of the torch
  chain).
- `tests/unit/test_execution.py::TestExecutionPlanner` — extended with
  decision-carrying `ExecutionPlan` assertions, custom `decision_model`
  injection, and `plan()` called with `position`/`portfolio`/`account`/
  `decision_memory`. Still uses `spec_from_file_location` per the file's
  existing torch-avoidance pattern.
