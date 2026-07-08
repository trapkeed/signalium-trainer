# Decision Engines → Context Providers (Before / After)

**Date:** 2026-07-08
**Related:** [`hardcoded_behavior_audit.md`](hardcoded_behavior_audit.md) — Section 8 ("Fixed Confidence Thresholds & Lifecycle Gating") and Section 3b ("Allocation Weight Formulas & Portfolio Constraints") documented the hardcoded gates and formulas this change addresses at the *decision-authority* level (who acts on the values), as a companion to the audit's focus on *where the values come from*.

**Objective:** Risk, Portfolio (allocator/optimizer/budget), Lifecycle, and Exposure engines previously behaved as autonomous gatekeepers — they could silently veto, force-rescale, or auto-apply a strategy decision. This change converts them into pure information/context providers. QDE is now the sole decision maker for strategy-level actions. The only remaining blocking behavior is (a) structural/graph invariants that have no valid alternative, and (b) true hard safety rails (margin/exchange limits), which are unaffected by this change and enforced elsewhere.

## Lifecycle (`trainer/lifecycle/`)

| | Before | After |
|---|---|---|
| `StateMachine` / `TransitionResult` | A single `allowed: bool` — soft contextual rule failures (news_risk, spread, session, confidence, momentum) and structural graph invalidity were indistinguishable and both blocked the transition. | Added `hard_block: bool`, `True` only when the target state isn't a valid edge in the FSM graph. Soft rule failures now set `allowed=False, hard_block=False` and are surfaced as advisory reasons. |
| `TradeLifecycle.transition_to()` | Rejected the transition whenever `result.allowed` was `False`, regardless of reason. | Rejects only when `result.hard_block` is `True`. Soft-rule advisory context is logged but does not block the caller's chosen transition. |
| `LifecycleAPI.auto_transition()` | Always executed the `TransitionEngine`'s recommended transition directly (`tl.transition_to(rec.recommended, ...)`) — an autonomous engine-decision path. | Defaults to `apply=False`: returns the engine's recommendation as context only (equivalent to `recommend_transition`), performing no state change. `apply=True` remains as an explicit, documented legacy/manual escape hatch that callers must opt into; it is not used by any QDE-driven flow. |
| `TransitionEngine`, `LifecycleAPI.recommend_transition()` | Recommendation output already existed but engines elsewhere in the flow could still act unilaterally. | Docstrings clarified: these are advisory-only; the decision layer (QDE) must call `transition()` explicitly with its own choice. |

## Portfolio (`trainer/portfolio/`)

| | Before | After |
|---|---|---|
| `PositionAllocator.allocate()` | Always force-rescaled the final weights to fit the `RiskBudgetEngine`'s cap — the budget engine could silently override the requested allocation. | `apply_budget_cap` parameter, default `False`. Returns the raw/requested weights as `final`, plus advisory `budget_feasible`, `budget_warnings`, and `suggested_capped_weights` fields so the caller (QDE) can decide whether/how to respond to budget pressure. `apply_budget_cap=True` preserves the old forced-rescale behavior for callers that still want it. |
| `PortfolioOptimizer.optimize()` | Candidate weight combinations that failed the risk budget check were filtered out of the search space entirely — the optimizer could never even present a budget-infeasible-but-otherwise-optimal candidate. | No feasibility filtering during search. `OptimizationResult.budget_check` reports budget status as advisory context on the winning candidate; QDE decides what to do with a budget-infeasible optimum. |
| `RiskBudgetEngine.scale_to_budget()` | (Bug, found while implementing the above) Mutated the caller's input weights dict in place when the gross-exposure branch was skipped, corrupting the "raw" weights a caller might keep alongside the capped/advisory version. | Always operates on a copy of the input dict, matching its documented pure-utility contract. Required once callers began keeping both the raw and advisory-capped weights side by side. |

## Risk / Exposure (`trainer/execution/risk_engine.py`, `trainer/portfolio/exposure_engine.py`)

Both were already pure information providers (no state mutation, no veto authority over trade execution). Added `get_context()` aliases and docstring clarifications so their role as info-only inputs to QDE is explicit and discoverable, matching the naming convention used by the other converted engines.

## What did not change

- Structural FSM graph validity (`StateMachine.can_transition`) still hard-blocks — there is no valid alternative state to fall back to, so this isn't a strategy decision to override.
- True hard safety rails (margin/exchange limits) are out of scope for this change and remain enforced at their existing layer.

## Test evidence

`python -m unittest tests.unit.test_portfolio tests.unit.test_lifecycle` — 309 tests pass, covering the new advisory-only budget/optimizer contracts and the lifecycle non-veto (`hard_block`) behavior described above.
