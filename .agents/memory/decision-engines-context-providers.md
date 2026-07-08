---
name: Decision engines converted to context providers
description: Risk/Portfolio/Lifecycle/Exposure engines are now pure info-only inputs to QDE, not autonomous gatekeepers
---

The Risk, Portfolio (allocator + optimizer + budget), Lifecycle (state
machine + trade lifecycle + transition engine), and Exposure engines were
converted from decision-makers/vetoers into pure context providers. QDE is
the sole decision maker over strategy-level actions.

**Why:** Previously these engines could silently override or reject a
strategy decision (forced budget rescaling, hard-filtered optimizer search
space, blocked lifecycle transitions on soft rule failures). That duplicated
decision authority and made it unclear who was actually deciding.

**How to apply:**
- Only two kinds of blocking remain legitimate:
  1. Structural/graph invariants (e.g. `StateMachine.can_transition` — an
     edge that doesn't exist in the FSM graph) — marked `hard_block=True` on
     `TransitionResult`.
  2. True hard safety rails (margin/exchange limits) — not part of this
     refactor, still enforced elsewhere.
- Everything else (confidence/news_risk/spread/trend/session rules, risk
  budget compliance, exposure alerts) is surfaced as advisory data
  (`to_dict()` fields, `suggested_capped_weights`, `budget_check`,
  `budget_warnings`) rather than auto-applied or vetoing.
- `PositionAllocator.allocate()` defaults to `apply_budget_cap=False` —
  returns the raw uncapped recommendation plus advisory budget context; pass
  `apply_budget_cap=True` for the old forced-rescale behavior.
- `PortfolioOptimizer.optimize()` no longer filters candidates by budget
  feasibility during search; `OptimizationResult.budget_check` reports it
  advisory-only.
- `TradeLifecycle.transition_to()` only blocks on `result.hard_block`, not on
  `result.allowed=False` from soft contextual rules.
