---
name: RiskBudgetEngine.scale_to_budget in-place mutation
description: scale_to_budget() could silently mutate a caller's weights dict when the gross-cap branch was skipped
---

`RiskBudgetEngine.scale_to_budget(weights, forecasts)` looped `for sym, w in
list(weights.items()): weights[sym] = w * scale` directly on the input dict.
When the initial gross-exposure check didn't trigger a rebind (`weights =
{...}`), that loop mutated the caller's original dict object in place instead
of a copy.

**Why:** Any caller that keeps a reference to the same dict object passed
into `scale_to_budget()` (e.g. to also use as its own "raw"/uncapped
recommendation) would see it corrupted after the call, even though the
function's contract implies a pure, non-mutating transform. This caused a
subtle test failure (weights not summing to 1.0) when `PositionAllocator`
was changed to treat the capped result as advisory-only context rather than
the enforced value.

**How to apply:** Any "compute an alternative/advisory version of X" utility
must defensively copy its input (`dict(weights)`) before any in-place
mutation, especially once callers start keeping the original alongside the
advisory alternative (a pattern introduced by the decision-engines-to-
context-providers refactor — see risk_budget.py, position_allocator.py,
optimizer.py).
