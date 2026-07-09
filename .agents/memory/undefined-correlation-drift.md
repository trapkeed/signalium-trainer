---
name: Undefined-correlation drift blind spot
description: When comparing a rolling statistical measure (e.g. correlation) across baseline vs recent windows for degradation, treat "measure became undefined" as a signal, not a no-op.
---

## The rule

When a drift/degradation check compares a statistic (correlation, calibration error, etc.) between a baseline window and a recent window, and that statistic can be legitimately undefined (e.g. Pearson correlation is undefined under zero variance), do not silently skip the comparison when only one side is defined.

- baseline defined, recent undefined → the signal *disappeared* — that is itself degradation (e.g. confidence flatlined and stopped tracking outcomes at all). Treat it as a full drop from the baseline value, and mark it explicitly in evidence (e.g. `..._became_undefined: true`) so alert consumers can distinguish "dropped" from "vanished."
- both baseline and recent undefined → no signal either way in either window; correctly skip that trigger.
- baseline undefined, recent defined → new information appeared; not degradation by itself, don't force a false trigger.

**Why:** In a QDE decision-quality drift check (S22B), the naive version only computed a drop when both baseline and recent correlations were non-`None`; if confidence became uninformative (constant) in the recent window, `corr_drop` silently defaulted to 0 and the check missed the exact failure mode it existed to catch. Caught by a code-review subagent, not by hand-written tests, until dedicated undefined/undefined and defined/undefined test cases were added.

**How to apply:** Any time you write `if a is not None and b is not None: compare(a, b)` for a rolling drift/degradation check, explicitly handle the `not None -> None` transition as a case, and add a test for it. Also add a test for the `None -> None` case to confirm it does NOT spuriously trigger.
