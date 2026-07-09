---
name: Execution-eligibility as a structural flag
description: Enforcing "only the champion/authorized decision can execute" requires two independent checks — a flag on the decision AND an identity re-check at the boundary.
---

When hardening an execution boundary so that only one specific
source's output (e.g. a champion model, as opposed to a shadow/challenger)
can ever be acted on, a single mechanism is not enough:

1. A flag set on the decision object by the producer that must stay
   observation-only (e.g. `is_shadow=True` set only by the shadow dispatcher)
   catches deliberate/known non-eligible paths, but a default of `False`
   means anything that never goes through that producer looks eligible by
   omission — including a miswired or corrupted pointer to the wrong model.
2. An independent identity re-check at the execution boundary itself
   (e.g. re-fetching "who is currently the active/champion version" and
   comparing it against the version that actually produced the decision,
   right before executing) catches the cases the flag can't.

**Why:** relying on only the flag leaves a gap where a non-eligible source
that never set the flag (e.g. because of a registry/pointer bug rather than
a deliberate shadow dispatch) can still pass through unflagged, defeating
the "structural, not conventional" guarantee the hardening was meant to
provide. This was caught by code review, not by initial test coverage.

**How to apply:** when asked to make an execution-eligibility guarantee
"enforced in code, not just by convention," implement both checks, and scope
the identity re-check narrowly (e.g. only when resolution goes through the
live registry, not when a caller/test intentionally injects a decision
source directly) to avoid false positives on legitimate direct-injection
paths.
