---
name: Champion status vs terminal deployment-state ordering
description: When re-promoting or re-activating an entry, order state mutations so an entry is never CHAMPION and in a terminal deployment_state at the same instant, or the registry's consistency invariant raises.
---

When a version/model registry enforces an invariant like "no entry may be
CHAMPION while its deployment_state is terminal (RETIRED/ROLLED_BACK/FAILED/
ARCHIVED)", two distinct failure modes appear depending on mutation order:

1. **Demoting the old champion**: if you flip champion status to the new
   entry first, then move the *old* champion into a terminal state, that's
   safe — it's no longer champion when it becomes terminal. But if you set
   the old entry's terminal state *before* revoking its champion status, the
   invariant trips.

2. **Promoting from a terminal state**: re-activating/re-promoting an entry
   that is *currently* sitting in a terminal state (e.g. re-activating a
   previously RETIRED or ROLLED_BACK version, or re-promoting a previously
   ARCHIVED model) requires moving it OUT of the terminal state before
   granting champion status — not after. Granting champion status first,
   while the entry's stored deployment_state is still terminal, trips the
   invariant even though the caller's intent is clearly a valid promotion.

**Why:** this is easy to miss because normal "first activation" (entry starts
in a non-terminal stage) never exercises path 2 — it only surfaces once a
rollback/rollback-forward or re-promotion-after-archival cycle is tested.

**How to apply:** the reliable ordering for any promote/reactivate operation
is: (a) mutate the target entry's deployment_state to a non-terminal value
and persist, (b) grant champion status, (c) demote the previous champion to
a non-champion status, (d) THEN move the previous entry into its terminal
state and persist. As a defense in depth, also make the registry's generic
`update()` auto-demote champion status when a caller pushes an entry into a
terminal deployment_state directly (rather than raising), since business
logic call sites won't always go through a dedicated promotion helper.
