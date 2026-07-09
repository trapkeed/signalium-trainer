---
name: Recommendation→experiment bridge dedup
description: How to dedup an auto-created record linked to another record across that record's full lifecycle, not just its "open" statuses.
---

When one workflow step auto-creates a linked record in another lifecycle (e.g. approving a `RetrainingRecommendation` auto-creates a linked `ExperimentRecord`), the dedup check for "does a linked record already exist" must scan **all** statuses of the target lifecycle, not just the "still open" ones (e.g. `PROPOSED`/`PENDING`).

**Why:** A retried or re-invoked bridge call (crash recovery, double-click, admin re-approval attempt) happens at an arbitrary point in time. By then the previously-created linked record may have already progressed past the open statuses (e.g. to `RUNNING`, `COMPLETED`, `FAILED`). Checking only the open statuses misses it and creates a duplicate.

**How to apply:** Tag the created record with a stable dedup key referencing the source record's ID (e.g. `f"from_recommendation:{rec_id}"` in experiment tags). When checking for an existing link, iterate every enum value of the target status type, not a hardcoded subset. Add a test that advances the linked record through its full lifecycle and re-invokes the bridge to confirm no duplicate is created.
