---
name: Recommendation status is PROPOSED not PENDING
description: The retraining-recommendation lifecycle status vocabulary, to avoid silently-always-zero filters in new aggregators.
---

`RetrainingRecommendation` (trainer/research_lab/recommendation_models.py)
lifecycles through exactly three statuses: `PROPOSED` → `APPROVED` or
`DISMISSED`. There is no `PENDING` status anywhere in this model, even
though other lifecycle-ish tables in the same codebase (e.g. approval-gate
requests) do use `PENDING`.

**Why:** it's easy to assume vocabulary consistency across sibling tables
(pipeline jobs, approval requests, recommendations) and write
`list_recommendations(status="PENDING")` by analogy — this doesn't raise an
error, it just returns an empty list forever, so the bug only surfaces as
"pending recommendations always shows 0" in dashboards/summaries.

**How to apply:** any new code that wants "recommendations awaiting an
admin decision" must filter on `status="PROPOSED"`, not `"PENDING"`. Add a
test that inserts a `PROPOSED` recommendation and asserts a non-zero count,
so a regression back to "PENDING" fails loudly instead of silently.
