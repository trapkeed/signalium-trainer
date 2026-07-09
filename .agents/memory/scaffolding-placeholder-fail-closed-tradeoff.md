---
name: Scaffolding placeholder must stay unguarded from fail-closed checks
description: Hardening an execution/dispatch guard to "fail closed" can break a documented pre-training baseline path — verify what the empty/None state actually means before tightening
---

## Rule
When adding a "fail closed" identity/authorization check to a dispatch or execution path (e.g. "only the champion model may execute"), don't assume every occurrence of an empty/None/missing identity field is an attack or bug. Some codebases have a documented scaffolding placeholder (e.g. a heuristic baseline model used before any trained artifact is registered) that legitimately reports an empty version id and runs with no "champion" configured yet. Tightening the guard to reject that state breaks the default/bootstrap path, not a real vulnerability.

**Why:** Found via code review on a "champion-only execution" hardening task — an initial fix that failed closed on `active_version is None` or empty `model_version_id` broke 39 pre-existing tests, because `HeuristicQDEDecisionModel` (explicitly documented as the pre-training placeholder) has `version_id == ""` and runs before any champion is configured. The correct guard only fires on a genuine *mismatch* — a decision whose `model_version_id` is present, non-empty, AND differs from the active version — never on "field missing/empty."

**How to apply:** Before hardening any guard around "only X may act," (1) find what produces the empty/default state today and read its own docstring/comments — is it a documented bootstrap/placeholder path? (2) run the full relevant test suite after the change, not just the new test, before concluding the guard is correct. If tests that exercise the default/no-real-model path start failing, that's a signal you've over-scoped the check, not that the tests are wrong.
