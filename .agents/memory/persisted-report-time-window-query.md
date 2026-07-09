---
name: Persisted-report time-window query completeness
description: A compute/generate endpoint accepting since/until doesn't mean the paired list/query endpoint does too
---

## Rule
When a monitoring/report module exposes both (a) a "compute and persist a report over telemetry in [since, until]" endpoint and (b) a separate "list persisted reports" endpoint, don't assume time-window filtering on (a) implies it exists on (b). They're independent code paths (compute filters the *input telemetry*, list filters the *stored reports*) and it's easy to wire since/until into one and forget the other.

**Why:** Found via code-review on a "queryable per model version and time window" requirement — `compute_report(since=, until=)` worked, but `list_reports()`/`list_regime_coverage_reports()` only filtered by `model_version`, silently failing the stated requirement despite the feature "looking" queryable.

**How to apply:** When a task requires findings to be "queryable by time window" (or similar), trace the full chain end-to-end (DB method → monitor/manager method → HTTP endpoint) for the *listing* path specifically, not just the compute/generate path. Add a direct test that seeds two records with different `created_at` and asserts `since`-only and `until`-only filters each return the right subset.
