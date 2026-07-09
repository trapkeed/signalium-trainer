---
name: Feature health monitoring blind spots
description: Two ways a per-feature pipeline health check can look complete but miss real degradation
---

## Rule
A per-feature health check (missing/stale/corrupted/delayed) that only compares an *expected* feature set against the *observed* set, and only measures staleness relative to the window's own latest timestamp, has two blind spots:
1. **Intermittent missingness** — a feature that shows up in some but not most events is in the "observed" set, so expected-vs-observed set difference never flags it. Need a presence-rate-per-feature check (n_present / n_events_in_window) with its own threshold, independent of the expected/observed set-difference check.
2. **Whole-stream going dark** — per-feature staleness measured as `window_end - last_seen` is always ~0 for any feature seen in the window's most recent event, even if `window_end` itself is arbitrarily stale relative to wall-clock time. Needs an explicit wall-clock `now` passed in (not `time.time()` inside a pure function — thread it from the caller, same convention as other health_monitor.py dimension checks) and a stream-level staleness check separate from the per-feature one.

**Why:** Caught via code-review subagent loop on a "S22B feature health monitoring" task where the implementation looked complete (all four conditions named) but was silently inert for these two cases. Same review rigor previously caught a Sharpe-formula parity bug on a sibling task — implementations that look formula-complete still need adversarial checking against realistic data shapes (dominant-loss series, intermittent presence, stale windows), not just happy-path fixtures.

**How to apply:** When building any "detect X missing/stale" pipeline-health check, explicitly write a test where the condition is *partial* (not fully present, not fully absent) and a test where the check's own reference window is itself stale relative to wall-clock time — these are the two cases naive implementations skip.
