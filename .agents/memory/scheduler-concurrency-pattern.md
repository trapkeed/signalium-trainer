---
name: In-process scheduler concurrency pattern
description: Safe pattern for a lightweight in-process daily/weekly/monthly cycle scheduler — locking, thread lifecycle, and env parsing pitfalls.
---

When building a lightweight in-process scheduler (daemon thread polling "is this cycle due?" then running it), three pitfalls recur:

1. **TOCTOU between due-check and execution.** If `is_due(name)` and `run(name)` are separate calls without a shared lock, two callers (the background poller and a manual API trigger, or two racing poll ticks) can both see "due" and both execute — duplicating DB writes and side effects. Fix: re-check `is_due()` *inside* the same lock that guards execution (e.g. `_run_if_still_due()` acquires the lock, rechecks, then calls a `_locked` execute variant). Never call the locking public method recursively from inside itself — split into a public locking wrapper and a `_locked` body.

2. **Background thread stop/restart race.** `stop_background()` must not unconditionally clear the thread handle after `thread.join(timeout=...)` — if the join times out because the worker is still mid-cycle, clearing the handle lets `start_background()` spawn a second concurrent poller. Only clear the handle when `thread.is_alive()` is false after the join; otherwise log a warning and leave it running.

3. **Numeric env-var parsing must reject non-finite values.** `float(raw)` happily parses `"nan"`, `"inf"`, `"-inf"` as valid — a poll interval of `nan` effectively busy-loops, `inf` stalls forever. Guard with `math.isfinite(value) and value > 0`, falling back to a safe default with a logged warning on any invalid input (never raise from import-time config parsing).

**Why:** these scheduler/threading bugs are easy to introduce because unit tests using a single thread never exercise the race window, and `float()` "succeeding" on nan/inf looks like valid input. A code-review pass specifically targeting concurrency and env-parsing edge cases caught all three in one session (S24 Research Lab scheduler).

**How to apply:** any time you add a background poller or "auto-run recurring task" feature, write an explicit multi-thread concurrency test that races the runner against itself and asserts exactly one execution per due item, plus parser edge-case tests for none/empty/non-numeric/zero/negative/nan/inf.
