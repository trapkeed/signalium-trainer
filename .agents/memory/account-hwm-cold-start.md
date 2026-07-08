---
name: Account high-water-mark cold-start bug
description: Why a history-derived running peak (drawdown, HWM, baseline, etc.) must be reconciled against the source's own authoritative metric, not trusted alone when history is thin
---

When a feature/derivation depends on a rolling history table (e.g. equity
high-water-mark computed from `account_history`), a cold start (no rows yet,
or only the current tick) makes the "peak so far" trivially equal to the
current value — which silently fabricates a false-positive signal (e.g. "new
equity high") even when the underlying account is genuinely in drawdown.

**Why:** Discovered building the Phase S8 QDE account-risk pipeline
(`trainer/qde/qde_account_pipeline.py`): a fresh account with
`current_drawdown_pct=0.07` but empty history computed
`drawdown_from_peak_pct=0.0` from history alone, because the HWM list had
only one entry (itself). This flipped a should-be `NO_TRADE` into an
`AGGRESSIVE` recommendation.

**How to apply:** Whenever a derived rolling metric (HWM, baseline, moving
average, streak) is computed purely from an internal history table, check
whether the upstream domain object already carries its own authoritative
version of that same fact (e.g. a broker-reported `current_drawdown_pct`).
If so, reconcile — e.g. take the *more conservative* of the two views
(worse drawdown, not better) — rather than trusting the internally-derived
value in isolation. This generalizes beyond trading: any "first N
observations" cold start for a rolling stat is a place to double check
against a ground-truth field if one exists.
