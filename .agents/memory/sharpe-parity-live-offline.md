---
name: Sharpe ratio live/offline parity pitfalls
description: Subtle ways a "matching" pure-Python Sharpe reimplementation silently diverges from TradingMetricsEngine
---

## Rule
When porting `TradingMetricsEngine`'s Sharpe formula to a pure-Python live-monitoring path, three details are each independently easy to get wrong and must all match exactly:
1. Normalization denominator is `max(abs(pnls.max()), 1.0)` — the absolute value of the single *largest* (most positive) pnl — not `max(abs(p) for p in pnls)` (largest-*magnitude* pnl). These only agree when the biggest winner outweighs the biggest loser; a loss-dominated or all-negative series diverges.
2. Std must be population std (ddof=0, matching `np.std()`'s default), not sample std (ddof=1).
3. Do not special-case `n < 2` to return 0.0 — the engine doesn't either; a single trade falls through to std=0 and an epsilon-stabilized (`+1e-9`) large value on both sides.

**Why:** All three bugs are "close enough" that naive tests (small mixed-sign fixtures, zero risk-free rate) pass anyway — Sharpe is scale-invariant at `risk_free_rate=0.0`, which also hides the denominator bug. Caught via a code-review subagent loop on Task #4 (S22B autonomous performance monitoring); took 3 review rounds to fully pin down.

**How to apply:** Any time you reimplement/port an offline metrics formula for a live/streaming path, write a direct parity test against the offline implementation using a non-zero risk-free-rate/non-default-annualization fixture (so the comparison is actually denominator/scale-sensitive), plus loss-dominated, all-negative, and single-point edge cases — not just a "happy path" mixed series.
