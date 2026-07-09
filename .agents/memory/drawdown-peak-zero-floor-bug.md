---
name: Drawdown peak must not floor at zero
description: Pure-Python ports of numpy max-drawdown formulas must seed the running peak from the equity curve itself, never from a floored 0.0 baseline.
---

## The rule

The reference numpy formula is:

```python
equity = np.cumsum(pnls)
peak = np.maximum.accumulate(equity)   # cummax INCLUDING the current point
drawdowns = (peak - equity) / np.where(peak != 0, np.abs(peak), 1.0)
```

A naive pure-Python port that initializes `peak = 0.0` before the loop and does `peak = max(peak, equity)` diverges from this whenever the equity curve starts negative (a loss-first sequence, or an all-losses sequence). The numpy version's peak at index 0 is `equity[0]` itself (so drawdown there is always exactly 0); the naive port's peak stays floored at 0.0 until equity turns positive, which manufactures a phantom drawdown against a peak that was never actually reached.

Correct pure-Python form: seed `peak` from the first equity value (`peak = equity if peak is None else max(peak, equity)`), not from a literal `0.0`.

**Why:** Caught by code review while porting `trainer/validation/trading.py`'s numpy `TradingMetricsEngine` drawdown formula to a live, torch/numpy-free `performance_monitor.py` (S22B). The bug silently inflated `max_drawdown` (and anything derived from it, e.g. risk-adjusted return, drift-alert thresholds) specifically for losing streaks — exactly the case the health-monitor degradation check exists to catch correctly.

**How to apply:** Any time you port a numpy `cumsum` + `maximum.accumulate` drawdown-style formula to pure Python, write a unit test for a loss-first / all-negative pnl sequence and assert the first point has zero drawdown — that single assertion catches this whole class of bug.
