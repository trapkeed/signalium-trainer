---
name: Evaluator window overlap bug
description: Naive slice-based baseline/recent window splitting overlaps when the dataset is smaller than baseline_window + rolling_window, suppressing true threshold crossings.
---

## The bug

When computing rolling-vs-baseline metrics (Sharpe, confidence drift, reasoning drift), naively slicing:
```python
baseline = data[:baseline_window]   # e.g. first 200
recent   = data[-rolling_window:]   # e.g. last 50
```
works correctly only when `len(data) >= baseline_window + rolling_window`. With fewer points, the two slices overlap — often heavily — making `baseline ≈ recent` and suppressing real threshold crossings.

## The fix

Use the `_split_windows(data, baseline_window, rolling_window, min_samples)` helper in `trainer/research_lab/dimension_evaluators.py`. It:
- If `len ≥ baseline_window + rolling_window`: uses strict prefix/suffix (disjoint).
- Elif `len ≥ 2 * min_samples`: splits 67%/33% (oldest = baseline, newest = recent), guaranteed disjoint.
- Else: returns `(None, None)` → caller returns a not-triggered result.

## Why

The `min_trades_for_evaluation` guard was set much lower than `baseline_window + rolling_window`, so evaluators could fire with only a handful of points — all of which appeared in both windows.

## How to apply

Every new evaluator that compares a historical baseline against a recent rolling window must call `_split_windows()` instead of slicing directly. Unit tests should cover both the full-window case and the short-data 67/33 fallback.
