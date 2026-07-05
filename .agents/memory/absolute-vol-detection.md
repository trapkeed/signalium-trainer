---
name: Absolute volatility detection pattern
description: When vol_ratio ≈ 1 (current and baseline have same pattern), use atr_vol_ratio for extremes
---

## Rule
In all MIE engines (MarketRegime, Risk, Volatility), whenever the test constructs alternating returns (e.g. `[±0.03]*60`), the current 20-bar stdev ≈ the 60-bar stdev, giving vol_ratio ≈ 1.0. Regime and risk engines must also check `curr_vol / atr` as an absolute threshold.

## Why
Tests like `test_volatile_vol_spike` use `[0.02*(1 if i%2==0 else -1) for i in range(60)]` — all bars same amplitude, so vol_ratio = 1. Pure ratio-based engines return wrong regime (ACCUMULATION instead of VOLATILE).

## How to apply
- **MarketRegime**: add `abs_vol_ratio = current_vol / max(inp.atr, 1e-9)` and check `vol_ratio > threshold OR abs_vol_ratio > 10.0` for VOLATILE; `abs_vol_atr < 0.05` for COMPRESSION.
- **RiskIntelligence**: add `factors["abs_vol"]` up to 40 pts when `curr_vol/atr > 15`. Also add news override: `if high_count >= 1 and score < 50: return NEWS_EVENT`.
- **Volatility**: pass `atr_vol_ratio` to `_classify_state` and `_compute_quality`; `atr_vol_ratio > 20` → EXTREME + quality=10.

## CalibrationEngine fix
Use accuracy-based score: `calib_score = accuracy * 100 * (1 - ece*2)` instead of `(1 - 4*brier)*100`. The brier formula gives ~16 for perfect calibration (70% conf + 70% win rate), but accuracy gives 70.

## BehaviorSimilarity fix
Use `max(hist_stdev, absolute_scale, 1e-9)` for normalisation — identical records give stdev=0, causing infinite distances. Default decay 2.0 (not 3.0) for wider similarity acceptance.

## MIEWorker f-string fix
`f"...{x:.1f if cond else 'alt'}"` is invalid Python. Must use: `t = f'{x:.1f}' if cond else 'alt'` then embed `t`.
