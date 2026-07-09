---
name: Champion-challenger scoring design
description: Dimension baseline must be 0.5 (neutral), not 1.0; max_drawdown must handle all-negative pnl_diff series.
---

# Champion-Challenger Dimension Scoring Design Rules

## Baseline score must be neutral (0.5), not the champion's perfect score (1.0)

**Why:** If `champion_score = 1.0` is the reference point for any dimension (e.g. RISK_QUALITY using "diff from champion = 0 → perfect"), then `challenger_score` can never exceed `champion_score`, making `score_delta` always ≤ 0 and `challenger_wins` always False — even when the challenger is genuinely good.

**Rule:** Use `champion_score = 0.5` (neutral midpoint) for dimensions measured by comparison-to-reference. The challenger wins when its score exceeds 0.5, which means `challenger_wins ↔ score_delta > 0` is always consistent.

**Example (RISK_QUALITY):**
- Scale: `challenger_score = max(0, 1 - abs_diff / 0.05)` → 0% diff = 1.0, ≥5% = 0.0
- Baseline: `champion_score = 0.5`
- Wins when: `abs_diff < 0.025` (challenger score > 0.5)

## max_drawdown must handle all-negative pnl_diff series

**Why:** If the cumulative pnl_diff series never rises above 0 (all decisions were net-negative), the standard "peak-to-trough / peak" formula silently returns 0.0 drawdown because `peak == 0` and the guard `if peak > 0` prevents any computation.

**Fix:**
```python
if peak > 1e-9:
    dd = (peak - cum) / peak          # percentage from positive peak
else:
    dd = max(0.0, -cum)               # absolute drop below 0 (initial equity)
```

This ensures an all-negative pnl_diff series yields max_drawdown > 0.

## HOLD_QUALITY win criterion

Use an absolute score threshold (`challenger_score > 0.6`), NOT `challenger_cov < champion_cov AND score > 0.6`. When champion confidence variance is ~0 (stable champion), the relative condition is unwinnable even when the challenger is also perfectly consistent.
