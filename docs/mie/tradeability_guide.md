# Tradeability Engine — Developer Guide
**Phase 24.9 | Signalium Proprietary Metric**

---

## Overview

The **Tradeability Score** is a proprietary Signalium metric that measures how attractive the current market environment is for trading.

**Tradeability is NOT confidence.** A prediction can have 90% confidence but a tradeability score of 5/100 (e.g., 2 minutes before a black-swan event, in an illiquid market, with a spread of 10× the ATR).

---

## Score Range and Grades

| Score | Grade | Recommended Action |
|-------|-------|--------------------|
| 85-100 | A+ | Full allocation — excellent conditions |
| 70-84 | A | Standard allocation |
| 55-69 | B | Reduce to 75% allocation |
| 40-54 | C | Reduce to 50% or wait |
| 25-39 | D | Minimal size only |
| 0-24 | Avoid | Do not enter |

---

## 15 Factors

### 1. Prediction Confidence (weight: 12%)
Peak quality at 0.70. Penalties applied below 0.30 (insufficient signal) and above 0.95 (overconfidence risk).

### 2. Expected Risk:Reward (10%)
- ≥ 3.0 → 100 pts
- ≥ 2.0 → 80-100 pts  
- ≥ 1.5 → 60-80 pts
- ≥ 1.0 → 30-60 pts
- < 1.0 → 0-30 pts

### 3. Risk Regime (10%)
Risk multiplier (0.10-1.20) mapped proportionally to 0-100.

### 4. Market Regime (10%)
Each of the 16 regimes has a pre-defined base tradeability. STRONG_TREND = 85, BLACK_SWAN = 5.

### 5. Liquidity Quality (8%)
Directly from the Liquidity Intelligence Engine.

### 6. Execution Quality (7%)
Composite of session quality + liquidity quality.

### 7. Expected Move (6%)
Expected move / ATR ratio — targets ≥ 1.5×.

### 8. Historical Similarity (6%)
From the Behavior Similarity Engine — how similar the current setup is to past winning trades.

### 9. Calibration Quality (6%)
From the Calibration Engine — how accurately the model assigns probabilities.

### 10. Spread Cost (5%)
Spread / ATR ratio. ≤ 2% = 100 pts, ≥ 60% = 0 pts.

### 11. Portfolio Exposure (5%)
≤ 20% exposure = 100 pts. Reduces linearly to 0 pts at 100% exposure.

### 12. Expected Holding Time (5%)
Ideal: 5-30 bars. Penalties for very short (< 3) or very long (> 100).

### 13. News Calendar (4%)
100 - news_impact_score. HIGH upcoming event = large penalty.

### 14. Trend Stability (4%)
From the Trend Intelligence Engine quality score.

### 15. Momentum Stability (2%)
From the Momentum Intelligence Engine quality score.

---

## Mathematical Formula

```
Tradeability = Σ(factor_score_i × weight_i)  for i in 1..15
             ∈ [0, 100]
```

where `Σ weights_i = 1.0` (enforced by assertion at module load).

---

## Integration

The Tradeability Engine requires results from other MIE engines to produce its most accurate score. Pass available results as optional arguments:

```python
t = engine.score(
    inp,
    regime=regime_result,
    risk=risk_result,
    liquidity=liquidity_result,
    ...
)
```

Missing optional results fall back to conservative defaults (50/100 for most factors).
