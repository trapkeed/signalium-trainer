# MIE Feature Engineering Documentation
**Phase 24.9 | Deep Learning Integration**

---

## Purpose

The Feature Store Exporter produces a 12-dimensional normalised feature vector for every MIE run. These features are appended to the model's input matrix, giving the Deep Learning model direct awareness of current market conditions.

---

## Feature Vector (12 Dimensions)

All dimensions are normalised to [0, 1].

| Index | Feature | Source | Normalisation |
|-------|---------|--------|--------------|
| 0 | `market_regime_enc` | `MarketRegimeEngine` | ordinal / 15 |
| 1 | `risk_regime_enc` | `RiskIntelligenceEngine` | ordinal / 11 |
| 2 | `tradeability` | `TradeabilityEngine` | score / 100 |
| 3 | `execution_quality` | `MarketQualityEngine` | score / 100 |
| 4 | `market_quality` | `MarketQualityEngine` | score / 100 |
| 5 | `prediction_reliability` | `PredictionReliabilityEngine` | score / 100 |
| 6 | `behavior_similarity` | `BehaviorSimilarityEngine` | score / 100 |
| 7 | `calibration` | `CalibrationEngine` | score / 100 |
| 8 | `historical_success` | `BehaviorSimilarityEngine` | win_rate 0-1 |
| 9 | `trend_quality` | `TrendIntelligenceEngine` | score / 100 |
| 10 | `momentum_quality` | `MomentumIntelligenceEngine` | score / 100 |
| 11 | `liquidity_quality` | `LiquidityIntelligenceEngine` | score / 100 |

---

## Regime Encoding

`MarketRegime` ordinals (0-15):
```
TRENDING=0, STRONG_TREND=1, WEAK_TREND=2, RANGE=3, COMPRESSION=4,
EXPANSION=5, BREAKOUT=6, FALSE_BREAKOUT=7, REVERSAL=8, ACCUMULATION=9,
DISTRIBUTION=10, VOLATILE=11, ILLIQUID=12, NEWS_DRIVEN=13, BLACK_SWAN=14,
UNKNOWN=15
```

`RiskRegime` ordinals (0-11):
```
OPTIMAL=0, LOW_RISK=1, NORMAL=2, ELEVATED=3, HIGH_RISK=4, EXTREME=5,
FLASH_VOLATILITY=6, LOW_LIQUIDITY=7, MARKET_STRESS=8, NEWS_EVENT=9,
BLACK_SWAN=10, UNKNOWN=11
```

---

## Usage in Training Pipeline

```python
# At every prediction point:
api = MIEAPI()
report = api.recalculate(market_input)
feature_vec = report.features.to_feature_vector()  # 12-dim list[float]

# Prepend / append to model's feature matrix:
model_features = existing_features + feature_vec
```

---

## Expected Impact on Model Performance

MIE features provide the model with:
- **Regime context**: avoids wrong-regime predictions (e.g., trend-following in a range)
- **Risk awareness**: reduces position sizing during dangerous conditions
- **Calibration signal**: allows the model to self-correct for overconfidence
- **Liquidity constraint**: avoids illiquid conditions
- **Historical grounding**: similarity to past winning setups improves generalisation
