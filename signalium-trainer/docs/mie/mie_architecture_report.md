# Signalium Market Intelligence Engine — Architecture Report
**Phase 24.9 | Classification: Enterprise Internal**

---

## 1. Executive Summary

The Market Intelligence Engine (MIE) is the central intelligence layer of the Signalium Quant platform. It sits between Market Memory and the Deep Learning model, providing real-time market condition assessment before any prediction is consumed by execution, portfolio, or signal-delivery systems.

Every AI prediction, every live analysis update, every API response, every EA response, and every Android signal passes through MIE.

---

## 2. System Architecture

```
Market Data
    ↓
Feature Store
    ↓
Market Memory
    ↓
Market Intelligence Engine (Phase 24.9)
├── Market Regime Engine          (16 regimes)
├── Risk Intelligence Engine      (12 risk regimes)
├── Volatility Intelligence Engine
├── Momentum Intelligence Engine
├── Trend Intelligence Engine
├── Liquidity Intelligence Engine
├── Session Intelligence Engine
├── News Intelligence Engine
├── Behavior Similarity Engine
├── Prediction Reliability Engine
├── Calibration Engine
├── Tradeability Engine           (0-100 score, 6 grades)
├── Execution Intelligence Engine
├── Market Quality Engine         (9 sub-scores)
├── MIE Portfolio Risk Engine
├── AI Commentary Generator
├── Feature Store Exporter        (12-dim feature vector)
├── MIE Database                  (SQLite, 10 tables)
├── MIE Dashboard                 (9 views)
├── MIE API                       (9 endpoints)
└── MIE Worker                    (6 trigger types)
    ↓
Deep Learning Model
    ↓
Execution Intelligence → Portfolio Intelligence → Signal Builder
    ↓
Android | API | EA | Website | Telegram
```

---

## 3. Module Inventory

| Module | Class | Purpose |
|--------|-------|---------|
| `mie_types.py` | Enums + helpers | 10 enums, type helpers, lookup tables |
| `mie_models.py` | Dataclasses | 18 dataclasses covering all I/O contracts |
| `market_regime_engine.py` | `MarketRegimeEngine` | 16-regime market classification |
| `risk_intelligence.py` | `RiskIntelligenceEngine` | 12-regime risk assessment |
| `volatility_intelligence.py` | `VolatilityIntelligenceEngine` | Vol state + quality score |
| `momentum_intelligence.py` | `MomentumIntelligenceEngine` | Momentum state + quality |
| `trend_intelligence.py` | `TrendIntelligenceEngine` | Trend state + quality |
| `liquidity_intelligence.py` | `LiquidityIntelligenceEngine` | Liquidity state + quality |
| `session_intelligence.py` | `SessionIntelligenceEngine` | Session classification + quality |
| `news_intelligence.py` | `NewsIntelligenceEngine` | News impact scoring |
| `behavior_similarity.py` | `BehaviorSimilarityEngine` | Historical pattern matching |
| `prediction_reliability.py` | `PredictionReliabilityEngine` | 4-factor reliability score |
| `calibration_engine.py` | `CalibrationEngine` | Brier score + ECE calibration |
| `tradeability_engine.py` | `TradeabilityEngine` | **Proprietary 15-factor score** |
| `execution_intelligence.py` | `ExecutionIntelligenceEngine` | Entry mode + zones + SL/TP |
| `market_quality_engine.py` | `MarketQualityEngine` | 9 sub-scores + overall quality |
| `portfolio_risk_engine.py` | `MIEPortfolioRiskEngine` | Portfolio risk constraints |
| `ai_commentary.py` | `AICommentaryGenerator` | Template-based AI commentary |
| `feature_store.py` | `FeatureStoreExporter` | 12-dim normalised feature vector |
| `mie_database.py` | `MIEDatabase` | SQLite, 10 tables, historical analytics |
| `mie_dashboard.py` | `MIEDashboard` | 9 dashboard view generators |
| `mie_api.py` | `MIEAPI` | Orchestrator + 9 endpoint handlers |
| `mie_worker.py` | `MIEWorker` | Background job queue processor |

---

## 4. Market Regime Classification (16 Regimes)

Decision hierarchy (top = highest priority):

1. **BLACK_SWAN** — vol_ratio > 6× or spread > 80% of ATR
2. **ILLIQUID** — volume < 10% of average or spread > 50% ATR
3. **NEWS_DRIVEN** — HIGH-impact news + vol spike > 1.5×
4. **VOLATILE** — vol_ratio > 2.5×
5. **REVERSAL** — volume spike + price direction opposite to prior trend
6. **BREAKOUT** — volume spike + price move > 2×ATR + ADX > 25
7. **FALSE_BREAKOUT** — volume spike + price move > 2×ATR + ADX < 25
8. **STRONG_TREND** — ADX ≥ 40
9. **EXPANSION** — ADX ≥ 25 + ATR expanding > 30%
10. **TRENDING** — ADX ≥ 25
11. **ACCUMULATION** — ADX 15-25 + higher highs + RSI < 60
12. **DISTRIBUTION** — ADX 15-25 + lower lows + RSI > 40
13. **WEAK_TREND** — ADX 15-25
14. **COMPRESSION** — vol_ratio < 0.5×
15. **RANGE** — ADX < 15
16. **UNKNOWN** — no clear signal

---

## 5. Risk Intelligence (12 Regimes)

| Regime | Multiplier | Trigger |
|--------|-----------|---------|
| OPTIMAL | 1.20 | risk_score < 10 |
| LOW_RISK | 1.10 | risk_score < 20 |
| NORMAL | 1.00 | risk_score < 35 |
| ELEVATED | 0.80 | risk_score < 50 |
| NEWS_EVENT | 0.70 | high news + score < 60 |
| HIGH_RISK | 0.60 | risk_score < 60 |
| LOW_LIQUIDITY | 0.40 | vol_frac < 0.2 |
| EXTREME | 0.30 | risk_score ≥ 60 |
| MARKET_STRESS | 0.25 | risk_score ≥ 60, no flash vol |
| FLASH_VOLATILITY | 0.20 | risk_score ≥ 60 + curr_vol/base > 2.5× |
| BLACK_SWAN | 0.10 | risk_score ≥ 75 + vol > 4× |
| UNKNOWN | 0.50 | fallback |

---

## 6. Tradeability Score — Proprietary Signalium Metric

**Tradeability ≠ Confidence.**

Tradeability measures how attractive the current market environment is for trading, regardless of prediction direction. A high-confidence prediction in an illiquid, news-driven, low-quality market should receive a LOW tradeability score.

### 15 Factors and Weights

| Factor | Weight | Description |
|--------|--------|-------------|
| prediction_confidence | 12% | Peak at 0.70 confidence |
| expected_rr | 10% | Expected risk-to-reward ratio |
| risk_regime | 10% | Risk multiplier → quality |
| market_regime | 10% | Regime base tradeability |
| liquidity | 8% | Liquidity quality score |
| execution_quality | 7% | Session + liquidity composite |
| expected_move | 6% | Expected move vs. ATR |
| historical_similarity | 6% | Similarity to past winning setups |
| calibration | 6% | Prediction calibration quality |
| spread | 5% | Spread cost as fraction of ATR |
| portfolio_exposure | 5% | Current exposure penalty |
| expected_holding | 5% | Ideal holding duration |
| news_calendar | 4% | News event proximity penalty |
| trend_stability | 4% | Trend quality contribution |
| momentum_stability | 2% | Momentum quality contribution |

### Grade Thresholds

| Score | Grade |
|-------|-------|
| ≥ 85 | A+ |
| ≥ 70 | A |
| ≥ 55 | B |
| ≥ 40 | C |
| ≥ 25 | D |
| < 25 | Avoid |

---

## 7. Market Quality Engine (9 Sub-Scores)

Each score: 0-100. Overall Market Quality is a weighted composite:

| Dimension | Weight | Source |
|-----------|--------|--------|
| Execution Quality | 15% | Spread + liquidity + session |
| Prediction Quality | 15% | PredictionReliabilityEngine |
| Liquidity Quality | 15% | LiquidityIntelligenceEngine |
| Volatility Quality | 15% | VolatilityIntelligenceEngine |
| Momentum Quality | 10% | MomentumIntelligenceEngine |
| Trend Quality | 10% | TrendIntelligenceEngine |
| Session Quality | 10% | SessionIntelligenceEngine |
| News Impact (inverse) | 10% | 100 - NewsIntelligenceEngine.impact_score |

---

## 8. Feature Store Export (12 Dimensions)

All features normalised 0-1 for the Deep Learning model:

```
[0]  market_regime_enc      (regime ordinal / 15)
[1]  risk_regime_enc        (regime ordinal / 11)
[2]  tradeability           (score / 100)
[3]  execution_quality      (score / 100)
[4]  market_quality         (score / 100)
[5]  prediction_reliability (score / 100)
[6]  behavior_similarity    (score / 100)
[7]  calibration            (score / 100)
[8]  historical_success     (win_rate 0-1)
[9]  trend_quality          (score / 100)
[10] momentum_quality       (score / 100)
[11] liquidity_quality      (score / 100)
```

---

## 9. Database Schema (10 Tables)

| Table | Primary Purpose |
|-------|----------------|
| `market_intelligence` | Master intelligence records per report |
| `market_regime_history` | Historical regime transitions |
| `risk_history` | Risk score and factor history |
| `tradeability_history` | Tradeability score time series |
| `market_quality` | 9 quality sub-score history |
| `execution_quality` | Execution recommendation history |
| `prediction_reliability` | Model reliability tracking |
| `behavior_similarity` | Similarity search results |
| `ai_commentary` | Generated commentary archive |
| `market_scores` | Denormalised composite scores for analytics |

All tables have `(symbol, timestamp)` compound indexes for efficient historical analytics.

---

## 10. API Endpoints

| Method | Path | Handler |
|--------|------|---------|
| GET | /api/v1/intelligence/current | `current()` |
| GET | /api/v1/intelligence/history | `history()` |
| GET | /api/v1/intelligence/tradeability | `tradeability_endpoint()` |
| GET | /api/v1/intelligence/risk | `risk_endpoint()` |
| GET | /api/v1/intelligence/regime | `regime_endpoint()` |
| GET | /api/v1/intelligence/execution | `execution_endpoint()` |
| GET | /api/v1/intelligence/quality | `quality_endpoint()` |
| GET | /api/v1/intelligence/commentary | `commentary_endpoint()` |
| POST | /api/v1/intelligence/recalculate | `recalculate()` |

---

## 11. Worker Triggers

| Trigger | When |
|---------|------|
| NEW_CANDLE | Every new OHLCV candle received |
| ECONOMIC_EVENT | On economic calendar event |
| NEW_PREDICTION | After model generates prediction |
| MODEL_UPDATE | After model retrain / update |
| REGIME_TRANSITION | When regime changes detected |
| MANUAL_REFRESH | User or system refresh request |

---

## 12. SOLID Design Principles

- **Single Responsibility**: Each engine handles exactly one intelligence dimension
- **Open/Closed**: All engines accept optional sub-results; extending with new engines does not break existing ones
- **Liskov Substitution**: All engines are independently replaceable — the API accepts Optional results
- **Interface Segregation**: `MarketInput` carries all required context; engines only read what they need
- **Dependency Inversion**: `MIEAPI` depends on abstract engine contracts, not concrete implementations

---

*Generated: Phase 24.9 | Signalium Quant Platform*
