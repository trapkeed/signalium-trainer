# QDE Architectural Violations Report

**Scope:** Full audit of trainer/ codebase against the QDE design contract  
**Files audited:** 427 Python files across trainer/  
**Date:** 2026-07-08  
**Standard:** One model. One decision maker. One intelligence. No sub-engines. No rule systems. No hardcoded strategies, exits, risk, or market regimes. The QDE learns these behaviours.

---

## Summary

| Category | Violation Count | Severity |
|---|---|---|
| V-01 Multiple decision makers | 1 structural | Critical |
| V-02 Sub-engine proliferation | 40+ engine classes | Critical |
| V-03 Hardcoded market regimes | 18 thresholds | Critical |
| V-04 Hardcoded exits | 10 constants | Critical |
| V-05 Hardcoded risk parameters | 20+ constants | Critical |
| V-06 Hardcoded order strategy rules | 8 rule branches | High |
| V-07 Hardcoded promotion gates | 5 constants | High |
| V-08 Hardcoded regime routing | 3 policies | High |
| V-09 Parallel sizing methods | 10 competing formulas | High |
| V-10 Hardcoded momentum classification | 7 thresholds | Medium |
| V-11 Hardcoded trend classification | 8 thresholds | Medium |
| V-12 Hardcoded statistical constants | 2 constants | Medium |

**Total violations: 12 categories. Every core principle is breached.**

---

## V-01 — Multiple Decision Makers

**Contract:** One decision maker.

**Violation:** The system has N independent decision makers running in parallel.

`trainer/ensemble/engine.py` — `EnsembleEngine` maintains a registry of `ExpertModel` instances. On every prediction call it:

1. Routes a subset of experts based on regime/instrument/timeframe
2. Runs each expert's `predict()` independently
3. Combines their outputs via one of 8 combination strategies
4. Optionally overrides with a meta-model

```python
# engine.py:280–288
candidates = self.registry.active_experts()
expert_predictions = [e.predict({**features, **ctx}) for e in experts]
ensemble_result = self.strategy.combine(expert_predictions, routing.weights)
meta_prediction = self.meta_model.predict(expert_predictions, routing.weights, ctx)
final: Any = meta_prediction if self.meta_model.is_trained else ensemble_result
```

`trainer/ensemble/expert.py` line 18 makes the multiple-model contract explicit:

```
"The existing Signalium Quant model is Expert 1 and is never removed."
```

The QDE version lifecycle (`qde_ab_testing.py`) simultaneously operates a CONTROL version and a CHALLENGER version in SHADOW and CANARY modes. Two decision makers are live at the same time.

**Required:** One model produces one output. No voting. No ensemble. No meta-model override.

---

## V-02 — Sub-Engine Proliferation

**Contract:** No sub-engines. One intelligence.

**Violation:** 40+ named engine/intelligence classes across 5 subsystems. Each is an independent decision system with its own logic, thresholds, and output.

### MIE — 13 engines (trainer/mie/)

| Class | File | Responsibility |
|---|---|---|
| `MarketRegimeEngine` | market_regime_engine.py | Classifies 16 market states |
| `MomentumIntelligenceEngine` | momentum_intelligence.py | Momentum state |
| `TrendIntelligenceEngine` | trend_intelligence.py | Trend state |
| `VolatilityIntelligenceEngine` | volatility_intelligence.py | Volatility analysis |
| `LiquidityIntelligenceEngine` | liquidity_intelligence.py | Liquidity analysis |
| `RiskIntelligenceEngine` | risk_intelligence.py | Risk intelligence |
| `SessionIntelligenceEngine` | session_intelligence.py | Trading session |
| `NewsIntelligenceEngine` | news_intelligence.py | News events |
| `ExecutionIntelligenceEngine` | execution_intelligence.py | Execution quality |
| `MarketQualityEngine` | market_quality_engine.py | Market quality |
| `TradeabilityEngine` | tradeability_engine.py | Tradeability scoring |
| `CalibrationEngine` | calibration_engine.py | Signal calibration |
| `BehaviorSimilarityEngine` | behavior_similarity.py | Behavior similarity |

### Execution — 7 engines (trainer/execution/)

| Class | File | Responsibility |
|---|---|---|
| `StopLossEngine` | stop_loss_engine.py | Recommends 6 stop levels |
| `TakeProfitEngine` | take_profit_engine.py | Recommends 5 TP levels |
| `RiskEngine` | risk_engine.py | Computes risk profile |
| `OrderStrategyEngine` | order_strategy.py | Selects order type |
| `EntryZoneEngine` | entry_zones.py | Detects entry zones |
| `EntryQualityEngine` | entry_quality.py | Scores entry quality |
| `PositionSizer` | position_sizer.py | Position sizing wrapper |

### ICARI — 8 engines (trainer/icari/)

| Class | File | Responsibility |
|---|---|---|
| `PositionSizingEngine` | position_sizing_engine.py | 10 sizing methods |
| `DynamicSizer` | dynamic_sizer.py | 10-factor adjustment |
| `AllocationEngine` | allocation_engine.py | Portfolio allocation |
| `CorrelationEngine` | correlation_engine.py | Correlation analysis |
| `PortfolioRiskEngine` | portfolio_risk_engine.py | Portfolio-level risk |
| `RiskBudgetEngine` | risk_budget_engine.py | Risk budget management |
| `ScenarioEngine` | scenario_engine.py | Scenario analysis |
| `CapitalHealthEngine` | capital_health.py | Capital health |

### Ensemble — 12+ classes (trainer/ensemble/)

`EnsembleEngine`, `ExpertRouter`, `DynamicWeightEngine`, `EnsembleMetaModel`, `ExpertCalibrator`, `ExpertFailureAnalyzer`, `OnlineScorer`, `EnsembleLeaderboard` plus 8 combination strategy classes.

**Required:** One class produces the decision. Market context, risk, sizing, and exits are outputs of what the model has learned, not of separate rule engines.

---

## V-03 — Hardcoded Market Regimes

**Contract:** No hardcoded market regimes. The QDE learns these.

**Violation:** 16 regime labels are enumerated up-front. All boundaries are compile-time constants.

### trainer/mie/market_regime_engine.py

```python
# Lines 79–91 — constructor defaults
adx_strong: float = 40.0     # ADX ≥ 40  → STRONG_TREND
adx_trending: float = 25.0   # ADX ≥ 25  → TRENDING / EXPANSION
adx_weak: float = 15.0       # ADX ≥ 15  → WEAK_TREND / ACCUMULATION / DISTRIBUTION
vol_extreme: float = 4.0     # vol_ratio ≥ 4.0 → BLACK_SWAN
vol_high: float = 2.5        # vol_ratio ≥ 2.5 → VOLATILE
vol_compress: float = 0.5    # vol_ratio < 0.5 → COMPRESSION
```

```python
# Line 55 — hardcoded volume spike multiplier
def _volume_spike(volumes, threshold: float = 2.5) -> bool: ...

# Line 140 — hardcoded illiquid volume threshold
if avg_vol > 0 and (inp.volume < avg_vol * 0.10 or spread > atr * 0.5):
    return MarketRegime.ILLIQUID, ...

# Line 150 — hardcoded absolute vol threshold
if vol_ratio > self._vol_high or abs_vol_ratio > 10.0:
    return MarketRegime.VOLATILE, ...

# Line 169 — hardcoded RSI direction split
direction = "UP" if rsi > 55 else "DOWN"

# Line 182 — hardcoded ATR expansion threshold
if recent_range > older_range * 1.3:
    return MarketRegime.EXPANSION, ...

# Lines 190–193 — hardcoded accumulation/distribution RSI bounds
if _has_higher_highs(highs) and rsi < 60:
    return MarketRegime.ACCUMULATION, ...
if _has_lower_lows(lows) and rsi > 40:
    return MarketRegime.DISTRIBUTION, ...

# Line 198 — hardcoded compression absolute threshold
if vol_ratio < self._vol_compress or abs_vol_atr < 0.05:
    return MarketRegime.COMPRESSION, ...
```

### trainer/mie/trend_intelligence.py

```python
# Lines 93–109 — hardcoded RSI direction boundaries
if rsi < 40:   return "DOWN"
if rsi > 60:   return "UP"
if rsi > 55:   return "UP"
if rsi < 45:   return "DOWN"

# Lines 101–104 — hardcoded EMA gap threshold
if ema_f > ema_s * 1.001:  return "UP"
if ema_f < ema_s * 0.999:  return "DOWN"
```

The 16 regime labels themselves (`STRONG_TREND`, `TRENDING`, `EXPANSION`, `WEAK_TREND`, `ACCUMULATION`, `DISTRIBUTION`, `COMPRESSION`, `RANGE`, `VOLATILE`, `BREAKOUT`, `FALSE_BREAKOUT`, `REVERSAL`, `NEWS_DRIVEN`, `BLACK_SWAN`, `ILLIQUID`, `UNKNOWN`) are hardcoded in `trainer/mie/mie_types.py`. The taxonomy is fixed. The model cannot discover a regime the designer did not name.

**Required:** The model learns what conditions precede good/bad trades. No ADX thresholds. No RSI splits. No vol_ratio buckets. No regime enum.

---

## V-04 — Hardcoded Exit Conditions

**Contract:** No hardcoded exits. The QDE learns when to exit.

**Violation:** All exit levels are ATR multiplier formulas with compile-time constants.

### trainer/execution/stop_loss_engine.py

```python
# Lines 88–92 — five hardcoded ATR multipliers
_INITIAL_MULT    = 1.50   # stop at entry ± 1.5× ATR
_ADAPTIVE_MULT   = 1.20   # stop at entry ± 1.2× ATR
_ATR_MULT        = 2.00   # stop at entry ± 2.0× ATR
_VOLATILITY_MULT = 2.50   # stop at entry ± 2.5× ATR (scaled by vol)
_EMERGENCY_MULT  = 3.50   # stop at entry ± 3.5× ATR
```

```python
# Line 111 — volatility scaling constant
volatility_stop = stop(self._VOLATILITY_MULT * (1 + forecast.volatility * 5))
#                                                                           ^ hardcoded
```

The system always produces exactly 6 stops. The labels (`initial`, `adaptive`, `atr`, `structure`, `volatility`, `emergency`) are pre-defined. Exit geometry is a function of human judgment about ATR multipliers, not learned trade outcomes.

### trainer/execution/take_profit_engine.py

```python
# Lines 115–117 — hardcoded risk:reward ratios
tp1 = entry + sign * 1.0 * stop_dist   # 1:1 RR
tp2 = entry + sign * 1.5 * stop_dist   # 1.5:1 RR
tp3 = entry + sign * 2.0 * stop_dist   # 2:1 RR
```

```python
# Lines 43–46 — hardcoded RR as dataclass defaults
rr_tp1: float = 1.0
rr_tp2: float = 1.5
rr_tp3: float = 2.0
rr_runner: float = 3.0
```

```python
# Lines 131–135 — hardcoded probability decay
decay = 0.70
p_tp1 = base * 0.90
p_tp2 = base * decay        # base × 0.70
p_tp3 = base * decay**2     # base × 0.49
p_run  = base * decay**3    # base × 0.34
```

The docstring at line 86–90 states the entire exit philosophy as a fixed formula rather than as learned behaviour:

```
TP1:    entry + 1.0 × stop_distance (1:1 RR)
TP2:    entry + 1.5 × stop_distance (1.5:1 RR)
TP3:    entry + 2.0 × stop_distance (2:1 RR)
```

**Required:** Exit timing and level come from the model's learned understanding of how trades unfold. 1:1 RR, 1.5:1 RR, 2:1 RR are not intrinsic truths — they are the designer's prior.

---

## V-05 — Hardcoded Risk Parameters

**Contract:** No hardcoded risk. The QDE learns appropriate risk.

**Violation:** Risk is computed by a rule system of thresholds and multipliers.

### trainer/icari/position_sizing_engine.py

```python
# Lines 23–24 — hardcoded global risk limits
_MIN_UNITS    = 0.001
_MAX_RISK_FRAC = 0.05    # never risk more than 5% per trade

# Lines 33–35 — hardcoded defaults
risk_frac  = 0.01        # default 1% risk
kelly_frac = 0.5         # fractional kelly multiplier
atr_mult   = 2.0         # stop = atr * 2.0

# Line 116 — arbitrary volatility scaling formula
vol_mult = max(0.25, 1.0 - req.volatility * 10)

# Line 131 — arbitrary EV scaling formula
ev_mult  = max(0.1,  min(2.0, 1.0 + req.expected_value * 50))
```

10 independent sizing formulas each produce a different answer. None of these formulas was learned from outcomes — they are mathematical identities (Kelly Criterion, fixed fractional, etc.) applied with hardcoded coefficients.

### trainer/icari/dynamic_sizer.py

```python
# Lines 34–36 — hardcoded drawdown limits
drawdown_soft_limit = 0.05   # 5% → begin reducing
drawdown_hard_limit = 0.10   # 10% → emergency 0.25× position

# Lines 43–49 — hardcoded session multipliers
"LONDON":   1.00,
"NEW_YORK": 1.00,
"TOKYO":    0.80,   # 20% size reduction in Tokyo
"SYDNEY":   0.70,   # 30% size reduction in Sydney
"OVERLAP":  1.10,

# Lines 151–158 — hardcoded exposure reduction tiers
if exposure >= 0.90: return 0.25
if exposure >= 0.70: return 0.60
if exposure >= 0.50: return 0.85
return 1.0

# Lines 163–165 — hardcoded volatility reduction tiers
if vol >= 0.05: return 0.40
if vol >= 0.02: return clamp(1.0 - (vol - 0.01) * 15, 0.40, 1.0)

# Lines 169–172 — hardcoded correlation reduction
if corr >= self._max_corr: return 0.50
if corr >= 0.50:           return 0.75

# Line 21 — hardcoded adjustment bounds
_ADJUSTMENT_BOUNDS = (0.10, 2.0)
```

The composite multiplier is the geometric mean of 10 independently-designed adjustment factors — each expressing a human belief about how a single variable should affect size. The model's own view of risk is never consulted.

**Required:** Risk per trade is a function of what the model has learned from its own prediction accuracy on this instrument in this regime, not a formula designed by a human.

---

## V-06 — Hardcoded Order Strategy (Rule System)

**Contract:** No hardcoded strategies. The QDE learns these behaviours.

**Violation:** `trainer/execution/order_strategy.py` implements a complete if/else strategy tree.

```python
# Line 100 — hardcoded zone proximity threshold
at_zone = dist_to_primary <= atr * 0.15

# Line 114 — hardcoded confidence threshold for market orders
if forecast.confidence >= 0.70 and broker.spread_ok(forecast.spread):
    primary = OrderType.MARKET
    fill_conf = 0.95
    expiry = 0.0

# Lines 125–131 — hardcoded expiry and fill confidence
expiry = 4.0     # 4-hour limit
fill_conf = 0.75

# Lines 138–158 — hardcoded 8h and 24h expiry by case
expiry = 8.0    # session limit
expiry = 24.0   # GTC limit

# Lines 161–169 — hardcoded 12h expiry
expiry = 12.0

# Line 173 — hardcoded momentum threshold for order type switch
if aligned_mom > 0.60 and primary == OrderType.LIMIT:
    alt = OrderType.STOP
```

Six branches, six sets of magic numbers. The model has no say in order type selection — the rule tree decides.

**Required:** The model learns what order type and timing produces the best fill quality and trade outcomes. Confidence thresholds, expiry windows, and zone proximity thresholds are learned, not specified.

---

## V-07 — Hardcoded Promotion Gates

**Contract:** No hardcoded rule systems.

**Violation:** `trainer/qde/qde_models.py` — `QDEPromotionCriteria` defines 5 absolute floor constants that a new QDE version must pass regardless of anything the model has learned.

```python
# Lines 86–89 — hardcoded minimum performance floors
min_sharpe:       float = 0.5
min_profit_factor: float = 1.1
max_drawdown_abs: float = 0.30   # 30% drawdown ceiling
min_win_rate:     float = 0.45   # 45% win rate floor
max_drawdown_allowed: float = 0.25
```

These numbers are designer opinions about what constitutes acceptable trading performance. They are not derived from the characteristics of the instruments, regimes, or capital being traded. A model that has genuinely learned profitable behaviour in a low-win-rate/high-reward regime (e.g. trend following) will fail the `min_win_rate = 0.45` gate even if its expectancy is excellent.

**Required:** Promotion criteria are configurable by the administrator and reflect the specific deployment context, or are derived from the statistical properties of the incumbent's performance distribution — not hardcoded.

---

## V-08 — Hardcoded Regime-Based Expert Routing

**Contract:** No hardcoded strategies.

**Violation:** `trainer/ensemble/routing.py` lines 13–16 document a hardcoded routing policy:

```
Trending Market  → Transformer receives highest weight
Ranging Market   → LSTM receives highest weight
High Volatility  → TCN receives highest weight
```

This is a hand-designed prior: "Transformers are better for trends, LSTMs are better for ranges." Whether this is empirically true for the specific instruments and timeframes in production is never verified. The routing policy is baked in, not learned.

Combined with V-01, the routing system means:
- Market regime is classified by rules (V-03)
- Expert selection is determined by those regime labels via a fixed policy (V-08)
- Experts combine their votes by a fixed strategy (V-01)

Three layers of hardcoded decisions before a single learned weight is consulted.

**Required:** If multiple experts must exist at all (which V-01 already prohibits), their weights are determined entirely by recent prediction accuracy on this instrument in this context — not by regime labels.

---

## V-09 — 10 Parallel Sizing Methods

**Contract:** No sub-engines. One intelligence.

**Violation:** `trainer/icari/position_sizing_engine.py` implements 10 independent sizing algorithms and exposes `compare_methods()` which runs all 10 simultaneously:

```python
# Lines 52–62 — 10 methods in dispatch table
SizingMethod.FIXED_FRACTIONAL:   self._fixed_fractional
SizingMethod.FIXED_RISK:          self._fixed_risk
SizingMethod.ATR_ADJUSTED:        self._atr_adjusted
SizingMethod.VOLATILITY_ADJUSTED: self._volatility_adjusted
SizingMethod.EXPECTED_VALUE:      self._expected_value_adjusted
SizingMethod.CONFIDENCE_ADJUSTED: self._confidence_adjusted
SizingMethod.RISK_BUDGET_ADJUSTED: self._risk_budget_adjusted
SizingMethod.KELLY:               self._kelly
SizingMethod.FRACTIONAL_KELLY:    self._fractional_kelly
SizingMethod.CUSTOM:              self._custom
```

Each formula produces a different position size for the same input. There is no mechanism to determine which formula is correct for a given situation. The `DynamicSizer` then applies 10 additional adjustment factors on top. The final size is the product of a human-designed formula plus 10 human-designed multipliers. The model's view is absent.

**Required:** The model learns what fraction of capital to risk on a given trade based on its own confidence signal and historical outcomes on similar trades. No formulas. No 10-method comparison.

---

## V-10 — Hardcoded Momentum Classification

**Contract:** No hardcoded market regimes. The QDE learns these.

**Violation:** `trainer/mie/momentum_intelligence.py` classifies momentum using a lookup table and fixed thresholds:

```python
# Lines 119–124 — hardcoded quality score lookup table
MomentumState.ACCELERATING: 85.0
MomentumState.STEADY:        70.0
MomentumState.DECELERATING:  45.0
MomentumState.REVERSING:     25.0
MomentumState.FLAT:          20.0

# Lines 101–110 — hardcoded classification thresholds
if abs_str < 0.0001:                               return FLAT
if (strength > 0 and rsi < 40) or (...rsi > 60):  return REVERSING
if acceleration > 0.0001:                          return ACCELERATING
if acceleration < -0.0001 and abs_str > 0.0005:   return DECELERATING
```

The designer decided that ACCELERATING momentum is worth 85 quality points and FLAT is worth 20. These numbers encode a directional view about momentum that is not derived from backtested outcomes.

---

## V-11 — Hardcoded Trend Classification

**Contract:** No hardcoded market regimes.

**Violation:** `trainer/mie/trend_intelligence.py` — quality scoring is an explicit formula with hardcoded weights:

```python
# Lines 150–164 — hardcoded quality scoring formula
score += clamp(adx / self._adx_strong * 60.0, 0.0, 60.0)  # ADX worth 60 pts
# ...
score += 30.0   # structure confirmation: fixed 30 points
score += 10.0   # partial structure: fixed 10 points
score += clamp(gap * 1000.0, 0.0, 10.0)  # EMA gap: up to 10 points
```

ADX contributes 60% of the score by design, not by learning. A model trained on these signals could discover that ADX is irrelevant for a particular instrument — but the scoring formula will always weight it at 60%.

---

## V-12 — Hardcoded Statistical Constants

**Contract:** No hardcoded risk.

**Violation:** `trainer/execution/risk_engine.py` uses two hardcoded statistical constants that embed assumptions about the return distribution:

```python
# Line 110 — 1.645 is the z-score for the 95th percentile of a normal distribution
tail_risk = round(min(1.645 * sigma, struct_dist), 5)

# Line 117 — 60/40 weighting is designer judgment, not learned
expected_dd = round(0.6 * stop_dist + 0.4 * tail_risk, 5)
```

The assumption that returns are normally distributed (justified by the 1.645 z-score) is false for financial returns. The 60/40 blend is an arbitrary choice with no empirical basis.

---

## Root Cause

All violations share a common pattern: **the system was designed as a rule engine that delegates to a model for signal, rather than as a model that learns behaviour end-to-end.**

The model answers "buy or sell?" — but the system answers "with what size, at what price, with what stop, with what target, under what regime classification." Those five questions should also be answered by the model.

The current architecture has:

```
Market Data
    ↓
Rule Engine A (regime classification — V-03, V-10, V-11)
    ↓
Rule Engine B (momentum, trend, volatility, liquidity — V-02)
    ↓
N Models (ensemble voting — V-01)
    ↓
Rule Engine C (entry zones, order type — V-06)
    ↓
Rule Engine D (stop/TP placement — V-04)
    ↓
Rule Engine E (position sizing — V-05, V-09)
    ↓
Decision
```

**What is required:**

```
Market Data
    ↓
One Model
    ↓
Decision (direction, size, entry, exit — all learned)
```

---

## Priority Order for Remediation

| Priority | Violation | Reason |
|---|---|---|
| P0 | V-01 — Multiple decision makers | Architecturally incompatible with the contract |
| P0 | V-02 — Sub-engine proliferation | 40+ engines cannot coexist with "one intelligence" |
| P1 | V-03 — Hardcoded market regimes | 16 regime labels are a fixed taxonomy; model cannot discover new structure |
| P1 | V-04 — Hardcoded exits | Stop/TP geometry is a rule, not a learned behaviour |
| P1 | V-05 — Hardcoded risk | Risk is 10 formulas + 10 adjustments, not learned |
| P2 | V-06 — Hardcoded order strategy | Order type selection is an if/else tree |
| P2 | V-08 — Hardcoded regime routing | Regime-to-expert mapping is a fixed policy |
| P2 | V-09 — 10 parallel sizing methods | 10 competing algorithms with no learned selector |
| P3 | V-07 — Hardcoded promotion gates | Absolute floors encode designer priors |
| P3 | V-10 — Hardcoded momentum classification | Lookup table quality scores |
| P3 | V-11 — Hardcoded trend classification | Hardcoded ADX/structure weighting |
| P3 | V-12 — Hardcoded statistical constants | Normality assumption + arbitrary blend weights |
