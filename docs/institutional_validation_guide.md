# Institutional Research Validation Platform Guide

## Phase 15 — Signalium Quant

---

## Overview

The Institutional Research Validation Platform is the **official validation framework** for Signalium Quant. No model may be deployed to production unless it successfully passes this platform.

This framework replaces traditional validation and becomes the **quantitative research standard** for Signalium.

**Key principle:** Evaluate only. Never modify models. Never retrain automatically.

---

## Quick Start

```python
from trainer.validation import (
    InstitutionalValidationEngine,
    EngineConfig,
    PromotionPolicy,
)

# 1. Configure promotion rules
policy = PromotionPolicy(
    min_accuracy=0.55,
    min_sharpe=1.0,
    max_drawdown=0.20,
    min_profit_factor=1.5,
    max_ece=0.10,
    min_trade_count=100,
    require_manual_approval=True,
)

# 2. Configure the engine
config = EngineConfig(
    strategy="expanding",           # expanding or sliding
    min_train_size=2000,
    val_size=500,
    step_size=100,
    n_mc_simulations=1000,
    export_formats=["json", "csv", "html"],
)

# 3. Run validation
engine = InstitutionalValidationEngine(
    db_path="experiments/validation.db",
    leaderboard_path="experiments/leaderboard.json",
    policy=policy,
    export_dir="reports/",
    config=config,
)

result = engine.validate(
    model=trained_model,
    x=data_tensor,          # (N, seq_len, n_features)
    y=labels_tensor,        # (N,)
    model_id="quant_v1",
    model_version="2.0.0",
)

print(result.summary())
```

---

## Validation Pipeline (15 Steps)

The engine executes these steps in sequence:

| Step | Component | What It Does |
|------|-----------|--------------|
| 1 | LeakageDetector | Detects future/target/feature leakage; aborts if critical |
| 2 | WalkForward | Generates time-ordered folds (no shuffling) |
| 3 | StatisticalMetricsEngine | Computes 17 statistical metrics |
| 4 | TradingMetricsEngine + TradeSimulator | Computes 21 trading metrics |
| 5 | CalibrationValidator | ECE, MCE, reliability diagram, temperature recommendation |
| 6 | PredictionHeadValidator | Per-head evaluation for all 16 prediction heads |
| 7 | MonteCarloValidator | 1000-path equity curve simulation |
| 8 | RobustnessTester | 7 stress scenarios |
| 9 | DriftAnalyzer | PSI + JSD across features, predictions, labels, regimes |
| 10 | FailureAnalyzer | Worst instruments, timeframes, regimes, heads |
| 11 | PromotionPolicy | Checks all configured thresholds |
| 12 | VisualizationDataBuilder | Generates all chart data |
| 13 | InstitutionalLeaderboard | Adds entry and ranks |
| 14 | ValidationDatabase | Persists everything to SQLite |
| 15 | ReportExporter | Writes JSON, CSV, HTML reports |

---

## Time Series Safety

The platform **never shuffles data**. Chronological ordering is always preserved.

### Leakage Detection

```python
from trainer.validation import LeakageDetector

detector = LeakageDetector(
    abort_on_detect=True,
    target_correlation_threshold=0.95,
    future_correlation_threshold=0.90,
)
report = detector.check(x_train, y_train, x_val, y_val, timestamps=ts)
```

Detects:
- **Future leakage** — features containing future information
- **Target leakage** — features trivially correlated with the label (|r| > 0.95)
- **Timestamp issues** — non-monotone, duplicate, or gapped timestamps
- **Duplicate samples** — identical rows appearing in both train and val
- **NaN/Inf features** — invalid values in the feature tensor

---

## Walk-Forward Validation Strategies

### Expanding Window (default)
Training window grows on each fold. Recommended for most use cases.

```python
from trainer.validation import ExpandingWindowValidator

v = ExpandingWindowValidator(
    min_train_size=2000,
    val_size=500,
    test_size=100,   # optional hold-out
    step_size=100,
    gap_size=10,     # purge gap between train and val
)
```

### Sliding Window
Fixed-size training window slides forward. Simulates periodic model retraining.

```python
from trainer.validation import SlidingWindowValidator

v = SlidingWindowValidator(
    train_size=2000,
    val_size=500,
    step_size=100,
)
```

### Anchored Walk-Forward
Training always starts from t=0. Maximum history available at every fold.

```python
from trainer.validation import AnchoredWalkForward

v = AnchoredWalkForward(
    initial_train_size=2000,
    val_size=500,
    step_size=100,
)
```

### Nested Validation
Outer expanding-window loop with inner sliding-window for hyper-parameter search.

```python
from trainer.validation import NestedValidator

v = NestedValidator(
    outer_min_train=3000, outer_val_size=500, outer_step=200,
    inner_train_size=1000, inner_val_size=250, inner_step=100,
    gap_size=10,
)
```

---

## Market Regime Classification

```python
from trainer.validation import MarketRegimeClassifier

clf = MarketRegimeClassifier(
    trend_window=20,
    volatility_window=14,
)
regimes = clf.classify(ohlcv_tensor, timestamps=ts)
```

Classified regimes:
- `trending_strong`, `trending_weak`
- `ranging`, `breakout`, `pullback`
- `high_volatility`, `low_volatility`
- `compression`, `expansion`
- `accumulation`, `distribution`
- `news_period`, `session_open`, `session_close`, `weekend_gap`

---

## Statistical Metrics (17)

| Metric | Description |
|--------|-------------|
| accuracy | Classification accuracy |
| precision_macro | Macro-averaged precision |
| recall_macro | Macro-averaged recall |
| f1_macro | Macro-averaged F1 |
| roc_auc | One-vs-rest ROC AUC |
| pr_auc | Average precision (PR AUC) |
| log_loss | Cross-entropy loss |
| cross_entropy | Identical to log_loss for softmax |
| mae | Mean absolute error on probabilities |
| rmse | Root mean squared error |
| mape | Mean absolute percentage error |
| r_squared | Explained variance of confidence |
| ece | Expected Calibration Error |
| brier_score | Multi-class Brier score |
| prediction_entropy | Mean predictive entropy |
| confidence_mean / std / p10 / p50 / p90 | Confidence distribution |
| mce | Maximum Calibration Error |

---

## Trading Metrics (21)

| Metric | Description |
|--------|-------------|
| win_rate | Fraction of profitable trades |
| profit_factor | Gross profit / gross loss |
| avg_rr | Average risk/reward ratio |
| avg_holding_bars | Mean holding duration |
| trade_frequency | Trades per 100 bars |
| avg_mfe / avg_mae | Average MFE / MAE in pips |
| expectancy | Expected P&L per trade |
| max_drawdown | Maximum equity drawdown |
| avg_drawdown | Mean drawdown |
| recovery_factor | Total return / max drawdown |
| sharpe_ratio | Annualised Sharpe |
| sortino_ratio | Annualised Sortino |
| calmar_ratio | CAGR / max drawdown |
| ulcer_index | RMS of drawdown sequence |
| mar_ratio | Annualised return / max drawdown |
| risk_of_ruin | Probability of 50% capital loss |
| capital_efficiency | Return / max capital at risk |
| return_skewness / return_kurtosis | Distribution shape |
| tail_risk_cvar5 | CVaR at 5th percentile |
| tp_before_sl_accuracy | TP hit rate |

---

## Monte Carlo Simulation

```python
from trainer.validation import MonteCarloValidator

mc = MonteCarloValidator(
    n_simulations=1000,
    seed=42,
    slippage_std_pips=0.5,
    spread_expansion_prob=0.05,
    initial_capital=10_000.0,
)
result = mc.run(trades)
print(result.summary())
```

Outputs confidence bands:
- P5 (worst), P25, P50 (median), P75, P95 (best)
- Ruin probability
- Max drawdown distribution

---

## Robustness Testing

Seven stress scenarios tested automatically:

| Scenario | What It Tests |
|----------|---------------|
| MissingCandles | Model stability with zeroed bars |
| NoisyData | Gaussian noise on all features |
| SpreadExpansion | Wider bid-ask spread impact |
| LatencyDelay | Stale data (shifted input) |
| DataGaps | NaN gaps in sequences |
| VolatilitySpike | Scaled outlier bars |
| MarketCrash | Reversed return window |

---

## Export Formats

Reports are automatically exported after each validation run:

| Format | Contents |
|--------|----------|
| JSON | Complete machine-readable results |
| CSV | Flat tabular metrics |
| HTML | Interactive dashboard |
| PDF (text) | Formatted text summary |

---

## API Reference

```python
# Access the internal API
api = engine.api

# Retrieve a report
resp = api.get_report(run_id)

# Query the leaderboard
resp = api.get_leaderboard(sort_by="sharpe_ratio", limit=20)

# Compare two models
resp = api.compare_models("model_a", "model_b")

# Approve a promotion
resp = api.approve_promotion(run_id, approver="researcher_name")

# Retire a model
resp = api.retire_model(model_id, reason="Superseded by v3")
```
