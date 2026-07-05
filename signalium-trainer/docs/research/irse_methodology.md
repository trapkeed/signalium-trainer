# IRSE Scoring Methodology

## Scoring Philosophy

The IRSE does not use accuracy alone.  A model can have high classification
accuracy and yet destroy capital if its confidence is poorly calibrated, its
strategy collapses under spread expansion, or it has never been tested on
unseen instruments.

The Research Score is a **composite quality signal** designed to surface
institutional-grade models.

---

## Category Methodology

### 1. Validation Quality (25%)

Evaluates the reliability of the training–validation protocol.

| Sub-metric                   | Target            | Weight |
|------------------------------|-------------------|--------|
| Walk-forward accuracy        | ≥ 0.75            | 25%    |
| Walk-forward F1 macro        | ≥ 0.75            | 15%    |
| Fold coverage (n_folds)      | ≥ 12              | 10%    |
| Cross-regime stability       | std ≤ 0.01        | 15%    |
| Cross-instrument stability   | std ≤ 0.01        | 10%    |
| Cross-timeframe stability    | std ≤ 0.01        | 10%    |
| Consistency (pct folds ≥ thr)| ≥ 90%             | 10%    |
| Prediction reliability        | ≥ 85%             | 5%     |

**Normalisation:** Accuracy is linearly mapped from [0.50, 0.75] → [0, 100].
Stability metrics are inverted: std = 0.0 → 100, std ≥ 0.10 → 0.

---

### 2. Trading Performance (25%)

Evaluates profitability and risk-adjusted returns.

| Sub-metric              | Target         | Weight |
|-------------------------|----------------|--------|
| Profit factor           | ≥ 2.0          | 15%    |
| Expectancy              | ≥ 0.50 R       | 10%    |
| Sharpe ratio            | ≥ 2.5          | 20%    |
| Sortino ratio           | ≥ 3.0          | 10%    |
| Calmar ratio            | ≥ 3.0          | 10%    |
| Max drawdown            | ≤ 3%           | 15%    |
| Win rate                | ≥ 65%          | 8%     |
| Average R multiple      | ≥ 2.0          | 5%     |
| TP before SL accuracy   | ≥ 75%          | 4%     |
| Capital efficiency      | ≥ 1.0          | 3%     |

**Normalisation:** Sharpe is mapped from [0.5, 2.5] → [0, 100]. Max drawdown
is inverted: 3% → 100, 25% → 0.

---

### 3. Robustness (15%)

All robustness inputs are fractions [0, 1] measuring performance retention
under adversarial conditions.

| Sub-metric                    | Scenario              | Weight |
|-------------------------------|-----------------------|--------|
| Monte Carlo stability         | Random path shuffling | 20%    |
| Stress test pass rate         | Extreme regime events | 25%    |
| Noise resistance              | Signal noise          | 15%    |
| Data gap tolerance            | Missing bars          | 10%    |
| Spread expansion tolerance    | 2× spread             | 15%    |
| Execution delay tolerance     | 1–3 bar delay         | 7%     |
| Volatility shift tolerance    | Vol × 2               | 8%     |

---

### 4. Calibration (10%)

Measures the reliability of probability estimates.

| Sub-metric                  | Target          | Weight |
|-----------------------------|-----------------|--------|
| Expected Calibration Error  | ≤ 0.02          | 35%    |
| Brier score                 | ≤ 0.12          | 25%    |
| Confidence reliability      | ≥ 90%           | 20%    |
| Prediction entropy          | ≤ 0.20 bits     | 10%    |
| Probability calibration     | R² ≥ 0.95       | 10%    |

---

### 5. Generalization (10%)

Tests whether the model works beyond its training distribution.

| Sub-metric                      | Weight |
|---------------------------------|--------|
| Unseen instrument performance   | 15%    |
| Unseen timeframe performance    | 15%    |
| Unseen regime performance       | 20%    |
| Out-of-sample (true) performance| 25%    |
| Recent data (last 6mo)          | 15%    |
| Historical stability            | 10%    |

**Normalisation:** Performance metrics mapped from [0.50, 0.75] → [0, 100].

---

### 6. Efficiency (5%)

Evaluates resource consumption.

| Sub-metric            | Target  | Weight |
|-----------------------|---------|--------|
| Inference latency     | ≤ 5ms   | 30%    |
| Training time         | ≤ 1h    | 15%    |
| GPU utilization       | 85%+    | 15%    |
| CPU utilization       | —       | 5%     |
| RAM usage             | ≤ 2GB   | 15%    |
| Model size            | ≤ 50MB  | 10%    |
| Storage footprint     | ≤ 1GB   | 10%    |

---

### 7. Explainability (5%)

| Sub-metric                   | Weight |
|------------------------------|--------|
| Feature attribution coverage | 30%    |
| Attention quality            | 20%    |
| Prediction traceability      | 20%    |
| Decision transparency        | 15%    |
| Documentation completeness   | 15%    |

---

### 8. Operational Readiness (5%)

| Sub-metric            | Weight |
|-----------------------|--------|
| Deployment readiness  | 25%    |
| Monitoring coverage   | 20%    |
| Health checks         | 20%    |
| Drift monitoring      | 15%    |
| Rollback readiness    | 10%    |
| Audit completeness    | 10%    |

---

## Determinism

All category scorers are pure functions with no random state.  The same inputs
always produce the same output — a requirement for reproducible institutional
scoring.

## Extensibility

Add new sub-metrics by extending the corresponding `*Inputs` dataclass and
updating the scorer's `_weighted_avg` call.  The composite score formula
remains unchanged.
