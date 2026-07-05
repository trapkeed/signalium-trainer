# Institutional Capital Allocation & Risk Intelligence Architecture Report
## ICARI — Phase 24

---

## Executive Summary

Phase 24 delivers ICARI — Signalium's Institutional Capital Allocation & Risk Intelligence Platform. ICARI transforms the Signalium quantitative platform from a forecasting engine into a full institutional portfolio management system.

Key capabilities:
- Dynamic 15-factor capital allocation with regime-aware multipliers
- 10 configurable position sizing methods (Fixed, ATR, Kelly, Fractional Kelly, Volatility, etc.)
- Adaptive drawdown protection with 6 progressive modes (Normal → Lockdown)
- Portfolio VaR/CVaR/Expected Shortfall via historical simulation
- 11 macro scenario stress tests
- 8-objective portfolio optimization (Max Sharpe, Risk Parity, Capital Preservation, etc.)
- 7 composite capital health scores → institutional grade
- Pure Python — zero external ML dependencies

**Absolute constraints:** Never modifies forecasting models. Never executes trades. Never connects to brokers.

---

## 1. Architecture

### 1.1 Position in Signalium Stack

```
┌────────────────────────────────────────────────────────────────┐
│                 Signalium Quant Platform                        │
├────────────────────────────────────────────────────────────────┤
│  Phase 20: Portfolio Analytics        ─┐                        │
│  Phase 21: Execution Analytics         ├── Feeds ICARI          │
│  Phase 21.5: Lifecycle Analytics      ─┘                        │
│  Phase 21.6: Decision Memory          ─┐                        │
│  Phase 22: Microstructure              ├── Knowledge for SAQRA  │
│  Phase 23: SAQRA (AI Analyst)         ─┘                        │
├────────────────────────────────────────────────────────────────┤
│  Phase 24: ICARI ← Capital Allocation Layer (THIS PHASE)        │
│                                                                  │
│  Reads: forecasts, scores, execution recommendations            │
│  Writes: allocation decisions, risk budgets, health scores      │
│  Never: modifies models, executes trades, connects brokers      │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 Module Architecture

```
IcariAPI (unified entry point)
├── AllocationEngine         ← 15-factor dynamic allocation
├── RiskBudgetEngine         ← 12 configurable risk budgets
├── PositionSizingEngine     ← 10 sizing methods
├── DynamicSizer             ← 10-factor adaptive adjustment
├── CapitalEfficiencyEngine  ← utilisation, idle, efficiency metrics
├── PortfolioRiskEngine      ← VaR/CVaR/ES/volatility (hist. sim.)
├── DrawdownManager          ← 6-mode adaptive drawdown protection
├── CorrelationEngine        ← rolling, dynamic, tail, clustering
├── ScenarioEngine           ← 11 macro stress scenarios
├── AllocationOptimizer      ← 8-objective grid-search optimizer
├── CapitalHealthEngine      ← 7 health scores → overall grade
├── IcariDatabase            ← SQLite (8 tables)
└── IcariDashboard           ← 7 dashboard views
```

---

## 2. Allocation Engine

### 2.1 Factor Weights

| Factor | Weight |
|---|---|
| forecast_confidence | 15% |
| research_score | 12% |
| execution_score | 10% |
| expected_value | 10% |
| expected_return | 10% |
| model_confidence | 10% |
| ensemble_confidence | 8% |
| decision_quality | 8% |
| liquidity | 7% |
| volatility_adj | 5% |
| correlation_adj | 5% |
| **Total** | **100%** |

### 2.2 Allocation Pipeline

```
AllocationRequest (15 factors)
    ↓ compute factor scores
Composite Confidence Score
    ↓ < 0.30 → DENIED
Regime Multiplier (0.65–1.10)
    ↓ × base_frac (10%)
Exposure Penalty (>50% portfolio exposure)
    ↓ Expected Drawdown Penalty (>10%)
Cap at 25% of capital
    ↓ Check correlation/exposure recommendations
AllocationResult (status, frac, usd, confidence, rationale, recommendations)
```

---

## 3. Risk Framework

### 3.1 Risk Budget Types (12)

DAILY, WEEKLY, MONTHLY, PORTFOLIO, INSTRUMENT, CURRENCY, COMMODITY, CRYPTO, INDEX, STRATEGY, CORRELATED, OPEN_TRADE.

All budgets are configurable. Default values are conservative fractions of total capital.

### 3.2 Drawdown Protection Modes

| Mode | Trigger | Risk Multiplier |
|---|---|---|
| NORMAL | < 5% DD | 1.00 |
| SOFT_LIMIT | 5–10% DD | 0.75 |
| HARD_LIMIT | 10–20% DD | 0.50 |
| PRESERVATION | Manual | 0.25 |
| RECOVERY | Improving from Hard | 0.60 |
| EMERGENCY_LOCKDOWN | > 20% DD | 0.00 |

### 3.3 Portfolio Risk (Historical Simulation)

```
Returns time series (per instrument)
    ↓ weighted portfolio return series
Sort ascending → percentile lookup
    ↓
VaR 95% = -percentile(5%)
VaR 99% = -percentile(1%)
CVaR 95% = -mean(tail returns below 5th percentile)
Portfolio Vol = std(port_returns) × √252
Max Expected DD = 2 × daily_std × √N
Diversification Ratio = weighted_vol_sum / port_vol
Risk Concentration = Σ(risk_contribution_i²)   [Herfindahl]
```

---

## 4. Sizing Engine

### 4.1 Methods

| Method | Formula |
|---|---|
| FIXED_FRACTIONAL | units = (capital × risk_frac) / ATR |
| FIXED_RISK | units = risk_amount / (ATR × atr_mult) |
| ATR_ADJUSTED | Same as Fixed Risk — stop = ATR × mult |
| VOLATILITY_ADJUSTED | vol_mult = max(0.25, 1 - vol×10); risk_amount × vol_mult |
| EXPECTED_VALUE | ev_mult = clamp(1 + EV×50, 0.1, 2.0); scale risk_amount |
| CONFIDENCE_ADJUSTED | conf = mean(forecast, model, ensemble); scale risk_amount |
| RISK_BUDGET_ADJUSTED | rf = risk_frac × (1 - exposure) |
| KELLY | f* = (p×(b+1)-1) / b; p=win_prob, b=win/loss |
| FRACTIONAL_KELLY | f* × kelly_frac (default 0.5) |
| CUSTOM | Administrator-provided callable |

### 4.2 Dynamic Adjustments

```
base_units × composite_multiplier
composite = geometric_mean(drawdown_adj, exposure_adj, vol_adj, liquidity, corr_adj,
                           spread_adj, session_mult, hist_perf, model_health, forecast_quality)
bounds: [0.10, 2.0]
```

---

## 5. Optimization Pipeline

### 5.1 Objectives

MAX_SHARPE, MAX_SORTINO, MIN_DRAWDOWN, MAX_DIVERSIFICATION, RISK_PARITY, BALANCED, CAPITAL_PRESERVATION, EXPECTED_UTILITY.

### 5.2 Algorithm

Grid-search over weight candidates (pure Python, no scipy):
1. Generate candidate vectors: uniform, single-dominant, pair-dominant, alpha sweep
2. Score each candidate with the objective function
3. Return the weight vector maximizing the score

No external solver required. Suitable for N ≤ 20 instruments.

---

## 6. Scenario Engine

### 6.1 Stress Factors

| Scenario | Return | Vol | Liquidity |
|---|---|---|---|
| FLASH_CRASH | -40% | 3.0× | 0.20 |
| LIQUIDITY_CRISIS | -20% | 2.0× | 0.10 |
| HIGH_VOLATILITY | -10% | 2.5× | 0.60 |
| CRYPTO_CRASH | -50% | 4.0× | 0.30 |
| RISK_ON | +5% | 0.7× | 1.20 |

Each scenario applies instrument-level stress using weights, volatilities, currency sensitivities, and liquidity adjustments. Resilience score = max(0, 1 + portfolio_return).

---

## 7. Capital Health Framework

### 7.1 Health Score Components

| Component | Weight |
|---|---|
| Capital Efficiency | 20% |
| Risk Score | 20% |
| Allocation Quality | 15% |
| Diversification | 15% |
| Portfolio Stability | 15% |
| Capital Utilisation | 15% |
| **Total** | **100%** |

### 7.2 Health Grades

| Score | Grade |
|---|---|
| ≥ 0.85 | EXCELLENT |
| ≥ 0.70 | GOOD |
| ≥ 0.55 | FAIR |
| ≥ 0.40 | POOR |
| < 0.40 | CRITICAL |

---

## 8. Database Design

### 8.1 Tables (8)

```
capital_allocation     — allocation decisions
risk_budgets           — current budget state
allocation_history     — append-only allocation log
portfolio_risk         — time series of risk snapshots
scenario_results       — stress test outputs
capital_health         — health score history
optimization_results   — optimizer outputs (JSON weights/metrics)
drawdown_events        — drawdown event log with recovery flag
```

### 8.2 Persistence

SQLite with persistent `:memory:` connection for testing. All tables auto-created on first use. Indexed on timestamp, instrument, status, and scenario type.

---

## 9. Operational Readiness

| Capability | Status |
|---|---|
| 15-factor capital allocation | ✅ |
| 10 position sizing methods | ✅ |
| 10-factor dynamic sizing | ✅ |
| 12 risk budget types | ✅ |
| 6-mode drawdown protection | ✅ |
| Portfolio VaR/CVaR/ES (historical) | ✅ |
| Diversification ratio | ✅ |
| Risk concentration (HHI) | ✅ |
| Rolling & tail correlations | ✅ |
| Correlation clustering | ✅ |
| 11 macro scenarios | ✅ |
| 8-objective optimizer | ✅ |
| 7 health scores → grade | ✅ |
| Capital efficiency analysis | ✅ |
| 8-table SQLite persistence | ✅ |
| 7 dashboard views | ✅ |
| Backward compatibility | ✅ |
| stdlib-only (no torch/numpy) | ✅ |
| Test coverage | ✅ 160+ tests |

---

## 10. Institutional Readiness

- **Transparency:** Every allocation includes rationale, factor scores, and recommendations
- **Auditability:** All decisions persisted with timestamps; full allocation history
- **Configurability:** All limits, methods, and budgets configurable without code changes
- **Safety:** Never executes trades; never modifies forecasting models
- **Resilience:** Automatic drawdown protection; emergency lockdown at 20% DD
- **Extensibility:** New sizing methods via `custom_fn`; new scenarios via `SCENARIO_FACTORS`

---

*Report generated: July 2025 — Phase 24 ICARI Implementation*
