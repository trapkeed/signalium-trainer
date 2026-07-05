# Portfolio Intelligence Guide — Phase 20

## Overview

Phase 20 transforms Signalium from a single-instrument forecasting system into a
portfolio-aware quantitative intelligence platform.

The engine evaluates all model forecasts **collectively** before producing final
portfolio recommendations. It operates as a pure analytics layer on top of the
existing forecasting system — it never modifies forecasting models, never retrains
any model, and has zero new runtime dependencies (stdlib + loguru only).

---

## Architecture

```
Model Forecasts (List[ModelForecast])
         │
         ▼
 ┌─────────────────┐
 │ ConflictResolver│ ← detect conflicting signals
 └────────┬────────┘
          │ adjusted_forecasts
          ▼
 ┌──────────────────┐
 │CorrelationEngine │ ← compute dynamic correlations
 └────────┬─────────┘
          │ CorrelationMatrix
          ▼
 ┌──────────────────┐   ┌──────────────────┐
 │  RegimeDetector  │   │  ExposureEngine  │
 └────────┬─────────┘   └────────┬─────────┘
          │ RegimeAssessment     │ ExposureReport
          ▼                      │
 ┌──────────────────┐            │
 │PositionAllocator │ ◄──────────┘
 └────────┬─────────┘
          │ AllocationResult (signed weights)
          ▼
 ┌──────────────────┐
 │ PortfolioScorer  │ ← Sharpe, Sortino, Drawdown, Diversification
 └────────┬─────────┘
          │ PortfolioMetrics
          ▼
 ┌────────────────────────────────────────┐
 │ Optimizer · ScenarioAnalyser · Database│
 └────────────────────────────────────────┘
          │
          ▼
  PortfolioApi → PortfolioDashboard
```

---

## Quick Start

```python
from trainer.portfolio import PortfolioApi, ModelForecast

api = PortfolioApi(db_path="data/portfolio.db")

forecasts = [
    ModelForecast("EURUSD", "BUY",  confidence=0.72, expected_return=0.08, expected_risk=0.10),
    ModelForecast("GBPUSD", "BUY",  confidence=0.65, expected_return=0.07, expected_risk=0.12),
    ModelForecast("XAUUSD", "BUY",  confidence=0.80, expected_return=0.12, expected_risk=0.15),
    ModelForecast("NAS100", "SELL", confidence=0.60, expected_return=0.18, expected_risk=0.22),
    ModelForecast("BTCUSD", "SELL", confidence=0.55, expected_return=0.25, expected_risk=0.35),
]

# Full portfolio recommendation
resp = api.recommend(forecasts, method="hybrid")
print(resp["data"]["portfolio_metrics"])

# Scenario analysis
sc = api.scenario(forecasts)
print(sc["data"]["worst_scenario"])

# Portfolio optimization
opt = api.optimize(forecasts, objective="max_sharpe", compare_all=True)
print(opt["data"]["best"]["objective_value"])
```

---

## Key Concepts

### ModelForecast

The entry point to the engine. Created by the forecasting system.

```python
ModelForecast(
    instrument="EURUSD",
    direction="BUY",        # "BUY" | "SELL" | "NEUTRAL"
    confidence=0.72,        # 0–1
    expected_return=0.08,   # annualized
    expected_risk=0.10,     # annualized volatility
)
```

### Signed Return

`signed_return()` = `expected_return` for BUY, `-expected_return` for SELL, `0` for NEUTRAL.
This is what drives all portfolio arithmetic.

---

## Module Reference

| Module | Purpose |
|--------|---------|
| `portfolio_instruments` | Registry + ModelForecast definition |
| `correlation_engine` | Rolling, rank, tail, exponential correlations |
| `exposure_engine` | Asset class / currency / region exposure |
| `dependency_graph` | Cluster detection, centrality |
| `regime_detector` | Market regime from return series |
| `risk_budget` | Configurable budget enforcement |
| `conflict_resolver` | Signal conflict detection |
| `portfolio_scorer` | Sharpe, Sortino, diversification |
| `position_allocator` | Normalized allocation weights |
| `scenario_analysis` | Macro stress testing |
| `optimizer` | Objective-based weight optimization |
| `portfolio_database` | SQLite persistence (6 tables) |
| `portfolio_api` | Unified entry point (16 endpoints) |
| `portfolio_dashboard` | 8 visualisation data structures |

---

## Backward Compatibility

- Phase 20 adds a new `trainer/portfolio/` package.
- Nothing in `trainer/research/`, `trainer/models/`, or any other Phase is modified.
- The `trainer/__init__.py` is not changed.
- All Phase 20 names are accessible via `from trainer.portfolio import ...`.
