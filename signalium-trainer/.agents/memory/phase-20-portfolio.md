---
name: Phase 20 portfolio package
description: Design rules and gotchas for the trainer/portfolio/ engine.
---

## The Package

`trainer/portfolio/` — 14 modules, stdlib + loguru only, zero model changes.

## Dependency load order (for spec_from_file_location tests)

```
portfolio_instruments → correlation_engine → exposure_engine → dependency_graph
→ regime_detector → risk_budget → conflict_resolver → portfolio_scorer
→ position_allocator → scenario_analysis → optimizer → portfolio_database
→ portfolio_dashboard → portfolio_api
```

## Key Design Rule: weight renormalization after scale_to_budget

`RiskBudgetEngine.scale_to_budget()` may cap individual instrument weights
without renormalizing the full vector. After calling it, always renormalize:

```python
total_abs = sum(abs(v) for v in scaled.values())
if total_abs > 1e-9:
    scaled = {k: v / total_abs for k, v in scaled.items()}
```

**Why:** per-instrument cap (e.g., max_single_instrument=0.30) reduces some weights
but does not proportionally increase others. Without renorm, sum(abs) < 1.0,
which breaks downstream Sharpe/Sortino arithmetic that assumes normalized weights.

## Key Design Rule: BudgetStatus WARNING threshold

WARNING fires at utilisation >= 0.85 (not 0.80). Test values must exceed 0.85.
Example: limit=0.50, current=0.44 → utilisation=0.88 → WARNING.

## Database key names

`PortfolioDatabase.stats()` returns: `recommendations`, `correlation_matrices`,
`exposures`, `optimizations`, `scenarios`, `allocations`, `average_sharpe`.
Note the key is `correlation_matrices` (not `correlations`).

## 16 default instruments

EURUSD GBPUSD USDJPY USDCHF USDCAD AUDUSD NZDUSD EURJPY GBPJPY
XAUUSD XAGUSD NAS100 US30 SPX500 BTCUSD ETHUSD
