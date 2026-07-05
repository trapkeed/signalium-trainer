# ICARI Developer Guide — Phase 24

## Architecture

All modules in `trainer/icari/`. Pure Python + stdlib + loguru. Zero external ML dependencies.

```
trainer/icari/
├── __init__.py              — 40+ public exports
├── icari_types.py           — Enums, constants, scenario factors
├── icari_models.py          — Dataclasses (AllocationRequest, AllocationResult, ...)
├── allocation_engine.py     — 15-factor capital allocator
├── risk_budget_engine.py    — 12-type risk budget manager
├── position_sizing_engine.py — 10 sizing methods
├── dynamic_sizer.py         — 10-factor adaptive sizing
├── capital_efficiency.py    — Utilisation & efficiency metrics
├── portfolio_risk_engine.py — VaR/CVaR/ES (historical simulation)
├── drawdown_manager.py      — 6-mode drawdown protection
├── correlation_engine.py    — Rolling, dynamic, tail, clustering
├── scenario_engine.py       — 11 macro scenario stress tests
├── allocation_optimizer.py  — 8-objective grid-search optimizer
├── capital_health.py        — 7-score health framework
├── icari_database.py        — SQLite persistence (8 tables)
├── icari_dashboard.py       — 7 dashboard views
└── icari_api.py             — Unified entry point
```

## Entry Point

```python
from trainer.icari import IcariAPI

api = IcariAPI(
    capital       = 1_000_000.0,
    db_path       = ":memory:",           # or "/data/icari.db"
    sizing_method = SizingMethod.KELLY,
    soft_dd_limit = 0.05,
    hard_dd_limit = 0.10,
)
```

## Adding a New Sizing Method

1. Add enum member to `SizingMethod` in `icari_types.py`
2. Add private method `_my_method(self, req)` in `PositionSizingEngine`
3. Add to `dispatch` dict in `PositionSizingEngine.size()`
4. Add test case to `TestPositionSizingEngine`

## Adding a New Scenario

1. Add enum member to `ScenarioType` in `icari_types.py`
2. Add stress factors `(return_mult, vol_mult, liq_mult)` to `SCENARIO_FACTORS`
3. (Optional) Add currency sensitivity to `_CCY_SENSITIVITY` in `scenario_engine.py`
4. Add test assertion in `TestScenarioEngine`

## Database Schema

```sql
capital_allocation   — primary allocation decisions
risk_budgets         — current budget state (upserted)
allocation_history   — append-only log of all decisions
portfolio_risk       — time series of risk snapshots
scenario_results     — stress test outputs
capital_health       — health score snapshots
optimization_results — optimizer runs (weights + metrics as JSON)
drawdown_events      — events with recovery flag
```

## Testing

```bash
cd signalium-trainer
python3 -m unittest tests/unit/test_icari.py -v
```

All tests use `spec_from_file_location` (no torch import).

## Configuration

`conf/icari/icari.yaml` — all limits, methods, and budget defaults.

Override at runtime:
```python
api.set_budget(RiskBudgetType.DAILY, 0.015)
# or at construction:
engine = AllocationEngine(max_allocation_frac=0.20, min_confidence=0.40)
```

## Safety Constraints

ICARI never:
- Modifies any forecasting model
- Executes trades
- Connects to brokers
- Returns BUY/SELL signals

ICARI always:
- Logs every allocation decision with rationale
- Persists all decisions for audit
- Respects drawdown lockdown before any allocation
- Caps position risk at 5% per trade
- Caps allocation at 25% of capital per instrument
