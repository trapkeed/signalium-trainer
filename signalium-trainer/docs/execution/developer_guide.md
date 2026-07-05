# Developer Guide — Phase 21 Execution Intelligence

## Repository Layout

```
signalium-trainer/
├── trainer/execution/
│   ├── __init__.py              # 65 exports
│   ├── execution_types.py       # Enums, ExecutionForecast, BrokerConstraints
│   ├── entry_zones.py           # EntryZoneEngine
│   ├── entry_quality.py         # EntryQualityEngine
│   ├── stop_loss_engine.py      # StopLossEngine
│   ├── take_profit_engine.py    # TakeProfitEngine
│   ├── position_sizer.py        # PositionSizer, RiskLimits
│   ├── trade_manager.py         # TradeManager
│   ├── execution_timing.py      # ExecutionTiming
│   ├── risk_engine.py           # RiskEngine
│   ├── order_strategy.py        # OrderStrategyEngine
│   ├── trade_scorer.py          # TradeScorer
│   ├── execution_simulator.py   # ExecutionSimulator
│   ├── execution_planner.py     # ExecutionPlanner (orchestrator)
│   ├── execution_database.py    # ExecutionDatabase (SQLite)
│   ├── execution_dashboard.py   # ExecutionDashboard (8 charts)
│   └── execution_api.py         # ExecutionApi (16 endpoints)
├── tests/unit/
│   └── test_execution.py        # 130+ tests
├── conf/execution/
│   └── execution.yaml
└── docs/execution/
    ├── execution_intelligence_guide.md
    ├── risk_engine_guide.md
    ├── sizing_guide.md
    ├── trade_optimization_guide.md
    ├── developer_guide.md
    └── api_documentation.md
```

## Dependencies

**Runtime:** Python 3.9+ stdlib only + `loguru` (optional, falls back to `logging`)
**No torch, no numpy, no pandas** — pure analytics layer.

## Design Rules

### 1. No Trade Execution
No broker API calls, no order placement, no live market data connections. All outputs are recommendation dicts.

### 2. Persistent Connection for :memory: SQLite
`ExecutionDatabase` uses `self._persistent_conn` for `:memory:` databases. A new `sqlite3.connect(":memory:")` always creates a fresh empty database — reusing one connection avoids this.

### 3. Renormalization After Budget Enforcement
The `PositionSizer` enforces `min_lot` and `max_lot` constraints that may change the raw calculated size. This is validated against broker constraints, not renormalized (unlike portfolio weights) — the sizing result reflects the actual lot count after constraints.

### 4. Weight Normalization in Scorer
`TradeScorer` weights sum to exactly 1.0:
```
0.20 + 0.20 + 0.15 + 0.10 + 0.10 + 0.10 + 0.10 + 0.05 = 1.00
```

### 5. Module Load Order (for spec_from_file_location tests)

```
execution_types → entry_zones → entry_quality
→ stop_loss_engine → take_profit_engine → risk_engine
→ position_sizer → trade_manager → execution_timing
→ order_strategy → trade_scorer → execution_simulator
→ execution_planner → execution_database
→ execution_dashboard → execution_api
```

## Running Tests

```bash
python3 -m unittest tests/unit/test_execution.py -v
```

Expected: **130+ tests, 0 failures**.

## Adding a New Execution Engine

1. Create `trainer/execution/my_engine.py` with a dataclass result and engine class.
2. Add it to `execution_planner.py` — instantiate in `__init__`, call in `plan()`.
3. Add the result to `ExecutionPlan`.
4. Export from `__init__.py`.
5. Add an API endpoint to `execution_api.py`.
6. Add tests to `test_execution.py`.

## Testing Pattern

All tests use `importlib.util.spec_from_file_location` to avoid triggering the torch import chain from `trainer/__init__.py`. `unittest discover` will show "0 tests" for these files — this is expected and correct.

```python
_BASE = Path(__file__).resolve().parents[2] / "trainer" / "execution"

def _load(mod_key, filename):
    path = _BASE / filename
    spec = importlib.util.spec_from_file_location(mod_key, str(path))
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    spec.loader.exec_module(mod)
    return mod
```
