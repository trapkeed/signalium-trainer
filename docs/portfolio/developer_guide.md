# Developer Guide — Phase 20 Portfolio Intelligence

## Package Structure

```
signalium-trainer/
├── trainer/
│   └── portfolio/
│       ├── __init__.py                # 60 exports
│       ├── portfolio_instruments.py   # InstrumentRegistry, ModelForecast
│       ├── correlation_engine.py      # 6 correlation methods
│       ├── exposure_engine.py         # Asset class / currency / region exposure
│       ├── dependency_graph.py        # Cluster detection, centrality
│       ├── regime_detector.py         # Market regime detection
│       ├── risk_budget.py             # Budget enforcement
│       ├── conflict_resolver.py       # Signal conflict resolution
│       ├── portfolio_scorer.py        # Portfolio metrics
│       ├── position_allocator.py      # Allocation engine
│       ├── scenario_analysis.py       # Macro stress testing
│       ├── optimizer.py               # Portfolio optimization
│       ├── portfolio_database.py      # SQLite persistence
│       ├── portfolio_api.py           # Unified API
│       └── portfolio_dashboard.py     # Visualisation data
├── tests/
│   └── unit/
│       └── test_portfolio.py          # 130 tests
├── conf/
│   └── portfolio/
│       └── portfolio.yaml             # Full YAML configuration
└── docs/
    └── portfolio/
        ├── portfolio_intelligence_guide.md
        ├── correlation_engine_guide.md
        ├── optimization_guide.md
        ├── scenario_analysis_guide.md
        ├── developer_guide.md          ← this file
        └── api_documentation.md
```

---

## Design Principles

1. **Zero model changes** — Phase 20 never touches forecasting models.
2. **Zero new dependencies** — stdlib + loguru only.
3. **Pure analytics** — consumes `ModelForecast` objects, produces recommendations.
4. **Full backward compatibility** — no existing package modified.
5. **SQLite persistence** — all results stored, including `:memory:` support for testing.

---

## Adding a New Instrument

```python
from trainer.portfolio import InstrumentRegistry, InstrumentInfo, AssetClass, Region

registry = InstrumentRegistry.default()
registry.register(InstrumentInfo(
    symbol="USDMXN",
    asset_class=AssetClass.FOREX,
    base_currency="USD",
    quote_currency="MXN",
    region=Region.LATIN_AMERICA,
    description="US Dollar / Mexican Peso",
    tags=["usd", "mxn", "emerging"],
))
```

---

## Adding a New Conflict Rule

Edit `trainer/portfolio/conflict_resolver.py`:

```python
_DIRECT_CONFLICTS.append((
    "EURUSD", "USDJPY",
    "EUR/USD long = USD short; USD/JPY long = USD long → USD conflict"
))
```

---

## Testing Pattern

All portfolio tests use `spec_from_file_location` to avoid importing torch
through the `trainer/__init__.py` chain:

```python
def _load(mod_key: str, filename: str):
    path = _BASE / filename
    spec = importlib.util.spec_from_file_location(mod_key, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    spec.loader.exec_module(mod)
    return mod
```

Always stub loguru before loading any module:

```python
import types
_loguru = types.ModuleType("loguru")
class _Logger:
    def info(self, *a, **k): pass
    # ...
_loguru.logger = _Logger()
sys.modules.setdefault("loguru", _loguru)
```

---

## Module Load Order

Modules must be loaded in dependency order (each before its dependents):

```
portfolio_instruments
→ correlation_engine
→ exposure_engine
→ dependency_graph
→ regime_detector
→ risk_budget
→ conflict_resolver
→ portfolio_scorer
→ position_allocator
→ scenario_analysis
→ optimizer
→ portfolio_database
→ portfolio_dashboard
→ portfolio_api
```

---

## Running Tests

```bash
cd signalium-trainer
python3 -m pytest tests/unit/test_portfolio.py -v
```
