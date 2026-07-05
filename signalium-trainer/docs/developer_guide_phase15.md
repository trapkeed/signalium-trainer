# Developer Guide — Phase 15

## Institutional Research Validation Platform

---

## Overview

This guide is for developers extending or integrating with the Phase 15 Institutional Validation Platform. It covers the internal architecture, extension points, design decisions, and testing patterns.

---

## Package Structure

```
trainer/validation/
├── __init__.py                  # Public API surface (60+ exports)
├── engine.py                    # InstitutionalValidationEngine (orchestrator)
├── windows.py                   # WindowFold + 4 walk-forward strategies
├── leakage.py                   # LeakageDetector + LeakageReport
├── regime.py                    # MarketRegimeClassifier + RegimeDistribution
├── instrument.py                # INSTRUMENTS, TIMEFRAMES, MultiAssetValidator
├── statistical.py               # StatisticalMetricsEngine (17 metrics, numpy)
├── trading.py                   # TradingMetricsEngine (21 metrics) + TradeSimulator
├── monte_carlo.py               # MonteCarloValidator (numpy random)
├── robustness.py                # RobustnessTester (7 scenarios)
├── calibration_validator.py     # CalibrationValidator (ECE, MCE)
├── drift.py                     # DriftAnalyzer (PSI, JSD)
├── head_validator.py            # PredictionHeadValidator (16 heads)
├── comparison.py                # ModelComparison (head-to-head)
├── leaderboard.py               # InstitutionalLeaderboard (thread-safe JSON)
├── promotion.py                 # PromotionPolicy + PromotionDecision
├── failure_analysis.py          # FailureAnalyzer + FailureReport
├── database.py                  # ValidationDatabase (SQLite)
├── visualization.py             # VisualizationDataBuilder (11 chart types)
├── export.py                    # ReportExporter (JSON, CSV, HTML, PDF)
└── api.py                       # ValidationAPI (11 endpoints)
```

---

## Design Principles

### 1. Evaluate Only — Never Modify Models

```python
# Always call model.eval() before inference
model.eval()
with torch.no_grad():
    logits = model(x_val)
```

The engine explicitly calls `model.eval()` at the start of every validation run and never calls `.train()` on the input model.

### 2. No External Dependencies Beyond PyTorch and NumPy

All statistical computation uses numpy. No sklearn, scipy, or plotting libraries are required.

```python
# Correct: use numpy directly
import numpy as np
auc = float(np.trapz(tpr, fpr))

# Incorrect: don't import sklearn
from sklearn.metrics import roc_auc_score  # NOT allowed
```

### 3. JSON-Safe Chart Data

All visualization outputs return plain Python dicts:

```python
# All chart methods return dict, not matplotlib figures
data = VisualizationDataBuilder.build_equity_curves(mc_result)
# data is a dict with lists of floats — JSON-safe
json.dumps(data)  # always works
```

### 4. Thread-Safe Storage

Both `InstitutionalLeaderboard` and `ValidationDatabase` use `threading.Lock()`:

```python
class InstitutionalLeaderboard:
    def __init__(self):
        self._lock = threading.Lock()

    def add(self, entry):
        with self._lock:
            self._entries.append(entry)
            self._persist()
```

### 5. SQLite With No Extra Dependencies

```python
import sqlite3  # built-in only

conn = sqlite3.connect(str(self._path), check_same_thread=False)
```

---

## Extending the Platform

### Adding a New Walk-Forward Strategy

1. Implement a class with a `.split(x, y)` method that yields `WindowFold` objects.
2. Set `fold.strategy = "your_strategy_name"`.
3. The engine accepts any strategy object — no subclassing required (duck typing).

```python
class CustomStrategy:
    def split(self, x: torch.Tensor, y: torch.Tensor) -> Iterator[WindowFold]:
        # Your logic here
        yield WindowFold(
            fold_index=0,
            train_start=0, train_end=500,
            val_start=510, val_end=600,
            x_train=x[:500], y_train=y[:500],
            x_val=x[510:600], y_val=y[510:600],
            strategy="custom",
        )
```

Pass it to the engine:

```python
config = EngineConfig(strategy="custom")
engine = InstitutionalValidationEngine(config=config)
# The engine uses ExpandingWindow or SlidingWindow based on config.strategy string.
# To use a custom object, pass it to engine._strategy directly or subclass engine.
```

### Adding a New Stress Scenario

Add to `RobustnessTester._apply_*` method style, then call `_make_result()`:

```python
class RobustnessTester:
    def evaluate(self, model, x_val, y_val):
        # Add after existing scenarios:
        for level in [0.10, 0.20]:
            x_stressed = self._apply_bid_ask_noise(x_val.clone(), level)
            scenarios.append(self._make_result(
                "BidAskNoise", level, baseline_acc, model, x_stressed, y_val
            ))
```

### Adding a New Instrument

```python
from trainer.validation import register_instrument

register_instrument(
    symbol="USDZAR",
    instrument_type="forex",
    pip_size=0.001,
    description="US Dollar / South African Rand",
)
```

### Adding a New Promotion Rule

```python
from trainer.validation import PromotionPolicy, PromotionRule

policy = PromotionPolicy(min_accuracy=0.55)
policy.add_rule(PromotionRule(
    metric="ulcer_index",
    threshold=0.15,
    is_max=True,
    label="Maximum Ulcer Index",
    critical=True,
))
```

### Adding a New Prediction Head

The `PredictionHeadValidator` uses duck-typing:
1. Add the head attribute to your model.
2. Make `model.forward_multi_task(x)` return a dict containing the head output.
3. Register the head in `_HEAD_REGISTRY`:

```python
PredictionHeadValidator._HEAD_REGISTRY["my_custom_head"] = (
    "regression",
    "My custom prediction head",
)
```

---

## Testing Patterns

### Fixtures Used

```python
@pytest.fixture
def small_xy():
    """Standard (600, 10, 8) dataset for fast tests."""
    x = torch.randn(600, 10, 8)
    y = torch.randint(0, 3, (600,))
    return x, y

@pytest.fixture
def simple_model():
    """Minimal 3-class model that works with (N, 10, 8) input."""
    return nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(80, 32),
        nn.ReLU(),
        nn.Linear(32, 3),
    )

@pytest.fixture
def tmp_db(tmp_path):
    """SQLite database in a temporary directory."""
    db = ValidationDatabase(tmp_path / "test.db")
    yield db
    db.close()
```

### Test File Location

```
tests/unit/test_validation_platform.py
```

### Running Tests

```bash
# All Phase 15 tests
pytest tests/unit/test_validation_platform.py -v

# Specific class
pytest tests/unit/test_validation_platform.py::TestPromotionPolicy -v

# Fast tests only (no Monte Carlo, no full engine)
pytest tests/unit/test_validation_platform.py -v -k "not Engine"
```

### Test Organization

Tests follow the module order:

| Class | Module Tested | Test Count |
|-------|--------------|------------|
| TestSlidingWindowValidator | windows.py | 9 |
| TestExpandingWindowValidator | windows.py | 2 |
| TestAnchoredWalkForward | windows.py | 2 |
| TestNestedValidator | windows.py | 1 |
| TestLeakageDetector | leakage.py | 5 |
| TestMarketRegimeClassifier | regime.py | 5 |
| TestInstrumentRegistry | instrument.py | 5 |
| TestStatisticalMetricsEngine | statistical.py | 8 |
| TestTradingMetricsEngine | trading.py | 7 |
| TestTradeSimulator | trading.py | 2 |
| TestMonteCarloValidator | monte_carlo.py | 7 |
| TestRobustnessTester | robustness.py | 4 |
| TestCalibrationValidator | calibration_validator.py | 6 |
| TestDriftAnalyzer | drift.py | 5 |
| TestModelComparison | comparison.py | 5 |
| TestInstitutionalLeaderboard | leaderboard.py | 8 |
| TestPromotionPolicy | promotion.py | 9 |
| TestFailureAnalyzer | failure_analysis.py | 4 |
| TestValidationDatabase | database.py | 8 |
| TestPredictionHeadValidator | head_validator.py | 4 |
| TestVisualizationDataBuilder | visualization.py | 5 |
| TestReportExporter | export.py | 3 |
| TestValidationAPI | api.py | 8 |
| TestInstitutionalValidationEngine | engine.py | 8 |

---

## Module Dependency Graph

```
engine.py
├── windows.py          (strategy selection)
├── leakage.py          (step 1)
├── statistical.py      (step 3)
├── trading.py          (step 4)
├── calibration_validator.py (step 5)
├── head_validator.py   (step 6)
├── monte_carlo.py      (step 7)
├── robustness.py       (step 8)
├── drift.py            (step 9)
├── failure_analysis.py (step 10)
├── promotion.py        (step 11)
├── visualization.py    (step 12)
├── leaderboard.py      (step 13)
├── database.py         (step 14)
├── export.py           (step 15)
└── api.py              (public access layer)

comparison.py
├── statistical.py
└── trading.py

instrument.py
└── windows.py          (strategy duck-typing)

__init__.py
└── (all of the above)
```

---

## Data Flow

```
(x, y) → LeakageDetector → folds (WindowFold list)
                                         │
                        ┌────────────────┤ final fold
                        │                │
                        ▼                ▼
                  x_train,y_train   x_val, y_val
                        │                │
                        │          model.forward()
                        │                │
                        │           probs (N, C)
                        │                │
                        │   ┌────────────┤
                        │   │            │
                        ▼   ▼            ▼
                   DriftAnalyzer    StatisticalMetrics
                                         │
                                    TradeSimulator → trades
                                                        │
                                               TradingMetrics
                                               MonteCarloResult
                                                        │
                                        ┌───────────────┘
                                        │
                                   CalibrationReport
                                   HeadResults
                                   RobustnessReport
                                        │
                                   PromotionDecision
                                   LeaderboardEntry
                                   ValidationRunRecord → SQLite
                                        │
                                   ValidationResult (returned)
```

---

## Backward Compatibility

Phase 15 introduces a completely new package `trainer/validation/`.

**Zero changes** to any existing module:
- `trainer/research/validation/` — unchanged
- `trainer/research/leaderboard/` — unchanged
- `trainer/research/champion/` — unchanged
- `trainer/calibration/` — unchanged
- `trainer/evaluation/` — unchanged
- `trainer/training/` — unchanged
- `trainer/models/` — unchanged

The new `trainer/validation/` package coexists alongside all existing code with no conflicts.

Existing imports continue to work unchanged:

```python
from trainer.research import WalkForwardValidator  # unchanged
from trainer.research import Leaderboard           # unchanged
from trainer.calibration.metrics import compute_ece # unchanged
```

---

## Configuration Reference

```python
@dataclass
class EngineConfig:
    strategy: str = "expanding"          # "expanding" | "sliding"
    min_train_size: int = 500
    val_size: int = 100
    test_size: int = 0
    step_size: int = 50
    gap_size: int = 0                    # purge gap (leakage buffer)
    max_folds: int = 0                   # 0 = unlimited

    n_classes: int = 3
    n_calibration_bins: int = 15

    pip_size: float = 0.0001
    spread_pips: float = 1.5
    sl_pips: float = 20.0
    tp_pips: float = 40.0
    direction_threshold: float = 0.50   # min confidence to enter trade

    n_mc_simulations: int = 500
    mc_seed: int = 42

    robustness_tolerance: float = 0.20  # max acceptable degradation
    robustness_seed: int = 42

    abort_on_leakage: bool = True
    psi_threshold: float = 0.20         # drift detection threshold

    export_formats: list[str] = ["json", "csv", "html"]

    architecture: str = ""
    backbone: str = ""
    phase: str = "15"
    n_parameters: int = 0
    training_time_seconds: float = 0.0
    dataset_version: str = ""
    feature_version: str = ""
    label_version: str = ""
    experiment_id: str = ""
    git_commit: str = ""
    notes: str = ""
```
