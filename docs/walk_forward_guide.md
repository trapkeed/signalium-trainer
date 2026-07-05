# Walk-Forward Validation Guide

## Phase 15 — Signalium Quant

---

## Why Walk-Forward Validation?

Traditional k-fold cross-validation shuffles data randomly — this causes **temporal leakage** in time-series: future data leaks into training, producing artificially inflated evaluation scores.

Walk-forward validation preserves **chronological ordering**:
- Training data always precedes validation data
- No shuffling ever occurs
- Each fold simulates a real deployment scenario

---

## Available Strategies

### 1. Expanding Window (default)

Training set grows on each fold. More historical data = better signal extraction.

```
Fold 0:  |<------- 2000 ------->|<- 500 ->|
Fold 1:  |<----------- 2100 ---------->|<- 500 ->|
Fold 2:  |<--------------- 2200 ------------->|<- 500 ->|
```

```python
from trainer.validation import ExpandingWindowValidator

validator = ExpandingWindowValidator(
    min_train_size=2000,   # first fold training size
    val_size=500,          # validation window
    test_size=100,         # optional hold-out
    step_size=100,         # fold advancement
    gap_size=10,           # purge gap (leakage buffer)
)
```

**Best for:** Most production validation scenarios.

---

### 2. Sliding Window

Fixed training window slides forward. Simulates periodic model retraining on recent data.

```
Fold 0:  |<---- 2000 --->|<- 500 ->|
Fold 1:             |<---- 2000 --->|<- 500 ->|
Fold 2:                        |<---- 2000 --->|<- 500 ->|
```

```python
from trainer.validation import SlidingWindowValidator

validator = SlidingWindowValidator(
    train_size=2000,
    val_size=500,
    step_size=100,
    gap_size=10,
)
```

**Best for:** Models retrained periodically on the most recent N bars.

---

### 3. Anchored Walk-Forward

Training always starts from t=0. Maximises history available at each fold.

```
Fold 0:  |<----- 2000 ----->|<- 500 ->|
Fold 1:  |<------- 2100 ------>|<- 500 ->|
Fold 2:  |<--------- 2200 ------->|<- 500 ->|
```

```python
from trainer.validation import AnchoredWalkForward

validator = AnchoredWalkForward(
    initial_train_size=2000,
    val_size=500,
    step_size=100,
    anchor_start=0,     # always start from index 0
)
```

**Best for:** Models that benefit from maximum training history.

---

### 4. Nested Validation

Two-level validation for hyper-parameter search without look-ahead leakage.

- **Outer loop** — expanding window (simulates real deployment)
- **Inner loop** — sliding window within outer training data (hyper-parameter search)

```
Outer Fold 0:  |<------ Outer Train ------>|<-- Outer Val -->|
                         ↓ Inner splits:
               |<- Inner Train ->|<- Inner Val ->|
               |       |<- Inner Train ->|<- Inner Val ->|
```

```python
from trainer.validation import NestedValidator

validator = NestedValidator(
    outer_min_train=3000, outer_val_size=500, outer_step=200,
    inner_train_size=1000, inner_val_size=250, inner_step=100,
    gap_size=10,
    max_outer_folds=5,
)

for outer_fold in validator.split(x, y):
    # Use outer_fold.inner_folds for hyper-parameter search
    best_params = grid_search(outer_fold.inner_folds)
    # Evaluate on outer_fold.x_outer_val with best_params
```

**Best for:** Hyper-parameter optimisation with proper temporal controls.

---

## WindowFold Data Structure

All strategies yield `WindowFold` objects:

```python
@dataclass
class WindowFold:
    fold_index: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int       # 0 when no test set
    test_end: int         # 0 when no test set
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor  # empty when no test set
    y_test: torch.Tensor
    strategy: str

    @property
    def train_size(self) -> int: ...
    @property
    def val_size(self) -> int: ...
    @property
    def test_size(self) -> int: ...
    @property
    def has_test(self) -> bool: ...
```

---

## Gap / Purge Period

The `gap_size` parameter adds a buffer between the training end and validation start.  This prevents leakage when features use any look-forward smoothing.

```
Without gap:  [...train...][...val...]
With gap_size=10:  [...train...][GAP][...val...]
```

**Rule of thumb:** Set `gap_size` equal to the maximum look-back of any feature that could indirectly leak target information.

---

## Counting Folds

Check how many folds a dataset will produce before running:

```python
validator = ExpandingWindowValidator(
    min_train_size=2000,
    val_size=500,
    step_size=100,
)
n = validator.num_folds(n_samples=10_000)
print(f"Will generate {n} folds")
```

---

## Integration with the Validation Engine

The engine runs the walk-forward pipeline automatically:

```python
from trainer.validation import InstitutionalValidationEngine, EngineConfig

config = EngineConfig(
    strategy="expanding",      # "expanding" | "sliding"
    min_train_size=2000,
    val_size=500,
    step_size=100,
    gap_size=10,
    max_folds=20,              # cap at 20 folds
)
engine = InstitutionalValidationEngine(config=config, ...)
result = engine.validate(model, x, y, model_id="quant_v1")
```

The engine always uses the **last fold** for deep metric evaluation (statistical, trading, calibration, drift) while reporting fold count in `result.n_folds`.
