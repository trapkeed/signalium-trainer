# HRSA Developer Guide — Phase 19.3

## Architecture

```
trainer/research/
├── raw_metrics.py          # Level 1: RawMetricsRegistry, RawMetricsBundle
├── normalization.py        # Level 1→2: Normaliser, CategoryNormaliser
├── hrsa_categories.py      # Level 2: HRSACategoryScorer, CategoryResult
├── hrsa_grades.py          # Grade engine (institutional, research, promotion)
├── hrsa_scorer.py          # Level 3: HierarchicalResearchScorer
├── hrsa_benchmarks.py      # Benchmark engine
├── hrsa_trends.py          # Trend tracking and regression detection
├── hrsa_explanations.py    # Score explainer and recommendations
├── hrsa_whatif.py          # What-if simulation
├── hrsa_database.py        # SQLite persistence (8 tables)
├── hrsa_api.py             # Unified API entry point
└── hrsa_dashboard.py       # 8 visualisation types
```

## Module Dependency Graph

```
raw_metrics ──────────────────────────────────┐
normalization ────────────────────────────────┤
                                              ↓
hrsa_categories (uses raw_metrics, norm)  ──→ hrsa_scorer
hrsa_grades                               ──→ hrsa_scorer
                                              ↓
hrsa_scorer ──→ hrsa_benchmarks (consumes HierarchicalScore)
           ──→ hrsa_trends
           ──→ hrsa_explanations
           ──→ hrsa_whatif
           ──→ hrsa_database
           ──→ hrsa_api (wraps all)
           ──→ hrsa_dashboard
```

## Hierarchical Flow Contract

**Rule**: Level 3 must be computed from Level 2 scores only.
No raw metric values may bypass the category layer.

In code: `HierarchicalResearchScorer._aggregate()` receives only
`Dict[str, CategoryResult]` — never the raw bundle directly.

## Adding a New Metric

1. Add to `_DEFAULT_METRICS` in `raw_metrics.py`:
```python
dict(name="my_new_metric", direction="higher", category="trading_performance",
     unit="ratio", lo=0.0, hi=5.0, description="My new metric"),
```

2. Add a `MetricMapping` to the relevant category in `hrsa_categories.py`:
```python
CategoryDefinition(
    name="trading_performance",
    ...
    metric_mappings=[
        ...
        MetricMapping("my_new_metric", weight=1.5),
    ],
)
```

3. No other changes needed. The scoring pipeline auto-discovers registered metrics.

## Adding a New Category

1. Add metric definitions to `_DEFAULT_METRICS` with the new `category` value.
2. Add a `CategoryDefinition` to `_default_categories()` in `hrsa_categories.py`.
3. Update `category_weights` in `conf/research/hrsa.yaml` (weights must sum to 1.0).
4. Re-normalise the weight distribution across all categories.

## Thread Safety

All scorers are stateless. `HierarchicalResearchScorer.score()` has no
shared mutable state — it is safe to call concurrently from multiple threads.

The `HRSADatabase` uses SQLite with WAL mode, which supports concurrent reads.
One writer at a time. For multi-process environments, use a dedicated DB process.

## Extending the Grade System

```python
from trainer.research import HRSAGradeEngine

engine = HRSAGradeEngine.default()
engine.define_custom_grade(
    name="Proprietary Elite",
    min_score=97.0,
    max_score=100.0,
    description="Firm's highest internal standard.",
    color="platinum",
)
```

Custom grades are inserted into the institutional grade table in score order.

## Extending the Normaliser

```python
from trainer.research import Normaliser, NormMethod

def my_log_norm(value: float) -> float:
    import math
    return min(1.0, math.log1p(value) / math.log1p(10))

n = Normaliser()
result = n.normalise(7.3, method=NormMethod.CUSTOM,
                     direction="higher_is_better",
                     custom_fn=my_log_norm)
```

## Reproducibility Contract

Every `HierarchicalScore` is deterministic given the same `RawMetricsBundle`.
No randomness is introduced at any layer. Scorers use pure arithmetic only.

To reproduce a score: save the `RawMetricsBundle.to_dict()` alongside the
`HierarchicalScore.to_dict()` in the database.

## Zero New Runtime Dependencies

Phase 19.3 adds **zero** new runtime dependencies beyond:
- Python 3.8+ stdlib (`dataclasses`, `sqlite3`, `json`, `uuid`, `math`)
- `loguru` (already in the project)

No torch, numpy, scipy, pandas, or matplotlib required.

## Data Flow for a Complete Scoring Run

```python
# 1. Collect (Level 1)
bundle = RawMetricsBundle("exp_001")
bundle.set("sharpe_ratio", 1.82)
# ...

# 2. Category scoring (Level 2)
cat_scorer = HRSACategoryScorer.default()
cat_results = cat_scorer.score_all_categories(bundle)

# 3. Aggregate (Level 3)
scorer = HierarchicalResearchScorer()
hs = scorer.score(bundle, name="my_model")

# 4. Grade
grade = HRSAGradeEngine.default().full_assessment(
    hs.level3_composite,
    confidence=hs.confidence,
    category_scores=hs.category_scores_dict(),
)

# 5. Explain
explainer = HRSAExplainer()
explanation = explainer.explain(hs)

# 6. Simulate
sim = HRSAWhatIfSimulator(scorer=scorer)
what_if = sim.simulate_category(hs, "calibration", new_score=85.0)

# 7. Persist
db = HRSADatabase("data/hrsa.db")
db.save(hs)
db.save_explanation(explanation)

# 8. Visualise
dashboard = HRSADashboard(db=db)
charts = dashboard.build_all(hs)
```

## Testing

```bash
python3 tests/unit/test_hrsa.py
```

The test suite is self-contained: stubs `loguru` and uses `:memory:` SQLite.
No torch/numpy/pytest required.
