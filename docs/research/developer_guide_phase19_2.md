# Developer Guide — Phase 19.2: IRSE

## Architecture

```
trainer/research/
├── weights.py          ScoringWeights — configurable weight system
├── grades.py           GradeSystem, Grade, GradeResult
├── categories.py       8 category input dataclasses + scorers
├── confidence.py       ConfidenceCalculator, ConfidenceInputs/Result
├── scorer.py           ResearchScorer, ResearchScoreInputs, ResearchScore
├── trends.py           ScoreTrendTracker, ScoreSnapshot, ChampionChange
├── leaderboard_irse.py IRSELeaderboard, IRSELeaderboardEntry
├── benchmarking.py     IRSEBenchmarker, BenchmarkRecord, BenchmarkComparison
├── score_database.py   IRSEScoreDatabase — SQLite WAL, 6 tables
├── score_api.py        IRSEApi — 30+ structured endpoints
├── score_dashboard.py  IRSEScoreDashboard — 7 dashboard types
└── score_export.py     IRSEScoreExporter — JSON/CSV/HTML/Markdown
```

---

## Data Flow

```
ResearchScoreInputs
    │
    ▼
ResearchScorer.score()
    ├── ValidationQualityScorer.score(inputs.validation)    → (score, sub_scores)
    ├── TradingPerformanceScorer.score(inputs.trading)       → (score, sub_scores)
    ├── RobustnessScorer.score(inputs.robustness)            → (score, sub_scores)
    ├── CalibrationScorer.score(inputs.calibration)          → (score, sub_scores)
    ├── GeneralizationScorer.score(inputs.generalization)    → (score, sub_scores)
    ├── EfficiencyScorer.score(inputs.efficiency)            → (score, sub_scores)
    ├── ExplainabilityScorer.score(inputs.explainability)    → (score, sub_scores)
    ├── OperationalReadinessScorer.score(inputs.operational) → (score, sub_scores)
    ├── ConfidenceCalculator.compute(inputs.confidence)      → ConfidenceResult
    └── GradeSystem.assign(composite)                        → GradeResult
    │
    ▼
ResearchScore
    ├── composite: float (0–100)
    ├── grade_label: str
    ├── promotion_eligible: bool
    ├── deployment_eligible: bool
    ├── category_scores: List[CategoryScore]
    └── confidence: ConfidenceResult
    │
    ▼
IRSEApi.score_experiment()
    ├── db.save_research_score()
    ├── lb.add(rs)
    └── trends.record_from_research_score(rs)
```

---

## Adding a New Category

1. Add an `inputs` dataclass to `categories.py`:

```python
@dataclass
class MyNewCategoryInputs:
    my_metric: float = 0.0
    another_metric: float = 0.0
```

2. Add a scorer class:

```python
class MyNewCategoryScorer:
    def score(self, inputs: MyNewCategoryInputs) -> Tuple[float, Dict[str, float]]:
        sub = {
            "my_metric":      _linear(inputs.my_metric, 0.0, 100.0),
            "another_metric": _clamp(inputs.another_metric * 100.0),
        }
        composite = _weighted_avg([
            (sub["my_metric"],      0.60),
            (sub["another_metric"], 0.40),
        ])
        return _clamp(composite), sub
```

3. Add the new weight to `ScoringWeights` in `weights.py`.

4. Add the category name to `ALL_CATEGORIES` in `weights.py`.

5. Add the field to `ResearchScoreInputs` in `scorer.py`.

6. Register the scorer and input mapping in `ResearchScorer.__init__()`.

7. Update `conf/research/scoring.yaml` with the new weight.

---

## Adding a New Sub-metric

Within an existing category:

1. Add the field to the `*Inputs` dataclass (with a default).
2. Compute the sub-score in the scorer's `score()` method.
3. Add to `_weighted_avg()` pairs — adjust weights to keep the sum ≤ 1.
4. The composite category score automatically reflects the change.

---

## Normalisation Patterns

```python
from trainer.research.categories import _linear, _inverse_linear, _clamp, _weighted_avg

# Higher is better: [lo, hi] → [0, 100]
score = _linear(value, lo=0.5, hi=2.5)

# Lower is better: [lo, hi] → [100, 0]
score = _inverse_linear(value, lo=0.0, hi=0.10)

# Direct fraction: 0–1 → 0–100
score = _clamp(fraction * 100.0)

# Weighted average
composite = _weighted_avg([
    (score_a, 0.60),
    (score_b, 0.40),
])
```

---

## Thread Safety

- `IRSEScoreDatabase` uses a `threading.Lock` around all SQLite operations.
- `ResearchScorer` is stateless — safe to call from any thread.
- `IRSELeaderboard` and `ScoreTrendTracker` are **not** thread-safe.
  Use one instance per thread or wrap in a lock if needed.

---

## Testing

```bash
# Syntax check all Phase 19.2 modules
for f in trainer/research/{weights,grades,categories,confidence,scorer,trends,leaderboard_irse,benchmarking,score_database,score_api,score_dashboard,score_export}.py; do
  python3 -c "import ast; ast.parse(open('$f').read())" && echo "OK: $f"
done

# Run test suite (no external deps)
python3 tests/unit/test_research_scoring.py
```

The test runner is self-contained: it loads all modules via `importlib`
bypassing `__init__.py` (which requires `torch`), stubs `loguru` if not
installed, and reports pass/fail to stdout.

---

## Database Schema

```sql
research_scores          -- one row per scored experiment
category_scores          -- 8 rows per research_score (FK cascade delete)
score_history            -- time-series snapshots for trend tracking
weight_configurations    -- named weight presets
benchmark_records        -- registered models for benchmarking
benchmark_comparisons    -- results of pairwise comparisons
```

---

## Zero New Runtime Dependencies

Phase 19.2 adds **zero new runtime dependencies**.

Runtime: Python stdlib only (`sqlite3`, `csv`, `io`, `json`, `uuid`,
`threading`, `dataclasses`, `typing`) + existing `loguru`.

No `torch`, `numpy`, `matplotlib`, `plotly`, `pandas`, or any other
third-party package.
