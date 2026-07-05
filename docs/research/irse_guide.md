# Institutional Research Scoring Engine (IRSE)

## Overview

The IRSE produces a single, reproducible **Research Score (0–100)** for every
experiment, model, dataset, feature set, validation run, and production
deployment in the Signalium ecosystem.

The Research Score is the official ranking metric.  It replaces accuracy as the
primary quality signal and drives all promotion, deployment, and champion-change
decisions.

---

## Research Score Formula

```
Research Score = Σ (category_score_i × weight_i)
```

Where each `category_score_i` is in [0, 100] and `Σ weight_i = 1.0`.

### Default Weights

| Category              | Weight | Rationale |
|-----------------------|--------|-----------|
| Validation Quality    |  25%   | Walk-forward fidelity is the strongest predictor of live performance |
| Trading Performance   |  25%   | Profit factor, Sharpe, drawdown — directly tied to capital efficiency |
| Robustness            |  15%   | Monte Carlo and stress tests reveal hidden fragility |
| Calibration           |  10%   | Miscalibrated confidence is operationally dangerous |
| Generalization        |  10%   | OOS generalisation prevents overfitting to the training distribution |
| Efficiency            |   5%   | Latency and resource cost matter at production scale |
| Explainability        |   5%   | Regulatory and audit requirements |
| Operational Readiness |   5%   | Monitoring, rollback, and health check completeness |

---

## Grade System

| Grade              | Score Range | Promotion | Deployment |
|--------------------|-------------|-----------|------------|
| Institutional Elite | 95–100      | ✓         | ✓          |
| Institutional       | 90–94       | ✓         | ✓          |
| Production Ready    | 85–89       | ✓         | ✓          |
| Candidate           | 80–84       | ✓         | ✗          |
| Experimental        | 70–79       | ✗         | ✗          |
| Research Only       | < 70        | ✗         | ✗          |

Grades and thresholds are configurable in `conf/research/scoring.yaml`.

---

## Quick Start

```python
from trainer.research.scorer import ResearchScorer, ResearchScoreInputs
from trainer.research.categories import (
    ValidationQualityInputs,
    TradingPerformanceInputs,
    RobustnessInputs,
)

scorer = ResearchScorer()

inputs = ResearchScoreInputs(
    name="my_transformer_v3",
    experiment_id="exp_2026_001",
    validation=ValidationQualityInputs(
        walk_forward_accuracy=0.692,
        n_folds=12,
        cross_regime_stability=0.022,
        consistency_score=0.85,
    ),
    trading=TradingPerformanceInputs(
        sharpe_ratio=1.82,
        max_drawdown=0.067,
        win_rate=0.573,
        profit_factor=1.65,
    ),
    robustness=RobustnessInputs(
        monte_carlo_stability=0.87,
        stress_test_pass_rate=0.88,
    ),
    instrument="ES",
    timeframe="1h",
    backbone="transformer",
)

result = scorer.score(inputs)
print(result.summary())
# Research Score: 82.4/100  [Candidate]
# Promotion: YES
# Confidence: 78%
```

---

## Full IRSE Pipeline

```python
from trainer.research.scorer import ResearchScorer
from trainer.research.leaderboard_irse import IRSELeaderboard
from trainer.research.trends import ScoreTrendTracker
from trainer.research.benchmarking import IRSEBenchmarker, ModelRole
from trainer.research.score_database import IRSEScoreDatabase
from trainer.research.score_api import IRSEApi
from trainer.research.score_dashboard import IRSEScoreDashboard
from trainer.research.score_export import IRSEScoreExporter

# Build the system
scorer  = ResearchScorer()
lb      = IRSELeaderboard()
trends  = ScoreTrendTracker()
bm      = IRSEBenchmarker()
db      = IRSEScoreDatabase("data/research/irse_scores.db")

api     = IRSEApi(scorer, lb, trends, bm, db)
dash    = IRSEScoreDashboard(lb, trends, bm, db)
exporter = IRSEScoreExporter(lb, trends, bm, db)

# Score an experiment
response = api.score_experiment(inputs)
print(response["data"]["grade_label"])

# Leaderboard
lb_report = api.get_leaderboard(axis="research_score", limit=20)

# Benchmark
api.register_benchmark("current_champion", champion_rs, role=ModelRole.CHAMPION)
api.register_benchmark("new_candidate",    candidate_rs, role=ModelRole.CANDIDATE)
comparison = api.compare_experiments("new_candidate", "current_champion")

# Export
html = exporter.export_html(top_n=20)
csv  = exporter.export_csv(top_n=100)

# Dashboards
all_dashboards = dash.build_all(top_n=10, sample_experiment_id="exp_2026_001")
```

---

## Confidence Score

Each Research Score includes an independent **Confidence Score** (0–1) that
reflects how much trust to place in the composite score.

| Level     | Confidence |
|-----------|-----------|
| Very High | ≥ 90%     |
| High      | 75–89%    |
| Moderate  | 55–74%    |
| Low       | 35–54%    |
| Very Low  | < 35%     |

Factors affecting confidence:
- Number of validation folds (target: 12+)
- Number of trade samples (target: 20,000+)
- Regime coverage (target: 80%+)
- Instrument diversity (target: 5+)
- Statistical p-value (target: ≤ 0.01)
- Fold score standard deviation (target: ≤ 2%)
- Data quality score
- Reproducibility score

---

## Partial Inputs

All category input dataclasses have sensible defaults (zeros).
You may provide only the metrics you have — unset inputs score 0 on that
sub-metric but do not raise errors.

```python
# Minimal — only what you have
inputs = ResearchScoreInputs(
    name="quick_check",
    trading=TradingPerformanceInputs(sharpe_ratio=1.5, max_drawdown=0.10),
)
result = scorer.score(inputs)
# Scores all other categories at 0 — composite will be low
```

---

## Next Steps

- [Scoring Methodology](irse_methodology.md)
- [Weight Configuration Guide](irse_weights_guide.md)
- [Benchmark Guide](irse_benchmark_guide.md)
- [Developer Guide](developer_guide_phase19_2.md)
- [API Documentation](irse_api_docs.md)
