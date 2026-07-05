# HRSA — Hierarchical Research Scoring Architecture Guide

## Overview

Phase 19.3 replaces the flat IRSE with a three-level Hierarchical Research
Scoring Architecture (HRSA). Every score flows through a strict hierarchy —
no shortcutting:

```
Level 1  →  Raw Metrics (80+ measurable values)
    ↓
Level 2  →  Category Scores (15 scored dimensions)
    ↓
Level 3  →  Institutional Research Score (0–100)
```

## Quick Start

```python
from trainer.research import (
    RawMetricsBundle, HierarchicalResearchScorer, HRSAApi,
)

# 1. Collect raw metrics (Level 1)
bundle = RawMetricsBundle("experiment_2026_001")
bundle.set("sharpe_ratio", 1.82)
bundle.set("max_drawdown", 0.067)
bundle.set("walk_forward_accuracy", 0.692)
# ... set all available metrics

# 2. Score through the hierarchy (Level 1 → 2 → 3)
scorer = HierarchicalResearchScorer()
result = scorer.score(bundle, name="my_model", instrument="ES")
print(result.summary())

# 3. Or use the full API
api = HRSAApi(db_path="data/hrsa.db")
resp = api.score(bundle, name="my_model", instrument="ES")
```

## The Three Levels

### Level 1 — Raw Metrics

The `RawMetricsRegistry` defines 80+ metrics across 15 categories.
Each metric has:
- **name**: unique identifier
- **direction**: `"higher_is_better"` or `"lower_is_better"`
- **category**: which Level 2 category it feeds
- **normalisation_lo / hi**: expected range for min-max normalisation
- **unit**: measurement unit for documentation

```python
from trainer.research import RawMetricsRegistry

reg = RawMetricsRegistry.default()
defn = reg.get("sharpe_ratio")
print(defn.direction)   # "higher_is_better"
print(defn.category)    # "trading_performance"
```

### Level 2 — Category Scores

15 categories aggregate Level 1 metrics into a scored dimension (0–100):

| Category | Weight |
|----------|--------|
| Validation Quality | 15% |
| Trading Performance | 15% |
| Generalization | 10% |
| Robustness | 10% |
| Risk Management | 8% |
| Calibration | 7% |
| Data Quality | 5% |
| Efficiency | 5% |
| Explainability | 5% |
| Operational Readiness | 5% |
| Feature Quality | 4% |
| Model Stability | 4% |
| Research Reproducibility | 4% |
| Security | 2% |
| Compliance | 1% |

Each category score includes:
- **score** (0–100)
- **confidence** (0–1): based on metric coverage, count, and variance
- **grade**: Exceptional / Strong / Good / Adequate / Weak / Critical
- **weighted_score**: score × category weight

### Level 3 — Institutional Research Score

Computed as a coverage-adjusted weighted mean of Level 2 category scores:

```
composite = Σ (category_score_i × effective_weight_i) / Σ effective_weight_i

effective_weight_i = category_weight_i × coverage_i
```

Coverage-adjustment ensures categories with partial data have less influence.

## Result Structure

```python
result = scorer.score(bundle, name="my_model")

result.level3_composite        # 0–100 overall score
result.confidence              # 0–1 weighted confidence
result.institutional_grade     # "Production Ready", etc.
result.research_grade          # "High Quality Research", etc.
result.promotion_recommendation # "GO" | "REVIEW" | "HOLD" | "NO"
result.risk_rating             # "LOW" | "MEDIUM" | "HIGH" | "VERY_HIGH"
result.future_potential        # "STRONG" | "MODERATE" | "LIMITED" | "DECLINING"

# Category breakdown
for cat, cat_result in result.level2_categories.items():
    print(f"{cat}: {cat_result.score:.1f}/100 [{cat_result.grade}]")

# Level 1 summary
print(f"Metrics provided: {result.level1_metric_count}")
```

## Normalisation

Default normalisation is **Min-Max** per category, using the registered
`normalisation_lo` and `normalisation_hi` bounds for each metric.

```python
from trainer.research import Normaliser, NormMethod

n = Normaliser()
score = n.normalise(1.82, lo=0.5, hi=2.5,
                    method=NormMethod.MIN_MAX, direction="higher_is_better")
print(score.score_100)  # e.g. 66.0
```

Supported methods: `min_max`, `z_score`, `percentile`, `rank_scaling`,
`robust_scaling`, `custom`.

## Grades

### Institutional Grade
| Score | Grade |
|-------|-------|
| ≥95 | Institutional Elite |
| ≥90 | Institutional |
| ≥85 | Production Ready |
| ≥80 | Candidate |
| ≥70 | Experimental |
| <70 | Research Only |

### Research Grade
Based on composite × confidence:

| Adjusted Score | Grade |
|----------------|-------|
| ≥90 | Exceptional Research |
| ≥80 | High Quality Research |
| ≥65 | Standard Research |
| ≥50 | Preliminary Research |
| <50 | Insufficient Research |

## Promotion Rules

| Score | Condition | Recommendation |
|-------|-----------|----------------|
| ≥85 + grade rank ≤2 | No critical weaknesses | GO |
| ≥85 + grade rank ≤2 | Critical category <50 | REVIEW |
| ≥80 | — | REVIEW |
| ≥70 | — | HOLD |
| <70 | — | NO |

Critical categories: `trading_performance`, `robustness`, `risk_management`.
