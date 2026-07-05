# Self-Evaluation Guide — Phase 21.6

## Overview

The Self-Evaluation Engine generates periodic institutional AI self-assessments. It analyses the full decision history and produces structured reports covering strengths, weaknesses, common mistakes, failure modes, best/worst conditions, and improvement opportunities.

This system does **not** retrain models automatically. It produces advisory assessments for human review and long-term quality management.

## Assessment Structure

```
SelfAssessment
├── period                  # Evaluation period label (MONTHLY, Q1-2025, etc.)
├── total_decisions         # Total decisions in scope
├── decisions_with_outcome  # Completed decisions
├── overall_win_rate        # Fraction of positive outcomes
├── overall_quality_score   # Average DecisionScore across scored decisions
├── strengths               # List[str] — what is working well
├── weaknesses              # List[str] — what needs improvement
├── common_mistakes         # List[str] — systematic errors
├── failure_modes           # List[str] — recurring failure patterns
├── best_conditions         # List[str] — where performance is strongest
├── worst_conditions        # List[str] — where performance is weakest
├── opportunities           # List[str] — specific improvement actions
├── grade                   # EXCELLENT | GOOD | ACCEPTABLE | POOR | REJECT
└── summary                 # One-line overall summary
```

## Running Self-Evaluation

```python
from trainer.memory import MemoryAPI

api = MemoryAPI(db_path="decisions.db")

# Current period assessment
assessment = api.self_evaluate(period="CURRENT")
print(assessment["grade"])        # GOOD
print(assessment["win_rate"])     # 0.62
print(assessment["strengths"])    # ["Win rate of 62% exceeds threshold", ...]
print(assessment["opportunities"]) # ["Reduce SL hit rate by ...", ...]
```

## Evaluation Periods

| Period | Description |
|---|---|
| CURRENT | All time (default) |
| MONTHLY | Last 30 days |
| QUARTERLY | Last 91 days |
| ANNUAL | Last 365 days |
| LIFETIME | All decisions ever recorded |

## Interpretation Guide

### Grade Thresholds

| Grade | Quality Score | Win Rate Typical |
|---|---|---|
| EXCELLENT | ≥ 0.85 | > 70% |
| GOOD | ≥ 0.70 | > 60% |
| ACCEPTABLE | ≥ 0.55 | > 48% |
| POOR | ≥ 0.40 | > 38% |
| REJECT | < 0.40 | < 38% |

### Strengths Generated

The engine identifies the following as strengths:
- Win rate above 60% threshold
- TP hit rate above 50%
- High-confidence decisions winning at > 65%
- Session win rate above 60%

### Weaknesses Generated

The engine flags:
- Win rate below 45%
- SL hit rate above 40%
- Sessions with < 40% win rate
- Overconfidence: > 20% of high-confidence (>80%) decisions are losers

### Common Mistakes

The engine flags:
- Taking low-confidence (<55%) entries that result in losses
- Repeated incorrect directional forecasts

### Failure Modes

The engine flags:
- ≥ 2 missed opportunities (NO_TRADE when market moved favourably)
- ≥ 2 decisions disrupted by unexpected market events

### Improvement Opportunities

Always generated:
- SL sizing adjustment if SL hit rate > 35%
- Confidence threshold increase if win rate < 55%
- Sample size note if < 20 completed decisions
- Counterfactual TP review recommendation

## Example Assessment Output

```python
{
    "period": "MONTHLY",
    "total_decisions": 45,
    "decisions_with_outcome": 38,
    "overall_win_rate": 0.63,
    "overall_quality_score": 0.71,
    "strengths": [
        "Win rate of 63% exceeds 60% threshold",
        "TP hit rate of 55% — good target selection",
        "High-confidence decisions win at 72% — confidence is well-calibrated"
    ],
    "weaknesses": [
        "NY session: only 38% win rate — consistently weak"
    ],
    "common_mistakes": [
        "5 low-confidence entries that resulted in losses"
    ],
    "failure_modes": [
        "No recurring failure modes identified"
    ],
    "best_conditions": [
        "TRENDING_UP regime: 70% win rate",
        "LONDON session: 68% win rate"
    ],
    "worst_conditions": [
        "NY session: only 38% win rate"
    ],
    "opportunities": [
        "Increase minimum confidence threshold from 0.55 to 0.65 before entry",
        "Review counterfactual analysis for systematic improvements to TP sizing"
    ],
    "grade": "GOOD",
    "summary": "Self-evaluation [MONTHLY]: 45 decisions, 38 completed, win_rate=63%, quality=0.71, grade=GOOD."
}
```

## Long-Term Learning Timeline

```python
# Track improvement across months
timeline = api.get_dashboard("learning_timeline")
# Returns monthly win_rate and avg_quality:
# {
#   "2025-01": {"avg_quality": 0.62, "win_rate": 0.55, "total": 18},
#   "2025-02": {"avg_quality": 0.67, "win_rate": 0.58, "total": 22},
#   "2025-03": {"avg_quality": 0.71, "win_rate": 0.63, "total": 38},
# }
```

## Pattern Discovery Integration

Self-evaluation works alongside Pattern Discovery to provide a complete picture:

```python
# Discover patterns first
patterns = api.discover_patterns()
for p in patterns["patterns"]:
    if p["direction"] == "NEGATIVE":
        print(f"Weak pattern: {p['description']} — win_rate={p['win_rate']:.0%}")

# Then evaluate
assessment = api.self_evaluate()
print(assessment["worst_conditions"])
```
