# AI Research Analyst Guide — Phase 23

## Overview

SAQRA (Signalium AI Quant Research Analyst) is an institutional-grade AI research assistant powered by a configurable LLM with Retrieval-Augmented Generation (RAG).

**Mission:** Help researchers understand Signalium's quantitative platform — why models perform as they do, which experiments succeeded, what changed, and what to try next.

**Absolute constraints:**
- NEVER generates BUY or SELL signals
- NEVER overrides production forecasting
- NEVER replaces deterministic model logic
- ALWAYS grounds responses in retrieved research evidence
- ALWAYS cites experiment, model, dataset, and report IDs

## Quick Start

```python
from trainer.analyst import AnalystAPI, KnowledgeSourceType, ReportType, Role

# Create the analyst (uses MockLLMProvider by default)
api = AnalystAPI(db_path="analyst.db")

# Index your research documents
api.index_document(
    source_type = KnowledgeSourceType.EXPERIMENT,
    artifact_id = "EXP-001",
    title       = "EURUSD Transformer Experiment",
    content     = "Sharpe 2.3, max_drawdown 8.5%, accuracy 0.72, status PASSED. "
                  "Architecture: Transformer. Dataset: DATASET-v18.",
    tags        = ["eurusd", "transformer", "passed"],
    metadata    = {"sharpe": 2.3, "max_drawdown": 8.5, "status": "PASSED"},
)

# Ask a research question
result = api.chat("Why did Model 42 outperform Model 37?", researcher="alice")
print(result["response"]["content"])
print(result["response"]["citations"])
```

## Supported Research Questions

SAQRA can answer questions such as:

| Question | Query Type |
|---|---|
| "Why did Model 42 outperform Model 37?" | COMPARISON |
| "Which feature contributed most to EURUSD?" | ANALYSIS |
| "Why did calibration deteriorate?" | ANALYSIS |
| "Why was this model promoted?" | ANALYSIS |
| "Which experiments failed?" | SEARCH |
| "Which regime is weakest?" | ANALYSIS |
| "What changed between two datasets?" | COMPARISON |
| "What explains the increase in drawdown?" | ANALYSIS |
| "Which features appear redundant?" | ANALYSIS |
| "Which labels are degrading?" | ANALYSIS |
| "Which market regimes require more data?" | RECOMMENDATION |

## Knowledge Sources (20 types)

SAQRA has read-only access to all 20 knowledge source types:

```
RESEARCH_REPORT, EXPERIMENT, TRAINING_RUN, VALIDATION_REPORT,
WALK_FORWARD, MODEL_REGISTRY, RESEARCH_SCORE, FEATURE_IMPORTANCE,
DATASET_VERSION, DECISION_MEMORY, COUNTERFACTUAL, KNOWLEDGE_GRAPH,
CHAMPION_HISTORY, MODEL_COMPARISON, PORTFOLIO_ANALYTICS,
EXECUTION_ANALYTICS, MARKET_MEMORY, LIFECYCLE_ANALYTICS,
DRIFT_REPORT, AUDIT_LOG
```

## Security Model

| Role | Permissions |
|---|---|
| ADMIN | READ, EXPORT, SUMMARIZE, COMPARE, AUDIT |
| RESEARCHER | READ, SUMMARIZE, COMPARE |
| VIEWER | READ only |

```python
from trainer.analyst import Role, Permission, has_permission
assert has_permission(Role.RESEARCHER, Permission.COMPARE)  # True
assert not has_permission(Role.VIEWER, Permission.SUMMARIZE)  # True
```

## Report Generation

```python
result = api.report(ReportType.WEEKLY_REPORT, period="2025-W27")
result = api.report(ReportType.MODEL_HEALTH)
result = api.report(ReportType.DEPLOYMENT_READINESS)
```

All 12 report types: DAILY_SUMMARY, WEEKLY_REPORT, MONTHLY_REPORT, QUARTERLY_REVIEW, ANNUAL_REPORT, EXECUTIVE_SUMMARY, MODEL_HEALTH, EXPERIMENT_REVIEW, PERFORMANCE_REVIEW, DEPLOYMENT_READINESS, FEATURE_ANALYSIS, RESEARCH_PROGRESS.

## Recommendations

```python
# Ask for recommendations
recs = api.recommend("What should we prioritise for next quarter?")

# Automatically scan knowledge base for recommendation triggers
recs = api.auto_recommend()

# View pending recommendations
pending = api.pending_recommendations()
```

## LLM Provider Switching

```python
from trainer.analyst import LLMProviderType

# Switch to OpenAI (requires OPENAI_API_KEY)
api.switch_provider(LLMProviderType.OPENAI, api_key="sk-...")

# Switch to Anthropic
api.switch_provider(LLMProviderType.ANTHROPIC, api_key="sk-ant-...")

# Switch to local Ollama
api.switch_provider(LLMProviderType.OLLAMA, model_name="llama3")

# Revert to mock for testing
api.switch_provider(LLMProviderType.MOCK)
```

No other code changes are required — all sub-systems update automatically.
