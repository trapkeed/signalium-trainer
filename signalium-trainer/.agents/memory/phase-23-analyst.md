---
name: Phase 23 SAQRA analyst package
description: Design rules and gotchas for the trainer/analyst/ SAQRA RAG system.
---

## The Package

`trainer/analyst/` — 15 modules (+ __init__), stdlib + loguru only.
Institutional AI Research Analyst powered by configurable LLM + RAG.
Never generates BUY/SELL signals. Never overrides production.

## Module Load Order (for spec_from_file_location tests)

```
analyst_types → analyst_models → llm_provider → knowledge_index →
retrieval_engine → prompt_engine → citation_engine →
research_chat → research_search → model_comparator →
report_generator → recommendation_engine →
analyst_database → analyst_dashboard → analyst_api
```

## LLM Providers (6)

MockLLMProvider (fully functional, default), OpenAIProvider, AnthropicProvider,
GeminiProvider, OllamaProvider, AzureOpenAIProvider.
Hot-swap via `api.switch_provider(LLMProviderType.X)` — updates all 5 sub-systems atomically.

## Safety Critical Gotcha

`MockLLMProvider._SAFETY_DISCLAIMER` MUST NOT contain words from PROHIBITED_PATTERNS
(BUY, SELL, LONG, SHORT, etc.) — the disclaimer is appended to every response,
and `is_safe_response()` scans the full response including the disclaimer.
If the disclaimer triggers safety, the ENTIRE response is replaced with a safety notice.

## Test Helper Gotcha

`make_chunk()` positional arg 1 is `source_type: KnowledgeSourceType`, NOT `artifact_id`.
Always use keyword args: `make_chunk(artifact_id="EXP-001")`.

## Source Type Inference Order

In `ResearchSearch._infer_source_type()`, DATASET must be checked before EXPERIMENT.
Queries like "find experiments using Dataset v18" mention both — if experiment is
checked first, it returns EXPERIMENT instead of DATASET_VERSION.

## Knowledge Sources (20)

RESEARCH_REPORT, EXPERIMENT, TRAINING_RUN, VALIDATION_REPORT, WALK_FORWARD,
MODEL_REGISTRY, RESEARCH_SCORE, FEATURE_IMPORTANCE, DATASET_VERSION,
DECISION_MEMORY, COUNTERFACTUAL, KNOWLEDGE_GRAPH, CHAMPION_HISTORY,
MODEL_COMPARISON, PORTFOLIO_ANALYTICS, EXECUTION_ANALYTICS, MARKET_MEMORY,
LIFECYCLE_ANALYTICS, DRIFT_REPORT, AUDIT_LOG. All read-only.

## Security Model

ADMIN=all, RESEARCHER=READ+SUMMARIZE+COMPARE, VIEWER=READ.
Permission denied messages say "permission required" not "denied" — test assertions
must check for "permission" OR "denied" in content.lower().

## Report Types (12)

DAILY_SUMMARY, WEEKLY_REPORT, MONTHLY_REPORT, QUARTERLY_REVIEW, ANNUAL_REPORT,
EXECUTIVE_SUMMARY, MODEL_HEALTH, EXPERIMENT_REVIEW, PERFORMANCE_REVIEW,
DEPLOYMENT_READINESS, FEATURE_ANALYSIS, RESEARCH_PROGRESS.

## Citation Format

`[TYPE:ID]` e.g. `[EXP:001]`, `[MODEL:42]`, `[DATASET:v18]`.
CitationEngine extracts, validates (1.0 if in retrieved chunks, 0.60 if not), and enriches.

## Database (6 tables)

chat_sessions, research_queries, generated_reports, recommendation_history,
prompt_templates, llm_usage. SQLite :memory: uses persistent connection (same pattern as prior phases).

## Dashboard Views (5)

research_chat, experiment_explorer, knowledge_graph, recommendation_queue, executive_dashboard.

## API Endpoints (10)

chat, search, compare, rank, report, recommend, auto_recommend, pending_recommendations,
knowledge_search, get_dashboard, system_stats, index_document, switch_provider.

## Test Count

- Phase 23 analyst: 157 tests
- Total across all phases: 918 tests (all green)
