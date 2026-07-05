# Developer Guide — Phase 23

## Module Inventory

```
trainer/analyst/
├── analyst_types.py          — Enums, constants, permissions, safety checks
├── analyst_models.py         — Data models (Chunk, Citation, Response, Query, Session, Report, Recommendation)
├── llm_provider.py           — LLM provider abstraction + MockLLMProvider + 5 stubs
├── knowledge_index.py        — KnowledgeIndex: in-memory document store
├── retrieval_engine.py       — RetrievalEngine: BM25-inspired RAG retrieval
├── prompt_engine.py          — PromptEngine: structured prompt construction
├── citation_engine.py        — CitationEngine: extraction, validation, enrichment
├── research_chat.py          — ResearchChat: conversational research interface
├── research_search.py        — ResearchSearch: natural language search
├── model_comparator.py       — ModelComparator: comparison reports
├── report_generator.py       — ReportGenerator: 12 report types
├── recommendation_engine.py  — RecommendationEngine: evidence-based suggestions
├── analyst_database.py       — AnalystDatabase: SQLite (6 tables)
├── analyst_dashboard.py      — AnalystDashboard: 5 dashboard views
├── analyst_api.py            — AnalystAPI: 10-endpoint unified API
└── __init__.py               — 40+ public exports
```

## Dependency Stack

```
analyst_types          ← no dependencies
analyst_models         ← analyst_types
llm_provider           ← analyst_types, analyst_models
knowledge_index        ← analyst_types, analyst_models
retrieval_engine       ← analyst_types, analyst_models, knowledge_index
prompt_engine          ← analyst_types, analyst_models
citation_engine        ← analyst_types, analyst_models
research_chat          ← all of the above
research_search        ← all of the above
model_comparator       ← all of the above
report_generator       ← all of the above
recommendation_engine  ← all of the above
analyst_database       ← analyst_types, analyst_models
analyst_dashboard      ← analyst_types, analyst_models
analyst_api            ← everything above
__init__.py            ← analyst_api (re-exports all)
```

## Key Design Rules

### 1. No torch, no numpy, no pandas
Pure Python + stdlib + loguru. The package imports and runs anywhere.

### 2. Spec-from-file-location imports in tests
```python
import importlib.util
spec = importlib.util.spec_from_file_location("trainer.analyst.analyst_api", path)
```
Never `importlib.import_module("trainer.analyst.*")`.

### 3. SQLite `:memory:` persistence
`AnalystDatabase` keeps a `_persistent_conn` for `:memory:` mode. Same pattern as Phases 20–22.

### 4. Provider hot-swap
`AnalystAPI.switch_provider()` updates the provider in all sub-systems atomically. No restart required.

### 5. Safety is non-negotiable
Every response passes `is_safe_response()`. If it fails, the content is replaced with a safety notice. This is enforced in `MockLLMProvider.generate()` and should be enforced in all future real providers.

### 6. Grounding is enforced
`LLMResponse.grounded = True` only when context was provided. Dashboard and API surface this flag.

## Adding a New LLM Provider

1. Create a class inheriting from `LLMProvider`
2. Implement `generate()`, `provider_type()`, `is_available()`
3. Add to `_REGISTRY` in `create_provider()`
4. Add the provider type to `LLMProviderType` enum
5. Add to `analyst.yaml` providers section

## Adding a New Knowledge Source

1. Add to `KnowledgeSourceType` enum in `analyst_types.py`
2. Add to `analyst.yaml` knowledge_sources list
3. Call `api.index_document(source_type=KnowledgeSourceType.NEW_TYPE, ...)`
4. Update `CitationEngine._source_to_citation()` mapping if needed

## Database Schema

```sql
chat_sessions          (session_id PK, researcher, role, message_count, active, created_at, updated_at)
research_queries       (query_id PK, session_id FK, text, query_type, role, timestamp, response_preview)
generated_reports      (report_id PK, report_type, title, period, word_count, citation_count, grounded, generated_at, content)
recommendation_history (rec_id PK, rec_type, title, description, priority, implemented, citation_count, timestamp)
prompt_templates       (template_id AUTOINCREMENT, name UNIQUE, query_type, template, created_at, updated_at)
llm_usage              (usage_id AUTOINCREMENT, provider, model_name, prompt_tokens, response_tokens, latency_ms, grounded, safe, query_type, timestamp)
```

## Running Tests

```bash
cd signalium-trainer
python3 -m unittest tests/unit/test_analyst.py -v
```

## Integration with Phase 21.6 (Decision Memory)

```python
from trainer.analyst import AnalystAPI, KnowledgeSourceType
from trainer.memory import MemoryAPI

memory = MemoryAPI()
analyst = AnalystAPI()

# Index decision memory records
for rec in memory.get_all_decisions():
    analyst.index_document(
        source_type = KnowledgeSourceType.DECISION_MEMORY,
        artifact_id = rec["decision_id"],
        title       = f"Decision: {rec['decision_type']} {rec['instrument']}",
        content     = str(rec),
        tags        = [rec["instrument"], rec["decision_type"]],
        metadata    = rec,
    )

# Now ask questions about decision history
result = analyst.chat("Which decisions were most often incorrect?")
```

## Integration with Phase 22 (Microstructure)

```python
from trainer.analyst import AnalystAPI, KnowledgeSourceType
from trainer.microstructure import MicrostructureAPI

micro   = MicrostructureAPI()
analyst = AnalystAPI()

# Analyse bars and index the market state
result = micro.analyse(bars, "EURUSD", "H1")
analyst.index_document(
    source_type = KnowledgeSourceType.MARKET_MEMORY,
    artifact_id = result["features"]["bar_id"],
    title       = f"EURUSD H1 Microstructure",
    content     = str(result["market_state"]),
    tags        = ["eurusd", result["market_state"]["state"].lower()],
    metadata    = result["score"],
)
```
