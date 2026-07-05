# Knowledge Base Guide — Phase 21.6

## Overview

The Knowledge Base is a searchable institutional memory linking every decision to related trades, experiments, models, features, market regimes, research reports, lifecycle events, and portfolio states.

## Entry Types

| Type | Description |
|---|---|
| DECISION | A recorded Signalium decision |
| TRADE | A completed trade linked to one or more decisions |
| EXPERIMENT | A research or model training experiment |
| MODEL | A trained model version |
| FEATURE | A feature set or feature importance record |
| REGIME | A market regime characterisation |
| RESEARCH | A research report or analysis |
| LIFECYCLE_EVENT | A significant trade lifecycle event |
| PORTFOLIO_STATE | A portfolio state snapshot |

## Knowledge Entry Structure

```
KnowledgeEntry
├── entry_id     # Short UUID
├── entry_type   # See entry types above
├── title        # Human-readable title
├── description  # Full description
├── tags         # List[str] — searchable labels
├── links        # List[str] — entry_ids of related entries
├── decision_ids # List[str] — linked decision UUIDs
├── metadata     # Dict — additional structured data
└── created_at   # Unix epoch
```

## Adding Entries

```python
from trainer.memory import MemoryAPI, KnowledgeBase, KnowledgeEntry

api = MemoryAPI(db_path="decisions.db")

# Auto-index a decision (most common pattern)
api.index_decision_in_knowledge_base(record.decision_id)

# Add a custom model entry
api.knowledge.create_entry(
    entry_type  = "MODEL",
    title       = "Ensemble Champion v3.2",
    description = "Champion model using 12 experts, trained on 2Y of EURUSD data.",
    tags        = ["ensemble", "EURUSD", "champion", "v3.2"],
    metadata    = {"accuracy": 0.724, "trained_on": "2024-01-01", "epochs": 120},
)

# Add a research report
api.knowledge.create_entry(
    entry_type  = "RESEARCH",
    title       = "London Session Breakout Study Q1-2025",
    description = "Analysis of 340 breakout decisions during LONDON session.",
    tags        = ["research", "LONDON", "breakout", "Q1-2025"],
)
```

## Linking Entries

```python
# Link a model entry to an experiment entry
api.knowledge.link_entries(model_entry.entry_id, experiment_entry.entry_id)
```

## Searching the Knowledge Base

```python
# Search by text
results = api.knowledge_search(query="EURUSD breakout", limit=20)

# Search by type
results = api.knowledge_search(entry_type="MODEL")

# Search by tags
results = api.knowledge_search(tags=["champion", "LONDON"])

# Combined search
results = api.knowledge_search(
    query      = "pullback",
    entry_type = "DECISION",
    tags       = ["trend_continuation"],
    limit      = 50,
)
```

## Decision Auto-Indexing

When `auto_index_decisions: true` in config, every decision is automatically indexed:

```python
# Auto-index generates a DECISION entry with:
# - Title: "BUY EURUSD @ 2025-03-15 09:30"
# - Description: Summary of regime, session, confidence, grade
# - Tags: [decision_type, instrument, market_regime, session, volatility_regime, ...tags]
# - decision_ids: [record.decision_id]
# - metadata: record.summary()
api.index_decision_in_knowledge_base(record.decision_id)
```

## Linking a Decision to Its Trade

```python
# After a trade closes, link decision entries to the trade entry
trade_entry = api.knowledge.create_entry(
    entry_type   = "TRADE",
    title        = "EURUSD Long — 2025-03-15",
    description  = "Trend-following long on EURUSD, closed at TP.",
    tags         = ["EURUSD", "LONG", "trend"],
    decision_ids = [entry_decision_id, exit_decision_id],
)

# Link decisions ↔ trade
api.knowledge.link_entries(entry_decision_id, trade_entry.entry_id)
api.knowledge.link_entries(exit_decision_id, trade_entry.entry_id)
```

## Knowledge Graph Dashboard

```python
graph = api.get_dashboard("knowledge_graph")
# Returns:
# {
#   "view": "KNOWLEDGE_GRAPH",
#   "nodes": [
#     {"id": "dec12345", "type": "DECISION", "label": "BUY@EURUSD"},
#     {"id": "instr_EURUSD", "type": "INSTRUMENT", "label": "EURUSD"},
#     {"id": "reg_TRENDING_UP", "type": "REGIME", "label": "TRENDING_UP"},
#     ...
#   ],
#   "edges": [
#     {"from": "dec12345", "to": "instr_EURUSD", "rel": "ON"},
#     {"from": "dec12345", "to": "reg_TRENDING_UP", "rel": "DURING"},
#     ...
#   ]
# }
```

## Institutional Search Examples

```python
# All models used in champion configuration
results = api.knowledge_search(entry_type="MODEL", tags=["champion"])

# All research reports about breakout trades
results = api.knowledge_search(entry_type="RESEARCH", query="breakout")

# All EURUSD decisions in LONDON session
results = api.knowledge_search(
    entry_type = "DECISION",
    tags       = ["EURUSD", "LONDON"]
)
```

## Statistics

```python
stats = api.knowledge.stats()
# {
#   "total_entries": 342,
#   "by_type": {"DECISION": 150, "MODEL": 20, "RESEARCH": 15, ...},
#   "indexed_decisions": 150,
# }

# All tags in the knowledge base
tags = api.knowledge.all_tags()
```
