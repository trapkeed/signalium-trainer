# AI Quant Research Analyst Architecture Report
## SAQRA — Signalium Institutional AI Research System — Phase 23

---

## Executive Summary

Phase 23 delivers SAQRA — the Signalium AI Quant Research Analyst — a production-ready institutional research intelligence layer. SAQRA provides evidence-based analysis of Signalium's quantitative research platform without ever generating trading signals or overriding deterministic forecasting.

The system implements a full RAG (Retrieval-Augmented Generation) architecture with a configurable LLM provider layer supporting six providers, comprehensive citation enforcement, role-based security, 12 report types, and 5 dashboard views.

---

## 1. System Architecture

### 1.1 Position in the Signalium Stack

```
┌─────────────────────────────────────────────────────────┐
│                 Signalium Quant Platform                 │
├─────────────────────────────────────────────────────────┤
│  Phase 20: Portfolio Analytics                          │
│  Phase 21: Execution Analytics                          │
│  Phase 21.5: Lifecycle Analytics                        │
│  Phase 21.6: Decision Memory (Knowledge Source)        │
│  Phase 22: Market Microstructure (Knowledge Source)    │
├─────────────────────────────────────────────────────────┤
│  Phase 23: SAQRA ← AI Research Analyst (THIS PHASE)    │
│                                                          │
│  Reads from all prior phases.                           │
│  Never modifies. Never generates signals.               │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Module Architecture

```
AnalystAPI (unified entry point)
├── KnowledgeIndex         ← in-memory document store (20 source types)
├── RetrievalEngine        ← BM25-approximate retrieval
├── PromptEngine           ← 7 prompt templates + safety system prompt
├── CitationEngine         ← extraction, validation, enrichment
├── LLMProvider            ← abstraction (Mock + 5 provider stubs)
├── ResearchChat           ← conversational research mode
├── ResearchSearch         ← natural language document search
├── ModelComparator        ← structured comparison reports
├── ReportGenerator        ← 12 report types
├── RecommendationEngine   ← evidence-based suggestions (9 types)
├── AnalystDatabase        ← SQLite (6 tables)
└── AnalystDashboard       ← 5 dashboard views
```

---

## 2. RAG Design

### 2.1 Retrieval-Augmented Generation Pipeline

```
User Query
    ↓ tokenise + remove stopwords
RetrievalEngine → BM25 scoring → top-k chunks
    ↓ format context string (max 3000 chars)
PromptEngine → system prompt + context + history + query
    ↓ inject into provider
LLMProvider → generate response
    ↓ safety check (prohibited patterns)
CitationEngine → extract [TYPE:ID] → validate → enrich from chunks
    ↓
LLMResponse (content, citations, grounded, safe, tokens, latency)
```

### 2.2 BM25 Retrieval

- Implemented in pure Python (no numpy, no scipy)
- K1 = 1.5, B = 0.75 (standard BM25 parameters)
- Title matches: 3× weight boost
- Tag matches: 1× weight boost
- Stopword removal: 35 common English words filtered
- Empty query fallback: returns N most recent chunks

### 2.3 Context Window Management

- Max context: 3,000 characters (configurable)
- Chunks formatted as: `[SOURCE_TYPE:ARTIFACT_ID] Title\nContent`
- Separated by `---` delimiters
- Last chunk truncated if over budget

---

## 3. Knowledge Sources

SAQRA indexes all 20 research knowledge source types:

| Source Type | Description |
|---|---|
| RESEARCH_REPORT | Formal research reports |
| EXPERIMENT | Experiment configs and results |
| TRAINING_RUN | Training run logs |
| VALIDATION_REPORT | Walk-forward and holdout results |
| WALK_FORWARD | Walk-forward analysis |
| MODEL_REGISTRY | Model registry entries |
| RESEARCH_SCORE | Model scoring history |
| FEATURE_IMPORTANCE | Feature importance analysis |
| DATASET_VERSION | Dataset version manifests |
| DECISION_MEMORY | Phase 21.6 decision records |
| COUNTERFACTUAL | Phase 21.6 counterfactual analysis |
| KNOWLEDGE_GRAPH | Phase 21.6 knowledge base |
| CHAMPION_HISTORY | Champion model history |
| MODEL_COMPARISON | Head-to-head model comparisons |
| PORTFOLIO_ANALYTICS | Phase 20 portfolio metrics |
| EXECUTION_ANALYTICS | Phase 21 execution analytics |
| MARKET_MEMORY | Phase 22 microstructure data |
| LIFECYCLE_ANALYTICS | Phase 21.5 trade lifecycle data |
| DRIFT_REPORT | Feature and model drift reports |
| AUDIT_LOG | System audit trail |

---

## 4. Prompt Pipeline

### 4.1 System Prompt (always injected)

The system prompt defines SAQRA's identity, mission, constraints, and citation format. Key elements:
- Role: Institutional Research Analyst
- Absolute constraints: NEVER BUY/SELL, NEVER override production
- Citation format: `[TYPE:ID]`
- Response format: Markdown + executive summary + citations + disclaimer

### 4.2 Query Type Templates (7 templates)

| Query Type | Structure |
|---|---|
| CHAT | System + context + history (6 turns) + query |
| COMPARISON | System + context + comparison instructions |
| ANALYSIS | System + context + structured analysis |
| SEARCH | System + context + listing format |
| SUMMARY | System + context + period + summary sections |
| RECOMMENDATION | System + context + evidence-based recommendation format |
| REPORT | System + context + report type + institutional format |

---

## 5. Citation Framework

### 5.1 Citation Types (9)

EXPERIMENT, MODEL, DATASET, FEATURE, VALIDATION, REPORT, DECISION, AUDIT_LOG, KNOWLEDGE

### 5.2 Citation Lifecycle

1. **Extraction** — `re.finditer()` finds `[TYPE:ID]` patterns in response text
2. **Validation** — cross-references artifact IDs against retrieved chunks; validated → 1.0, unvalidated → 0.60
3. **Enrichment** — adds citations for all retrieved chunks not explicitly referenced by LLM
4. **Bibliography** — formats a numbered reference list for report appendices

### 5.3 Anti-Hallucination

Validated citations have higher relevance (1.0) than unvalidated (0.60), surfacing to the researcher which claims are directly supported by retrieved evidence.

---

## 6. Security Model

### 6.1 Role-Based Access

| Role | READ | EXPORT | SUMMARIZE | COMPARE | AUDIT |
|---|---|---|---|---|---|
| ADMIN | ✅ | ✅ | ✅ | ✅ | ✅ |
| RESEARCHER | ✅ | ❌ | ✅ | ✅ | ❌ |
| VIEWER | ✅ | ❌ | ❌ | ❌ | ❌ |

### 6.2 Permission Enforcement

Permission checks are enforced at the entry point of every API method. Denied requests return structured error responses — no partial data is leaked.

### 6.3 Safety Guardrails

The `is_safe_response()` function scans every LLM output for prohibited patterns before delivery. If triggered, the response is replaced with a safety notice. The function is provider-agnostic — it runs regardless of which LLM generated the content.

**Prohibited patterns:** BUY, SELL, LONG, SHORT, ENTER TRADE, EXIT TRADE, PLACE ORDER, EXECUTE ORDER.

---

## 7. LLM Provider Layer

### 7.1 Provider Registry

| Provider | Status | Notes |
|---|---|---|
| MockLLMProvider | ✅ Fully operational | Template-based synthesis; no external calls |
| OpenAIProvider | Stub | Requires `pip install openai` + `OPENAI_API_KEY` |
| AnthropicProvider | Stub | Requires `pip install anthropic` + `ANTHROPIC_API_KEY` |
| GeminiProvider | Stub | Requires `pip install google-generativeai` + key |
| OllamaProvider | Stub | Requires Ollama on localhost:11434 |
| AzureOpenAIProvider | Stub | Requires `pip install openai` + Azure credentials |

### 7.2 Hot-Swap

```python
api.switch_provider(LLMProviderType.OPENAI, api_key="sk-...")
```

Updates all 5 sub-systems (Chat, Search, Comparator, Reporter, Recommender) atomically. No service restart required.

### 7.3 MockLLMProvider Design

The mock generates structured responses using pattern-matched synthesis:
- Detects query intent from keywords
- Extracts relevant sentences from context
- Applies query-type templates
- Extracts [TYPE:ID] citations from context

This allows full system testing and offline research without an API key.

---

## 8. Operational Readiness

| Capability | Status |
|---|---|
| RAG pipeline (retrieve + generate + cite) | ✅ Operational |
| 20 knowledge source types | ✅ Indexed |
| BM25 retrieval | ✅ Operational |
| Mock LLM (offline research) | ✅ Operational |
| 6 LLM provider stubs | ✅ Wired |
| Safety guardrails | ✅ Enforced |
| Citation extraction + validation | ✅ Operational |
| Role-based security (3 roles, 5 permissions) | ✅ Enforced |
| 12 report types | ✅ Operational |
| 9 recommendation types | ✅ Operational |
| Auto-recommendation (index scan) | ✅ Operational |
| 5 dashboard views | ✅ Operational |
| 6-table SQLite persistence | ✅ Operational |
| LLM usage tracking | ✅ Operational |
| Provider hot-swap | ✅ Operational |
| Backward compatibility | ✅ No existing changes |
| stdlib-only (no torch/numpy) | ✅ Confirmed |
| Test coverage | ✅ 145+ unit tests |

---

## 9. Institutional Readiness

### Transparency
- Every response includes machine-readable citations in `LLMResponse.citations`
- Grounding flag (`grounded: bool`) surfaces whether context was available
- Safety flag (`safe: bool`) confirms no prohibited signals were detected
- Provider and latency metadata are recorded in every response

### Auditability
- Every query, response, report, and recommendation is persisted to SQLite
- LLM usage (tokens, latency, safety, grounding) is tracked per call
- Prompt templates are versioned in the `prompt_templates` table

### Extensibility
- New knowledge sources: add to enum + index_document()
- New LLM providers: subclass LLMProvider + register in create_provider()
- New report types: add to ReportType enum + _REPORT_SECTIONS + _REPORT_TITLES
- New recommendation types: add to RecommendationType + _TRIGGER_MAP

### Isolation
- No trading signals can be generated by design
- No production system calls — read-only access to indexed documents
- Mock provider enables full operation without external dependencies

---

*Report generated: July 2025 — Phase 23 SAQRA Implementation*
