# RAG Architecture Guide — Phase 23

## Overview

SAQRA uses Retrieval-Augmented Generation (RAG) to ground every response in retrieved research evidence. No information is hallucinated — all claims must be supported by indexed artifacts.

## RAG Pipeline

```
User Query
    ↓
RetrievalEngine.retrieve()    ← BM25-approximate scoring
    ↓
Retrieved KnowledgeChunks (top-k)
    ↓
RetrievalEngine.build_context()   ← Formats chunks into context string
    ↓
PromptEngine.build()              ← Injects: system + context + history + query
    ↓
LLMProvider.generate()            ← Mock / OpenAI / Anthropic / Gemini / Ollama / Azure
    ↓
CitationEngine.extract()          ← Parses [TYPE:ID] citations from response
CitationEngine.validate()         ← Cross-references against retrieved chunks
CitationEngine.add_from_chunks()  ← Ensures all retrieved chunks are cited
    ↓
LLMResponse (content + citations + metadata)
```

## Retrieval Algorithm

The RetrievalEngine uses a BM25-inspired scoring approach implemented in pure Python (no numpy required).

### Scoring Formula

```
score(q, d) = Σ IDF(t) × [ TF(t,d) × (K1+1) ] / [ TF(t,d) + K1 × (1-B + B×|d|/avgdl) ]
```

Where:
- `K1 = 1.5` (term-frequency saturation)
- `B = 0.75` (length normalisation)
- Title matches are boosted 3×
- Tag matches are boosted 1×

### Index Structure

The KnowledgeIndex maintains three indices for O(1) retrieval:
1. `chunk_id → KnowledgeChunk` — primary store
2. `source_type → [chunk_ids]` — source-filtered retrieval
3. `artifact_id → [chunk_ids]` — artifact lookup

## Context Building

Retrieved chunks are formatted into a context string:

```
[EXPERIMENT:EXP-001] EURUSD Transformer Experiment
Sharpe 2.3, max_drawdown 8.5%, accuracy 0.72...
---
[MODEL_REGISTRY:MODEL-42] Champion Transformer Model
Sharpe 2.8, drawdown 6.2%...
```

Max context: 3000 characters (configurable). Long chunks are truncated with `...`.

## Citation Framework

### Citation Extraction

The CitationEngine extracts `[TYPE:ID]` patterns from LLM responses:

| Pattern | Citation Type |
|---|---|
| `[EXP:001]` or `[EXP-001]` | EXPERIMENT |
| `[MODEL:42]` | MODEL |
| `[DATASET:v18]` | DATASET |
| `[FEATURE:rsi_14]` | FEATURE |
| `[VAL:2024Q2]` | VALIDATION |
| `[REPORT:Q2]` | REPORT |
| `[DECISION:abc]` | DECISION |
| `[AUDIT:xyz]` | AUDIT_LOG |
| `[KB:abc]` | KNOWLEDGE |

### Citation Validation

Citations are cross-referenced against the retrieved chunks:
- **Validated citation** (artifact in retrieved chunks): relevance = 1.0
- **Unvalidated citation** (artifact referenced but not retrieved): relevance = 0.60

### Citation Enrichment

After extraction, `CitationEngine.add_from_chunks()` ensures that every retrieved chunk is cited, even if the LLM did not explicitly reference it. This prevents unreferenced evidence.

## Groundedness

A response is `grounded = True` when:
1. Context was provided (at least one chunk was retrieved)
2. The LLM response references the context

A response is `grounded = False` when:
1. No chunks were retrieved (empty index or no matches)
2. The query was rejected for safety reasons

## Prompt Templates

Each query type has a dedicated prompt template:

| Query Type | Template Contents |
|---|---|
| CHAT | System + context + history + query |
| COMPARISON | System + context + side-by-side instructions |
| ANALYSIS | System + context + analysis structure |
| SEARCH | System + context + listing instructions |
| SUMMARY | System + context + period + summary structure |
| RECOMMENDATION | System + context + evidence-based recommendation format |
| REPORT | System + context + report type + professional structure |

## Provider Abstraction

The LLMProvider interface allows hot-swapping providers:

```python
class LLMProvider(ABC):
    def generate(prompt, context, temperature, max_tokens) -> LLMResponse
    def provider_type() -> LLMProviderType
    def is_available() -> bool
```

All providers share the same interface. Switching providers is one line:
```python
api.switch_provider(LLMProviderType.OPENAI, api_key="sk-...")
```

## Safety Layer

Every generated response passes through `is_safe_response()` before delivery. Prohibited patterns include: BUY, SELL, LONG, SHORT, ENTER TRADE, EXIT TRADE, PLACE ORDER, EXECUTE ORDER.

If a response contains prohibited patterns, the MockLLMProvider replaces the response with a safety notice.
