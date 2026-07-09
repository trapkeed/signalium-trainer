---
name: S28 decision traceability & lineage design
description: Where QDE decision-level traceability, model lineage resolution, and replay/audit live, and the key design choices (resolver pattern, injected inference_fn) that keep them dependency-light.
---

`trainer/research_lab/decision_traceability.py` is the S28 home for per-decision traceability (`DecisionTraceRecord`), model lineage resolution (`LineageResolver` / `ModelLineageChain`), deterministic inference replay (`replay_decision`), and historical reconstruction (`reconstruct_historical_decision`). Persisted via new `decision_traces` / `decision_audits` tables in `rl_database.py`, exposed through `ResearchLabAPI`.

**Why the resolver pattern instead of importing the model runtime:** `replay_decision` takes an injected `inference_fn: Callable[[dict], dict]` rather than importing the QDE inference stack directly — this module has to stay usable in environments where torch is broken/uninstalled (see `torch-broken-lazy-registry-import.md`), and it decouples "can we prove reproducibility" from "which model runtime is loaded."

**Why lineage links track `resolved: bool` per hop:** unlike the older in-memory `trainer/memory/decision_lineage.py` (which stores bare identifier strings with no verification), `LineageResolver` marks each hop resolved only if a real persisted record (training run / experiment) was found. A chain can be "complete" (all 6 hop types present) but only partially "resolved" — both signals matter for audit credibility.

**How to apply:** any new per-decision record schema should reuse `compute_input_feature_hash()` (wraps `trainer.traceability.hasher.hash_config`) rather than re-deriving hashing, so replay/reproducibility checks stay consistent with the rest of the traceability stack. New administrative-only Research Lab features should add a `Visibility` enum entry in `sso.py` and explicitly grant it per-role in `ROLE_VISIBILITY` — don't assume `OBSERVABILITY_DASHBOARD` or a blanket admin check covers a new feature's intended audience (e.g. Auditor needed `DECISION_TRACEABILITY` even though it has no other dashboard visibility).
