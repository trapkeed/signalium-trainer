---
name: QDE ↔ Research Lab dependency direction
description: Which package may import which when a feature needs to bridge trainer.qde and trainer.research_lab
---

`trainer.qde` may import data classes from `trainer.research_lab.rl_models` (no cycle). `trainer.research_lab` must never import anything from `trainer.qde` — not even for type hints.

**Why:** `trainer.research_lab` is a standalone, torch-free, no-cycle module tested via plain unittest without import stubbing (see `tests/unit/test_research_lab_persistence.py`). `trainer.qde` pulls in `trainer.execution` → `trainer.registry`, a heavier chain. Keeping the dependency one-directional preserves Research Lab's lightweight testability.

**How to apply:** Any live-system logic that needs both QDE decision models and Research Lab data models (e.g. dual-dispatch shadow testing) belongs in a new module under `trainer/qde/` (e.g. `trainer/qde/qde_shadow.py`), not under `trainer/research_lab/`. That module imports `rl_models` dataclasses directly; it never round-trips through `research_lab_api.py`'s DB-backed CRUD layer for its core logic — callers glue the two APIs together (e.g. `QDELifecycleAPI.evaluate_shadow_promotion` takes a plain dict, not a `ResearchLabAPI` instance).
