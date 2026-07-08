---
name: QDE output-heads vs. decision translation split
description: How QDE's raw model output (signal heads) relates to the execution-facing decision vocabulary, and the rule for adding new consumers.
---

QDE's model produces a raw, strategy-agnostic vector of signal heads (`trainer/qde/qde_outputs.py::QDEOutputHeads` — direction + 9 float signals such as `confidence`, `reduce_signal`, `scale_signal`, `exit_horizon`, `management_confidence`, `risk_signal`, `attention_focus`). These heads are the single source of truth for "what does the model think right now."

A **separate, pure translation function** (`derive_decision_fields()` in `trainer/qde/qde_inference.py`) maps heads → the entry-side execution vocabulary (`EntryAction`/`ExitAction`/`ScaleAction`/`ManagementAction`) that `ExecutionPlanner` already knows how to execute. This is deliberately the *only* place that translation happens for entries.

**Why:** keeping "produce heads" and "translate heads into an action vocabulary" as two separate steps means the heads stay a genuine, independently-loggable/inspectable model output (useful for shadow testing, monitoring, and any new consumer) rather than being baked directly into one hardcoded action set. It also means the mapping logic itself can be swapped (e.g. learned instead of threshold-based) without touching the heads schema or any downstream consumer.

**How to apply:** when adding a new consumer of QDE's output that needs a *different* action vocabulary than the entry-side one (e.g. `trainer/qde/qde_position_manager.py`, which needs HOLD/REDUCE/ADD/CLOSE/MOVE_TO_BREAKEVEN/TRAIL/LOCK_PROFIT for live position management, not entry/exit/scale), do NOT reuse or extend `derive_decision_fields()`. Instead, write a second pure translation function that reads directly from `QDEOutputHeads` (plus whatever consumer-specific hard-constraint inputs it needs, e.g. position P&L or free margin) and produces its own decision type. Mirror `derive_decision_fields()`'s shape (heads-in, enum-out, pure, first-match-wins threshold ladder, hard safety overrides checked first) rather than inventing a different pattern. Keep a mapping-table doc note cross-referencing the different action vocabularies in play so future readers don't assume they're interchangeable.
