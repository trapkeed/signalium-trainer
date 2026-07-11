---
name: C6 Conscious Decision Engine
description: Design rules and test gotchas for trainer/qde/brain/decision/ (Phase C6)
---

## Design rules

- `CognitiveDecision` is brain vocabulary; `QDEDecision` is execution vocabulary — they are separate and C6 files must never import `QDEDecision`. Grep check must use a regex that matches real `from X import Y` lines, not docstring sentences that start with "from".
- Five actions: BUY, SELL, WAIT, MONITOR, IGNORE. WAIT and MONITOR are first-class decisions stored in memory identically to BUY/SELL.
- Decision priority: IGNORE (EXTREME uncertainty or no cognitive basis) → WAIT (HIGH uncertainty, daily loss limit, overconfidence+inconclusive) → MONITOR (no debate, inconclusive debate, conviction < ACTION_THRESHOLD) → BUY/SELL (conclusive + conviction ≥ 0.45).
- `hold_horizon = persistence_strength × 14400s`. A brand-new hypothesis has age=0 → persistence=0 → hold_horizon=0.0. This is correct, not a bug; test with `>= 0`, not `> 0`.
- All C2 imports (EpisodicMemory, WorkingMemory) are soft `try/except ImportError`.

## Test fixture gotchas

- `TrendMaturity` values: NONE / EARLY / ESTABLISHED / LATE / EXHAUSTED. There is no MODERATE.
- `make_debate_record()` in `debate_store.py` calls `.to_dict()` on the overconfidence/ignored-evidence args — pass actual `OverconfidenceEvent` / `IgnoredEvidenceRecord` objects, not raw dicts.
- For a "low conviction" test case use `reinforcement_count=0, confidence=0.20, sup=[], con=[...]` — with `reinforcement_count=4`, `compute_conviction` returns ≈0.52 even at confidence=0.30, which exceeds `CONVICTION_ACTION_THRESHOLD=0.45`.

**Why:** conviction is structural (reinforcement × evidence × maturity × persistence), not just raw hypothesis confidence; the test must drive the structural inputs low, not just the confidence field.

## __all__ location

C6 exports live in `brain/__init__.py` just above the "Cognitive memory layer (Phase C2)" block comment.
