---
name: C4 hypothesis set design
description: Key design decisions and gotchas for the Phase C4 hypothesis competing-scenarios layer under trainer/qde/brain/hypothesis/
---

## The rule
`HypothesisAPI` delegates all `create/reinforce/weaken/invalidate` to `HypothesisEngine`; it never reimplements those operations.

## Critical gotcha: weaken auto-invalidates at ≤0.15 confidence
`Hypothesis.weaken()` (in `brain_models.py`) auto-sets status to INVALIDATED when confidence drops to ≤0.15.
`update_probabilities()` calls `HypothesisEngine.weaken()` for negative evidence, which carries a default 0.20 penalty.
Any seed/hypothesis with confidence ≤0.35 can become terminal after a single negative evidence update.

**How to apply:**
- In `update_probabilities()`, always check `live_hyp.is_terminal()` before calling reinforce/weaken, and skip terminal hypotheses.
- Tests that call `update_probabilities` then `invalidate_hypothesis` must use seeds with initial confidence ≥0.40+ so one weaken call doesn't push them terminal.

## invalidate_hypothesis handles already-terminal case
If a hypothesis was auto-invalidated by the weaken chain before the explicit `invalidate_hypothesis` call:
- Skip the `HypothesisEngine.invalidate()` call (already done).
- Still record the failure in `HypothesisFailureTracker`.
- Still update the lineage node.
- Return `True` (the hypothesis IS invalidated — the caller's goal is achieved).

**Why:** callers should not have to defensively check `is_terminal()` before calling `invalidate_hypothesis`. The API absorbs this complexity.

## Stub brain for HypothesisEngine tests
`HypothesisEngine._check_milestone()` calls `self._brain.record_experience(...)`.
Test stubs must include `record_experience(instrument, kind, description, outcome_quality, regime='', tags=None)` (no-op is fine) alongside `get_state()` and `all_states()`, or tests fail with AttributeError.

## File locations
All 6 new files: `trainer/qde/brain/hypothesis/`
- `hypothesis_set.py` — `HypothesisSet` / `HypothesisScenario` / `ProbabilitySnapshot` / `make_hypothesis_set()`
- `hypothesis_probability.py` — `HypothesisProbabilityEngine` / `ProbabilityUpdateResult`
- `hypothesis_failure_tracker.py` — `HypothesisFailureTracker` / `HypothesisFailureRecord` / `FailureMode`
- `hypothesis_lineage.py` — `HypothesisLineage` / `LineageNode`
- `hypothesis_memory_bridge.py` — `HypothesisMemoryBridge` / `HypothesisMemoryContext`
- `hypothesis_api.py` — `HypothesisAPI` / `HypothesisGenerationResult`
