---
name: C5 debate subpackage design
description: Design decisions and gotchas for the Phase C5 internal debate engine under trainer/qde/brain/debate/
---

## What already existed (do not reimplement)
- `debate_engine.py` — `DebateEngine.evaluate()` / `DebateVerdict` / `clears_for_execution()`
- `conviction_engine.py` — `compute_conviction()` / `ConvictionScore` / `ConvictionProfile`
- `thesis_manager.py` — `ThesisManager.sync()` / `Thesis` (bull/bear arguments per Opportunity)

## What C5 adds (trainer/qde/brain/debate/)
Five new files — none reimplement the above:
- `debate_quality.py` — `measure_quality()` pure function: 5-dim quality score (density, balance, margin_clarity, conviction_alignment, timeliness)
- `overconfidence_tracker.py` — 3 triggers: LOW_MARGIN / POOR_EVIDENCE_BALANCE / CONVICTION_CONFIDENCE_SPLIT
- `ignored_evidence_tracker.py` — records dominant thesis's conflicting evidence when conviction ≥ 0.55
- `debate_store.py` — `DebateRecord` (complete snapshot) + `DebateMemoryBridge` (soft C2 episodic import)
- `debate_api.py` — `DebateAPI`: `create_debate()` / `calculate_conviction()` / `store_debate()`

## Key design flow for create_debate()
After `DebateEngine.evaluate()` is called, it mutates `opp.bull_thesis` / `opp.bear_thesis` via ThesisManager internally. So `create_debate()` reads the synced Thesis objects directly from `state.opportunities[opp_id]` after the evaluate call — no second sync needed.

## Round-trip rounding gotcha
`to_dict()` rounds floats to 4 decimal places. Any test asserting round-trip equality must use `< 1e-4` tolerance (not `< 1e-9`).

## OverconfidenceTracker thresholds
- `HIGH_CONVICTION_THRESHOLD = 0.70` — conviction floor for LOW_MARGIN and POOR_EVIDENCE_BALANCE triggers
- `DEFAULT_DOMINANCE_MARGIN = 0.20` — margin floor; below this with high conviction → LOW_MARGIN event
- CONVICTION_CONFIDENCE_SPLIT fires at any conviction level when profile == HIGH_CONFIDENCE_LOW_CONVICTION

## IgnoredEvidenceTracker definition
"Ignored evidence" = `dominant_thesis.conflicting_evidence` when conviction ≥ 0.55 and dominant_side is not None.
Threshold: `IGNORED_EVIDENCE_MIN_CONVICTION = 0.55`.

## DebateMemoryBridge fallback
When episodic memory is not available, `store_debate()` uses a local `deque(maxlen=500)` per instrument and returns `record.debate_id` (not an episode_id). Always check `ep_id != record.debate_id` to know whether episodic storage succeeded.

## Stub brain requirements for tests
Same as C4: stub must have `get_state()`, `all_states()`, `record_experience()`. The stub DebateEngine must call `ThesisManager.sync()` and log to `WorkingMemory` to match real behaviour.
