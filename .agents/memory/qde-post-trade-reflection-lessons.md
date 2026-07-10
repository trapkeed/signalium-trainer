---
name: QDE post-trade reflection/surprise/lesson pipeline
description: How per-trade reflection, surprise scoring, and lesson extraction are split and wired, to avoid re-duplicating or colliding with existing periodic reflection.
---

There are now two distinct "reflection" concepts in `trainer/qde/brain/` — do not merge or confuse them:

- `reflection.ReflectionEngine` (pre-existing): periodic, rolling-window synthesis across *many* decisions for a symbol, rate-limited hourly via `QDEBrain.reflect()`.
- `reflection_engine.TradeReflectionEngine` (new): per-trade post-mortem on *one* just-closed `DecisionRecord` — analyzes assumptions, reasoning quality (reused from `DecisionScore`, not recomputed), ignored evidence (`contradicting_factors` present but not weighted), false evidence (`primary_reasons`/`supporting_factors` that didn't hold up), and surprise magnitude.

`surprise_engine.compute_surprise` is a new prediction-error concept (didn't exist anywhere in `trainer/`) — compares `DecisionSnapshot` forecast fields against `DecisionOutcome` realized fields (direction, PnL vs expected_move, holding time, outcome_type).

`lesson_extractor.LessonExtractor` routes a `TradeReflection` into the Brain's existing learning surfaces rather than introducing new ones: personality via `LongTermMemory.record_experience` (reused), a new `BrainState.calibration: CalibrationProfile` (conviction_bias/uncertainty_bias/confidence_bias, EMA-nudged) since no calibration-adjustment state existed before, active hypotheses' `contradicting_evidence` on losses, and a `LESSON`-kind working-memory item (via existing `WorkingMemory.add`) for future hypothesis formation context.

**Why:** `conviction_engine`/`uncertainty_engine` are deliberately pure/stateless (their docstrings say so) — they must never gain new tracked fields. Calibration bias needed a home, so it lives on `BrainState` instead, as a correction layer callers can add to those engines' raw output.

**How to apply:** entry point is `QDEBrain.reflect_on_trade(record, regime="")` — call it once per completed trade (after `DecisionRecord.outcome` is set), analogous to how `_check_milestone`/`position_just_closed` already trigger `record_experience`. It returns `None` if the record has no outcome yet.
