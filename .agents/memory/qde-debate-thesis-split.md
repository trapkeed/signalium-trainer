---
name: QDE debate engine / thesis manager split
description: How bull/bear thesis tracking and dominance-comparison were added to QDE Brain's opportunity queue without duplicating existing evidence/confidence infrastructure.
---

`thesis_manager.py` (`Thesis`, `ThesisManager`) and `debate_engine.py`
(`DebateEngine`, `DebateVerdict`) give every `opportunity_tracker.Opportunity`
a competing bull thesis and bear thesis, and a way to check whether one side
clearly dominates before execution.

**Why this split, not a bigger change to `Hypothesis`:** `Hypothesis.
supporting_evidence`/`contradicting_evidence` already existed but only for
one hypothesis's own direction. Rather than adding a second evidence store,
`ThesisManager.sync()` *reprojects* that same evidence onto both sides: a
hypothesis's own contradicting evidence for its direction is, by definition,
supporting evidence for the opposite side. It also folds in evidence from
sibling hypotheses of the opposite direction on the same symbol. Confidence
derives from `Hypothesis.confidence` (or its complement); "historical
outcomes" reuses `SymbolPersonality.regime_outcome_quality` and
`PersonalityProfile.trend_persistence`/`reversal_tendency` (already learned
elsewhere) — no parallel outcome tracker was introduced.

**How to apply:** `DebateEngine.clears_for_execution(symbol, opportunity_id)`
is a consult-only seam (same convention as `SetupQueue.ready_opportunities()`)
— it is *not* auto-wired into `OpportunityTracker.mark_executing()` or
`SetupQueue`. A decision layer must call it explicitly before treating a
READY opportunity as safe to execute. It returns true only when the debate
margin clears `DEFAULT_DOMINANCE_MARGIN` *and* the dominant side agrees with
the opportunity's own tracked direction (a dominant thesis opposing the
setup withholds, not confirms, execution).
