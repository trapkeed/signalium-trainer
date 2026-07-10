---
name: Confidence vs conviction vs uncertainty in QDE Brain
description: Where the three-way distinction lives and how it gates trade selection
---

QDE Brain distinguishes three separate concepts that are easy to conflate:
- **Confidence** — probability the directional call is correct (already existed, forecast-level).
- **Conviction** — strength of belief in the *setup* (evidence quality/consistency for a hypothesis), computed in `trainer/qde/brain/conviction_engine.py` via `compute_conviction`, reusing `hypothesis_lifecycle` rather than re-deriving it. `classify_profile` labels combinations (e.g. high-confidence/low-conviction, low-confidence/high-conviction).
- **Uncertainty** — market ambiguity (news, volatility, liquidity, regime conflict, structural quality, evidence conflict across active hypotheses), computed in `trainer/qde/brain/uncertainty_engine.py` via `compute_uncertainty`. `does_not_understand_market()` is the explicit "QDE doesn't understand this market" signal; `prefers_no_trade()` is the softer gate.

**Why:** the prior brain only tracked confidence; without separating conviction and uncertainty, a high-confidence forecast in an ambiguous market had no principled way to be downgraded to NO_TRADE.

**How to apply:**
- `brain_context` (produced by `QDEBrain`/`_synthesize_thought`, flowed through `to_memory_context`/`enrich_inference_input` into `QDEInferenceInput.brain_context`) was previously an inert integration seam — nothing downstream consumed it. It is now read by `HeuristicQDEDecisionModel._market_uncertainty()` in `qde_inference.py`.
- Uncertainty composite >= 0.75 (EXTREME) is a hard REJECT gate in `derive_decision_fields(market_uncertainty=...)`; values below that only dampen conviction/confidence, they never force REJECT alone.
- EXTREME uncertainty is only reachable when hypothesis-conflict signals (regime_conflict/evidence_conflict) are nonzero — pure market-condition signals (news/vol/liquidity/structural) cap out around HIGH given the current weighting. Don't write tests expecting EXTREME from environmental factors alone; combine with conflicting directional hypotheses.
