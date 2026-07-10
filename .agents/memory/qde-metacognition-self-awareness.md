---
name: Metacognition/self-awareness caution-vs-expansion asymmetry
description: Why caution and performance-driven expansion must use separate signals, not a single symmetric scale, when converting a self-assessment score into behavioral multipliers.
---

`MetacognitiveAssessment.caution_level` is 0=confident/neutral..1=maximally cautious — 0 is
the *neutral default*, not a midpoint. A prior design treated 0.5 as neutral and used
`caution - 0.5` as a signed expansion/reduction delta; this was wrong because the composite
caution formula (weighted shortfall + calibration_error + overconfidence) rarely produces
values above ~0.5 even during a genuine losing streak, so "poor recent performance" could
still land on the *expansion* side of a 0.5 midpoint — the opposite of the intended behavior.

**Fix:** reduction and expansion are driven by independent signals:
- `reduction_pull = caution` directly (unconditional — any caution above 0 pulls multipliers
  down toward their floor).
- `expansion_pull` requires an explicit positive-performance signal (recent_win_rate above
  baseline, optionally reinforced by a positive trend), and is damped by `(1 - caution)` so
  any live caution suppresses it. Being merely "not cautious" is not sufficient to expand.

**How to apply:** when converting any single composite health/confidence score into both a
"tighten" and "loosen" behavioral response, don't assume the score's neutral point is its
midpoint — check what value the score actually takes under "nothing wrong" vs "something
mildly wrong" scenarios, and consider using two independent signals (one for tightening, one
for loosening) rather than a signed delta around an assumed center.
