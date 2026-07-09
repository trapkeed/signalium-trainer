---
name: Composite QDE health score design
description: How the 13 health-monitor drift dimensions map into the 5-category composite QDE health score, and where shadow-observation degradation actually gets wired into Research Lab tasks.
---

The 13 `DriftDimension` values computed by `ResearchLabHealthMonitor` partition
disjointly into exactly 5 composite categories (calibration, profitability,
stability, robustness, drift) for the composite QDE Health Score — every
dimension belongs to exactly one category, unlike the coarse
`EvaluationDimension` mapping (which intentionally reuses categories and
relies on evidence-field dedup, see `health-monitor-dedup-collision.md`).

**Universal task-creation gotcha:** "every degradation finding must create a
Research Lab review task" is easy to satisfy for health-monitor's own 13
dimensions (extend the CRITICAL-only trigger to also fire on WARNING, at LOW
urgency) but easy to miss for shadow-test rejection. The function that looks
like the natural place to wire this — `qde_shadow.py`'s
`evaluate_and_maybe_promote` — is dead code, not called from any production
server path. The actual live shadow-evaluation path is
`ChampionChallengerTracker.evaluate_promotion` (`champion_challenger.py`),
wired to `POST /shadow/evaluate`. Any future "shadow observation degradation"
requirement must hook there, not into `qde_shadow.py`.

**Why:** confirmed via grep that `evaluate_and_maybe_promote` has zero
callers outside its own module and tests; `qde_api.py`'s
`evaluate_shadow_promotion` wrapper exists but is itself never invoked by
`research_lab_server.py` or `qde_server.py`.

**How to apply:** when adding new "degradation finding → review task" wiring,
first grep for the *actual* server-endpoint call graph, not just the
function whose name suggests it's the right place.
