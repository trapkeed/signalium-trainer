# Autonomous Learning Loop (Phase 29b)

This document describes the full set of 9 trigger events the learning loop
watches for, and the always-on scheduler that keeps them under continuous
observation without a manual call to kick off each cycle.

It extends Phase 29 (`trainer/learning_loop/actuators.py`,
`LearningLoopActuator`), which is described from the "hardcoded behavior"
angle in [`hardcoded_behavior_audit.md`](./hardcoded_behavior_audit.md#relationship-to-the-learning-loop-actuator-layer).
That document explains *why* the actuator layer exists (turning evaluation
findings into training requests instead of hand-edited constants). This
document catalogs *what* it watches and *how* it keeps watching.

## Hard constraint: proposals only, never auto-deploy

Nothing described below writes to live trading logic, promotes a model, or
starts a retrain by itself. Every trigger produces one of:

- A `ResearchLabAPI` record (`FailureEvent`, `Experiment`, `ParameterSearchRun`,
  `ShadowTestRun`, `FeatureCandidate`) — all pending human/pipeline review.
- A `RetrainingAdvisor.evaluate(...)` recommendation — `RetrainingAdvisor` is
  hard-pinned `auto_retrain=False`; recommendations require explicit human
  action, and every one it emits is tagged "ADVISORY ONLY".

Promotion to production still goes exclusively through `QDEPromotionEngine`
and its existing gates (see [`promotion_procedures.md`](./promotion_procedures.md)
and [`version_lifecycle.md`](./version_lifecycle.md)). The autonomous loop
described here changes *when* detection happens (continuously, not on
manual demand) — it does not change *what* is allowed to happen as a result.

## The 9 trigger events

| # | Trigger event | Source report | Actuator method | Typical output |
|---|---|---|---|---|
| 1 | Large drawdown / low quality score | `SelfAssessment` | `act_on_self_assessment` | Retraining recommendation (performance/confidence degradation) |
| 2 | Unexpected loss / recurring failure mode | `SelfAssessment` | `act_on_self_assessment` | Investigation (`FailureEvent`) |
| 3 | Missed opportunity / strong pattern found | `PatternReport` | `act_on_pattern_report` | Feature discovery request, or dataset enrichment for weak/undersampled patterns |
| 4 | Regime failure (negative regime pattern) | `PatternReport`, `OptimizationResult` | `act_on_pattern_report`, `act_on_portfolio_result` | Retraining recommendation; investigation on non-convergence |
| 5 | Premature exit / late exit edge | `CounterfactualResult` batch | `act_on_counterfactual_batch` | Parameter search (SL/TP/trail geometry) or retraining recommendation |
| 6 | Lifecycle inefficiency (breakeven/trailing/scaling/timing) | `OptimisationReport` | `act_on_lifecycle_report` | Parameter search, experiment proposal, or investigation |
| 7 | **Slippage** (expected vs. actual fill price) | `SlippageReport` | `act_on_slippage_report` | Investigation (`EXECUTION_FAILURE`) + parameter search over execution-quality controls |
| 8 | **Scaling failure** (scale-in/out actions hurting P&L) | `ScalingReport` | `act_on_scaling_report` | Investigation (`EXECUTION_FAILURE`) + shadow-test candidate for a corrected scaling policy |
| 9 | **Capital inefficiency** (idle capital + skipped high-confidence signals) | `CapitalUtilizationReport` | `act_on_capital_utilization` | Investigation (`MARKET_CONDITION_FAILURE`) + parameter search over allocation caps |

Triggers 1–6 were already covered as of Phase 29. Triggers 7–9 close the
previously-missing gaps and are new in this phase.

### 7. Slippage

`SlippageRecord` holds one filled order's expected vs. actual price plus the
ATR at fill time; `slippage_atr()` normalises adverse slippage by ATR so the
same raw price gap is judged differently on a quiet pair vs. a volatile one.
A single bad fill is noise — `act_on_slippage_report` only fires once a
large-enough share of a batch (`SLIPPAGE_SYSTEMATIC_RATIO`, default 50% of
at least `SLIPPAGE_MIN_SAMPLE` fills) clears the adverse-ATR bar
(`SLIPPAGE_ATR_THRESHOLD`, default 0.15 ATR). On trigger it opens an
`EXECUTION_FAILURE` investigation and proposes a Bayesian parameter search
over execution-quality knobs (limit-order offset, acceptable spread) —
never a direct change to order routing.

### 8. Scaling failures

`ScalingOutcome` records the realised P&L impact of one scale-in or
scale-out action. `act_on_scaling_report` groups outcomes by action type and
flags a type only when both the negative-outcome ratio
(`SCALING_FAILURE_RATIO`, default 55%) and the average P&L impact are
negative across at least `SCALING_MIN_SAMPLE` samples. This is deliberately
separate from `act_on_lifecycle_report`'s existing scaling-*improvement*
path (which fires on a positive finding from `LifecycleOptimizer` about an
opportunity to add scaling): this method looks at scaling actions already
taken and catches the case where the policy itself is misfiring. On trigger
it opens an investigation and proposes a **shadow test**
(`ResearchLabAPI.create_shadow_test`) validating a corrected scaling policy
against the current champion — the corrected policy only ever runs in
shadow, never promoted directly.

### 9. Capital inefficiency

`CapitalUtilizationReport` is a portfolio-level snapshot of deployed vs.
idle capital, plus how many high-confidence signals were skipped while
capital sat idle. Idle capital alone is not flagged — it can be prudent risk
management. `act_on_capital_utilization` only fires when idle capital
exceeds `CAPITAL_IDLE_THRESHOLD` (default 35%) *and* at least one
high-confidence signal was skipped, i.e. an opportunity cost rather than a
deliberate risk-off stance. On trigger it opens a `MARKET_CONDITION_FAILURE`
investigation and proposes a parameter search over allocation caps
(`max_allocation_frac`, `max_total_gross`).

## `run_cycle`: unified entry point

`LearningLoopActuator.run_cycle(...)` accepts all 9 report kinds as optional
keyword arguments (`self_assessment`, `pattern_report`,
`counterfactual_batches`, `lifecycle_report`, `portfolio_result` +
`portfolio_baseline_objective`, `outcome_summary`, `slippage_report`,
`scaling_report`, `capital_utilization`). Any subset may be supplied — a
caller (or the scheduler below) is never required to gather all 9 on every
tick.

## Autonomous trigger/scheduler loop

Before this phase, `run_cycle` only ran when something called it — a human,
a test, or a one-off script. `trainer/learning_loop/scheduler.py` adds
`LearningLoopScheduler`, which drives `run_cycle` continuously so all 9
trigger events are watched with no manual call required:

```python
from trainer.learning_loop import LearningLoopActuator, LearningLoopScheduler

actuator = LearningLoopActuator(model_id="qde-live", qde_version="v2.3.0")

def collector() -> dict:
    """Gather whatever of the 9 reports are available right now."""
    return {
        "self_assessment": self_eval_engine.latest_assessment(),
        "slippage_report": execution_monitor.recent_slippage_report(),
        "scaling_report": trade_manager.recent_scaling_report(),
        "capital_utilization": portfolio_allocator.utilization_report(),
        # ... any of the other 5 report kinds, or omit what isn't ready yet
    }

scheduler = LearningLoopScheduler(actuator, collector, interval_seconds=300)
scheduler.start()   # runs run_cycle every 5 minutes on a background thread
...
scheduler.stop()    # graceful shutdown
```

Design points:

- **`collector` is the only integration point.** The scheduler does not know
  how to compute any of the 9 reports itself; it calls `collector()` fresh
  on every tick and passes whatever it returns straight into `run_cycle`.
  This keeps the scheduler decoupled from wherever the underlying
  observation modules (`SelfEvaluationEngine`, `PatternDiscovery`,
  execution/trade-manager telemetry, `PortfolioAllocator`, etc.) actually
  live.
- **A bad tick cannot kill the loop.** `run_once()` catches any exception
  raised by the collector or by `run_cycle`, logs it, and records it in
  `history()` — the background thread keeps ticking on the configured
  interval regardless.
- **`run_once()` is also the deterministic test/manual-trigger entry
  point.** Tests and operators can call it directly without touching the
  background thread or real sleeping.
- **Observability without digging through logs.** `status()` reports
  whether the loop is running, how many cycles have run, total actions
  produced, and the most recent run's outcome; `history(limit)` returns the
  last N cycle records (timestamp, action count, action-type breakdown,
  success/error).
- **No promotion authority.** The scheduler only ever calls
  `LearningLoopActuator.run_cycle`, which only ever produces the same
  proposal-only outputs described above. It has no path to
  `QDEPromotionEngine`, no path to starting a retrain, and no path to
  deploying a shadow-test winner — those remain separate, gated steps.

### Wiring into a live process

`LearningLoopScheduler` is a plain Python class with a background
`threading.Thread` — it does not assume any particular process model. It can
be started inside the trainer's long-running process (e.g. alongside the
existing scheduling/orchestration code), or driven from a dedicated
lightweight worker. Either way, starting it once at process startup with a
`collector` wired to the live observation modules is sufficient to keep all
9 triggers under continuous, autonomous (but non-authoritative) watch.
