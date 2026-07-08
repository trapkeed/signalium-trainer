---
name: Hindsight training-signal labelers
description: How trainer/training_signals/ derives training labels from realized future outcomes rather than live heuristic rules
---

`trainer/training_signals/` is a dedicated package (separate from `execution/`,
`icari/`, `portfolio/`, `lifecycle/`) that converts fully-realized historical
price paths into hindsight-optimal training labels for QDE's output heads.

**Why:** Phase S1 already made `execution/`, `icari/`, `portfolio/`,
`lifecycle/` pure advisory/telemetry providers at runtime (see
`decision-engines-context-providers.md`). Their heuristic rules (ATR stop
multiples, fixed RR-ratio targets, fixed-fractional/Kelly sizing,
equal-weight/risk-parity allocation, confidence/news-risk transition
thresholds) are appropriate for a live advisory recommendation but are the
wrong source for a *training label* — a label should reflect what the best
possible call actually would have been, given the full realized future, not
re-encode the same heuristic the model is meant to learn to beat.

**How to apply:**
- All 5 labelers consume a shared `TradeOutcomePath` (reuses
  `trainer.microstructure.micro_models.OHLCBar`), never a live snapshot.
- Stop-loss label = tightest stop distance that survives to the realized MFE
  bar. Take-profit label = the realized MFE price itself. Position-size label
  = realized return ÷ realized max adverse excursion (risk-adjusted return),
  saturated into `[0,1]` via `tanh`. Allocation label = drawdown-penalized
  realized-return score per instrument, normalized to weights (all-zero if
  every instrument lost money in hindsight). Lifecycle label = a
  `LifecycleState` transition sequence keyed off fractions of the eventual
  MFE (protect at 50%, runner/trailing at 85% and at the MFE bar).
- None of these labelers run at inference/live time — they are a
  training-data pipeline concern only. Wiring them into an actual
  historical-bar ETL job is intentionally deferred (matches the S1→S2→S3
  pattern of defining a contract before building its pipeline).
- `mie/` needed no labeler — it was already pure telemetry/feature output
  before this phase.
