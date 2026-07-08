# Phase S4 Migration Report — Decision Engines as Training-Signal Generators

**Date:** 2026-07-08
**Related:** [`decision_engines_context_providers.md`](decision_engines_context_providers.md) (S1-era conversion of these same engines from decision-makers to runtime context providers), [`phase_s1_migration_report.md`](phase_s1_migration_report.md), [`phase_s2_output_contract.md`](phase_s2_output_contract.md) (`QDEOutputHeads` — the 10-head schema these labels ultimately train).

## Objective

Convert `execution/`, `icari/`, `portfolio/`, `lifecycle/`, and `mie/` from *runtime-adjacent heuristic engines* into *training-signal generators*: their outputs become labels, features, evaluation dimensions, and telemetry consumed by the QDE training pipeline — never runtime decisions, and never fed back into the live decision path. This continues, rather than repeats, the S1 conversion: S1 established that these engines must not gate/veto/mutate a live QDE decision; S4 establishes the positive half of the story — what they *are* for now that they don't decide anything — by giving each engine's category (stop-loss, take-profit, sizing, allocation, lifecycle) a hindsight-optimal label generator trained on actual realized future outcomes, not on the engines' own heuristic rules.

## Survey: current role of the five engine groups (T001)

Confirmed via code audit (no runtime call-site changes required — S1 already closed the gating paths):

| Group | Runtime role today | Decision authority |
|---|---|---|
| `execution/` (`stop_loss_engine.py`, `take_profit_engine.py`, `position_sizer.py`, `execution_planner.py`, ...) | Computes ATR/RR-ratio-based stop/target/size *recommendations*; `ExecutionPlanner` executes QDE's `QDEDecision`, not its own heuristics | Advisory / order-mechanics only (lot clipping, broker constraints) |
| `icari/` (`position_sizing_engine.py`, `allocation_engine.py`, `drawdown_manager.py`, ...) | Capital-at-risk circuit breakers (S1) + advisory sizing suggestions | Advisory + capital circuit-breaker only |
| `portfolio/` (`position_allocator.py`, `optimizer.py`, `exposure_engine.py`, ...) | Advisory allocation/exposure recommendations, `apply_budget_cap=False` by default (S1) | Advisory only |
| `lifecycle/` (`transition_engine.py`, `state_machine.py`, `trade_lifecycle.py`, ...) | FSM structural invariants (`hard_block`) + advisory confidence/news-risk transition recommendations (S1) | Structural block only; soft rules advisory |
| `mie/` (`market_regime_engine.py`, `momentum_intelligence.py`, `risk_intelligence.py`, ...) | Pure feature/telemetry providers — market microstructure intelligence surfaced to QDE as input features | None — already telemetry-only |

No hard-gating call sites were found outside the FSM structural invariant and broker safety rails documented in S1. This means Phase S4's job is additive: build the missing hindsight-label pipeline, not remove more runtime authority.

## New package: `trainer/training_signals/`

A new, dedicated package for hindsight-optimal label generation. It intentionally does **not** live inside `execution/`, `icari/`, `portfolio/`, or `lifecycle/` — those packages retain their live-advisory/telemetry role from S1, and mixing a backward-looking training-label pipeline into a forward-looking advisory engine would blur that boundary again. `trainer/training_signals/` consumes historical, fully-realized outcome data only; nothing in it runs at inference time.

### Shared input type — `outcome_path.py`

`TradeOutcomePath` is the single realized-future-price-path type all five labelers consume: instrument, direction, entry price, and a list of future `OHLCBar`s (reusing the existing `trainer.microstructure.micro_models.OHLCBar` type rather than inventing a parallel one). It exposes `favorable_excursion`/`adverse_excursion` per bar and `max_favorable_excursion()` / `max_adverse_excursion_through()` helpers that every labeler builds on, so "what actually happened, in hindsight" has one canonical definition across all five label types.

### Label generators

| Label | File | Hindsight source (per spec) | NOT derived from |
|---|---|---|---|
| Stop-loss | `stop_loss_labels.py` — `HindsightStopLossLabeler` | Tightest stop distance that survives to the bar where the trade reached its actual max favorable excursion (MFE) | ATR multiples (`stop_loss_engine.py`'s live rule) |
| Take-profit | `take_profit_labels.py` — `HindsightTakeProfitLabeler` | Price level at the realized MFE itself | Fixed reward:risk ratios (`take_profit_engine.py`'s TP1-TP3 live rule) |
| Position-size | `position_size_labels.py` — `HindsightPositionSizeLabeler` | Risk-adjusted return = realized return ÷ realized max adverse excursion, mapped to a `[0,1]` `size_signal`-shaped label via a saturating `tanh` curve | Fixed-fractional / plain-Kelly / volatility-multiple sizing rules (`position_sizer.py`, `position_sizing_engine.py`) |
| Portfolio allocation | `portfolio_allocation_labels.py` — `HindsightAllocationLabeler` | Drawdown-penalized realized-return score per instrument (a portfolio-Sharpe/drawdown proxy), normalized to allocation weights across the supplied instrument set; all-zero weights when every instrument lost money hindsight-optimally | Equal-weight / risk-parity / Sharpe-scaled heuristics (`position_allocator.py`) |
| Lifecycle transitions | `lifecycle_transition_labels.py` — `HindsightLifecycleLabeler` | Sequence of `LifecycleState` transitions (`ACTIVE` → `PROTECTED` at 50% of eventual MFE → `RUNNER`/`TRAILING` at 85% of eventual MFE and at the MFE bar itself → `EXIT_READY` → `CLOSED`; or `ACTIVE` → `STOPPED` directly if the path never moved favorably) derived from the realized path shape | Confidence/news-risk threshold rules (`transition_engine.py`, `position_monitor.py`) |

Each labeler returns a small, serializable dataclass (`to_dict()`) with a human-readable `rationale` string explaining how the label was derived from the realized path — so every label is auditable back to the specific bars that produced it, not a black-box number.

Existing types were reused deliberately rather than duplicated: `OHLCBar` for price bars, `LifecycleState` (20-state enum, `trainer/lifecycle/lifecycle_states.py`) as the lifecycle label vocabulary. `TradeResult`/`TradingMetrics` (`trainer/validation/trading.py`) were surveyed but not reused directly — they summarize a single closed trade's realized metrics rather than a bar-by-bar future path, so they are a natural *downstream* consumer of these labels (e.g. for backtest cross-checking) rather than an input type.

## Test evidence

- `tests/unit/test_training_signals.py` — 18 new tests covering all 5 labelers plus `TradeOutcomePath`: winning/losing/flat/never-favorable/zero-risk paths, monotonic transition ordering, allocation weight normalization (including the all-losers zero-weight case and empty-input case). `pytest tests/unit/test_training_signals.py --noconftest -o addopts="" -q` → **18 passed**.
- Full regression of all five engine groups plus QDE, confirming the S4 additions introduced zero behavioral change to the existing advisory/telemetry runtime paths:
  - `tests/unit/test_execution.py` → 165 passed
  - `tests/unit/test_portfolio.py` → 132 passed
  - `tests/unit/test_icari.py` → 212 passed / 21 failed (pre-existing `AllocationRequest.ensemble_confidence` bug, documented as out-of-scope in the S1 migration report's Appendix — unrelated to S4, unchanged by S4)
  - `tests/unit/test_lifecycle.py` → 182 passed
  - `tests/unit/test_mie.py` → 197 passed
  - `tests/unit/test_qde_inference.py`, `test_qde_outputs.py`, `test_qde_position_manager.py`, `test_qde_rollback.py`, `test_qde_shadow.py`, `test_training_signals.py` (combined) → 104 passed

## Non-goals / explicitly deferred

- **Wiring these labelers into an actual training-data ETL job** (reading historical bars from the trainer's data store, batching per-instrument outcome paths, writing label tensors) is out of scope for S4 — this phase delivers the label-generation *logic* and its contract, matching the S1/S2/S3 pattern of defining the schema/contract in one phase and wiring the pipeline in a later one. `docs/qde/phase_s2_output_contract.md` describes the `QDEOutputHeads` targets these labels are ultimately meant to train against (`size_signal`, `reduce_signal`, etc.); mapping each label 1:1 onto a specific output head's loss function is deferred until the training loop itself is built.
- **`mie/` did not require a labeler** — it was already telemetry-only per the S1 survey table above, so it contributes features to the training pipeline (existing behavior), not new hindsight labels.
