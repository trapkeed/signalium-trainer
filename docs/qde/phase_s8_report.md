# Phase S8 — QDE Account Awareness

## Goal

Before Phase S8, QDE reasoned about symbols (S7: always-on per-symbol
observation) and open positions (S3-S6: entry/position-management/lifecycle),
but had no notion of the *trading account itself* — its balance trajectory,
margin health, streaks, or activity level. S8 adds that layer: QDE becomes
account-aware, producing a continuous `risk_signal` and `AccountAction`
posture that other engines read as context, the same way `ObserverState`
(S7) is context rather than a rule engine.

The model behaves like an experienced fund manager: **capital protection
first, profit generation second.** Concretely, a small number of hard
capital-protection lines (hard drawdown, margin-call territory, daily/weekly
loss limits) override everything else and force `NO_TRADE`, even against
high direction confidence. Everything short of those lines is a soft,
explainable, additive score — not a black box.

## Scope of this phase

Exactly the four deliverables requested: schema, telemetry schema,
historical database, and feature pipeline. No FastAPI/Express wiring and no
new state machine were added — this phase is the data/reasoning layer that
a future phase can expose via API and wire into the entry/position-sizing
engines, mirroring how S7's engine (`qde_observer.py`) came before S7's API
wiring.

## Modules

| Module | Purpose |
|---|---|
| `qde_account_types.py` | `AccountAction` enum (AGGRESSIVE/NORMAL/DEFENSIVE/REDUCE_RISK/NO_TRADE/WAIT) + every named risk-band threshold (drawdown soft/hard, daily/weekly loss limits, margin warning/critical, risk-utilization high/critical, overtrade multiplier, streak thresholds). No magic numbers elsewhere in the pipeline. |
| `qde_account_models.py` | `AccountSnapshot` (raw context schema — exactly the S8 input list: balance, equity, margin level, free margin, daily/weekly pnl, open exposure, portfolio correlation, drawdown, win/loss streaks, risk utilization, trade frequency), `AccountFeatures` (engineered/history-aware view), `AccountRiskAssessment` (risk_signal + action + `forces_no_trade` override + reason codes). |
| `qde_account_database.py` | SQLite (WAL) persistence: `account_state` (current snapshot+features+assessment per `account_id`, for restart rehydration) + `account_history` (append-only, used as the feature pipeline's lookback source). |
| `qde_account_pipeline.py` | `AccountFeaturePipeline` — pure `compute_features()` (snapshot + history → `AccountFeatures`) and `assess_risk()` (`AccountFeatures` → `AccountRiskAssessment`, the fund-manager heuristic). |
| `qde_account_telemetry.py` | `AccountTelemetryMonitor`/`AccountTelemetrySnapshot` — rolling-window posture distribution, risk-signal mean/volatility, NO_TRADE rate, forced-stop count, time-in-defensive-mode. |

## Account context schema (`AccountSnapshot`)

All 13 S8 input fields, flat and stateless (no thresholds baked in):
`balance`, `equity`, `margin_level_pct`, `free_margin`, `daily_pnl(+pct)`,
`weekly_pnl(+pct)`, `open_exposure`, `portfolio_correlation`,
`current_drawdown_pct`, `winning_streak`, `losing_streak`,
`risk_utilization_pct`, `trades_today`/`trades_this_week`. Keyed by
`account_id` so multiple accounts can be tracked independently.

## Feature engineering (`AccountFeatures`)

History-aware derivations the raw snapshot alone can't answer:

- **Equity high-water-mark & drawdown-from-peak** — computed from
  `account_history`, but reconciled against the snapshot's own
  authoritative `current_drawdown_pct` (via an implied peak) so a cold
  start (no/thin history) can never misreport "new equity high" for an
  account that is actually underwater — the true failure mode caught during
  verification of this phase.
- **PnL trend** — today's `daily_pnl_pct` vs. its own recent rolling average.
- **Streak signal** — signed win/loss streak, plus a severe-losing-streak flag.
- **Margin pressure** — 0-1 scale derived from `margin_level_pct` vs. the
  warning/critical bands.
- **Risk-utilization trend** — rate of change vs. recent history.
- **Trade-frequency ratio** — today's trade count vs. the account's own
  rolling baseline (not a fixed number — what counts as "a lot of trades"
  is account-specific).

## Risk assessment (`assess_risk`)

Hard stops (checked first, in order, any one forces `NO_TRADE` +
`forces_no_trade=True` regardless of the additive score below):

1. Hard drawdown limit breached (`DRAWDOWN_HARD_LIMIT_PCT`)
2. Margin level critical (`MARGIN_LEVEL_CRITICAL_PCT`)
3. Daily or weekly loss limit breached
4. Drawdown approaching the soft limit (`DRAWDOWN_APPROACH_BAND_PCT` window
   below `DRAWDOWN_SOFT_LIMIT_PCT`) — this is the exact S8 example: forces
   `NO_TRADE` even with high direction confidence, because direction
   confidence is never part of this model's inputs to begin with.

Otherwise, an additive score starting from a neutral baseline (0.5),
adjusted by losing/winning streaks, pnl trend, new-equity-high, margin
pressure, low free margin, risk-utilization level, overtrading, and high
portfolio correlation — clamped to [0, 1] and mapped to an `AccountAction`
band. A cold-start account with no evidence at all reports `WAIT` rather
than a falsely-confident `NORMAL`.

## Examples verified

| Scenario | Result |
|---|---|
| 6 losses in a row (isolated from drawdown) | `risk_signal≈0.15`, `REDUCE_RISK`, reason `SEVERE_LOSING_STREAK` — matches the spec's example value. |
| New equity high + winning streak | `risk_signal=0.75`, `AGGRESSIVE`, reasons `WINNING_STREAK`/`NEW_EQUITY_HIGH` — matches the spec's example direction (elevated signal). |
| Drawdown approaching soft limit | `action=NO_TRADE`, `forces_no_trade=True`, regardless of any other favorable inputs — matches the spec's example exactly. |

## Verification performed

- `py_compile` clean across all new `trainer/qde/qde_account_*.py` modules.
- End-to-end functional test: snapshot → `compute_features()` →
  `assess_risk()` → `QDEAccountDatabase.record()` → `AccountTelemetryMonitor.record()`
  for all three spec examples, asserting the expected qualitative and
  (where given) quantitative outputs.
- Persistence round-trip: `get_state()` returns the same action after a
  fresh `QDEAccountDatabase` instance re-opens the same file (simulated
  restart).
- Found and fixed a cold-start bug where a thin-history high-water-mark
  calculation could spuriously report "new equity high" for an account
  that was actually in drawdown; fixed by reconciling against the
  snapshot's own reported `current_drawdown_pct`.

## Deliberately out of scope (future phase)

- No FastAPI/Express endpoints — this phase is the schema/pipeline layer.
- No dashboard views (S7 has `ObserverDashboard` as the template to follow).
- No wiring into `qde_inference.py`/`qde_position_manager.py` to actually
  consume `AccountRiskAssessment` when sizing/gating trades — that
  integration is a natural next step now that the assessment exists.
