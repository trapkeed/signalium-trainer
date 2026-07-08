# Hardcoded Trading Behavior Audit — Signalium Trainer 2.0

**Date:** 2026-07-08
**Objective:** Identify hardcoded trading behavior (thresholds, multipliers, fixed rules) embedded in production decision logic that should instead be learned by the QDE (Quant Decision Engine). The trainer should produce training labels, objectives, evaluation metrics, benchmark datasets, and research experiments — not prescribe behavior.

**Scope:** `trainer/qde/`, `trainer/mie/`, `trainer/microstructure/`, `trainer/execution/`, `trainer/icari/`, `trainer/portfolio/`, `trainer/lifecycle/`, `trainer/analyst/`, `trainer/automl/`, `trainer/models/`, `trainer/sampling/`, `trainer/research/`, and `config/*.yaml`.

---

## Executive Summary

The codebase implements a **rules-based expert system** for the final trading decision layer, not a learned one. Hardcoded thresholds, multipliers, and if/else branches govern market regime detection, stop-loss/take-profit placement, position sizing, risk limits, session behavior, drawdown response, and trade lifecycle state transitions. Critically, `trainer/qde/qde_models.py` — the supposed "learned core" — itself contains hardcoded performance floors used as hard gates, meaning even the QDE's promotion logic is rule-based rather than learned.

None of these values are exposed as model outputs, learned parameters, or training targets today. They are class defaults, YAML config constants, or literal comparisons baked directly into decision code.

---

## 1. Market Regime & State Detection

| File | Lines | Hardcoded Behavior |
|---|---|---|
| `trainer/mie/market_regime_engine.py` | 79-84 | ADX thresholds: `adx_strong=40.0`, `adx_trending=25.0`, `adx_weak=15.0`; volatility ratios: `vol_extreme=4.0`, `vol_high=2.5`, `vol_compress=0.5` |
| `trainer/mie/market_regime_engine.py` | 135 | Black Swan detection: `vol_ratio > vol_extreme * 1.5` or `spread > atr * 0.8` |
| `trainer/mie/market_regime_engine.py` | 140 | Illiquid regime: `volume < avg_vol * 0.10` or `spread > atr * 0.5` |
| `trainer/mie/market_regime_engine.py` | 150 | Volatile regime: `abs_vol_ratio > 10.0` |
| `trainer/mie/market_regime_engine.py` | 156 | Breakout detection: `abs(price_move) > 2 * atr / close` |
| `trainer/mie/market_regime_engine.py` | 182 | Expansion regime: `recent_range > older_range * 1.3` |
| `trainer/microstructure/state_classifier.py` | 39-47 | 12 hardcoded market-state score thresholds, e.g. `HIGH_VOL_THRESHOLD = 0.75`, `BREAKOUT_THRESH = 0.70` |
| `trainer/microstructure/state_classifier.py` | 113 | Strong Trend: `ts >= 0.75` and `directional_persistence >= 0.70` |
| `trainer/microstructure/state_classifier.py` | 128 | Fake Breakout: `reversal_strength >= 0.60` and `directional_persistence < 0.45` |
| `trainer/microstructure/state_classifier.py` | 211-215 | Liquidity states: Vacuum `prob >= 0.70`; Deep `spread <= 0.20` and `candle_freq <= 0.15` |
| `config/mie/mie.yaml` | 36-37 | RSI thresholds: `rsi_overbought: 70.0`, `rsi_oversold: 30.0` |
| `config/mie/mie.yaml` | 127 | `drawdown_atr_mult: 0.80` for regime-based drawdown |
| `trainer/portfolio/regime_detector.py` | 30-64 | Static `REGIME_ADJUSTMENTS` matrix (e.g. Trending → `momentum_boost: 1.3`; High Volatility → `position_size: 0.6`) |
| `config/icari/icari.yaml` | 48-51 | Regime multipliers: `TRENDING: 1.10`, `RANGING: 0.85`, `VOLATILE: 0.65` |

## 2. Execution: Stop Loss & Take Profit

| File | Lines | Hardcoded Behavior |
|---|---|---|
| `trainer/execution/stop_loss_engine.py` | 88-92 | Fixed ATR multiples per stop type: INITIAL 1.5x, ADAPTIVE 1.2x, ATR 2.0x, VOLATILITY 2.5x, EMERGENCY 3.5x |
| `trainer/execution/stop_loss_engine.py` | 124 | Confidence floor formula: `0.5 + min(atr_spread_ratio / 20, 0.4)` |
| `trainer/execution/take_profit_engine.py` | 86-88 | Fixed reward:risk targets: TP1 1:1, TP2 1.5:1, TP3 2:1 |
| `trainer/execution/take_profit_engine.py` | 131 | Fixed hit-probability decay: `decay = 0.70` across TP levels |
| `trainer/execution/trade_manager.py` | 62 | `partial_close_tp2_pct: float = 0.30` |
| `trainer/execution/trade_manager.py` | 124 | Trailing stop: fixed `1.0 × ATR` distance |

## 3. Risk Management, Position Sizing & Exposure

| File | Lines | Hardcoded Behavior |
|---|---|---|
| `trainer/execution/position_sizer.py` | 32-35 | Fixed risk caps: `max_risk_per_trade_pct: 0.02`, `max_daily_risk_pct: 0.06`, `max_weekly_risk_pct: 0.12`, `max_portfolio_risk_pct: 0.20` |
| `trainer/execution/position_sizer.py` | 143 | `0.02` hard cap per trade |
| `trainer/execution/position_sizer.py` | 220 | Fixed-fractional fallback: `risk_amount = account * 0.01` |
| `trainer/execution/position_sizer.py` | 228 | Volatility adjustment clamp: `_clamp(vol_ratio, 0.1, 3.0)` |
| `trainer/execution/position_sizer.py` | 239 | Half-Kelly safety factor: `safe_kf = kf * 0.5` |
| `trainer/execution/execution_types.py` | 137 | `max_leverage: float = 30.0` |
| `config/execution/execution.yaml` | 19-20 | `max_leverage: 30.0`, `margin_rate: 0.033` |
| `trainer/icari/position_sizing_engine.py` | 24 | `_MAX_RISK_FRAC = 0.05` (never risk more than 5% per trade) |
| `trainer/icari/position_sizing_engine.py` | 116 | `vol_mult = max(0.25, 1.0 - volatility * 10)` |
| `trainer/icari/position_sizing_engine.py` | 131 | `ev_mult = max(0.1, min(2.0, 1.0 + expected_value * 50))` |
| `trainer/icari/dynamic_sizer.py` | 36 | `max_correlation: float = 0.70` |
| `trainer/icari/dynamic_sizer.py` | 154 | `if exposure >= 0.70:` |
| `trainer/icari/capital_health.py` | 110-111 | `var_penalty = min(0.30, var_95 * 3)`, `dd_penalty = min(0.30, drawdown * 3)` |
| `trainer/portfolio/risk_budget.py` | 58-69 | Fixed exposure caps: `max_usd_exposure: 0.60`, `max_gold_exposure: 0.25`, `max_crypto_exposure: 0.20`, `max_total_gross: 1.50` |
| `trainer/portfolio/risk_budget.py` | 74-98 | Hardcoded CONSERVATIVE vs AGGRESSIVE presets (e.g. Aggressive `max_total_gross: 2.00`) |
| `trainer/portfolio/risk_budget.py` | 34-36 | Utilization gates: `>= 1.0`, `0.85` |
| `config/portfolio/portfolio.yaml` | 36-37, 53, 64 | `max_single_instrument: 0.30`; `max_correlated_exposure` at `0.50` / `0.40` / `0.70` per tier |

## 4. Fixed Drawdown Rules

| File | Lines | Hardcoded Behavior |
|---|---|---|
| `trainer/icari/drawdown_manager.py` | 29-32 | Drawdown limits: `soft_limit: 0.05`, `hard_limit: 0.10`, `emergency_limit: 0.20` |
| `trainer/icari/drawdown_manager.py` | 103-110 | Risk multipliers by mode: SOFT 0.75, HARD 0.50, PRESERVATION 0.25, RECOVERY 0.60 |
| `trainer/qde/qde_models.py` | 72 | `max_drawdown_allowed: float = 0.25` |
| `trainer/qde/qde_models.py` | 85-88 | `min_sharpe: 0.5`, `min_profit_factor: 1.1`, `max_drawdown_abs: 0.30`, `min_win_rate: 0.45` — hard promotion gates |

## 5. Fixed Session Multipliers

| File | Lines | Hardcoded Behavior |
|---|---|---|
| `config/execution/execution.yaml` | 87-91 | Session multipliers: `OVERLAP: 1.00`, `LONDON: 0.85`, `NEW_YORK: 0.80`, `TOKYO: 0.55` |

## 6. Fixed Scaling Rules

| File | Lines | Hardcoded Behavior |
|---|---|---|
| `config/execution/execution.yaml` | 107-109 | `scale_in_confidence: 0.75`, `scale_in_lots_pct: 0.50` |
| `trainer/execution/trade_manager.py` | 62 | `partial_close_tp2_pct: 0.30` |

## 7. Fixed Strategy Selection

| File | Lines | Hardcoded Behavior |
|---|---|---|
| `trainer/models/quant_v1/hybrid/fusion.py` | 52, 60, 73 | if/elif on `FusionStrategy` enum (CONCAT / WEIGHTED / ATTENTION) selecting fusion logic |
| `trainer/sampling/strategy.py` | 192, 210, 227 | if/elif on `BalancingStrategy` (WEIGHTED_SAMPLER / CLASS_WEIGHTS / OVERSAMPLE) |
| `trainer/portfolio/position_allocator.py` | 115-128 | if/elif chain selecting allocation method (EQUAL_WEIGHT, RISK_PARITY, etc.) |
| `trainer/portfolio/regime_detector.py` | 30-64 | Regime → strategy-parameter map (`REGIME_ADJUSTMENTS`) determines trading behavior directly from a fixed lookup, not a learned mapping |

## 8. Fixed Confidence Thresholds & Lifecycle Gating

| File | Lines | Hardcoded Behavior |
|---|---|---|
| `trainer/lifecycle/state_machine.py` | 55, 67, 79, 91 | Confidence gates for state transitions: IDEA→WATCHLIST 0.40, WATCHLIST→PREPARATION 0.55, PREPARATION→ENTRY_READY 0.65 |
| `trainer/lifecycle/state_machine.py` | 315-344 | Contextual gates: `news_risk < 0.40`, `spread < atr * 0.30`, `momentum > 0.20` |
| `config/lifecycle/lifecycle.yaml` | 38-44 | Confidence gates: Watchlist 0.50, Entry Ready 0.65, Scale In 0.75 |
| `config/lifecycle/lifecycle.yaml` | 76-77 | Volatility thresholds: `high: 0.020`, `extreme: 0.035` |
| `config/memory/memory.yaml` | 7, 98 | `min_confidence_threshold: 0.40`, `overconfidence_threshold: 0.80` |

## 9. Other Hardcoded Numeric Behavior (Adjacent Systems)

These aren't direct trading rules but reflect the same "hardcode instead of learn" pattern in adjacent research/AutoML tooling:

| File | Lines | Hardcoded Behavior |
|---|---|---|
| `trainer/portfolio/scenario_analysis.py` | 44-153 | Fixed return-shock dictionaries for 12+ stress scenarios |
| `trainer/portfolio/scenario_analysis.py` | 159-172 | Fixed volatility multipliers per scenario (e.g. `VOLATILITY_SPIKE: 2.5`) |
| `trainer/portfolio/scenario_analysis.py` | 264 | `RISK_FREE_RATE = 0.045` |
| `trainer/analyst/retrieval_engine.py` | 75-76 | Fixed BM25 constants `K1 = 1.5`, `B = 0.75` |
| `trainer/analyst/citation_engine.py` | 94, 123 | Fixed relevance scores `0.60` / `0.80` |
| `trainer/automl/objectives.py` | 111-128 | Hardcoded metric clip ranges (`profit_factor` 0-5, `sharpe_ratio` -3 to 5) |
| `trainer/automl/resource_manager.py` | 240-243 | Utilization caps: RAM `* 0.9`, GPU `* 0.95` |

---

## Key Finding: The QDE Core Is Not Actually Learned

`trainer/qde/qde_models.py` — meant to be the learned decision core — hardcodes its own promotion gates (Sharpe, drawdown, win rate, profit factor floors) as class defaults rather than exposing them as configurable evaluation metrics or learned/tunable parameters. This means the "trained" model is still ultimately accepted or rejected by a fixed rule, not a learned criterion.

---

## Recommended Conversion Plan

For each category above, the fixed rule should be replaced with one of the following trainer-owned artifacts instead of remaining as inline production logic:

| Hardcoded System | Convert To |
|---|---|
| ADX / RSI / volatility regime thresholds | **Training labels** — regime classification learned from labeled historical windows |
| ATR stop/TP multipliers, R:R targets, decay | **Training objective** — the model outputs stop/TP distances directly, trained against realized MFE/MAE |
| Fixed risk %, exposure caps, Kelly factors | **Evaluation metric + benchmark constraint** — risk sizing becomes a learned output, evaluated against a risk-adjusted-return objective, with the old fixed caps demoted to a safety-rail benchmark used only in evaluation, not as decision logic |
| Fixed drawdown limits / mode multipliers | **Training objective + evaluation metric** — a drawdown-aware objective (e.g. CVaR-penalized) instead of static mode switches |
| Fixed session multipliers | **Training label/feature** — session becomes an input feature; its effect is learned, not multiplied in |
| Fixed scaling/partial-close percentages | **Training label** — scale-in/out amount predicted by the model, benchmarked against optimal realized scaling in research experiments |
| Fixed strategy selection (if/else, regime maps) | **Research experiment** — treat strategy choice as a learned policy output (e.g. softmax over strategies) evaluated via backtest benchmark datasets, not hardcoded branching |
| Fixed confidence thresholds / lifecycle gates | **Training label + evaluation metric** — confidence calibration learned and validated via calibration curves, not fixed cutoffs |
| QDE promotion gates (Sharpe/DD/win-rate floors) | **Evaluation metric / benchmark dataset** — promotion decided by comparison against a benchmark suite and statistical significance testing, not fixed literal floors baked into `qde_models.py` |

---

## Next Steps

This report is a discovery/audit deliverable only — no production code has been modified. Suggested follow-up:
1. Prioritize by risk-of-blast-radius: stop-loss/take-profit and position sizing first (direct capital exposure), then regime detection, then lifecycle gating.
2. For each item, define the training label schema and evaluation metric before touching the production code path.
3. Migrate hardcoded YAML values into benchmark/eval configs (kept for backtesting comparison) while the QDE begins outputting these as learned values behind a feature flag, so old rule-based behavior can be A/B compared against the learned version before full cutover.
