---
name: Execution package design
description: trainer/qde/brain/execution/ — learned per-symbol execution quality (spread risk, slippage, fill probability, order timing/type selection).
---

## Rule
Three modules in `trainer/qde/brain/execution/`. No QDEDecision imports anywhere. No hardcoded broker assumptions. Reuses `OrderType`, `TimingRecommendation`, `ExecutionGrade` from `trainer.execution.execution_types`. All cost fields in ATR units (instrument-agnostic).

## Architecture (3 layers)
1. **Structural** — `SlippagePredictor` and `FillProbabilityEngine` are stateless; pure functions of conditions. No learning state.
2. **Historical** — `ExecutionMemory` (per-symbol bounded FIFO, 2000 records). L2-distance nearest-neighbour over 6-dim observation vector.
3. **Integration** — `ExecutionQualityEngine` wraps both; produces 4 downstream signals.

## slippage_predictor.py (STATELESS)
- LIMIT orders: spread_cost_atr = 0 (passive order avoids paying spread; gets adverse selection instead).
- MARKET: spread_cost = 0.5 * SAR; impact = aggressiveness * SAR * 0.30 * vol_factor * thin_factor.
- worst_case = expected * stress_multiplier where stress = 1 + (1-session_quality) * (1-microstructure_quality).
- favorable_probability > 0 only for passive orders.
- Historical correction applied externally by ExecutionQualityEngine.
- Bug fix: `favorable_prob` variable was named `favorable_probability` in the return — now uses `favorable_prob`.

## fill_probability_engine.py (STATELESS)
- MARKET: fill_probability ≈ 1.0; quality base = 0.95 - min(0.50, SAR*0.50) (NOT 1.0 — MARKET always pays liquidity cost).
- LIMIT: fill_prob via logistic decay: 1/(1+(distance/vol_scale)^1.5); quality = 0.92 - adverse_selection.
- **LIMIT fill quality >= MARKET fill quality for same spread** (key invariant). Verified by test. MARKET base 0.95 not 1.0.
- recommend_order_type: SAR<0.25 → MARKET; SAR>=0.25 → LIMIT; SAR>=0.60 AND slip>0.5 → STOP_LIMIT.

## execution_quality_engine.py (STATEFUL, per-symbol)
- Observation vector (6-dim): [spread_atr_ratio, volatility, session_quality, atr_pct, abs_momentum, order_aggressiveness].
- Timing score formula: (1-spread_risk)*0.40 + fill_prob*0.30 + session_quality*0.20 + (1-slip_risk)*0.10.
- Historical adjustment: `hist_drag = (hist_quality - 0.5) * 2 * hist_conf`.
  - Bad fills (hist_quality < 0.5) → negative drag → lower timing score.
  - Good fills (hist_quality > 0.5) → positive drag → higher timing score.
  - **Sign is OPPOSITE to exposure/saturation packages** — exposure uses `(0.5 - hist_quality)` because higher saturation = more caution = correct response; here lower timing = more caution = correct response.
- Integration signals: `conviction_modifier = -0.25*(1-timing_score)`; `reward_modifier = -0.30*exec_cost_score`.
- `context_for_inference` → exactly 16 `exec_*` keys.
- One engine per watched symbol; EURUSD history must NOT influence XAUUSD.

## Learning loop
1. `build_observation(conditions, order_type, distance)` at order submission.
2. Store `ExecutionObservation`.
3. After fill: `ExecutionOutcome.compute_quality(actual_slippage_atr, fill_quality_score, was_filled, structural_slippage_atr)`.
   - No-fill → 0.30 (not 0.0; missed limit ≠ catastrophic slippage).
   - Excess slippage vs. structural prediction drives quality up or down.
4. `record_outcome(observation, outcome)`.
5. `memory_snapshot()` / `load_memory()` for persistence.

## Outstanding wiring (not done)
- Wire `exec_*` keys into `BrainStateManager.tick_symbol` (alongside `regime_*`, `liq_*`, `exp_*`, `opp_*`).
- Add `ExecutionQualityEngine` per-symbol instance to `BrainState` / `BrainStateManager`.
- Call `build_observation` at order submission, `record_outcome` at fill confirmation.
- Update `trainer/qde/brain/__init__.py` to re-export execution package.
