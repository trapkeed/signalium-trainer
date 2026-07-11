"""QDE Brain — reward_risk: learned trade geometry and reward/risk estimation.

This package gives the QDE distribution-aware estimates of trade geometry:
how far the price typically moves against a trade (MAE), how far it moves
in the trade's favor (MFE), and what the realized reward/risk ratio has
historically been — all expressed as *distributions*, not single averages.

Problem being solved
--------------------
The projection package (``ExpectedMoveEngine``) answers: "In similar
market conditions, what was the *average* trajectory?"  That is useful
for context, but stop placement and target placement are *tail* problems:

  * A stop set at the mean MAE will be triggered more than 50 % of the
    time by normal adverse excursion, even if the trade eventually moves
    in the right direction.
  * A take-profit set at the mean MFE requires the favorable excursion
    to exceed the mean — which happens less than half the time for the
    right-skewed MFE distribution.

This package answers the tail questions:

  "If I place a stop 1.2 ATR away, what is the probability it gets hit
   before the trade resolves?"  → ``stop_hit_probability``

  "Where is the 75th-percentile adverse excursion, so my stop survives
   at least 75 % of normal adverse moves?"  → ``mae_p75_atr``

  "What fraction of similar trades were profitable, and what fraction
   achieved a 2:1 reward/risk outcome?"  → ``win_rate``, ``win_rate_at_rr2``

  "How much of the maximum favorable excursion did similar trades
   actually capture as realized profit?"  → ``expected_capture_rate``

What this package provides that projection does not
---------------------------------------------------
+--------------------------+------------------+------------------+
| Signal                   | Projection pkg   | Reward/Risk pkg  |
+--------------------------+------------------+------------------+
| Mean MAE / MFE           | ✓ (weighted avg) | ✓ (blended)     |
| P50 / P75 / P90 MAE      | ✗                | ✓               |
| P50 / P75 MFE            | ✗                | ✓               |
| Stop-hit probability      | ✗                | ✓               |
| Target-hit probability    | ✗                | ✓               |
| Win rate                  | ✗                | ✓               |
| Win rate at ≥2:1 RR       | ✗                | ✓               |
| Capture rate              | ✗                | ✓               |
| Kelly-inspired sizing     | ✗                | ✓               |
| Structural no-history     | ✗                | ✓               |
+--------------------------+------------------+------------------+

Architecture (three modules)
----------------------------
``adverse_excursion_predictor``
    Stateless structural model for the MAE distribution.  Requires no
    trade history.  Used as a baseline when history is thin.

    * ``AdverseExcursionInput`` — conditions bag
    * ``AdverseExcursionEstimate`` — expected_mae_atr, mae_p50/p75/p90_atr,
      stop_hit_probability, log_sigma
    * ``AdverseExcursionPredictor.estimate(inp)`` → ``AdverseExcursionEstimate``

    Model: MAE ~ LogNormal(μ, σ).  Mean driven by spread, volatility,
    adverse momentum, session quality, microstructure.  Dispersion (σ)
    driven by session and microstructure uncertainty.  Stop-hit
    probability computed analytically from the log-normal CDF.

``favorable_excursion_predictor``
    Stateless structural model for the MFE distribution.

    * ``FavorableExcursionInput`` — conditions bag
    * ``FavorableExcursionEstimate`` — expected_mfe_atr, mfe_p50/p75/p90_atr,
      target_hit_probability, optimal_target_atr, log_sigma
    * ``FavorableExcursionPredictor.estimate(inp)`` → ``FavorableExcursionEstimate``

    Model: MFE ~ LogNormal(μ, σ).  Mean driven by favorable momentum,
    session quality, microstructure, volatility (minus spread drag).
    Target-hit probability computed from log-normal CDF.

``reward_risk_engine``
    Learned multi-symbol engine.  Wraps both structural predictors with
    ``RewardRiskMemory`` (per-symbol bounded FIFO, 2000 records).

    * ``RewardRiskConditions`` — 7-dim conditions (instrument, spread,
      atr_pct, volatility, session, momentum, direction, microstructure)
    * ``TradeGeometryOutcome`` — realized mae/mfe/reward/capture; factory:
      ``TradeGeometryOutcome.compute(mae, mfe, reward, stopped, hit_target)``
    * ``RewardRiskRecord`` — (conditions, outcome, timestamp)
    * ``RewardRiskMemory`` — bounded FIFO with snapshot persistence
    * ``RewardRiskSignal`` — frozen dataclass with all 18 fields
    * ``RewardRiskEngine`` — multi-symbol.
        ``score(conditions, stop_distance_atr, target_distance_atr)``
        → ``RewardRiskSignal``
        ``context_for_inference(...)`` → 18 ``rr_*`` keys
        ``record_outcome(conditions, outcome)`` → learning loop
        ``to_snapshot(instrument)`` / ``load_snapshot(instrument, data)``

Integration signals (18 ``rr_*`` keys)
---------------------------------------
These keys are merged into ``QDEInferenceInput.brain_context`` alongside
the existing ``proj_*``, ``exec_*``, ``regime_*``, ``liq_*``, ``exp_*``
keys:

    rr_expected_mae_atr        — mean MAE in ATR (blended)
    rr_mae_p75_atr             — 75th-pct MAE: stop placement floor
    rr_mae_p90_atr             — 90th-pct MAE: stress bound
    rr_expected_mfe_atr        — mean MFE in ATR
    rr_mfe_p50_atr             — median MFE: conservative take-profit
    rr_mfe_p75_atr             — 75th-pct MFE: aggressive take-profit
    rr_expected_rr             — expected_mfe / expected_mae
    rr_win_rate                — empirical win rate (0-1)
    rr_win_rate_at_rr2         — win rate achieving ≥2:1 reward/risk
    rr_stop_hit_probability    — P(MAE > stop_distance_atr)
    rr_target_hit_probability  — P(MFE > target_distance_atr)
    rr_expected_capture_rate   — captured fraction of MFE as reward
    rr_position_size_modifier  — Kelly-inspired 0.5..2.0
    rr_risk_appetite_modifier  — -0.50..+0.25
    rr_opportunity_rank_signal — 0-1 for trade selection
    rr_confidence              — 0-1 based on similar-record count
    rr_sample_count            — records used in estimate
    rr_is_structural           — 1.0 = pure structural; 0.0 = history used

Design rules
------------
* No ``QDEDecision`` import anywhere in this package.
* No import from ``trainer.qde.brain.projection`` (independent package).
* All magnitude outputs in ATR units (instrument-agnostic).
* Structural predictors are stateless; ``RewardRiskEngine`` is the only
  stateful component.
* ``RewardRiskEngine`` is shared across symbols (multi-symbol memory,
  keyed by instrument name).  Do not instantiate per-symbol.
"""

from __future__ import annotations

# ── Adverse excursion predictor ───────────────────────────────────────────────
from trainer.qde.brain.reward_risk.adverse_excursion_predictor import (
    AdverseExcursionEstimate,
    AdverseExcursionInput,
    AdverseExcursionPredictor,
)

# ── Favorable excursion predictor ─────────────────────────────────────────────
from trainer.qde.brain.reward_risk.favorable_excursion_predictor import (
    FavorableExcursionEstimate,
    FavorableExcursionInput,
    FavorableExcursionPredictor,
)

# ── Reward/risk engine ────────────────────────────────────────────────────────
from trainer.qde.brain.reward_risk.reward_risk_engine import (
    RewardRiskConditions,
    RewardRiskEngine,
    RewardRiskMemory,
    RewardRiskRecord,
    RewardRiskSignal,
    TradeGeometryOutcome,
)

__all__ = [
    # adverse excursion
    "AdverseExcursionInput",
    "AdverseExcursionEstimate",
    "AdverseExcursionPredictor",
    # favorable excursion
    "FavorableExcursionInput",
    "FavorableExcursionEstimate",
    "FavorableExcursionPredictor",
    # engine
    "RewardRiskConditions",
    "TradeGeometryOutcome",
    "RewardRiskRecord",
    "RewardRiskMemory",
    "RewardRiskSignal",
    "RewardRiskEngine",
]
