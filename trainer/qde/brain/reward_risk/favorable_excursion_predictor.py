"""QDE Brain — reward_risk: structural favorable excursion predictor.

``FavorableExcursionPredictor`` is a **stateless, structural** model
that estimates the *distribution* of Maximum Favorable Excursion (MFE)
from first principles — favorable momentum, session quality, volatility,
spread, and microstructure — without requiring any trade history.

Why a separate MFE predictor?
------------------------------
The projection package (``ExpectedMoveEngine``) computes a weighted
mean MFE from similar historical setups.  The mean alone cannot answer:

  "If I place a take-profit 1.5 ATR away, what is the probability it
   gets hit before the trade reverses?"

Answering that question requires the *distribution shape* (quantiles) of
the MFE, not just its mean.  This predictor supplies that shape.

The ``RewardRiskEngine`` blends these structural MFE estimates with
historical observations as trade history accumulates.

Model
-----
MFE is modelled as log-normally distributed (same family as MAE):

* MFE cannot be negative (it is a non-negative excursion magnitude).
* The distribution is right-skewed — most trades have modest favorable
  excursions but occasional large ones.

The log-normal *mean* (= ``expected_mfe_atr``) is estimated from an
additive structural formula:

  * ``momentum_contribution`` — positive (favorable) momentum is the
    primary driver of MFE.  When the market is already moving in the
    trade direction, the favorable excursion tends to be larger.

  * ``vol_contribution`` — volatility allows the price to travel further
    from entry in *any* direction, including the favorable one.

  * ``session_boost`` — active trading sessions sustain directional
    moves better than thin, off-hours sessions.

  * ``micro_boost`` — clean order flow (low spread noise, steady depth)
    supports directional price action without false reversals.

  * ``spread_drag`` — a wider spread reduces the *net* favorable
    excursion from the entry price.  A move of X ATR yields only
    X − spread_cost net favorable excursion.

Capture rate
------------
``optimal_target_atr`` is the P50 of the MFE distribution — the level
that approximately half of similar setups historically reach before
reversing.  Setting a take-profit at or below this level achieves a
> 50 % hit rate against the structural prior.

The ``RewardRiskEngine`` refines ``optimal_target_atr`` by learning
the *actual realized capture rate* (reward_atr / mfe_atr) from
completed trades.

Design rules
------------
* No import of ``QDEDecision`` or any execution-layer type.
* No import from ``trainer.qde.brain.projection`` (independent package).
* Entirely stateless — safe to instantiate per-call or once at startup.
* All magnitude outputs are in ATR units (instrument-agnostic).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from trainer.qde.brain.reward_risk.adverse_excursion_predictor import (
    _SQRT2,
    _VOL_CEIL,
    _lognormal_quantiles,
)

# ── Constants ─────────────────────────────────────────────────────────────────

_MFE_FLOOR = 0.05  # minimum structural MFE estimate (ATR units)

# Log-normal σ bounds for MFE.  MFE tends to be slightly less dispersed
# than MAE because it requires sustained directional price action,
# whereas MAE can arise from any short-term noise spike.
_MFE_SIGMA_MIN = 0.35
_MFE_SIGMA_MAX = 0.80

# Session quality factor bounds: [0.65, 1.35]
_SESSION_BOOST_LO = 0.65
_SESSION_BOOST_HI = 1.35

# Microstructure factor bounds: [0.80, 1.20]
_MICRO_BOOST_LO = 0.80
_MICRO_BOOST_HI = 1.20


# ── FavorableExcursionInput ───────────────────────────────────────────────────


@dataclass
class FavorableExcursionInput:
    """Structural inputs for the favorable excursion predictor.

    All fields are instrument-agnostic normalised values.

    ``spread_atr_ratio``
        Current bid-ask spread / ATR.  Reduces net favorable excursion
        from entry.

    ``atr_pct``
        ATR / price (relative volatility).  Retained for future extensions.

    ``volatility``
        Realised rolling volatility (fractional, e.g. 0.01 = 1%).

    ``session_quality``
        0-1 session timing quality.  1 = London-NY overlap; 0 = closed.

    ``momentum``
        Signed −1..+1.  Positive = upward price trend.

    ``direction``
        +1 = long, −1 = short.  Used to compute directed momentum.

    ``microstructure_quality``
        0-1 order-flow / spread cleanliness quality.

    ``target_distance_atr``
        Proposed take-profit distance in ATR units.  0 = not specified.
        When > 0, ``target_hit_probability`` is computed analytically.
    """

    spread_atr_ratio: float = 0.10
    atr_pct: float = 0.005
    volatility: float = 0.010
    session_quality: float = 0.80
    momentum: float = 0.0
    direction: int = 1
    microstructure_quality: float = 0.80
    target_distance_atr: float = 0.0

    @property
    def directed_momentum(self) -> float:
        """Positive when momentum is in trade direction; negative otherwise."""
        return self.momentum * self.direction


# ── FavorableExcursionEstimate ────────────────────────────────────────────────


@dataclass(frozen=True)
class FavorableExcursionEstimate:
    """Structural distribution estimate for Maximum Favorable Excursion.

    All magnitude fields are in ATR units.

    ``expected_mfe_atr``
        Log-normal mean.

    ``mfe_p50_atr``
        Median MFE.  Half of similar setups will produce at least this
        favorable excursion.  The natural conservative take-profit level:
        targeting here achieves > 50 % hit rate structurally.

    ``mfe_p75_atr``
        75th percentile.  Targeting here means only the strongest 25 %
        of similar setups will actually reach the target.

    ``mfe_p90_atr``
        90th percentile.  Aspirational take-profit for exceptional setups.

    ``target_hit_probability``
        P(MFE > ``target_distance_atr``).  0.0 when ``target_distance_atr``
        <= 0.

    ``optimal_target_atr``
        Structural median (P50) — the take-profit distance at which
        approximately half of similar setups are reached.

    ``log_sigma``
        Internal dispersion parameter for blending with historical data.
    """

    expected_mfe_atr: float = 0.0
    mfe_p50_atr: float = 0.0
    mfe_p75_atr: float = 0.0
    mfe_p90_atr: float = 0.0
    target_hit_probability: float = 0.0
    optimal_target_atr: float = 0.0
    log_sigma: float = 0.55

    def to_dict(self) -> dict:
        return {
            "expected_mfe_atr": round(self.expected_mfe_atr, 4),
            "mfe_p50_atr": round(self.mfe_p50_atr, 4),
            "mfe_p75_atr": round(self.mfe_p75_atr, 4),
            "mfe_p90_atr": round(self.mfe_p90_atr, 4),
            "target_hit_probability": round(self.target_hit_probability, 4),
            "optimal_target_atr": round(self.optimal_target_atr, 4),
            "log_sigma": round(self.log_sigma, 4),
        }


# ── FavorableExcursionPredictor ───────────────────────────────────────────────


class FavorableExcursionPredictor:
    """Stateless structural estimator for the MFE distribution.

    Instantiate once and call ``estimate()`` on every tick.  No internal
    state; calls are thread-safe.

    The structural model provides a baseline for ``RewardRiskEngine``
    before sufficient trade history exists.
    """

    def estimate(self, inp: FavorableExcursionInput) -> FavorableExcursionEstimate:
        """Compute the structural MFE distribution for these conditions.

        Parameters
        ----------
        inp:
            Live market conditions and optional proposed take-profit distance.

        Returns
        -------
        ``FavorableExcursionEstimate`` with expected MFE, quantiles,
        optimal target, and (if ``target_distance_atr > 0``) the
        probability of reaching that target.
        """
        expected_mfe = self._expected_mfe(inp)
        sigma = self._log_sigma(inp)
        p50, p75, p90 = _lognormal_quantiles(expected_mfe, sigma)
        target_hit = _target_hit_prob(inp.target_distance_atr, expected_mfe, sigma)

        return FavorableExcursionEstimate(
            expected_mfe_atr=round(expected_mfe, 6),
            mfe_p50_atr=round(p50, 6),
            mfe_p75_atr=round(p75, 6),
            mfe_p90_atr=round(p90, 6),
            target_hit_probability=round(target_hit, 4),
            optimal_target_atr=round(p50, 6),  # P50 = natural conservative target
            log_sigma=round(sigma, 4),
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    def _expected_mfe(self, inp: FavorableExcursionInput) -> float:
        """Structural estimate of the log-normal mean MFE in ATR units.

        Component breakdown:

        ``momentum_contribution``
            Favorable (in-direction) momentum is the dominant MFE driver.
            At directed_momentum = +1, this contributes 0.80 ATR.  At
            zero or adverse momentum it contributes nothing.

        ``vol_contribution``
            Volatility expands both directions.  At the ceiling (2% vol)
            it adds 0.50 ATR.

        ``session_boost``
            Active sessions sustain directional moves.  Ranges from 0.65
            (closed) to 1.35 (fully active overlap), applied as a
            multiplicative factor on (base + momentum + vol).

        ``micro_boost``
            Clean order flow supports sustained directional price action.
            Ranges from 0.80 (noisy) to 1.20 (clean), multiplicative.

        ``spread_drag``
            The bid-ask spread is paid at entry, reducing the *net*
            favorable excursion from entry mid-price.  Coefficient 0.30
            converts SAR to ATR fraction.
        """
        dm = inp.directed_momentum
        vol_norm = min(1.0, inp.volatility / _VOL_CEIL)

        mom_contribution = max(0.0, dm) * 0.80
        vol_contribution = vol_norm * 0.50

        session_boost = _SESSION_BOOST_LO + inp.session_quality * (
            _SESSION_BOOST_HI - _SESSION_BOOST_LO
        )
        micro_boost = _MICRO_BOOST_LO + inp.microstructure_quality * (
            _MICRO_BOOST_HI - _MICRO_BOOST_LO
        )
        spread_drag = inp.spread_atr_ratio * 0.30

        base = 0.20 + mom_contribution + vol_contribution
        expected_mfe = max(_MFE_FLOOR, base * session_boost * micro_boost - spread_drag)
        return expected_mfe

    def _log_sigma(self, inp: FavorableExcursionInput) -> float:
        """Log-normal dispersion parameter σ for MFE.

        Higher vol environments produce more variable MFE (some trades
        catch a big move; many do not).  Poor session quality adds
        additional variability.
        """
        vol_norm = min(1.0, inp.volatility / _VOL_CEIL)
        sigma = _MFE_SIGMA_MIN + vol_norm * 0.25 + (1.0 - inp.session_quality) * 0.15
        return min(_MFE_SIGMA_MAX, sigma)


# ── Local helper ──────────────────────────────────────────────────────────────


def _target_hit_prob(
    target_distance_atr: float,
    expected_mfe: float,
    sigma: float,
) -> float:
    """P(MFE > target_distance_atr) from the structural log-normal model.

    Returns 0.0 when ``target_distance_atr <= 0`` (target not specified).
    """
    if target_distance_atr <= 0.0:
        return 0.0
    expected_mfe = max(expected_mfe, 1e-9)
    mu = math.log(expected_mfe) - sigma**2 / 2.0
    z = (math.log(max(target_distance_atr, 1e-9)) - mu) / sigma
    return max(0.0, min(1.0, 0.5 * math.erfc(z / _SQRT2)))
