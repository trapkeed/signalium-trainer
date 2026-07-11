"""QDE Brain — reward_risk: structural adverse excursion predictor.

``AdverseExcursionPredictor`` is a **stateless, structural** model that
estimates the *distribution* of Maximum Adverse Excursion (MAE) from
first principles — spread, volatility, session quality, momentum
direction, and microstructure — without requiring any trade history.

Why a separate MAE predictor?
------------------------------
The projection package (``ExpectedMoveEngine``) computes a *weighted
mean* MAE from similar historical setups.  That single number is useful
as a summary but it hides the spread of outcomes.  Stop placement is
specifically a *tail-risk* problem: you need to know where the 75th or
90th percentile of the MAE distribution sits, not just its mean.

This predictor answers that question structurally.  The
``RewardRiskEngine`` blends these structural estimates with historical
observations as trade history accumulates.

Model
-----
MAE is modelled as log-normally distributed.  The intuition:

* MAE cannot be negative (it is a non-negative excursion magnitude).
* The distribution is right-skewed — most trades have modest adverse
  excursions, but a minority have very large ones.
* The log-normal family captures both properties naturally.

The log-normal *mean* (= ``expected_mae_atr``) is estimated from a
structural additive formula whose components each have a clear economic
interpretation:

  * ``spread_contribution`` — the bid-ask spread is an *immediate*
    adverse excursion for any MARKET order.  Half the spread is paid
    at entry; the full spread is the realized adverse move from
    mid-price.  Coefficient 0.50 converts spread → ATR fraction.

  * ``vol_contribution`` — higher realised volatility means larger
    price swings.  We normalise against a "typical high-volatility"
    ceiling (2 % rolling vol) so the coefficient is instrument-agnostic.

  * ``adverse_momentum`` — when price momentum is already against the
    trade direction, the adverse excursion tends to be larger.  This is
    the component most sensitive to direction choice.

  * ``session_penalty`` — thin or off-hours sessions produce gappier
    price action.  Off-hours MAE is empirically 20-30 % larger than
    London/NY-overlap MAE for most instruments.

  * ``microstructure_penalty`` — noisy order flow (wide quote changes,
    low depth) produces erratic short-term excursions even in calm
    directional markets.

The log-normal *dispersion* parameter (σ) is set as a function of
session quality and microstructure quality, capturing the idea that
unpredictable market conditions produce more *variable* excursions, not
merely larger average ones.

Output quantiles
----------------
Three quantile estimates are produced from the log-normal (μ, σ):

  * ``mae_p50_atr``  — median MAE.  Half of similar setups stay within
    this level.  The minimum sensible stop distance.
  * ``mae_p75_atr``  — 75th percentile.  A stop inside this level has a
    > 25 % probability of being hit by *normal* adverse excursion, even
    if the trade eventually moves in the right direction.
  * ``mae_p90_atr``  — 90th percentile.  "Stress" scenario for sizing
    maximum adverse portfolio impact.

Stop-hit probability
--------------------
For any proposed stop distance ``stop_distance_atr``, the predictor
returns ``stop_hit_probability`` = P(MAE > stop_distance_atr), computed
analytically from the log-normal CDF.

Design rules
------------
* No import of ``QDEDecision`` or any execution-layer type.
* No import from ``trainer.qde.brain.projection`` (no cross-package
  dependency; conditions are passed in directly).
* Entirely stateless — safe to instantiate per-call or once at startup.
* All magnitude outputs are in ATR units (instrument-agnostic).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ── Constants ─────────────────────────────────────────────────────────────────

_SQRT2 = math.sqrt(2.0)

# Volatility ceiling for normalisation: rolling vol above this saturates
# the vol contribution at 1.0.  2% is a typical "high-volatility" ceiling
# for major FX pairs; metals and indices will saturate more easily,
# but the structural formula is scale-invariant by construction.
_VOL_CEIL = 0.020

# Minimum structural MAE estimate (ATR units).  Even in perfect conditions
# there is always some irreducible noise component.
_MAE_FLOOR = 0.05

# Log-normal σ bounds: σ governs the *width* of the MAE distribution.
# A minimum of 0.40 gives a P90/mean ratio of ≈ 2.5 for typical params,
# consistent with empirical FX trade geometry.
_SIGMA_MIN = 0.40
_SIGMA_MAX = 0.90

# Standard-normal quantile z-scores
_Z_P50 = 0.000
_Z_P75 = 0.674
_Z_P90 = 1.282


# ── AdverseExcursionInput ─────────────────────────────────────────────────────


@dataclass
class AdverseExcursionInput:
    """Structural inputs for the adverse excursion predictor.

    All fields are instrument-agnostic normalised values; no prices, no
    pip counts, no lot sizes.

    ``spread_atr_ratio``
        Current bid-ask spread / ATR.  Primary cost-of-entry friction.
    ``atr_pct``
        ATR / price (relative volatility).  Used for scaling; currently
        retained for future structural extensions.
    ``volatility``
        Realised rolling volatility (fractional, e.g. 0.01 = 1%).
    ``session_quality``
        0-1 from the session-timing engine.  1 = London-NY overlap;
        0 = weekend / closed.
    ``momentum``
        Signed −1..+1.  Positive = price trending upward.
    ``direction``
        +1 = long, −1 = short.  Combined with momentum to determine
        whether momentum is *with* or *against* the trade.
    ``microstructure_quality``
        0-1 from the MIE / liquidity map.  1 = clean order flow.
    ``stop_distance_atr``
        Proposed stop distance in ATR units.  0 = not yet decided.
        When > 0, ``stop_hit_probability`` is computed analytically.
    """

    spread_atr_ratio: float = 0.10
    atr_pct: float = 0.005
    volatility: float = 0.010
    session_quality: float = 0.80
    momentum: float = 0.0
    direction: int = 1
    microstructure_quality: float = 0.80
    stop_distance_atr: float = 0.0

    @property
    def directed_momentum(self) -> float:
        """Positive when momentum is in trade direction; negative otherwise."""
        return self.momentum * self.direction


# ── AdverseExcursionEstimate ──────────────────────────────────────────────────


@dataclass(frozen=True)
class AdverseExcursionEstimate:
    """Structural distribution estimate for Maximum Adverse Excursion.

    All magnitude fields are in ATR units.

    ``expected_mae_atr``
        Log-normal mean = the "typical" adverse excursion.

    ``mae_p50_atr``
        Median (P50).  Half of setups with these conditions will have
        MAE below this level.  The minimum defensible stop distance.

    ``mae_p75_atr``
        75th percentile.  A stop tighter than this has > 25 % chance of
        being triggered by *normal* adverse excursion before the trade
        resolves.

    ``mae_p90_atr``
        90th percentile.  Stress-scenario sizing bound.

    ``stop_hit_probability``
        P(MAE > ``stop_distance_atr``).  0.0 when ``stop_distance_atr``
        <= 0.  Ranges from ~0.90 for very tight stops to near 0.0 for
        very wide stops.

    ``log_sigma``
        Internal dispersion parameter of the log-normal model.
        Retained for blending with historical estimates.
    """

    expected_mae_atr: float = 0.0
    mae_p50_atr: float = 0.0
    mae_p75_atr: float = 0.0
    mae_p90_atr: float = 0.0
    stop_hit_probability: float = 0.0
    log_sigma: float = 0.60

    def to_dict(self) -> dict:
        return {
            "expected_mae_atr": round(self.expected_mae_atr, 4),
            "mae_p50_atr": round(self.mae_p50_atr, 4),
            "mae_p75_atr": round(self.mae_p75_atr, 4),
            "mae_p90_atr": round(self.mae_p90_atr, 4),
            "stop_hit_probability": round(self.stop_hit_probability, 4),
            "log_sigma": round(self.log_sigma, 4),
        }


# ── AdverseExcursionPredictor ─────────────────────────────────────────────────


class AdverseExcursionPredictor:
    """Stateless structural estimator for the MAE distribution.

    Instantiate once and call ``estimate()`` on every tick.  No internal
    state is maintained; calls are thread-safe.

    The structural model establishes a baseline used by
    ``RewardRiskEngine`` before sufficient trade history exists, and
    re-anchors estimates when history is thin (< 5 similar records).
    """

    def estimate(self, inp: AdverseExcursionInput) -> AdverseExcursionEstimate:
        """Compute the structural MAE distribution for these conditions.

        Parameters
        ----------
        inp:
            Live market conditions and optional proposed stop distance.

        Returns
        -------
        ``AdverseExcursionEstimate`` with expected MAE, quantiles, and
        (if ``stop_distance_atr > 0``) stop-hit probability.
        """
        expected_mae = self._expected_mae(inp)
        sigma = self._log_sigma(inp)
        p50, p75, p90 = _lognormal_quantiles(expected_mae, sigma)
        stop_hit = _stop_hit_prob(inp.stop_distance_atr, expected_mae, sigma)

        return AdverseExcursionEstimate(
            expected_mae_atr=round(expected_mae, 6),
            mae_p50_atr=round(p50, 6),
            mae_p75_atr=round(p75, 6),
            mae_p90_atr=round(p90, 6),
            stop_hit_probability=round(stop_hit, 4),
            log_sigma=round(sigma, 4),
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    def _expected_mae(self, inp: AdverseExcursionInput) -> float:
        """Structural estimate of the log-normal mean MAE in ATR units.

        Component breakdown (all additive, all non-negative):

        ``irreducible``
            Baseline noise even in perfect conditions (0.15 ATR).

        ``spread_contribution``
            Half-spread → half-spread cost paid at entry is an immediate
            adverse excursion.  Coefficient 0.50 converts SAR units to
            ATR fraction contribution.

        ``vol_contribution``
            Normalised volatility.  At the ceiling (2 %) this adds 0.60
            ATR to MAE.  At 0.5 % (quiet) it adds 0.15 ATR.

        ``adverse_momentum``
            When momentum is against the trade direction, the price is
            already drifting toward the stop.  Max contribution 0.40 ATR
            at directed_momentum = −1.

        ``session_penalty``
            Off-hours trading adds gap risk.  Adds up to 0.20 ATR.

        ``micro_penalty``
            Erratic order flow adds noise excursions.  Adds up to 0.15 ATR.
        """
        dm = inp.directed_momentum
        vol_norm = min(1.0, inp.volatility / _VOL_CEIL)

        expected_mae = (
            0.15  # irreducible noise floor
            + inp.spread_atr_ratio * 0.50  # spread = immediate adverse move
            + vol_norm * 0.60  # vol contribution
            + max(0.0, -dm) * 0.40  # adverse momentum amplification
            + (1.0 - inp.session_quality) * 0.20  # thin-session gap risk
            + (1.0 - inp.microstructure_quality) * 0.15  # noisy order flow
        )
        return max(_MAE_FLOOR, expected_mae)

    def _log_sigma(self, inp: AdverseExcursionInput) -> float:
        """Log-normal dispersion parameter σ.

        Higher σ → fatter right tail → P90 is further above the mean.
        The dispersion is driven by environmental uncertainty, not the
        expected level: thin sessions and poor microstructure produce
        *variable* excursions, not just larger ones.
        """
        sigma = (
            _SIGMA_MIN
            + (1.0 - inp.session_quality) * 0.25
            + (1.0 - inp.microstructure_quality) * 0.20
        )
        return min(_SIGMA_MAX, sigma)


# ── Log-normal helpers ────────────────────────────────────────────────────────


def _lognormal_quantiles(
    mean_x: float,
    sigma: float,
) -> tuple[float, float, float]:
    """Compute P50, P75, P90 of LogNormal(μ, σ) given the mean and σ.

    The mean of a log-normal is exp(μ + σ²/2), so:
        μ = log(mean_x) - σ²/2

    Returns (p50, p75, p90) in the same units as mean_x.
    """
    mean_x = max(mean_x, 1e-9)
    mu = math.log(mean_x) - sigma**2 / 2.0
    p50 = math.exp(mu + _Z_P50 * sigma)
    p75 = math.exp(mu + _Z_P75 * sigma)
    p90 = math.exp(mu + _Z_P90 * sigma)
    return p50, p75, p90


def _stop_hit_prob(
    stop_distance_atr: float,
    expected_mae: float,
    sigma: float,
) -> float:
    """P(MAE > stop_distance_atr) from the structural log-normal model.

    Returns 0.0 when ``stop_distance_atr <= 0`` (stop not specified).
    Uses the complementary error function for the tail probability.
    """
    if stop_distance_atr <= 0.0:
        return 0.0
    expected_mae = max(expected_mae, 1e-9)
    mu = math.log(expected_mae) - sigma**2 / 2.0
    z = (math.log(max(stop_distance_atr, 1e-9)) - mu) / sigma
    return max(0.0, min(1.0, 0.5 * math.erfc(z / _SQRT2)))
