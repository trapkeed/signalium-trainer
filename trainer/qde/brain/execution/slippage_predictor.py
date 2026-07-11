"""QDE Brain — execution: structural slippage prediction.

``SlippagePredictor`` answers: given the current market conditions and the
order type being considered, how much execution cost (in ATR units) should
we expect?

Stateless by design
--------------------
This module is a pure structural model — no internal state, no learning
loop.  Historical calibration is the responsibility of
``ExecutionQualityEngine``, which holds the learned memory and adjusts the
structural estimates after each completed fill.

Why ATR units?
--------------
Slippage expressed as a fraction of the current ATR is
instrument-agnostic.  EURUSD with a 1-pip spread and SPX500 with a
0.3-point spread are incomparable in raw terms but immediately comparable
when normalised to their respective ATRs.  An estimate of ``0.10 ATR``
means "this entry will cost roughly 10% of the day's range in execution
friction" — meaningful regardless of price scale.

What "slippage" means here
---------------------------
Slippage = total execution friction above the theoretical mid-price entry:

    one-way spread cost  (half the bid-ask spread)
  + market impact         (price moves against you as your order hits)
  + timing friction       (thin sessions, post-news, weekend gaps)

For MARKET orders all three components are non-zero.  For LIMIT orders
the spread cost is avoided (the order rests and gets hit), but adverse
selection (price running away before the limit is touched) replaces it.
For STOP orders, the trigger is at a known price but the fill is at market
— gap risk adds a "worst-case" component that can be much larger than the
expected value.

Design rules
------------
* No hardcoded broker assumptions (no min-lot, no commission per lot).
* No hardcoded pip value or instrument-specific multipliers.
* No import of ``QDEDecision`` or any execution-layer action type.
* Reuses ``OrderType`` from ``trainer.execution.execution_types``.
* Structural estimates are calibrated to be "reasonable defaults that
  the learned overlay can correct" — not to be perfectly accurate in
  isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from trainer.execution.execution_types import OrderType

# ── Order-type aggressiveness map ─────────────────────────────────────────────
# 1.0 = filled immediately at market; 0.0 = purely passive.
# This scalar drives how much market-impact slippage is attributed.
_AGGRESSIVENESS: dict[OrderType, float] = {
    OrderType.MARKET: 1.00,
    OrderType.STOP: 0.80,  # triggers → fills like market
    OrderType.STOP_LIMIT: 0.50,  # trigger → limit; may miss fill in fast move
    OrderType.LIMIT: 0.20,  # rests passively; gets adverse selection instead
    OrderType.CONDITIONAL: 0.10,  # most passive
}
_DEFAULT_AGGRESSIVENESS = 0.60

# Normative spread-to-ATR ratios: below LOW = tight; above HIGH = very wide.
# Used only to shape the "expected vs worst-case" spread between estimates.
_SPREAD_ATR_LOW = 0.05  # e.g., EURUSD in London session
_SPREAD_ATR_HIGH = 0.50  # e.g., XAUUSD around news / thin session


# ── SlippageInput ─────────────────────────────────────────────────────────────


@dataclass
class SlippageInput:
    """Conditions that drive slippage prediction.

    All fields are instrument-agnostic and pre-normalised by the caller.

    ``spread_atr_ratio``
        Current bid-ask spread divided by the current ATR.  The primary
        driver of slippage for all order types.

    ``atr_pct``
        ATR as a fraction of the current price (e.g., 0.005 = 0.5%).
        Used to sanity-check whether the spread_atr_ratio is anomalously
        high (broken feed, pre-open, etc.).

    ``volatility``
        Realised rolling volatility (fractional, e.g. 0.01 = 1% / bar).
        Higher volatility → higher market impact for aggressive orders.

    ``session_quality``
        0-1 scalar from the session timing engine.  0 = closed / weekend;
        1 = London-NY overlap peak liquidity.  Thin sessions dramatically
        amplify worst-case slippage.

    ``momentum``
        Signed −1..+1.  Strong momentum in the trade direction increases
        adverse selection for LIMIT orders; against direction increases
        impact for MARKET orders.

    ``order_type``
        The order type being evaluated.  Drives ``order_aggressiveness``.

    ``microstructure_quality``
        0-1 quality signal from the MIE / liquidity map.  0 = stressed
        microstructure (wide spreads, thin depth); 1 = clean.
    """

    spread_atr_ratio: float = 0.10
    atr_pct: float = 0.005
    volatility: float = 0.01
    session_quality: float = 0.80
    momentum: float = 0.0  # −1..+1
    order_type: OrderType = OrderType.MARKET
    microstructure_quality: float = 0.5  # 0-1


# ── SlippageEstimate ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SlippageEstimate:
    """Structural slippage estimate for one order scenario.

    All cost fields are expressed in ATR units so they are directly
    comparable across instruments.

    ``expected_slippage_atr``
        Best estimate of total one-way execution cost in ATR.  Includes
        half-spread cost, market impact, and session friction.

    ``worst_case_slippage_atr``
        90th-percentile cost estimate — used for worst-case expected-
        reward calculations.  Always >= expected_slippage_atr.

    ``spread_cost_atr``
        Isolated half-spread component (instrument of friction).
        ``expected_slippage_atr`` - ``spread_cost_atr`` = pure market impact.

    ``favorable_probability``
        Probability of receiving a better-than-expected fill.  Non-zero
        mainly for LIMIT orders (can get adverse selection but also
        favorable resting).

    ``order_aggressiveness``
        The aggressiveness scalar used internally (0-1).  Retained in the
        output so callers can incorporate it into downstream calculations.

    ``spread_risk``
        0-1 normalised spread risk: 0 = trivially tight; 1 = dangerously
        wide.  A convenience field so callers don't have to re-normalise.
    """

    expected_slippage_atr: float = 0.0
    worst_case_slippage_atr: float = 0.0
    spread_cost_atr: float = 0.0
    favorable_probability: float = 0.0
    order_aggressiveness: float = 1.0
    spread_risk: float = 0.0

    def to_dict(self) -> dict:
        return {
            "expected_slippage_atr": round(self.expected_slippage_atr, 6),
            "worst_case_slippage_atr": round(self.worst_case_slippage_atr, 6),
            "spread_cost_atr": round(self.spread_cost_atr, 6),
            "favorable_probability": round(self.favorable_probability, 4),
            "order_aggressiveness": round(self.order_aggressiveness, 4),
            "spread_risk": round(self.spread_risk, 4),
        }


# ── SlippagePredictor ─────────────────────────────────────────────────────────


class SlippagePredictor:
    """Stateless structural slippage estimator.

    No internal state.  Instantiation is side-effect-free.  The learned
    overlay (historical correction) is applied externally by
    ``ExecutionQualityEngine``, which holds per-symbol execution history.

    Structural model
    ----------------
    Slippage for an aggressive order (MARKET, STOP) has three components:

    1. **Spread cost** — ``0.5 * spread_atr_ratio`` (half-spread is the
       floor cost of any market-side fill).

    2. **Market impact** — proportional to aggressiveness and volatility:
       ``impact = aggressiveness * spread_atr_ratio * vol_factor * thin_factor``
       where:
         vol_factor  = 1 + max(0, volatility / 0.02 - 1) * 0.5
         thin_factor = 1 + (1 - session_quality) * 0.8

    3. **Microstructure friction** — reduced by good microstructure quality:
       ``friction = (1 - microstructure_quality) * spread_atr_ratio * 0.2``

    For LIMIT orders the spread cost is replaced by adverse-selection cost
    (the order rests and gets hit when momentum is adverse).  The
    impact and friction components are scaled down by (1 - aggressiveness).

    Worst-case estimate amplifies expected by a liquidity-stress multiplier:
       ``worst = expected * (1 + (1 - session_quality) * (1 - microstructure_quality))``
    """

    def estimate(self, inp: SlippageInput) -> SlippageEstimate:
        """Compute the structural slippage estimate for the given conditions."""
        aggressiveness = _AGGRESSIVENESS.get(inp.order_type, _DEFAULT_AGGRESSIVENESS)
        sar = max(0.0, inp.spread_atr_ratio)

        # ── 1. Spread cost (half-spread) ──────────────────────────────────
        spread_cost = sar * 0.5
        # Passive orders (LIMIT) sit on the other side — they avoid the
        # spread cost but incur adverse selection instead.
        if aggressiveness < 0.5:
            spread_cost = 0.0

        # ── 2. Market impact ──────────────────────────────────────────────
        vol_factor = 1.0 + max(0.0, inp.volatility / 0.02 - 1.0) * 0.5
        thin_factor = 1.0 + (1.0 - max(0.0, min(1.0, inp.session_quality))) * 0.8
        impact = aggressiveness * sar * 0.30 * vol_factor * thin_factor

        # ── 3. Microstructure friction ────────────────────────────────────
        friction = (1.0 - max(0.0, min(1.0, inp.microstructure_quality))) * sar * 0.20

        # ── 4. Adverse selection for passive orders ────────────────────────
        # LIMIT orders rest and get hit by flow — momentum in the opposite
        # direction to the order increases adverse selection.
        adverse_selection = 0.0
        if aggressiveness < 0.5:
            # Adverse momentum increases adverse selection (price keeps going
            # away after the limit is hit, suggesting the fill was "wrong").
            mom_factor = max(0.0, -inp.momentum)  # worse when momentum < 0
            adverse_selection = (1.0 - aggressiveness) * sar * 0.15 * (1.0 + mom_factor)

        expected = spread_cost + impact + friction + adverse_selection
        expected = max(0.0, expected)

        # ── 5. Worst-case ─────────────────────────────────────────────────
        stress_mult = 1.0 + (1.0 - inp.session_quality) * (1.0 - inp.microstructure_quality)
        worst_case = expected * max(1.0, stress_mult)

        # ── 6. Favorable-fill probability (mostly for passive orders) ─────
        favorable_prob = max(
            0.0, (1.0 - aggressiveness) * inp.microstructure_quality * inp.session_quality * 0.25
        )

        # ── 7. Spread risk (normalized) ───────────────────────────────────
        spread_risk = _spread_risk(sar)

        return SlippageEstimate(
            expected_slippage_atr=round(expected, 6),
            worst_case_slippage_atr=round(worst_case, 6),
            spread_cost_atr=round(spread_cost, 6),
            favorable_probability=round(favorable_prob, 4),
            order_aggressiveness=round(aggressiveness, 4),
            spread_risk=round(spread_risk, 4),
        )

    def aggressiveness(self, order_type: OrderType) -> float:
        """Return the aggressiveness scalar for a given order type."""
        return _AGGRESSIVENESS.get(order_type, _DEFAULT_AGGRESSIVENESS)


# ── Shared helper ─────────────────────────────────────────────────────────────


def _spread_risk(spread_atr_ratio: float) -> float:
    """Normalise spread-to-ATR ratio to a 0-1 risk score.

    Uses a soft sigmoid so the risk scales smoothly:
      0.05 SAR → ~0.18 (very tight)
      0.20 SAR → ~0.45 (moderate)
      0.50 SAR → ~0.76 (wide)
      1.00 SAR → ~0.92 (very wide)
    """
    return 1.0 / (1.0 + math.exp(-4.0 * (spread_atr_ratio - 0.20)))
