"""QDE Brain — execution: structural fill-probability estimator.

``FillProbabilityEngine`` answers two questions:

1. **Will this order be filled?**  For MARKET orders the answer is almost
   always yes — the uncertainty is in the *price*, not the fill itself.
   For LIMIT and STOP_LIMIT orders the fill is uncertain: the price may
   never reach the target level, or a fast market may jump straight past
   a STOP_LIMIT trigger before the limit half can execute.

2. **How good will the fill quality be?**  Fill quality is a 0-1 score
   representing how close to the theoretical ideal execution the actual
   fill lands.  1.0 = filled at better than expected; 0.0 = catastrophic
   slippage or no fill.

Both are structural estimates — pure functions of the conditions in
``FillContext``.  Historical calibration is owned by
``ExecutionQualityEngine``, which adjusts the structural predictions using
its learned memory of past execution outcomes for the same instrument.

Fill-probability model
----------------------
For aggressive orders (MARKET, STOP):
    ``fill_probability ≈ 1.0`` structurally.  The small discount from 1.0
    accounts for extreme tail events (weekend gap, circuit breaker, broker
    reject).

For passive orders (LIMIT, STOP_LIMIT):
    Fill probability depends on how far the target price is from the
    current market, and how volatile the market is.  The intuition is:
    if volatility is high relative to the distance, the price is likely
    to touch the limit level before the order is cancelled; if the
    distance is large relative to volatility, the fill is unlikely without
    a sustained directional move.

    Structural model: logistic-decay over ``distance / vol_scale``:
        ``P(fill) = 1 / (1 + (distance / vol_scale) ^ 1.5)``
    where ``vol_scale = max(spread_atr_ratio + volatility * 2, 0.05)``.

Fill-quality model
------------------
Fill quality is driven by:

  * **MARKET orders** — quality is reduced by the bid-ask spread and any
    market impact.  ``quality = 1 − spread_atr_ratio × 0.4`` (half-spread
    is the pure friction, market impact is already in slippage).

  * **LIMIT orders** — if filled, quality starts high (≥0.90) because you
    rested and got hit at your price.  Adverse selection (momentum running
    against the fill) reduces quality by up to 15%.

  * **STOP orders** — quality starts at 0.75 and is improved by session
    quality (better liquidity → lower stop-fill slippage).

  * **STOP_LIMIT orders** — quality starts at 0.85 but has a potential
    non-fill risk embedded in the fill_probability already.

Order-type recommendation
-------------------------
``recommend_order_type`` provides a coarse recommendation based on current
spread conditions and the slippage budget the caller is willing to tolerate:

  * Tight spread + acceptable structural slippage → MARKET (immediacy)
  * Wide spread or adverse momentum → LIMIT (wait for retracement)
  * High volatility + high slippage risk → STOP_LIMIT (risk-aware trigger)
  * Extreme conditions → stay with MARKET but let slippage engine warn

Design rules
------------
* Stateless — no learning state here; all learning is in
  ``ExecutionQualityEngine``.
* No hardcoded broker constraints.
* No import of ``QDEDecision`` or any execution-layer action type.
* Reuses ``OrderType`` and ``TimingRecommendation`` from
  ``trainer.execution.execution_types``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from trainer.execution.execution_types import OrderType

# ── Order-type aggressiveness (mirrored from slippage_predictor) ──────────────
# Kept local so this module has no circular imports.
_AGGRESSIVENESS: dict[OrderType, float] = {
    OrderType.MARKET:      1.00,
    OrderType.STOP:        0.80,
    OrderType.STOP_LIMIT:  0.50,
    OrderType.LIMIT:       0.20,
    OrderType.CONDITIONAL: 0.10,
}

# Spread-ATR thresholds that drive order-type recommendations.
_SPREAD_ATR_LIMIT_THRESHOLD = 0.25   # above this → prefer LIMIT over MARKET
_SPREAD_ATR_STOPLIMIT_THRESHOLD = 0.60  # above this → prefer STOP_LIMIT

# Estimated fill-wait seconds for passive orders (rough structural baseline).
_LIMIT_WAIT_BASE_SECONDS = 120.0
_STOPLIMIT_WAIT_SECONDS  = 180.0


# ── FillContext ───────────────────────────────────────────────────────────────


@dataclass
class FillContext:
    """Per-order execution context for fill-probability estimation.

    ``order_type``
        The order type being evaluated (MARKET, LIMIT, STOP, etc.).

    ``distance_to_target_atr``
        Distance from the current mid-price to the order target price,
        expressed in ATR units.  0.0 for MARKET orders (fill at market).
        Positive = target is away from current price in the order
        direction (e.g. a limit buy 1.5 ATR below current price →
        distance = 1.5).

    ``spread_atr_ratio``
        Current bid-ask spread / ATR.  Primary driver of fill quality
        for aggressive orders.

    ``volatility``
        Realised rolling volatility (fractional).  High volatility →
        passive orders more likely to be touched.

    ``session_quality``
        0-1 session timing quality.  Thin sessions → worse stop-fill
        quality; lower fill probability for passives.

    ``momentum``
        Signed −1..+1.  Adverse momentum (negative for a buy limit)
        reduces fill quality via adverse selection.

    ``direction``
        +1 = long (buy), −1 = short (sell).  Used to determine whether
        momentum is favourable or adverse relative to this order.
    """

    order_type:              OrderType = OrderType.MARKET
    distance_to_target_atr:  float     = 0.0    # ≥ 0; 0 for market
    spread_atr_ratio:         float     = 0.10
    volatility:               float     = 0.01
    session_quality:          float     = 0.80
    momentum:                 float     = 0.0
    direction:                int       = 1      # +1 long, −1 short

    def _adverse_momentum(self) -> float:
        """Momentum magnitude that opposes this order's direction.

        A long order benefits from positive momentum (likely to run to
        target) but faces adverse selection risk when momentum is strongly
        negative (the price will keep falling after the limit is hit).
        """
        # For a long order, negative momentum = adverse; vice versa.
        signed = self.direction * self.momentum   # positive = aligned
        return max(0.0, -signed)


# ── FillProbabilitySignal ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class FillProbabilitySignal:
    """Structural fill-probability estimate for one order scenario.

    ``fill_probability``
        Probability that this order type, in these conditions, receives a
        fill before a typical cancellation window.  0-1.  Always 1.0 for
        MARKET orders structurally.

    ``expected_fill_quality``
        Expected fill-quality score (0-1): how close to the theoretical
        ideal price this fill is likely to land.  1.0 = filled at or
        better than limit; 0.0 = catastrophic slippage / no fill.

    ``estimated_wait_seconds``
        For passive orders (LIMIT, STOP_LIMIT): structural estimate of
        how many seconds until a fill.  0.0 for MARKET.  Used by the
        timing engine to decide whether to wait or switch to MARKET.

    ``recommended_order_type``
        Order type this engine considers optimal for the given conditions.
        May differ from the evaluated ``order_type`` in ``FillContext``.

    ``fill_probability_passive``
        Fill probability for a hypothetical LIMIT order at the same
        distance — retained so callers can compare "if I placed a limit
        instead, would I get filled?"
    """

    fill_probability:        float     = 1.0
    expected_fill_quality:   float     = 0.75
    estimated_wait_seconds:  float     = 0.0
    recommended_order_type:  OrderType = OrderType.MARKET
    fill_probability_passive: float    = 0.5

    def to_dict(self) -> dict:
        return {
            "fill_probability":         round(self.fill_probability, 4),
            "expected_fill_quality":    round(self.expected_fill_quality, 4),
            "estimated_wait_seconds":   round(self.estimated_wait_seconds, 1),
            "recommended_order_type":   self.recommended_order_type.value,
            "fill_probability_passive": round(self.fill_probability_passive, 4),
        }


# ── FillProbabilityEngine ─────────────────────────────────────────────────────


class FillProbabilityEngine:
    """Stateless structural fill-probability and quality estimator.

    No internal state — instantiation is side-effect-free.  Historical
    correction is applied by ``ExecutionQualityEngine``.

    Usage
    -----
    ::

        engine = FillProbabilityEngine()
        ctx = FillContext(
            order_type=OrderType.LIMIT,
            distance_to_target_atr=1.2,
            spread_atr_ratio=0.15,
            volatility=0.012,
            session_quality=0.85,
            momentum=-0.3,
            direction=1,
        )
        signal = engine.estimate(ctx)
        rec    = engine.recommend_order_type(
            spread_atr_ratio=0.15,
            expected_slippage_atr=0.08,
            timing_budget_seconds=60.0,
        )
    """

    # ── Public API ────────────────────────────────────────────────────────

    def estimate(self, ctx: FillContext) -> FillProbabilitySignal:
        """Compute structural fill probability and quality for the context."""
        agg          = _AGGRESSIVENESS.get(ctx.order_type, 0.6)
        sar          = max(0.0, ctx.spread_atr_ratio)
        sess         = max(0.0, min(1.0, ctx.session_quality))
        vol          = max(0.0, ctx.volatility)
        dist         = max(0.0, ctx.distance_to_target_atr)
        adverse_mom  = ctx._adverse_momentum()

        fill_prob  = self._fill_probability(ctx.order_type, dist, sar, vol, sess, agg)
        quality    = self._fill_quality(ctx.order_type, sar, sess, adverse_mom, agg)
        wait_secs  = self._estimated_wait(ctx.order_type, dist, sar, vol, sess)

        # Fill probability for a hypothetical limit at the same distance
        fp_passive = self._fill_probability(
            OrderType.LIMIT, dist, sar, vol, sess,
            _AGGRESSIVENESS[OrderType.LIMIT]
        )

        rec = self.recommend_order_type(
            spread_atr_ratio=sar,
            expected_slippage_atr=sar * 0.5 + agg * sar * 0.2,
            timing_budget_seconds=300.0,   # default window
        )

        return FillProbabilitySignal(
            fill_probability=round(fill_prob, 4),
            expected_fill_quality=round(quality, 4),
            estimated_wait_seconds=round(wait_secs, 1),
            recommended_order_type=rec,
            fill_probability_passive=round(fp_passive, 4),
        )

    def recommend_order_type(
        self,
        spread_atr_ratio:       float,
        expected_slippage_atr:  float,
        timing_budget_seconds:  float = 300.0,
    ) -> OrderType:
        """Recommend the optimal order type for the given cost conditions.

        Decision logic (in order of priority):

        1. Very wide spread AND slippage high → STOP_LIMIT
           (slow entry with a limit to cap worst-case fill)
        2. Wide spread → LIMIT
           (wait for a retracement to avoid paying the spread)
        3. Tight spread AND budget allows immediacy → MARKET
        4. Intermediate → LIMIT if budget allows waiting, else MARKET

        ``timing_budget_seconds``
            How long the caller is willing to wait for a passive fill
            before the opportunity expires.  Short budget → prefer MARKET.
        """
        sar = max(0.0, spread_atr_ratio)
        slip = max(0.0, expected_slippage_atr)

        if sar >= _SPREAD_ATR_STOPLIMIT_THRESHOLD and slip > 0.5:
            return OrderType.STOP_LIMIT

        if sar >= _SPREAD_ATR_LIMIT_THRESHOLD:
            # Wide spread — check if we have time to wait for a passive fill
            if timing_budget_seconds >= _LIMIT_WAIT_BASE_SECONDS:
                return OrderType.LIMIT
            # Too little time → swallow the spread
            return OrderType.MARKET

        # Tight spread — MARKET unless we have time AND slippage is meaningful
        if timing_budget_seconds >= 120.0 and slip > 0.15:
            return OrderType.LIMIT
        return OrderType.MARKET

    # ── Internal structural models ─────────────────────────────────────────

    @staticmethod
    def _fill_probability(
        order_type: OrderType,
        distance:   float,
        sar:        float,
        vol:        float,
        sess:       float,
        agg:        float,
    ) -> float:
        """Structural fill probability for the given order type and conditions."""
        if order_type == OrderType.MARKET:
            # Market orders: structurally certain; tiny tail discount for
            # extreme gaps or broker rejects.
            tail = max(0.0, 1.0 - sess) * 0.02
            return max(0.90, 1.0 - tail)

        if order_type == OrderType.STOP:
            # Stop orders fill once triggered — main risk is thin markets
            # causing a gap that means the trigger fires but the broker
            # can't fill near that level.
            gap_risk = (1.0 - sess) * 0.08
            return max(0.75, 1.0 - gap_risk)

        if order_type == OrderType.STOP_LIMIT:
            # Stop trigger fires → limit placed → may miss in fast move
            trigger_prob = max(0.75, 1.0 - (1.0 - sess) * 0.10)
            fill_given_trigger = max(0.70, sess * 0.90)
            return trigger_prob * fill_given_trigger

        # LIMIT / CONDITIONAL: logistic decay over distance / vol_scale
        vol_scale = max(0.05, sar + vol * 2.0 * sess)
        ratio     = distance / vol_scale
        fill_prob = 1.0 / (1.0 + ratio ** 1.5)
        # Session quality penalises even reachable targets in thin markets
        return max(0.05, fill_prob * (0.7 + 0.3 * sess))

    @staticmethod
    def _fill_quality(
        order_type:  OrderType,
        sar:         float,
        sess:        float,
        adverse_mom: float,
        agg:         float,
    ) -> float:
        """Expected fill quality (0-1) for the given order type.

        1.0 = filled at or better than theoretical; 0 = catastrophic.
        """
        if order_type == OrderType.MARKET:
            # Quality drops with spread and thin session.
            # Base is 0.95 (not 1.0) because MARKET orders always pay at
            # least some spread — a perfectly clean MARKET fill is still
            # slightly inferior to a LIMIT fill that rests and gets hit.
            base     = 0.95 - min(0.50, sar * 0.50)
            sess_pen = (1.0 - sess) * 0.10
            return max(0.20, base - sess_pen)

        if order_type == OrderType.STOP:
            # Stops fill at market once triggered; session quality matters
            return max(0.40, 0.75 + 0.20 * sess - sar * 0.15)

        if order_type == OrderType.STOP_LIMIT:
            # Higher quality if filled (limit price protects against extreme gaps)
            return max(0.55, 0.85 - (1.0 - sess) * 0.15)

        # LIMIT / CONDITIONAL
        # High base quality (you rested and got hit at your price) but
        # adverse selection reduces it: if momentum is strongly adverse
        # right when the limit is hit, it suggests the fill was an
        # information-driven event that will continue to hurt.
        adverse_pen = adverse_mom * 0.15
        return max(0.40, 0.92 - adverse_pen)

    @staticmethod
    def _estimated_wait(
        order_type: OrderType,
        distance:   float,
        sar:        float,
        vol:        float,
        sess:       float,
    ) -> float:
        """Structural estimate of seconds until fill (0 for market orders)."""
        if order_type in (OrderType.MARKET, OrderType.STOP):
            return 0.0

        if order_type == OrderType.STOP_LIMIT:
            # Stop triggers first, then limit places — add extra latency
            return _STOPLIMIT_WAIT_SECONDS * (1.0 + distance)

        # LIMIT: time to fill grows with distance and shrinks with volatility
        vol_scale = max(0.05, vol * 2.0 * sess)
        ratio     = distance / vol_scale
        wait      = _LIMIT_WAIT_BASE_SECONDS * (1.0 + ratio ** 1.2)
        return min(wait, 3600.0)   # cap at 1 hour structurally
