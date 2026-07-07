"""Phase 21.6 — Counterfactual Analysis Engine.

For every completed decision, evaluates alternative scenarios:
- What if BUY had become WAIT?
- What if TP was larger?
- What if SL was tighter?
- What if entry was delayed?
- What if trailing activated earlier?
- What if no trade was taken?
"""

from __future__ import annotations

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.memory.decision_models import CounterfactualResult, DecisionRecord
from trainer.memory.decision_types import DecisionType


class CounterfactualEngine:
    """Generates counterfactual scenarios for completed decisions."""

    _WAIT_PROB = 0.50  # P(market improves if we wait)
    _LARGE_TP_FACTOR = 1.5  # How much larger the alternative TP is
    _TIGHT_SL_FACTOR = 0.70  # How much tighter the alternative SL is

    def analyse(self, record: DecisionRecord) -> list[CounterfactualResult]:
        """Generate all applicable counterfactuals for a decision record."""
        results: list[CounterfactualResult] = []

        if not record.has_outcome:
            return results  # need actual outcome to compare against

        actual_pnl = record.outcome.actual_pnl
        snap = record.snapshot

        # ── 1. WAIT instead of action ─────────────────────────────────────────
        if record.decision_type in (DecisionType.BUY, DecisionType.SELL):
            wait_pnl = 0.0  # waited — no trade = 0 PnL
            results.append(
                CounterfactualResult(
                    scenario=f"WAIT instead of {record.decision_type.value}",
                    alternative_action="WAIT",
                    estimated_pnl=wait_pnl,
                    estimated_pnl_pct=0.0,
                    probability=self._WAIT_PROB,
                    better_than_actual=wait_pnl > actual_pnl,
                    delta_pnl=round(wait_pnl - actual_pnl, 5),
                    notes="WAIT assumes no position taken, PnL = 0",
                )
            )

        # ── 2. No trade ───────────────────────────────────────────────────────
        if record.decision_type != DecisionType.NO_TRADE:
            results.append(
                CounterfactualResult(
                    scenario="NO_TRADE taken instead",
                    alternative_action="NO_TRADE",
                    estimated_pnl=0.0,
                    estimated_pnl_pct=0.0,
                    probability=0.45,
                    better_than_actual=actual_pnl < 0.0,
                    delta_pnl=round(-actual_pnl, 5),
                    notes="Counterfactual: completely avoiding the trade",
                )
            )

        # ── 3. Larger TP ──────────────────────────────────────────────────────
        if record.is_entry_decision and snap.expected_move > 0:
            # Estimated: probability of hitting larger TP is lower
            larger_tp_prob = snap.tp_probability * 0.65
            larger_tp_pnl = snap.expected_move * self._LARGE_TP_FACTOR
            est_pnl_larger = (
                larger_tp_pnl * larger_tp_prob - snap.expected_move * snap.sl_probability
            )
            results.append(
                CounterfactualResult(
                    scenario="TP set at 1.5× (larger target)",
                    alternative_action="LARGER_TP",
                    estimated_pnl=round(est_pnl_larger, 5),
                    estimated_pnl_pct=round(est_pnl_larger / max(snap.price, 1e-9), 5),
                    probability=larger_tp_prob,
                    better_than_actual=est_pnl_larger > actual_pnl,
                    delta_pnl=round(est_pnl_larger - actual_pnl, 5),
                    notes="Larger TP reduces hit probability by 35%",
                )
            )

        # ── 4. Tighter SL ─────────────────────────────────────────────────────
        if record.is_entry_decision and snap.expected_move > 0:
            # Tighter SL: smaller loss if stopped, but higher chance of being stopped
            tighter_sl_pnl_loss = snap.expected_move * self._TIGHT_SL_FACTOR * snap.sl_probability
            est_pnl_tight_sl = snap.expected_move * snap.tp_probability - tighter_sl_pnl_loss
            results.append(
                CounterfactualResult(
                    scenario="SL set at 0.7× (tighter stop)",
                    alternative_action="TIGHTER_SL",
                    estimated_pnl=round(est_pnl_tight_sl, 5),
                    estimated_pnl_pct=round(est_pnl_tight_sl / max(snap.price, 1e-9), 5),
                    probability=snap.tp_probability * 0.80,  # slightly lower tp prob
                    better_than_actual=est_pnl_tight_sl > actual_pnl,
                    delta_pnl=round(est_pnl_tight_sl - actual_pnl, 5),
                    notes="Tighter SL reduces loss but increases stop frequency",
                )
            )

        # ── 5. Delayed entry ──────────────────────────────────────────────────
        if record.is_entry_decision:
            # Delayed entry: different fill, same direction; P(better entry) ≈ 0.40
            delayed_est_pnl = snap.expected_move * 1.10 * snap.tp_probability
            results.append(
                CounterfactualResult(
                    scenario="Entry delayed by 1–2 ATR pullback",
                    alternative_action="DELAYED_ENTRY",
                    estimated_pnl=round(delayed_est_pnl, 5),
                    estimated_pnl_pct=round(delayed_est_pnl / max(snap.price, 1e-9), 5),
                    probability=0.40,
                    better_than_actual=delayed_est_pnl > actual_pnl,
                    delta_pnl=round(delayed_est_pnl - actual_pnl, 5),
                    notes="Delayed entry may provide better risk/reward",
                )
            )

        # ── 6. Earlier trailing stop ──────────────────────────────────────────
        if record.has_outcome and record.outcome.hit_tp:
            # If TP was hit, earlier trail might have locked in more
            trail_pnl = snap.expected_move * 0.80
            results.append(
                CounterfactualResult(
                    scenario="Trailing stop activated at 50% of move",
                    alternative_action="EARLY_TRAIL",
                    estimated_pnl=round(trail_pnl, 5),
                    estimated_pnl_pct=round(trail_pnl / max(snap.price, 1e-9), 5),
                    probability=0.75,
                    better_than_actual=trail_pnl < actual_pnl,  # full TP > trail lock
                    delta_pnl=round(trail_pnl - actual_pnl, 5),
                    notes="Early trail locks profit but may miss full move",
                )
            )
        elif record.has_outcome and record.outcome.hit_sl:
            # If SL was hit, earlier trail would have protected more
            trail_pnl_protected = snap.expected_move * 0.30  # small lock before reversal
            results.append(
                CounterfactualResult(
                    scenario="Trailing stop activated at 30% of initial move",
                    alternative_action="EARLY_TRAIL",
                    estimated_pnl=round(trail_pnl_protected, 5),
                    estimated_pnl_pct=round(trail_pnl_protected / max(snap.price, 1e-9), 5),
                    probability=0.60,
                    better_than_actual=trail_pnl_protected > actual_pnl,
                    delta_pnl=round(trail_pnl_protected - actual_pnl, 5),
                    notes="Early trail could have prevented SL from being hit",
                )
            )

        # Attach to record
        record.counterfactuals = results
        logger.debug(
            f"CounterfactualEngine [{record.decision_id[:8]}]: "
            f"{len(results)} scenarios generated"
        )
        return results

    def best_alternative(self, record: DecisionRecord) -> CounterfactualResult | None:
        """Return the counterfactual with the highest estimated PnL."""
        if not record.counterfactuals:
            self.analyse(record)
        if not record.counterfactuals:
            return None
        return max(record.counterfactuals, key=lambda c: c.estimated_pnl)

    def worst_alternative(self, record: DecisionRecord) -> CounterfactualResult | None:
        if not record.counterfactuals:
            self.analyse(record)
        if not record.counterfactuals:
            return None
        return min(record.counterfactuals, key=lambda c: c.estimated_pnl)

    def better_alternatives_count(self, record: DecisionRecord) -> int:
        """How many alternatives would have been better than actual?"""
        if not record.counterfactuals:
            self.analyse(record)
        return sum(1 for c in record.counterfactuals if c.better_than_actual)
