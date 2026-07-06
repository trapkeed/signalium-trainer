"""Phase 21.5 — AI Reasoning Engine.

Generates institutional-quality natural language explanations for
lifecycle state, transitions, and management decisions.

Pure template + rule-based reasoning — no external AI dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.lifecycle.lifecycle_models import (
    LifecycleConfidence,
    MarketContext,
    PositionMetrics,
)
from trainer.lifecycle.lifecycle_states import LifecycleState


@dataclass
class ReasoningReport:
    """Institutional explanation for a lifecycle state."""

    trade_id: str
    instrument: str
    direction: str
    state: LifecycleState
    headline: str
    rationale: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    outlook: str = ""
    confidence_narrative: str = ""

    def to_dict(self) -> Dict:
        return {
            "trade_id": self.trade_id,
            "instrument": self.instrument,
            "direction": self.direction,
            "state": self.state.value,
            "headline": self.headline,
            "rationale": self.rationale,
            "risk_factors": self.risk_factors,
            "action_items": self.action_items,
            "outlook": self.outlook,
            "confidence_narrative": self.confidence_narrative,
        }


class AiReasoningEngine:
    """Generates institutional-grade explanations for every lifecycle state.

    No external AI dependencies — deterministic rule-based text generation.
    """

    # State headlines
    _HEADLINES: Dict[LifecycleState, str] = {
        LifecycleState.IDEA: "Market identified — initial screening in progress",
        LifecycleState.WATCHLIST: "Setup developing — monitoring for entry conditions",
        LifecycleState.PREPARATION: "Conditions approaching entry — preparation active",
        LifecycleState.ENTRY_READY: "All entry conditions satisfied — ready to execute",
        LifecycleState.PENDING_ENTRY: "Order pending — awaiting optimal fill",
        LifecycleState.ACTIVE: "Position active — trade management in progress",
        LifecycleState.POSITION_BUILDING: "Scaling into position — confidence warrants addition",
        LifecycleState.PROTECTED: "Stop at breakeven — minimum outcome is scratch",
        LifecycleState.PARTIAL_EXIT: "Partial profit secured — managing remainder",
        LifecycleState.RUNNER: "Runner position active — targeting full return",
        LifecycleState.TRAILING: "Dynamic trail active — profit protection engaged",
        LifecycleState.EXIT_READY: "Exit conditions met — closing imminently",
        LifecycleState.CLOSING: "Trade being finalised — all positions closing",
        LifecycleState.CLOSED: "Trade completed — lifecycle concluded",
        LifecycleState.INVALIDATED: "Setup invalidated — idea withdrawn",
        LifecycleState.CANCELLED: "Trade cancelled — execution aborted",
        LifecycleState.EXPIRED: "Order expired — conditions no longer met",
        LifecycleState.STOPPED: "Stop loss triggered — risk management executed",
        LifecycleState.FORCED_EXIT: "Forced exit — external condition required closure",
        LifecycleState.EMERGENCY_EXIT: "Emergency exit — critical condition triggered",
    }

    def generate(
        self,
        trade_id: str,
        instrument: str,
        direction: str,
        state: LifecycleState,
        confidence: float,
        context: Optional[MarketContext] = None,
        metrics: Optional[PositionMetrics] = None,
        lc_conf: Optional[LifecycleConfidence] = None,
        prev_confidence: float = 0.65,
    ) -> ReasoningReport:

        rationale: List[str] = []
        risks: List[str] = []
        actions: List[str] = []
        headline = self._HEADLINES.get(state, f"Trade in {state.value} state")

        # ── Confidence narrative ──────────────────────────────────────────────
        delta = confidence - prev_confidence
        if delta >= 0.08:
            conf_narrative = (
                f"Confidence increased from {prev_confidence:.0%} to {confidence:.0%} — "
                f"setup is strengthening."
            )
        elif delta <= -0.08:
            conf_narrative = (
                f"Confidence declined from {prev_confidence:.0%} to {confidence:.0%} — "
                f"conditions are weakening."
            )
        else:
            conf_narrative = f"Confidence stable at {confidence:.0%} — " f"conditions unchanged."

        # ── State-specific rationale ──────────────────────────────────────────
        if state == LifecycleState.ACTIVE:
            rationale.append(
                f"Trade remains {state.value} because risk management framework is engaged."
            )
            if context:
                if context.trend_valid:
                    rationale.append("Trend structure remains valid — directional bias intact.")
                if abs(context.momentum) > 0.3:
                    mom_dir = "improving" if context.momentum > 0 else "declining"
                    rationale.append(f"Momentum {mom_dir} — price action consistent with thesis.")
                if context.volatility < 0.020:
                    rationale.append("Volatility within acceptable range — execution risk low.")
            if confidence > 0.70:
                rationale.append(f"Confidence at {confidence:.0%} — trade thesis remains intact.")

        elif state == LifecycleState.PROTECTED:
            rationale.append("Stop moved to breakeven — minimum outcome is a scratch trade.")
            rationale.append("Risk is now asymmetric: upside open, downside capped at entry.")
            if lc_conf:
                rationale.append(
                    f"Expected remaining reward: {lc_conf.expected_reward:.1%} with "
                    f"{lc_conf.transition_prob:.0%} probability of advancing."
                )

        elif state == LifecycleState.RUNNER:
            rationale.append("Runner position active — allowing price to reach full target.")
            rationale.append("80% of position profit already secured.")
            if context and abs(context.momentum) > 0.3:
                rationale.append("Momentum remains aligned — runner thesis supported.")

        elif state == LifecycleState.TRAILING:
            rationale.append("Dynamic trailing stop protecting runner profit.")
            rationale.append("No manual intervention required — trail stop is managing.")

        elif state == LifecycleState.WATCHLIST:
            rationale.append("Setup pattern identified — monitoring for confirmation.")
            if context:
                if not context.trend_valid:
                    rationale.append("Note: trend currently unclear — waiting for clarity.")

        elif state == LifecycleState.PREPARATION:
            rationale.append("Entry zone approaching — all preparation conditions being monitored.")
            if context:
                if context.news_risk > 0.3:
                    rationale.append(
                        f"News risk at {context.news_risk:.0%} — deferring entry until risk clears."
                    )

        elif state == LifecycleState.ENTRY_READY:
            rationale.append("All required conditions satisfied — execution window open.")
            rationale.append("Entry quality, timing, spread, and risk budget all confirmed.")

        elif state in (LifecycleState.STOPPED, LifecycleState.EMERGENCY_EXIT):
            rationale.append("Risk management protocol executed as designed.")
            rationale.append("Loss within pre-defined parameters — no adjustment required.")

        elif state == LifecycleState.CLOSED:
            rationale.append("Trade lifecycle completed — performance review available.")

        elif state == LifecycleState.INVALIDATED:
            rationale.append("Setup conditions no longer valid — idea withdrawn cleanly.")
            rationale.append("No loss taken — discipline preserved.")

        else:
            rationale.append(f"Trade progressing through {state.value} phase as expected.")

        # ── Risk factors ──────────────────────────────────────────────────────
        if context:
            if context.news_risk > 0.40:
                risks.append(f"News risk at {context.news_risk:.0%} — monitor for impact")
            if context.volatility > 0.020:
                risks.append(f"Elevated volatility {context.volatility:.3f} — wider stops required")
            if not context.trend_valid:
                risks.append("Trend structure uncertain — reduce conviction")
            if context.session == "CLOSED":
                risks.append("Market session closed — reduced liquidity")
            if context.spread > context.atr * 0.25 and context.atr > 0:
                risks.append(f"Wide spread ({context.spread/context.atr:.0%} of ATR)")

        if confidence < 0.55:
            risks.append(f"Low confidence ({confidence:.0%}) — reduce position size")

        # ── Action items ──────────────────────────────────────────────────────
        if state == LifecycleState.ACTIVE:
            actions.append("Monitor for TP1 approach — prepare partial close execution plan")
            actions.append("Verify stop loss level remains structurally valid")
        elif state == LifecycleState.ENTRY_READY:
            actions.append("Review timing conditions — confirm session and spread")
            actions.append("Confirm risk budget availability before order placement")
        elif state == LifecycleState.WATCHLIST:
            actions.append("Set price alerts for entry zone approach")
            actions.append("Monitor news calendar for potential risk events")
        elif state == LifecycleState.PROTECTED:
            actions.append("Monitor price advance toward TP2")
            actions.append("Prepare runner activation plan if TP2 approaches")
        elif state == LifecycleState.RUNNER:
            actions.append("Allow trail stop to manage the position")
            actions.append("No manual intervention unless emergency event")

        # ── Outlook ───────────────────────────────────────────────────────────
        if lc_conf:
            if lc_conf.transition_prob > 0.75:
                outlook = (
                    f"HIGH probability ({lc_conf.transition_prob:.0%}) of advancing to next state. "
                    f"Expected duration: {lc_conf.expected_duration_h:.0f}h. "
                    f"Reward target: {lc_conf.expected_reward:.1%}."
                )
            elif lc_conf.transition_prob > 0.50:
                outlook = (
                    f"MODERATE probability ({lc_conf.transition_prob:.0%}) of progress. "
                    f"Failure probability: {lc_conf.failure_prob:.0%}."
                )
            else:
                outlook = (
                    f"UNCERTAIN outlook — transition probability only {lc_conf.transition_prob:.0%}. "
                    f"Monitor conditions closely."
                )
        else:
            outlook = f"Trade in {state.value} phase — standard monitoring applies."

        logger.debug(
            f"AiReasoning {instrument} [{state.value}]: "
            f"{len(rationale)} rationale, {len(risks)} risks"
        )

        return ReasoningReport(
            trade_id=trade_id,
            instrument=instrument,
            direction=direction,
            state=state,
            headline=headline,
            rationale=rationale,
            risk_factors=risks,
            action_items=actions,
            outlook=outlook,
            confidence_narrative=conf_narrative,
        )
