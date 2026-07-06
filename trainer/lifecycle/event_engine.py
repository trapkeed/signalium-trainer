"""Phase 21.5 — Event Engine.

Generates structured lifecycle events when significant conditions
are detected or state transitions occur.
"""

from __future__ import annotations

from typing import List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.lifecycle.lifecycle_models import (
    EventType,
    LifecycleEvent,
    MarketContext,
    PositionMetrics,
)
from trainer.lifecycle.lifecycle_states import LifecycleState


class EventEngine:
    """Detects significant conditions and emits lifecycle events."""

    # Volatility thresholds
    _VOL_HIGH = 0.020
    _VOL_EXTREME = 0.035
    # Momentum threshold
    _MOM_STRONG = 0.40
    # News risk threshold
    _NEWS_WARN = 0.40
    _NEWS_CRITICAL = 0.70
    # Spread threshold
    _SPREAD_WIDE = 0.25  # fraction of ATR

    def detect_events(
        self,
        trade_id: str,
        state: LifecycleState,
        context: Optional[MarketContext] = None,
        metrics: Optional[PositionMetrics] = None,
        prev_context: Optional[MarketContext] = None,
        prev_confidence: float = 0.65,
        curr_confidence: float = 0.65,
    ) -> List[LifecycleEvent]:
        """Detect events by comparing current vs previous conditions."""

        events: List[LifecycleEvent] = []

        if context is None:
            return events

        # ── Confidence changes ───────────────────────────────────────────────
        conf_delta = curr_confidence - prev_confidence
        if conf_delta >= 0.08:
            events.append(
                LifecycleEvent.create(
                    trade_id,
                    EventType.CONFIDENCE_RAISED,
                    state,
                    f"Confidence increased {prev_confidence:.0%} → {curr_confidence:.0%}",
                    severity="INFO",
                    data={"from": prev_confidence, "to": curr_confidence, "delta": conf_delta},
                )
            )
        elif conf_delta <= -0.08:
            events.append(
                LifecycleEvent.create(
                    trade_id,
                    EventType.CONFIDENCE_DROPPED,
                    state,
                    f"Confidence dropped {prev_confidence:.0%} → {curr_confidence:.0%}",
                    severity="WARN",
                    data={"from": prev_confidence, "to": curr_confidence, "delta": conf_delta},
                )
            )

        # ── Volatility events ─────────────────────────────────────────────────
        if context.volatility > self._VOL_EXTREME:
            events.append(
                LifecycleEvent.create(
                    trade_id,
                    EventType.VOLATILITY_EXPANDED,
                    state,
                    f"Extreme volatility detected: {context.volatility:.3f}",
                    severity="CRITICAL",
                    data={"volatility": context.volatility},
                )
            )
        elif context.volatility > self._VOL_HIGH:
            if prev_context is None or prev_context.volatility <= self._VOL_HIGH:
                events.append(
                    LifecycleEvent.create(
                        trade_id,
                        EventType.VOLATILITY_EXPANDED,
                        state,
                        f"Volatility expanded to {context.volatility:.3f}",
                        severity="WARN",
                        data={"volatility": context.volatility},
                    )
                )
        elif (
            prev_context
            and prev_context.volatility > self._VOL_HIGH
            and context.volatility <= self._VOL_HIGH
        ):
            events.append(
                LifecycleEvent.create(
                    trade_id,
                    EventType.VOLATILITY_CONTRACTED,
                    state,
                    f"Volatility normalised to {context.volatility:.3f}",
                    severity="INFO",
                    data={"volatility": context.volatility},
                )
            )

        # ── Momentum events ───────────────────────────────────────────────────
        if prev_context:
            mom_delta = context.momentum - prev_context.momentum
            if mom_delta >= 0.25 and context.momentum > self._MOM_STRONG:
                events.append(
                    LifecycleEvent.create(
                        trade_id,
                        EventType.MOMENTUM_INCREASED,
                        state,
                        f"Momentum strengthened to {context.momentum:.2f}",
                        severity="INFO",
                        data={"momentum": context.momentum, "delta": mom_delta},
                    )
                )
            elif mom_delta <= -0.25 and context.momentum < -self._MOM_STRONG:
                events.append(
                    LifecycleEvent.create(
                        trade_id,
                        EventType.MOMENTUM_DECREASED,
                        state,
                        f"Momentum weakened to {context.momentum:.2f}",
                        severity="WARN",
                        data={"momentum": context.momentum, "delta": mom_delta},
                    )
                )

        # ── News risk events ──────────────────────────────────────────────────
        if context.news_risk > self._NEWS_CRITICAL:
            events.append(
                LifecycleEvent.create(
                    trade_id,
                    EventType.NEWS_RISK_ELEVATED,
                    state,
                    f"Critical news risk: {context.news_risk:.0%}",
                    severity="CRITICAL",
                    data={"news_risk": context.news_risk},
                )
            )
        elif context.news_risk > self._NEWS_WARN:
            if prev_context is None or prev_context.news_risk <= self._NEWS_WARN:
                events.append(
                    LifecycleEvent.create(
                        trade_id,
                        EventType.NEWS_RISK_ELEVATED,
                        state,
                        f"News risk elevated: {context.news_risk:.0%}",
                        severity="WARN",
                        data={"news_risk": context.news_risk},
                    )
                )

        # ── Spread events ─────────────────────────────────────────────────────
        if context.atr > 0:
            spread_ratio = context.spread / context.atr
            if spread_ratio > self._SPREAD_WIDE:
                if prev_context is None or (
                    prev_context.atr > 0
                    and prev_context.spread / prev_context.atr <= self._SPREAD_WIDE
                ):
                    events.append(
                        LifecycleEvent.create(
                            trade_id,
                            EventType.SPREAD_WIDENED,
                            state,
                            f"Spread widened to {spread_ratio:.0%} of ATR",
                            severity="WARN",
                            data={
                                "spread": context.spread,
                                "atr": context.atr,
                                "ratio": spread_ratio,
                            },
                        )
                    )

        # ── Trend invalidation ────────────────────────────────────────────────
        if not context.trend_valid:
            if prev_context is None or prev_context.trend_valid:
                events.append(
                    LifecycleEvent.create(
                        trade_id,
                        EventType.TREND_INVALID,
                        state,
                        "Trend structure invalidated",
                        severity="CRITICAL",
                        data={"trend_valid": False},
                    )
                )

        # ── Liquidity events ──────────────────────────────────────────────────
        if context.session == "CLOSED":
            if prev_context and prev_context.session != "CLOSED":
                events.append(
                    LifecycleEvent.create(
                        trade_id,
                        EventType.LIQUIDITY_DETERIORATED,
                        state,
                        "Session closed — liquidity deteriorated",
                        severity="WARN",
                        data={"session": context.session},
                    )
                )

        # ── Metrics-based events ──────────────────────────────────────────────
        if metrics:
            if metrics.unrealised_pct > 0.015:
                events.append(
                    LifecycleEvent.create(
                        trade_id,
                        EventType.TARGET_APPROACHING,
                        state,
                        f"Unrealised gain {metrics.unrealised_pct:.2%} — target approaching",
                        severity="INFO",
                        data={"unrealised_pct": metrics.unrealised_pct},
                    )
                )
            if metrics.unrealised_pct < -0.010:
                events.append(
                    LifecycleEvent.create(
                        trade_id,
                        EventType.RISK_INCREASED,
                        state,
                        f"Position in drawdown {metrics.unrealised_pct:.2%}",
                        severity="WARN",
                        data={"unrealised_pct": metrics.unrealised_pct},
                    )
                )

        logger.debug(
            f"EventEngine [{state.value}] trade={trade_id}: {len(events)} event(s) generated"
        )
        return events

    def emit_state_change(
        self,
        trade_id: str,
        from_state: LifecycleState,
        to_state: LifecycleState,
        reason: str,
        confidence: float = 0.65,
    ) -> LifecycleEvent:
        """Always emit an event when state changes."""
        sev = "INFO"
        if to_state.value in ("STOPPED", "EMERGENCY_EXIT", "FORCED_EXIT"):
            sev = "CRITICAL"
        elif to_state.value in ("INVALIDATED", "CANCELLED", "EXPIRED"):
            sev = "WARN"

        return LifecycleEvent.create(
            trade_id,
            EventType.STATE_CHANGED,
            to_state,
            f"State {from_state.value} → {to_state.value}: {reason}",
            severity=sev,
            data={
                "from_state": from_state.value,
                "to_state": to_state.value,
                "confidence": confidence,
            },
        )
