"""Phase 24 — Drawdown Manager.

Adaptive drawdown protection:
Normal, Soft Limit, Hard Limit, Capital Preservation,
Recovery Mode, Emergency Capital Lockdown.

Phase S1 (capital protection) constraint: this is a capital circuit breaker,
not a decision override engine. QDE remains the sole decision maker for
direction, stop loss, take profit, targets, confidence, and hold duration —
this class never touches any of those. Its only allowed actions are:
reducing the global risk multiplier, pausing new trade creation, emergency
account lockdown, and disabling scaling / pyramiding (adding to winners).
"""

from __future__ import annotations

import time

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.icari.icari_models import DrawdownEvent
from trainer.icari.icari_types import DrawdownMode


class DrawdownManager:
    """Monitors drawdown and enforces adaptive risk reduction."""

    def __init__(
        self,
        capital: float = 100_000.0,
        soft_limit: float = 0.05,  # 5% drawdown → reduce risk
        hard_limit: float = 0.10,  # 10% → preservation mode
        emergency_limit: float = 0.20,  # 20% → lockdown
        recovery_target: float = 0.02,  # return to within 2% of peak
    ):
        self._capital = capital
        self._soft = soft_limit
        self._hard = hard_limit
        self._emergency = emergency_limit
        self._recovery_tgt = recovery_target

        self._peak = capital
        self._current = capital
        self._mode = DrawdownMode.NORMAL
        self._events: list[DrawdownEvent] = []
        self._active_event: DrawdownEvent | None = None

        # Circuit-breaker flags. Auto-derived from mode on every update(), but
        # may also be forced on manually (e.g. by an operator or a portfolio-
        # level safety signal) independent of drawdown mode.
        self._manual_scaling_disabled = False
        self._manual_pyramiding_disabled = False

    # Modes in which scaling/pyramiding are automatically disabled — the
    # account is already under stress, so adding to any position is forbidden
    # even though existing trades and new (smaller) trades may still open.
    _AUTO_RESTRICT_MODES = (
        DrawdownMode.HARD_LIMIT,
        DrawdownMode.PRESERVATION,
        DrawdownMode.EMERGENCY_LOCKDOWN,
    )

    # ── Equity update ─────────────────────────────────────────────────────────

    def update(self, equity: float) -> dict:
        """Update current equity and recompute drawdown mode."""
        self._current = equity
        if equity > self._peak:
            self._peak = equity
            self._close_event(equity)

        drawdown = self._drawdown_frac()
        prev_mode = self._mode
        self._mode = self._compute_mode(drawdown)

        if self._mode != DrawdownMode.NORMAL and self._active_event is None:
            self._open_event(equity, drawdown)
        elif self._mode == DrawdownMode.NORMAL and self._active_event is not None:
            self._close_event(equity)

        if self._mode != prev_mode:
            logger.warning(
                f"DrawdownManager: mode changed {prev_mode.value} → {self._mode.value} "
                f"drawdown={drawdown:.2%}"
            )

        return {
            "equity": round(equity, 2),
            "peak": round(self._peak, 2),
            "drawdown": round(drawdown, 4),
            "mode": self._mode.value,
            "risk_multiplier": round(self._risk_multiplier(), 4),
            "scaling_disabled": self.scaling_disabled,
            "pyramiding_disabled": self.pyramiding_disabled,
            "new_trades_paused": not self.can_trade(),
        }

    def _drawdown_frac(self) -> float:
        if self._peak <= 0:
            return 0.0
        return max(0.0, (self._peak - self._current) / self._peak)

    def _compute_mode(self, drawdown: float) -> DrawdownMode:
        if drawdown >= self._emergency:
            return DrawdownMode.EMERGENCY_LOCKDOWN
        if drawdown >= self._hard:
            return DrawdownMode.HARD_LIMIT
        if drawdown >= self._soft:
            return DrawdownMode.SOFT_LIMIT
        # Recovery mode: was in hard/emergency, now improving
        if self._mode in (
            DrawdownMode.HARD_LIMIT,
            DrawdownMode.EMERGENCY_LOCKDOWN,
            DrawdownMode.PRESERVATION,
        ):
            if drawdown > self._recovery_tgt:
                return DrawdownMode.RECOVERY
            return DrawdownMode.NORMAL
        return DrawdownMode.NORMAL

    def _risk_multiplier(self) -> float:
        """Return a multiplier for position sizing based on current mode."""
        mult = {
            DrawdownMode.NORMAL: 1.00,
            DrawdownMode.SOFT_LIMIT: 0.75,
            DrawdownMode.HARD_LIMIT: 0.50,
            DrawdownMode.PRESERVATION: 0.25,
            DrawdownMode.RECOVERY: 0.60,
            DrawdownMode.EMERGENCY_LOCKDOWN: 0.00,
        }
        return mult.get(self._mode, 1.0)

    # ── Events ────────────────────────────────────────────────────────────────

    def _open_event(self, equity: float, drawdown: float) -> None:
        event = DrawdownEvent(
            peak=self._peak,
            trough=equity,
            drawdown=drawdown,
            mode=self._mode,
        )
        self._active_event = event
        self._events.append(event)
        logger.info(f"DrawdownManager: opened event dd={drawdown:.2%} mode={self._mode.value}")

    def _close_event(self, equity: float) -> None:
        if self._active_event:
            self._active_event.end_ts = time.time()
            self._active_event.recovered = True
            logger.info(f"DrawdownManager: closed event recovered at equity={equity:.2f}")
            self._active_event = None

    # ── Queries ───────────────────────────────────────────────────────────────

    @property
    def mode(self) -> DrawdownMode:
        return self._mode

    @property
    def current_drawdown(self) -> float:
        return self._drawdown_frac()

    @property
    def risk_multiplier(self) -> float:
        return self._risk_multiplier()

    def is_locked(self) -> bool:
        return self._mode == DrawdownMode.EMERGENCY_LOCKDOWN

    def can_trade(self) -> bool:
        """False when new trade creation must be paused (emergency lockdown)."""
        return self._mode != DrawdownMode.EMERGENCY_LOCKDOWN

    @property
    def scaling_disabled(self) -> bool:
        """True if adding to (scaling into) any open position is forbidden.

        Auto-disabled once the account enters HARD_LIMIT/PRESERVATION/
        EMERGENCY_LOCKDOWN, or if manually forced on via ``disable_scaling()``.
        """
        return self._manual_scaling_disabled or self._mode in self._AUTO_RESTRICT_MODES

    @property
    def pyramiding_disabled(self) -> bool:
        """True if pyramiding (adding to winners) is forbidden."""
        return self._manual_pyramiding_disabled or self._mode in self._AUTO_RESTRICT_MODES

    def disable_scaling(self) -> None:
        """Manually force scaling off, independent of drawdown mode."""
        self._manual_scaling_disabled = True
        logger.warning("DrawdownManager: scaling manually disabled")

    def enable_scaling(self) -> None:
        """Clear the manual scaling override (mode-based auto-restriction still applies)."""
        self._manual_scaling_disabled = False

    def disable_pyramiding(self) -> None:
        """Manually force pyramiding off, independent of drawdown mode."""
        self._manual_pyramiding_disabled = True
        logger.warning("DrawdownManager: pyramiding manually disabled")

    def enable_pyramiding(self) -> None:
        """Clear the manual pyramiding override (mode-based auto-restriction still applies)."""
        self._manual_pyramiding_disabled = False

    def events(self, last_n: int = 20) -> list[DrawdownEvent]:
        return self._events[-last_n:]

    def max_drawdown(self) -> float:
        if not self._events:
            return self._drawdown_frac()
        return max(e.drawdown for e in self._events)

    def activate_preservation(self) -> None:
        self._mode = DrawdownMode.PRESERVATION
        logger.warning("DrawdownManager: PRESERVATION mode manually activated")

    def reset_peak(self, equity: float) -> None:
        """Manually reset the high-water mark (e.g. after capital injection)."""
        self._peak = equity
        self._current = equity
        self._mode = DrawdownMode.NORMAL
        self._active_event = None
        logger.info(f"DrawdownManager: peak reset to {equity:.2f}")

    def stats(self) -> dict:
        return {
            "peak": round(self._peak, 2),
            "current": round(self._current, 2),
            "drawdown": round(self._drawdown_frac(), 4),
            "mode": self._mode.value,
            "risk_multiplier": round(self._risk_multiplier(), 4),
            "total_events": len(self._events),
            "max_dd_ever": round(self.max_drawdown(), 4),
            "can_trade": self.can_trade(),
            "scaling_disabled": self.scaling_disabled,
            "pyramiding_disabled": self.pyramiding_disabled,
        }
