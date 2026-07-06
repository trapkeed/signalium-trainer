"""Phase 24 — Drawdown Manager.

Adaptive drawdown protection:
Normal, Soft Limit, Hard Limit, Capital Preservation,
Recovery Mode, Emergency Capital Lockdown.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

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
        self._events: List[DrawdownEvent] = []
        self._active_event: Optional[DrawdownEvent] = None

    # ── Equity update ─────────────────────────────────────────────────────────

    def update(self, equity: float) -> Dict:
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
        return self._mode != DrawdownMode.EMERGENCY_LOCKDOWN

    def events(self, last_n: int = 20) -> List[DrawdownEvent]:
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

    def stats(self) -> Dict:
        return {
            "peak": round(self._peak, 2),
            "current": round(self._current, 2),
            "drawdown": round(self._drawdown_frac(), 4),
            "mode": self._mode.value,
            "risk_multiplier": round(self._risk_multiplier(), 4),
            "total_events": len(self._events),
            "max_dd_ever": round(self.max_drawdown(), 4),
            "can_trade": self.can_trade(),
        }
