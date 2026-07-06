"""Phase 24.9 — Execution Intelligence."""

from __future__ import annotations

from typing import Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.mie.mie_types import ExecutionMode, RiskRegime, LiquidityState, clamp
from trainer.mie.mie_models import (
    MarketInput,
    RiskIntelligenceResult,
    LiquidityResult,
    TradeabilityResult,
    VolatilityResult,
    ExecutionResult,
)


class ExecutionIntelligenceEngine:
    """
    Recommends an execution mode (Market / Limit / Wait / Avoid) and
    computes optimal entry/exit zones, stop, and target levels.

    Decision logic:
    1. If RiskRegime is extreme / black swan → AVOID
    2. If tradeability < 30 → AVOID
    3. If liquidity is ILLIQUID → WAIT
    4. If tradeability < 50 → WAIT
    5. If spread > 0.20 × ATR → LIMIT_ORDER
    6. Otherwise → MARKET_ORDER

    Zones use ATR-based buffers.
    """

    def __init__(
        self,
        tradeability_avoid: float = 30.0,
        tradeability_wait: float = 50.0,
        spread_limit_ratio: float = 0.10,  # spread > X * ATR → limit order
        atr_entry_buffer: float = 0.30,  # entry zone ± 0.30 * ATR
        atr_stop_mult: float = 1.50,  # stop = price ± 1.5 * ATR
        atr_target_mult: float = 3.00,  # target = price ± 3.0 * ATR (2:1 RR)
        atr_exit_buffer: float = 0.20,
        max_holding_bars: int = 40,
        drawdown_atr_mult: float = 0.80,
    ) -> None:
        self._avoid_thresh = tradeability_avoid
        self._wait_thresh = tradeability_wait
        self._spread_limit = spread_limit_ratio
        self._entry_buf = atr_entry_buffer
        self._stop_mult = atr_stop_mult
        self._target_mult = atr_target_mult
        self._exit_buf = atr_exit_buffer
        self._max_hold = max_holding_bars
        self._dd_mult = drawdown_atr_mult

    # ── Public API ─────────────────────────────────────────────────────────

    def recommend(
        self,
        inp: MarketInput,
        risk: Optional[RiskIntelligenceResult] = None,
        liquidity: Optional[LiquidityResult] = None,
        tradeability: Optional[TradeabilityResult] = None,
        volatility: Optional[VolatilityResult] = None,
    ) -> ExecutionResult:

        mode, rationale = self._decide_mode(risk, liquidity, tradeability)
        price = inp.close or 1.0
        atr = max(inp.atr, 1e-9)

        # Direction: use RSI as proxy (>50 = bullish)
        is_long = inp.rsi >= 50.0

        entry_low, entry_high = self._entry_zone(price, atr, mode)
        exit_low, exit_high = self._exit_zone(price, atr, is_long)
        optimal_stop = self._optimal_stop(price, atr, is_long)
        optimal_target = self._optimal_target(price, atr, is_long)
        expected_dd = self._expected_drawdown(atr, price, volatility)
        max_hold = self._max_holding(inp, volatility)

        logger.debug(
            f"[EI] {inp.symbol} → {mode.value}  stop={optimal_stop:.6f}  "
            f"target={optimal_target:.6f}"
        )
        return ExecutionResult(
            mode=mode,
            entry_zone_low=entry_low,
            entry_zone_high=entry_high,
            exit_zone_low=exit_low,
            exit_zone_high=exit_high,
            max_holding_bars=max_hold,
            expected_drawdown=expected_dd,
            optimal_stop=optimal_stop,
            optimal_target=optimal_target,
            rationale=rationale,
        )

    # ── Mode decision ──────────────────────────────────────────────────────

    def _decide_mode(
        self,
        risk: Optional[RiskIntelligenceResult],
        liquidity: Optional[LiquidityResult],
        tradeability: Optional[TradeabilityResult],
    ) -> tuple:
        # 1. Extreme risk
        if risk is not None and risk.regime in (
            RiskRegime.EXTREME,
            RiskRegime.BLACK_SWAN,
            RiskRegime.FLASH_VOLATILITY,
        ):
            return ExecutionMode.AVOID, f"Risk regime: {risk.regime.value}"

        # 2. Very low tradeability
        t_score = tradeability.score if tradeability is not None else 50.0
        if t_score < self._avoid_thresh:
            return ExecutionMode.AVOID, f"Tradeability too low: {t_score:.1f}"

        # 3. Illiquid
        if liquidity is not None and liquidity.state == LiquidityState.ILLIQUID:
            return ExecutionMode.WAIT, "Illiquid market conditions"

        # 4. Below wait threshold
        if t_score < self._wait_thresh:
            return ExecutionMode.WAIT, f"Tradeability below threshold: {t_score:.1f}"

        # 5. Wide spread → limit order
        if risk is not None and risk.factors.get("spread", 0.0) >= 15.0:
            return ExecutionMode.LIMIT_ORDER, "Wide spread — use limit order"

        return ExecutionMode.MARKET_ORDER, "Conditions favour immediate execution"

    # ── Zone calculations ──────────────────────────────────────────────────

    def _entry_zone(self, price: float, atr: float, mode: ExecutionMode) -> tuple:
        buf = atr * self._entry_buf
        if mode == ExecutionMode.LIMIT_ORDER:
            buf *= 0.50  # tighter zone for limit orders
        return (
            round(price - buf, 6),
            round(price + buf, 6),
        )

    def _exit_zone(self, price: float, atr: float, is_long: bool) -> tuple:
        buf = atr * self._exit_buf
        if is_long:
            center = price + atr * self._target_mult
        else:
            center = price - atr * self._target_mult
        return (round(center - buf, 6), round(center + buf, 6))

    def _optimal_stop(self, price: float, atr: float, is_long: bool) -> float:
        dist = atr * self._stop_mult
        return round(price - dist if is_long else price + dist, 6)

    def _optimal_target(self, price: float, atr: float, is_long: bool) -> float:
        dist = atr * self._target_mult
        return round(price + dist if is_long else price - dist, 6)

    def _expected_drawdown(
        self,
        atr: float,
        price: float,
        volatility: Optional[VolatilityResult],
    ) -> float:
        base_dd = (atr * self._dd_mult) / max(price, 1e-9)
        if volatility is not None:
            # Scale by vol_ratio: higher vol → higher expected drawdown
            base_dd *= clamp(volatility.vol_ratio, 0.5, 3.0)
        return clamp(base_dd, 0.0, 0.50)

    def _max_holding(
        self,
        inp: MarketInput,
        volatility: Optional[VolatilityResult],
    ) -> int:
        bars = inp.expected_holding_bars or self._max_hold
        if volatility is not None and volatility.vol_ratio > 2.0:
            bars = max(5, int(bars * 0.6))
        return min(bars, self._max_hold * 3)
