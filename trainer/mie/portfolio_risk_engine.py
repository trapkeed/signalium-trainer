"""Phase 24.9 — MIE Portfolio Risk Engine.

Provides portfolio-level risk constraints and position sizing guidance
as part of the Market Intelligence Engine output.

This is the MIE-layer portfolio view — it is distinct from the ICARI
PortfolioRiskEngine (which operates at the institutional allocation layer).
The MIE version is lightweight and feeds the Feature Store and the
Tradeability Engine.
"""

from __future__ import annotations

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.mie.mie_models import (
    MarketInput,
    MIEPortfolioRisk,
    RiskIntelligenceResult,
    TradeabilityResult,
)
from trainer.mie.mie_types import RISK_REGIME_MULTIPLIERS, RiskRegime, clamp

# Maximum portfolio exposure thresholds
_MAX_EXPOSURE_NORMAL = 0.40
_MAX_EXPOSURE_ELEVATED = 0.25
_MAX_EXPOSURE_HIGH = 0.15
_MAX_EXPOSURE_EXTREME = 0.05

_BASE_POSITION_FRAC = 0.02  # 2% of capital per trade baseline

_MAX_OPEN_TRADES = {
    RiskRegime.OPTIMAL: 8,
    RiskRegime.LOW_RISK: 7,
    RiskRegime.NORMAL: 6,
    RiskRegime.ELEVATED: 4,
    RiskRegime.HIGH_RISK: 3,
    RiskRegime.EXTREME: 1,
    RiskRegime.FLASH_VOLATILITY: 0,
    RiskRegime.LOW_LIQUIDITY: 2,
    RiskRegime.MARKET_STRESS: 1,
    RiskRegime.NEWS_EVENT: 3,
    RiskRegime.BLACK_SWAN: 0,
    RiskRegime.UNKNOWN: 3,
}


class MIEPortfolioRiskEngine:
    """
    Derives portfolio-level risk parameters from current risk regime and
    market conditions for inclusion in the MarketIntelligenceReport.

    Outputs:
    - risk_budget_used    : fraction of daily risk budget consumed
    - max_exposure        : maximum allowable portfolio exposure
    - correlation_limit   : maximum allowable average pairwise correlation
    - diversification     : estimated diversification score 0-1
    - max_open_trades     : maximum simultaneous open positions
    - position_size_frac  : recommended position size as fraction of capital
    - capital_allocation  : recommended USD allocation for this signal
    """

    def __init__(
        self,
        daily_risk_budget: float = 0.02,  # 2% of capital
        corr_limit_normal: float = 0.70,
        corr_limit_high: float = 0.40,
    ) -> None:
        self._daily_budget = daily_risk_budget
        self._corr_normal = corr_limit_normal
        self._corr_high = corr_limit_high

    # ── Public API ─────────────────────────────────────────────────────────

    def evaluate(
        self,
        inp: MarketInput,
        risk: RiskIntelligenceResult | None = None,
        tradeability: TradeabilityResult | None = None,
    ) -> MIEPortfolioRisk:

        regime = risk.regime if risk is not None else RiskRegime.NORMAL
        mult = RISK_REGIME_MULTIPLIERS.get(regime, 1.0)

        max_exposure = self._max_exposure(regime)
        corr_limit = self._correlation_limit(regime)
        max_trades = _MAX_OPEN_TRADES.get(regime, 3)
        pos_frac = self._position_size(mult, inp, tradeability)
        diversification = self._diversification(inp)
        risk_used = self._risk_budget_used(inp)
        allocation_usd = pos_frac * inp.capital

        warnings: list[str] = []
        if inp.portfolio_exposure > max_exposure:
            warnings.append(
                f"Current exposure {inp.portfolio_exposure:.0%} exceeds "
                f"limit {max_exposure:.0%}"
            )
        if inp.correlation_avg > corr_limit:
            warnings.append(f"Correlation {inp.correlation_avg:.2f} exceeds limit {corr_limit:.2f}")
        if max_trades == 0:
            warnings.append("No new trades allowed in current risk regime")

        logger.debug(
            f"[MPR] {inp.symbol} regime={regime.value} pos={pos_frac:.4f} "
            f"max_exp={max_exposure:.2%} max_trades={max_trades}"
        )
        return MIEPortfolioRisk(
            risk_budget_used=clamp(risk_used, 0.0, 1.0),
            max_exposure=max_exposure,
            correlation_limit=corr_limit,
            diversification=diversification,
            max_open_trades=max_trades,
            position_size_frac=clamp(pos_frac, 0.0, 0.10),
            capital_allocation=clamp(allocation_usd, 0.0, inp.capital),
            warnings=warnings,
        )

    # ── Internal ───────────────────────────────────────────────────────────

    def _max_exposure(self, regime: RiskRegime) -> float:
        if regime in (
            RiskRegime.EXTREME,
            RiskRegime.BLACK_SWAN,
            RiskRegime.FLASH_VOLATILITY,
            RiskRegime.MARKET_STRESS,
        ):
            return _MAX_EXPOSURE_EXTREME
        if regime in (RiskRegime.HIGH_RISK,):
            return _MAX_EXPOSURE_HIGH
        if regime in (RiskRegime.ELEVATED, RiskRegime.NEWS_EVENT, RiskRegime.LOW_LIQUIDITY):
            return _MAX_EXPOSURE_ELEVATED
        return _MAX_EXPOSURE_NORMAL

    def _correlation_limit(self, regime: RiskRegime) -> float:
        if regime in (
            RiskRegime.HIGH_RISK,
            RiskRegime.EXTREME,
            RiskRegime.BLACK_SWAN,
            RiskRegime.FLASH_VOLATILITY,
        ):
            return self._corr_high
        return self._corr_normal

    def _position_size(
        self,
        mult: float,
        inp: MarketInput,
        tradeability: TradeabilityResult | None,
    ) -> float:
        """Scale _BASE_POSITION_FRAC by risk multiplier and tradeability."""
        t_scale = 1.0
        if tradeability is not None:
            t_scale = clamp(tradeability.score / 70.0, 0.2, 1.5)
        # Reduce if already exposed
        exp_scale = clamp(1.0 - inp.portfolio_exposure * 2.0, 0.1, 1.0)
        return _BASE_POSITION_FRAC * mult * t_scale * exp_scale

    def _diversification(self, inp: MarketInput) -> float:
        """Estimate diversification from correlation and open trades."""
        corr_factor = 1.0 - clamp(inp.correlation_avg, 0.0, 1.0)
        trade_factor = clamp(inp.open_trades / 5.0, 0.0, 1.0)
        return clamp((corr_factor * 0.60 + trade_factor * 0.40), 0.0, 1.0)

    def _risk_budget_used(self, inp: MarketInput) -> float:
        """Estimate fraction of daily risk budget consumed by exposure."""
        return clamp(inp.portfolio_exposure / _MAX_EXPOSURE_NORMAL, 0.0, 1.0)
