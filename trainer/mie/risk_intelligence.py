"""Phase 24.9 — Risk Intelligence (12 risk regimes)."""

from __future__ import annotations

import statistics
from typing import Dict, List, Tuple

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.mie.mie_models import MarketInput, RiskIntelligenceResult
from trainer.mie.mie_types import (
    RISK_REGIME_MULTIPLIERS,
    MarketRegime,
    RiskRegime,
    clamp,
)


def _mean(vals: List[float]) -> float:
    return statistics.mean(vals) if vals else 0.0


def _stdev(vals: List[float]) -> float:
    return statistics.stdev(vals) if len(vals) >= 2 else 0.0


class RiskIntelligenceEngine:
    """
    Converts market conditions into a RiskRegime with a numeric risk score
    (0 = safe, 100 = extreme risk) and a position-sizing multiplier.

    Factor scoring:
    - Volatility ratio       (0-25 pts)
    - Spread cost            (0-20 pts)
    - Liquidity stress       (0-20 pts)
    - News event presence    (0-15 pts)
    - Portfolio exposure     (0-10 pts)
    - Correlation pressure   (0-10 pts)
    """

    def __init__(
        self,
        vol_high_thresh: float = 2.0,
        vol_extreme_thresh: float = 4.0,
        spread_pct_warn: float = 0.002,
        spread_pct_extreme: float = 0.010,
    ) -> None:
        self._vol_high = vol_high_thresh
        self._vol_extreme = vol_extreme_thresh
        self._spread_warn = spread_pct_warn
        self._spread_ext = spread_pct_extreme

    # ── Public API ─────────────────────────────────────────────────────────

    def evaluate(self, inp: MarketInput) -> RiskIntelligenceResult:
        factors, warnings = self._score_factors(inp)
        risk_score = clamp(sum(factors.values()), 0.0, 100.0)
        regime = self._map_regime(risk_score, inp)
        mult = RISK_REGIME_MULTIPLIERS.get(regime, 1.0)
        logger.debug(f"[RI] {inp.symbol} → {regime.value} score={risk_score:.1f}")
        return RiskIntelligenceResult(
            regime=regime,
            risk_score=risk_score,
            multiplier=mult,
            factors=factors,
            warnings=warnings,
        )

    # ── Factor scoring ─────────────────────────────────────────────────────

    def _score_factors(
        self,
        inp: MarketInput,
    ) -> Tuple[Dict[str, float], List[str]]:
        factors: Dict[str, float] = {}
        warnings: List[str] = []

        # 1. Volatility ratio (0-25)
        returns = inp.returns or []
        curr_vol = _stdev(returns[-20:]) if len(returns) >= 2 else 0.0
        base_vol = _stdev(returns[-60:]) if len(returns) >= 10 else max(curr_vol, 1e-9)
        vol_ratio = curr_vol / max(base_vol, 1e-9)
        if vol_ratio >= self._vol_extreme:
            factors["volatility"] = 25.0
            warnings.append(f"Extreme volatility: ratio={vol_ratio:.2f}")
        elif vol_ratio >= self._vol_high:
            factors["volatility"] = (
                15.0
                + (vol_ratio - self._vol_high)
                / max(self._vol_extreme - self._vol_high, 1e-9)
                * 10.0
            )
        else:
            factors["volatility"] = clamp((vol_ratio - 1.0) * 10.0, 0.0, 15.0)

        # 1b. Absolute volatility vs ATR (0-40 extra pts) — catches cases where
        #     current vol == baseline vol but both are extremely high in absolute terms
        atr_vol = curr_vol / max(inp.atr, 1e-9)
        if atr_vol > 15:
            factors["abs_vol"] = 40.0
            warnings.append(f"Extreme absolute volatility: cv/atr={atr_vol:.1f}")
        elif atr_vol > 8:
            factors["abs_vol"] = 15.0
        elif atr_vol > 4:
            factors["abs_vol"] = 5.0
        else:
            factors["abs_vol"] = 0.0

        # 2. Spread cost as fraction of ATR (0-20)
        spread_frac = inp.spread / max(inp.atr, 1e-9)
        if spread_frac >= 0.5:
            factors["spread"] = 20.0
            warnings.append(f"Dangerous spread: {spread_frac:.2f}x ATR")
        elif spread_frac >= 0.2:
            factors["spread"] = 10.0 + (spread_frac - 0.2) / 0.3 * 10.0
        else:
            factors["spread"] = clamp(spread_frac * 50.0, 0.0, 10.0)

        # 3. Liquidity stress via volume vs. baseline (0-20)
        vols = inp.volumes or []
        avg_vol = _mean(vols[-20:]) if len(vols) >= 5 else inp.volume
        vol_frac = inp.volume / max(avg_vol, 1.0)
        if vol_frac < 0.1:
            factors["liquidity"] = 20.0
            warnings.append("Critically low volume — illiquid conditions")
        elif vol_frac < 0.3:
            factors["liquidity"] = 12.0
        elif vol_frac < 0.5:
            factors["liquidity"] = 6.0
        else:
            factors["liquidity"] = 0.0

        # 4. News events (0-15)
        news = inp.news_events or []
        high_count = sum(1 for e in news if e.get("impact", "NONE") == "HIGH")
        med_count = sum(1 for e in news if e.get("impact", "NONE") == "MEDIUM")
        news_pts = min(15.0, high_count * 7.0 + med_count * 3.0)
        factors["news"] = news_pts
        if news_pts >= 7.0:
            warnings.append(f"High-impact news: {high_count} events")

        # 5. Portfolio exposure (0-10)
        exp_pts = clamp(inp.portfolio_exposure * 20.0, 0.0, 10.0)
        factors["exposure"] = exp_pts
        if inp.portfolio_exposure > 0.6:
            warnings.append(f"High portfolio exposure: {inp.portfolio_exposure:.0%}")

        # 6. Correlation pressure (0-10)
        corr_pts = clamp(inp.correlation_avg * 10.0, 0.0, 10.0)
        factors["correlation"] = corr_pts
        if inp.correlation_avg > 0.7:
            warnings.append(f"High correlation: {inp.correlation_avg:.2f}")

        return factors, warnings

    # ── Regime mapping ─────────────────────────────────────────────────────

    def _map_regime(self, score: float, inp: MarketInput) -> RiskRegime:
        news = inp.news_events or []
        high_count = sum(1 for e in news if e.get("impact", "NONE") == "HIGH")
        vols = inp.volumes or []
        avg_vol = _mean(vols[-20:]) if len(vols) >= 5 else inp.volume
        vol_frac = inp.volume / max(avg_vol, 1.0)

        # High-impact news override — always at least NEWS_EVENT when score is low
        if high_count >= 1 and score < 50:
            return RiskRegime.NEWS_EVENT

        # Special overrides
        if score >= 75:
            returns = inp.returns or []
            curr_vol = _stdev(returns[-5:]) if len(returns) >= 2 else 0.0
            base_vol = _stdev(returns[-60:]) if len(returns) >= 10 else max(curr_vol, 1e-9)
            if curr_vol / max(base_vol, 1e-9) > 4.0:
                return RiskRegime.BLACK_SWAN
            return RiskRegime.EXTREME
        if score >= 60:
            returns = inp.returns or []
            curr_vol = _stdev(returns[-5:]) if len(returns) >= 2 else 0.0
            base_vol = _stdev(returns[-60:]) if len(returns) >= 10 else max(curr_vol, 1e-9)
            if curr_vol / max(base_vol, 1e-9) > 2.5:
                return RiskRegime.FLASH_VOLATILITY
            if vol_frac < 0.2:
                return RiskRegime.LOW_LIQUIDITY
            return RiskRegime.MARKET_STRESS
        if score >= 50:
            if high_count >= 1:
                return RiskRegime.NEWS_EVENT
            return RiskRegime.HIGH_RISK
        if score >= 35:
            return RiskRegime.ELEVATED
        if score >= 20:
            return RiskRegime.NORMAL
        if score >= 10:
            return RiskRegime.LOW_RISK
        return RiskRegime.OPTIMAL
