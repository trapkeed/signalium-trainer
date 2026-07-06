"""Phase 24.9 — Tradeability Engine (proprietary Signalium metric)."""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.mie.mie_models import (
    BehaviorSimilarityResult,
    CalibrationResult,
    LiquidityResult,
    MarketInput,
    MarketRegimeResult,
    MomentumResult,
    NewsIntelligenceResult,
    PredictionReliabilityResult,
    RiskIntelligenceResult,
    SessionResult,
    TradeabilityResult,
    TrendResult,
    VolatilityResult,
)
from trainer.mie.mie_types import (
    NEWS_IMPACT_PENALTY,
    REGIME_TRADEABILITY_BASE,
    RISK_REGIME_MULTIPLIERS,
    MarketRegime,
    RiskRegime,
    TradeabilityGrade,
    clamp,
    tradeability_grade,
)

# ── Factor weights (must sum to 1.0) ──────────────────────────────────────────
_FACTOR_WEIGHTS: Dict[str, float] = {
    "prediction_confidence": 0.12,
    "expected_rr": 0.10,
    "expected_move": 0.06,
    "expected_holding": 0.05,
    "risk_regime": 0.10,
    "market_regime": 0.10,
    "liquidity": 0.08,
    "execution_quality": 0.07,
    "spread": 0.05,
    "historical_similarity": 0.06,
    "calibration": 0.06,
    "portfolio_exposure": 0.05,
    "news_calendar": 0.04,
    "trend_stability": 0.04,
    "momentum_stability": 0.02,
}

assert (
    abs(sum(_FACTOR_WEIGHTS.values()) - 1.0) < 1e-9
), f"Tradeability factor weights must sum to 1.0, got {sum(_FACTOR_WEIGHTS.values()):.6f}"


def _score_confidence(conf: float) -> float:
    """Peak at 0.70 confidence; penalty for extremes. Returns 0-100."""
    if conf < 0.30:
        return conf / 0.30 * 40.0
    if conf > 0.95:
        return (1.0 - conf) / 0.05 * 60.0
    peak = 0.70
    dist = abs(conf - peak) / 0.30
    return clamp(100.0 - dist * 40.0, 0.0, 100.0)


def _score_rr(rr: float) -> float:
    """Expected R:R — ideal ≥ 2.0, good ≥ 1.5, poor < 1.0. Returns 0-100."""
    if rr >= 3.0:
        return 100.0
    if rr >= 2.0:
        return 80.0 + (rr - 2.0) * 20.0
    if rr >= 1.5:
        return 60.0 + (rr - 1.5) * 40.0
    if rr >= 1.0:
        return 30.0 + (rr - 1.0) * 60.0
    return clamp(rr * 30.0, 0.0, 30.0)


def _score_expected_move(move: float, atr: float) -> float:
    """Expected move vs. ATR. ≥ 1.5x ATR = 100, < 0.5x = 0. Returns 0-100."""
    if atr <= 0:
        return 50.0
    ratio = move / atr
    return clamp(ratio / 1.5 * 100.0, 0.0, 100.0)


def _score_holding_time(bars: int) -> float:
    """Ideal holding 5-30 bars. < 3 or > 100 is penalised. Returns 0-100."""
    if 5 <= bars <= 30:
        return 100.0
    if bars < 5:
        return bars / 5.0 * 80.0
    if bars <= 60:
        return 80.0 - (bars - 30) / 30.0 * 30.0
    return clamp(50.0 - (bars - 60) / 40.0 * 50.0, 0.0, 50.0)


def _score_exposure(exposure: float) -> float:
    """Lower portfolio exposure = higher tradeability. Returns 0-100."""
    if exposure >= 0.85:
        return 0.0  # dangerously concentrated — no tradeability
    if exposure <= 0.20:
        return 100.0
    if exposure <= 0.50:
        return 100.0 - (exposure - 0.20) / 0.30 * 40.0
    return clamp(60.0 - (exposure - 0.50) / 0.50 * 60.0, 0.0, 60.0)


def _score_spread(spread: float, atr: float) -> float:
    """Spread as fraction of ATR — lower is better. Returns 0-100."""
    if atr <= 0:
        return 50.0
    ratio = spread / atr
    if ratio <= 0.02:
        return 100.0
    if ratio <= 0.10:
        return 80.0
    if ratio <= 0.30:
        return 50.0
    if ratio <= 0.60:
        return 20.0
    return 0.0


class TradeabilityEngine:
    """
    Computes the proprietary Signalium Tradeability Score (0-100) and Grade.

    Tradeability ≠ Confidence.
    It measures how attractive the current market environment is for trading,
    weighing 15 independent factors across risk, regime, quality, and context.
    """

    # ── Public API ─────────────────────────────────────────────────────────

    def score(
        self,
        inp: MarketInput,
        regime: Optional[MarketRegimeResult] = None,
        risk: Optional[RiskIntelligenceResult] = None,
        liquidity: Optional[LiquidityResult] = None,
        volatility: Optional[VolatilityResult] = None,
        momentum: Optional[MomentumResult] = None,
        trend: Optional[TrendResult] = None,
        similarity: Optional[BehaviorSimilarityResult] = None,
        reliability: Optional[PredictionReliabilityResult] = None,
        calibration: Optional[CalibrationResult] = None,
        session: Optional[SessionResult] = None,
        news: Optional[NewsIntelligenceResult] = None,
    ) -> TradeabilityResult:

        factors = self._compute_factors(
            inp,
            regime,
            risk,
            liquidity,
            volatility,
            momentum,
            trend,
            similarity,
            reliability,
            calibration,
            session,
            news,
        )

        raw_score = sum(factors[k] * _FACTOR_WEIGHTS[k] for k in _FACTOR_WEIGHTS if k in factors)
        score = clamp(raw_score, 0.0, 100.0)
        grade = tradeability_grade(score)
        reasons, suggestions = self._build_narrative(factors, score, inp)

        logger.debug(f"[TE] {inp.symbol} tradeability={score:.1f} grade={grade.value}")
        return TradeabilityResult(
            score=score,
            grade=grade,
            factors=factors,
            reasons=reasons,
            suggestions=suggestions,
        )

    # ── Factor computation ─────────────────────────────────────────────────

    def _compute_factors(
        self,
        inp: MarketInput,
        regime: Optional[MarketRegimeResult],
        risk: Optional[RiskIntelligenceResult],
        liquidity: Optional[LiquidityResult],
        volatility: Optional[VolatilityResult],
        momentum: Optional[MomentumResult],
        trend: Optional[TrendResult],
        similarity: Optional[BehaviorSimilarityResult],
        reliability: Optional[PredictionReliabilityResult],
        calibration: Optional[CalibrationResult],
        session: Optional[SessionResult],
        news: Optional[NewsIntelligenceResult],
    ) -> Dict[str, float]:
        f: Dict[str, float] = {}

        # 1. Prediction confidence (0-100)
        f["prediction_confidence"] = _score_confidence(inp.prediction_confidence)

        # 2. Expected R:R (0-100)
        f["expected_rr"] = _score_rr(inp.expected_rr)

        # 3. Expected move vs. ATR (0-100)
        f["expected_move"] = _score_expected_move(inp.expected_move, inp.atr)

        # 4. Expected holding time (0-100)
        f["expected_holding"] = _score_holding_time(inp.expected_holding_bars)

        # 5. Risk regime (0-100 via multiplier → 0-100)
        if risk is not None:
            risk_mult = RISK_REGIME_MULTIPLIERS.get(risk.regime, 1.0)
            f["risk_regime"] = clamp(risk_mult / 1.20 * 100.0, 0.0, 100.0)
        else:
            f["risk_regime"] = 50.0

        # 6. Market regime base tradeability (0-100)
        if regime is not None:
            f["market_regime"] = REGIME_TRADEABILITY_BASE.get(regime.regime, 35.0)
        else:
            f["market_regime"] = 50.0

        # 7. Liquidity quality (0-100)
        f["liquidity"] = liquidity.quality if liquidity is not None else 50.0

        # 8. Execution quality proxy — mix of spread + liquidity + session
        if session is not None and liquidity is not None:
            f["execution_quality"] = session.quality * 0.40 + liquidity.quality * 0.60
        elif liquidity is not None:
            f["execution_quality"] = liquidity.quality
        else:
            f["execution_quality"] = 50.0

        # 9. Spread cost (0-100)
        f["spread"] = _score_spread(inp.spread, inp.atr)

        # 10. Historical similarity (0-100)
        if similarity is not None:
            f["historical_similarity"] = similarity.similarity_score
        else:
            f["historical_similarity"] = 50.0

        # 11. Calibration (0-100)
        if calibration is not None:
            f["calibration"] = calibration.calibration_score
        else:
            f["calibration"] = clamp(inp.calibration_score * 100.0, 0.0, 100.0)

        # 12. Portfolio exposure (0-100)
        f["portfolio_exposure"] = _score_exposure(inp.portfolio_exposure)

        # 13. News / economic calendar (0-100; high impact reduces tradeability)
        if news is not None:
            f["news_calendar"] = clamp(100.0 - news.impact_score, 0.0, 100.0)
        else:
            news_pen = sum(
                NEWS_IMPACT_PENALTY.get(
                    e.get("impact", "NONE"),  # type: ignore[arg-type]
                    0.0,
                )
                for e in (inp.news_events or [])
            )
            f["news_calendar"] = clamp(100.0 - news_pen, 0.0, 100.0)

        # 14. Trend stability (0-100)
        f["trend_stability"] = (
            trend.quality if trend is not None else clamp(inp.adx / 50.0 * 100.0, 0.0, 100.0)
        )

        # 15. Momentum stability (0-100)
        f["momentum_stability"] = momentum.quality if momentum is not None else 50.0

        return f

    # ── Narrative ──────────────────────────────────────────────────────────

    def _build_narrative(
        self,
        factors: Dict[str, float],
        score: float,
        inp: MarketInput,
    ) -> tuple:
        reasons: List[str] = []
        suggestions: List[str] = []

        low_factors = {k: v for k, v in factors.items() if v < 40.0}
        for k, v in sorted(low_factors.items(), key=lambda x: x[1]):
            reasons.append(f"{k.replace('_', ' ').title()} is weak ({v:.0f}/100)")

        if factors.get("risk_regime", 100) < 40:
            suggestions.append("Wait for risk conditions to normalise")
        if factors.get("liquidity", 100) < 40:
            suggestions.append("Avoid trading in low-liquidity conditions")
        if factors.get("news_calendar", 100) < 40:
            suggestions.append("Wait for high-impact news events to pass")
        if factors.get("expected_rr", 100) < 40:
            suggestions.append("Improve entry to increase R:R ratio")
        if factors.get("prediction_confidence", 100) < 40:
            suggestions.append("Wait for a higher-confidence signal")
        if score >= 70 and not suggestions:
            suggestions.append("Market conditions are favourable — proceed with normal sizing")

        return reasons, suggestions
