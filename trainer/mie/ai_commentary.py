"""Phase 24.9 — AI Commentary Generator."""

from __future__ import annotations

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.mie.mie_models import (
    AICommentaryResult,
    ExecutionMode,
    ExecutionResult,
    MarketInput,
    MarketQualityResult,
    MarketRegimeResult,
    RiskIntelligenceResult,
    TradeabilityResult,
)
from trainer.mie.mie_types import (
    MarketRegime,
    RiskRegime,
    TradeabilityGrade,
    clamp,
)

# ── Template fragments ────────────────────────────────────────────────────────

_REGIME_SUMMARIES: dict[str, str] = {
    MarketRegime.TRENDING: "The market is in a clear trending regime with sustained directional pressure.",
    MarketRegime.STRONG_TREND: "A strong, high-conviction trend is underway — momentum is dominant.",
    MarketRegime.WEAK_TREND: "A weak trend is present but lacks the strength for high-confidence entries.",
    MarketRegime.RANGE: "Price is consolidating in a range — breakout direction is uncertain.",
    MarketRegime.COMPRESSION: "Market is in a volatility compression phase — a large move may be imminent.",
    MarketRegime.EXPANSION: "ATR is expanding — the market is entering a higher-volatility expansion phase.",
    MarketRegime.BREAKOUT: "A breakout is in progress — volume confirms directional intention.",
    MarketRegime.FALSE_BREAKOUT: "A false breakout has been detected — price has reversed after the initial surge.",
    MarketRegime.REVERSAL: "A potential regime reversal is underway — prior trend structure is breaking.",
    MarketRegime.ACCUMULATION: "Smart money appears to be accumulating — upside bias with patience required.",
    MarketRegime.DISTRIBUTION: "Distribution phase detected — institutions may be reducing long exposure.",
    MarketRegime.VOLATILE: "High volatility conditions — signals are unreliable and spreads are wide.",
    MarketRegime.ILLIQUID: "Market depth is critically low — avoid large positions.",
    MarketRegime.NEWS_DRIVEN: "Price action is being driven by news events — fundamentals dominate.",
    MarketRegime.BLACK_SWAN: "Black swan conditions — extreme caution required; all positions at risk.",
    MarketRegime.UNKNOWN: "Regime cannot be classified — insufficient signal clarity.",
}

_RISK_SUMMARIES: dict[str, str] = {
    RiskRegime.OPTIMAL: "Risk conditions are optimal — size normally.",
    RiskRegime.LOW_RISK: "Below-average risk environment — slight position size increase acceptable.",
    RiskRegime.NORMAL: "Risk is within normal parameters.",
    RiskRegime.ELEVATED: "Risk is elevated — reduce position sizes by 20%.",
    RiskRegime.HIGH_RISK: "High-risk environment — halve standard position sizes.",
    RiskRegime.EXTREME: "Extreme risk — only minimal positions if any.",
    RiskRegime.FLASH_VOLATILITY: "Flash volatility spike — all entries on hold.",
    RiskRegime.LOW_LIQUIDITY: "Low liquidity — widen stops and use limit orders only.",
    RiskRegime.MARKET_STRESS: "Market stress detected — systematic risk is elevated.",
    RiskRegime.NEWS_EVENT: "High-impact news event in proximity — wait for resolution.",
    RiskRegime.BLACK_SWAN: "Black swan risk regime — capital preservation mode.",
    RiskRegime.UNKNOWN: "Risk regime uncertain — default to conservative sizing.",
}

_GRADE_RECOMMENDATIONS: dict[str, str] = {
    TradeabilityGrade.A_PLUS: "Excellent setup — full allocation recommended.",
    TradeabilityGrade.A: "High-quality setup — standard allocation.",
    TradeabilityGrade.B: "Above-average setup — consider 75% allocation.",
    TradeabilityGrade.C: "Marginal setup — reduce size to 50% or wait.",
    TradeabilityGrade.D: "Poor setup — minimal size only, consider waiting.",
    TradeabilityGrade.AVOID: "Avoid trading this setup — conditions are unfavourable.",
}

_MODE_NOTES: dict[str, str] = {
    ExecutionMode.MARKET_ORDER: "Conditions support immediate execution — use a market order.",
    ExecutionMode.LIMIT_ORDER: "Wide spread detected — use a limit order for better fill.",
    ExecutionMode.WAIT: "Wait for conditions to improve before entering.",
    ExecutionMode.AVOID: "Do not enter — risk or tradeability conditions are prohibitive.",
}


class AICommentaryGenerator:
    """
    Generates human-readable AI commentary from MIE sub-engine results.
    Uses deterministic template-based generation — no external LLM dependency.
    """

    # ── Public API ─────────────────────────────────────────────────────────

    def generate(
        self,
        inp: MarketInput,
        regime: MarketRegimeResult | None = None,
        risk: RiskIntelligenceResult | None = None,
        tradeability: TradeabilityResult | None = None,
        quality: MarketQualityResult | None = None,
        execution: ExecutionResult | None = None,
    ) -> AICommentaryResult:

        regime_key = regime.regime.value if regime else MarketRegime.UNKNOWN.value
        risk_key = risk.regime.value if risk else RiskRegime.NORMAL.value
        t_score = tradeability.score if tradeability else 50.0
        t_grade = tradeability.grade if tradeability else TradeabilityGrade.C
        exec_mode = execution.mode if execution else ExecutionMode.WAIT

        regime_note = _REGIME_SUMMARIES.get(
            regime_key, _REGIME_SUMMARIES[MarketRegime.UNKNOWN.value]
        )
        risk_note = _RISK_SUMMARIES.get(risk_key, _RISK_SUMMARIES[RiskRegime.NORMAL.value])
        trade_note = _MODE_NOTES.get(exec_mode.value, "")
        recommendation = _GRADE_RECOMMENDATIONS.get(
            t_grade.value, _GRADE_RECOMMENDATIONS[TradeabilityGrade.C.value]
        )

        summary = self._build_summary(
            inp,
            regime_key,
            risk_key,
            t_score,
            t_grade,
            quality,
        )
        tags = self._build_tags(regime_key, risk_key, t_grade)
        confidence = clamp(inp.prediction_confidence, 0.0, 1.0)

        logger.debug(f"[ACG] {inp.symbol} grade={t_grade.value} summary generated")
        return AICommentaryResult(
            summary=summary,
            regime_note=regime_note,
            risk_note=risk_note,
            trade_note=trade_note,
            recommendation=recommendation,
            confidence=confidence,
            tags=tags,
        )

    # ── Internal ───────────────────────────────────────────────────────────

    def _build_summary(
        self,
        inp: MarketInput,
        regime_key: str,
        risk_key: str,
        t_score: float,
        t_grade: TradeabilityGrade,
        quality: MarketQualityResult | None,
    ) -> str:
        mq = f"{quality.market_quality:.0f}/100" if quality else "N/A"
        return (
            f"{inp.symbol}: Tradeability {t_score:.0f}/100 ({t_grade.value}). "
            f"Market Regime: {regime_key.replace('_', ' ').title()}. "
            f"Risk: {risk_key.replace('_', ' ').title()}. "
            f"Market Quality: {mq}. "
            f"Confidence: {inp.prediction_confidence:.0%}."
        )

    def _build_tags(
        self,
        regime_key: str,
        risk_key: str,
        t_grade: TradeabilityGrade,
    ) -> list[str]:
        tags: list[str] = [regime_key, risk_key, f"GRADE_{t_grade.value}"]
        if t_grade in (TradeabilityGrade.A_PLUS, TradeabilityGrade.A):
            tags.append("HIGH_QUALITY")
        if risk_key in (RiskRegime.EXTREME.value, RiskRegime.BLACK_SWAN.value):
            tags.append("DANGER")
        if t_grade == TradeabilityGrade.AVOID:
            tags.append("AVOID")
        return tags
