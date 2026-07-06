"""Phase 24.9 — MIE Unified API.

The MIE API is the single entry point for all Market Intelligence Engine
operations. It orchestrates all sub-engines, persists results, and provides
response shapes for the 9 REST endpoints.

Endpoints simulated:
    GET  /api/v1/intelligence/current       → current()
    GET  /api/v1/intelligence/history       → history()
    GET  /api/v1/intelligence/tradeability  → tradeability()
    GET  /api/v1/intelligence/risk          → risk()
    GET  /api/v1/intelligence/regime        → regime()
    GET  /api/v1/intelligence/execution     → execution()
    GET  /api/v1/intelligence/quality       → quality()
    GET  /api/v1/intelligence/commentary    → commentary()
    POST /api/v1/intelligence/recalculate   → recalculate()
"""

from __future__ import annotations

from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.mie.ai_commentary import AICommentaryGenerator
from trainer.mie.behavior_similarity import BehaviorSimilarityEngine
from trainer.mie.calibration_engine import CalibrationEngine
from trainer.mie.execution_intelligence import ExecutionIntelligenceEngine
from trainer.mie.feature_store import FeatureStoreExporter
from trainer.mie.liquidity_intelligence import LiquidityIntelligenceEngine
from trainer.mie.market_quality_engine import MarketQualityEngine
from trainer.mie.market_regime_engine import MarketRegimeEngine
from trainer.mie.mie_dashboard import MIEDashboard
from trainer.mie.mie_database import MIEDatabase
from trainer.mie.mie_models import (
    MarketInput,
    MarketIntelligenceReport,
)
from trainer.mie.momentum_intelligence import MomentumIntelligenceEngine
from trainer.mie.news_intelligence import NewsIntelligenceEngine
from trainer.mie.portfolio_risk_engine import MIEPortfolioRiskEngine
from trainer.mie.prediction_reliability import PredictionReliabilityEngine
from trainer.mie.risk_intelligence import RiskIntelligenceEngine
from trainer.mie.session_intelligence import SessionIntelligenceEngine
from trainer.mie.tradeability_engine import TradeabilityEngine
from trainer.mie.trend_intelligence import TrendIntelligenceEngine
from trainer.mie.volatility_intelligence import VolatilityIntelligenceEngine


class MIEAPI:
    """
    Orchestrates the full Market Intelligence Engine pipeline.

    Usage:
        api = MIEAPI()
        report = api.recalculate(market_input)
        current = api.current("EURUSD")
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db = MIEDatabase(db_path)
        self._dash = MIEDashboard()

        # Sub-engines
        self._regime = MarketRegimeEngine()
        self._risk = RiskIntelligenceEngine()
        self._vol = VolatilityIntelligenceEngine()
        self._mom = MomentumIntelligenceEngine()
        self._trend = TrendIntelligenceEngine()
        self._liq = LiquidityIntelligenceEngine()
        self._session = SessionIntelligenceEngine()
        self._news = NewsIntelligenceEngine()
        self._similarity = BehaviorSimilarityEngine()
        self._reliability = PredictionReliabilityEngine()
        self._calibration = CalibrationEngine()
        self._tradeability = TradeabilityEngine()
        self._execution = ExecutionIntelligenceEngine()
        self._quality = MarketQualityEngine()
        self._portfolio = MIEPortfolioRiskEngine()
        self._commentary = AICommentaryGenerator()
        self._features = FeatureStoreExporter()

        logger.info("[MIEAPI] Initialised with all 16 engines")

    # ── Core pipeline ──────────────────────────────────────────────────────

    def recalculate(self, inp: MarketInput) -> MarketIntelligenceReport:
        """
        POST /api/v1/intelligence/recalculate

        Runs the full MIE pipeline and persists all results.
        """
        logger.info(f"[MIEAPI] Recalculating for {inp.symbol}")
        report = MarketIntelligenceReport(symbol=inp.symbol)

        # Layer 1: independent signal engines
        report.regime = self._regime.detect(inp)
        report.risk = self._risk.evaluate(inp)
        report.volatility = self._vol.analyse(inp)
        report.momentum = self._mom.analyse(inp)
        report.trend = self._trend.analyse(inp)
        report.liquidity = self._liq.analyse(inp)
        report.session = self._session.analyse(inp)
        report.news = self._news.evaluate(inp)
        report.similarity = self._similarity.analyse(inp)
        report.reliability = self._reliability.evaluate(inp)
        report.calibration = self._calibration.evaluate(inp)

        # Layer 2: synthesis engines (depend on layer 1)
        report.tradeability = self._tradeability.score(
            inp,
            regime=report.regime,
            risk=report.risk,
            liquidity=report.liquidity,
            volatility=report.volatility,
            momentum=report.momentum,
            trend=report.trend,
            similarity=report.similarity,
            reliability=report.reliability,
            calibration=report.calibration,
            session=report.session,
            news=report.news,
        )
        report.execution = self._execution.recommend(
            inp,
            risk=report.risk,
            liquidity=report.liquidity,
            tradeability=report.tradeability,
            volatility=report.volatility,
        )
        report.quality = self._quality.evaluate(
            inp,
            volatility=report.volatility,
            momentum=report.momentum,
            trend=report.trend,
            liquidity=report.liquidity,
            session=report.session,
            news=report.news,
            reliability=report.reliability,
        )
        report.portfolio_risk = self._portfolio.evaluate(
            inp,
            risk=report.risk,
            tradeability=report.tradeability,
        )

        # Layer 3: output / presentation
        report.commentary = self._commentary.generate(
            inp,
            regime=report.regime,
            risk=report.risk,
            tradeability=report.tradeability,
            quality=report.quality,
            execution=report.execution,
        )
        report.features = self._features.export(
            inp,
            regime=report.regime,
            risk=report.risk,
            tradeability=report.tradeability,
            quality=report.quality,
            reliability=report.reliability,
            similarity=report.similarity,
            calibration=report.calibration,
            trend=report.trend,
            momentum=report.momentum,
            liquidity=report.liquidity,
        )

        self._persist(report)
        return report

    # ── GET endpoint handlers ──────────────────────────────────────────────

    def current(self, symbol: str) -> dict[str, Any]:
        """GET /api/v1/intelligence/current"""
        rows = self._db.get_latest_intelligence(symbol, limit=1)
        if not rows:
            return {"error": f"No intelligence found for {symbol}"}
        return {"ok": True, "symbol": symbol, "data": rows[0]}

    def history(
        self,
        symbol: str,
        since: float = 0.0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """GET /api/v1/intelligence/history"""
        rows = self._db.get_regime_history(symbol, since=since, limit=limit)
        return {"ok": True, "symbol": symbol, "count": len(rows), "data": rows}

    def tradeability_endpoint(
        self,
        symbol: str,
        since: float = 0.0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """GET /api/v1/intelligence/tradeability"""
        rows = self._db.get_tradeability_history(symbol, since=since, limit=limit)
        return {"ok": True, "symbol": symbol, "count": len(rows), "data": rows}

    def risk_endpoint(
        self,
        symbol: str,
        since: float = 0.0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """GET /api/v1/intelligence/risk"""
        rows = self._db.get_risk_history(symbol, since=since, limit=limit)
        return {"ok": True, "symbol": symbol, "count": len(rows), "data": rows}

    def regime_endpoint(
        self,
        symbol: str,
        since: float = 0.0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """GET /api/v1/intelligence/regime"""
        rows = self._db.get_regime_history(symbol, since=since, limit=limit)
        return {"ok": True, "symbol": symbol, "count": len(rows), "data": rows}

    def execution_endpoint(self, symbol: str, limit: int = 10) -> dict[str, Any]:
        """GET /api/v1/intelligence/execution"""
        rows = self._db.get_scores_history(symbol, limit=limit)
        return {"ok": True, "symbol": symbol, "count": len(rows), "data": rows}

    def quality_endpoint(
        self,
        symbol: str,
        since: float = 0.0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """GET /api/v1/intelligence/quality"""
        rows = self._db.get_quality_history(symbol, since=since, limit=limit)
        return {"ok": True, "symbol": symbol, "count": len(rows), "data": rows}

    def commentary_endpoint(self, symbol: str, limit: int = 10) -> dict[str, Any]:
        """GET /api/v1/intelligence/commentary"""
        rows = self._db.get_commentary_history(symbol, limit=limit)
        return {"ok": True, "symbol": symbol, "count": len(rows), "data": rows}

    # ── Persistence ────────────────────────────────────────────────────────

    def _persist(self, report: MarketIntelligenceReport) -> None:
        ts = report.timestamp
        sym = report.symbol

        self._db.save_intelligence(
            {
                "report_id": report.report_id,
                "symbol": sym,
                "timestamp": ts,
                "tradeability": report.tradeability.score if report.tradeability else None,
                "trade_grade": report.tradeability.grade.value if report.tradeability else None,
                "market_quality": report.quality.market_quality if report.quality else None,
                "execution_quality": report.quality.execution_quality if report.quality else None,
                "regime": report.regime.regime.value if report.regime else None,
                "risk_regime": report.risk.regime.value if report.risk else None,
                "summary": report.commentary.summary if report.commentary else None,
            }
        )

        if report.regime:
            self._db.save_regime(
                {
                    "symbol": sym,
                    "timestamp": ts,
                    "regime": report.regime.regime.value,
                    "confidence": report.regime.confidence,
                    "sub_regime": report.regime.sub_regime,
                    "adx": report.regime.signals.get("adx"),
                    "vol_ratio": report.regime.signals.get("vol_ratio"),
                }
            )

        if report.risk:
            f = report.risk.factors
            self._db.save_risk(
                {
                    "symbol": sym,
                    "timestamp": ts,
                    "risk_regime": report.risk.regime.value,
                    "risk_score": report.risk.risk_score,
                    "multiplier": report.risk.multiplier,
                    "vol_factor": f.get("volatility"),
                    "spread_factor": f.get("spread"),
                    "liquidity_factor": f.get("liquidity"),
                }
            )

        if report.tradeability:
            f = report.tradeability.factors
            self._db.save_tradeability(
                {
                    "symbol": sym,
                    "timestamp": ts,
                    "score": report.tradeability.score,
                    "grade": report.tradeability.grade.value,
                    "confidence_f": f.get("prediction_confidence"),
                    "rr_f": f.get("expected_rr"),
                    "liquidity_f": f.get("liquidity"),
                    "risk_f": f.get("risk_regime"),
                    "regime_f": f.get("market_regime"),
                }
            )

        if report.quality:
            q = report.quality
            self._db.save_quality(
                {
                    "symbol": sym,
                    "timestamp": ts,
                    "market_quality": q.market_quality,
                    "execution_quality": q.execution_quality,
                    "prediction_quality": q.prediction_quality,
                    "liquidity_quality": q.liquidity_quality,
                    "volatility_quality": q.volatility_quality,
                    "momentum_quality": q.momentum_quality,
                    "trend_quality": q.trend_quality,
                    "session_quality": q.session_quality,
                    "news_impact": q.news_impact_score,
                }
            )

        if report.execution:
            e = report.execution
            self._db.save_execution(
                {
                    "symbol": sym,
                    "timestamp": ts,
                    "mode": e.mode.value,
                    "entry_low": e.entry_zone_low,
                    "entry_high": e.entry_zone_high,
                    "optimal_stop": e.optimal_stop,
                    "optimal_target": e.optimal_target,
                    "expected_dd": e.expected_drawdown,
                    "max_hold_bars": e.max_holding_bars,
                }
            )

        if report.reliability:
            r = report.reliability
            self._db.save_reliability(
                {
                    "symbol": sym,
                    "timestamp": ts,
                    "reliability": r.reliability_score,
                    "confidence_score": r.confidence_score,
                    "consistency": r.consistency_score,
                    "model_health": r.model_health,
                    "win_rate_adj": r.win_rate_adj,
                }
            )

        if report.similarity:
            s = report.similarity
            self._db.save_similarity(
                {
                    "symbol": sym,
                    "timestamp": ts,
                    "similarity": s.similarity_score,
                    "historical_win": s.historical_success,
                    "match_count": s.match_count,
                    "best_match": s.best_match_score,
                }
            )

        if report.commentary:
            c = report.commentary
            self._db.save_commentary(
                {
                    "symbol": sym,
                    "timestamp": ts,
                    "summary": c.summary,
                    "regime_note": c.regime_note,
                    "risk_note": c.risk_note,
                    "trade_note": c.trade_note,
                    "recommendation": c.recommendation,
                    "confidence": c.confidence,
                    "tags": c.tags,
                }
            )

        self._db.save_scores(
            {
                "symbol": sym,
                "timestamp": ts,
                "tradeability": report.tradeability.score if report.tradeability else None,
                "market_quality": report.quality.market_quality if report.quality else None,
                "risk_score": report.risk.risk_score if report.risk else None,
                "regime": report.regime.regime.value if report.regime else None,
                "risk_regime": report.risk.regime.value if report.risk else None,
                "session": report.session.session.value if report.session else None,
                "news_impact": report.news.impact_score if report.news else None,
                "volatility_state": report.volatility.state.value if report.volatility else None,
                "momentum_state": report.momentum.state.value if report.momentum else None,
                "trend_state": report.trend.state.value if report.trend else None,
            }
        )

        logger.debug(f"[MIEAPI] Persisted report {report.report_id} for {sym}")
