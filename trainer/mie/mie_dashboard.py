"""Phase 24.9 — MIE Dashboard views."""

from __future__ import annotations

from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.mie.mie_models import MarketIntelligenceReport


class MIEDashboard:
    """
    Generates structured dashboard view dicts from a MarketIntelligenceReport.

    Views:
    1. overview       — top-level snapshot (tradeability, regime, risk, quality)
    2. regime_view    — detailed regime intelligence
    3. risk_view      — risk regime and factor breakdown
    4. tradeability_view — full tradeability factor breakdown with grade
    5. quality_view   — 9 quality sub-scores
    6. execution_view — entry/exit zones, stop, target, execution mode
    7. commentary_view — AI commentary + recommendation
    8. feature_view   — 12-dim feature vector for model consumption
    9. live_analysis  — full live analysis summary (EA / Android compatible)
    """

    # ── Public API ─────────────────────────────────────────────────────────

    def overview(self, report: MarketIntelligenceReport) -> dict[str, Any]:
        t = report.tradeability
        ri = report.risk
        re = report.regime
        q = report.quality
        return {
            "report_id": report.report_id,
            "symbol": report.symbol,
            "timestamp": report.timestamp,
            "tradeability": round(t.score, 1) if t else None,
            "trade_grade": t.grade.value if t else None,
            "market_regime": re.regime.value if re else None,
            "risk_regime": ri.regime.value if ri else None,
            "risk_score": round(ri.risk_score, 1) if ri else None,
            "market_quality": round(q.market_quality, 1) if q else None,
            "execution_quality": round(q.execution_quality, 1) if q else None,
            "commentary": report.commentary.summary if report.commentary else None,
        }

    def regime_view(self, report: MarketIntelligenceReport) -> dict[str, Any]:
        re = report.regime
        if not re:
            return {"error": "No regime result"}
        v = report.volatility
        tr = report.trend
        mo = report.momentum
        return {
            "symbol": report.symbol,
            "regime": re.regime.value,
            "confidence": round(re.confidence, 3),
            "sub_regime": re.sub_regime,
            "signals": re.signals,
            "trend_state": tr.state.value if tr else None,
            "trend_direction": tr.direction if tr else None,
            "adx": tr.adx_value if tr else None,
            "momentum_state": mo.state.value if mo else None,
            "momentum_strength": round(mo.strength, 4) if mo else None,
            "volatility_state": v.state.value if v else None,
            "vol_ratio": round(v.vol_ratio, 3) if v else None,
        }

    def risk_view(self, report: MarketIntelligenceReport) -> dict[str, Any]:
        ri = report.risk
        if not ri:
            return {"error": "No risk result"}
        return {
            "symbol": report.symbol,
            "risk_regime": ri.regime.value,
            "risk_score": round(ri.risk_score, 1),
            "multiplier": round(ri.multiplier, 3),
            "factors": {k: round(v, 2) for k, v in ri.factors.items()},
            "warnings": ri.warnings,
        }

    def tradeability_view(self, report: MarketIntelligenceReport) -> dict[str, Any]:
        t = report.tradeability
        if not t:
            return {"error": "No tradeability result"}
        return {
            "symbol": report.symbol,
            "score": round(t.score, 1),
            "grade": t.grade.value,
            "factors": {k: round(v, 1) for k, v in t.factors.items()},
            "reasons": t.reasons,
            "suggestions": t.suggestions,
        }

    def quality_view(self, report: MarketIntelligenceReport) -> dict[str, Any]:
        q = report.quality
        if not q:
            return {"error": "No quality result"}
        return q.to_dict()

    def execution_view(self, report: MarketIntelligenceReport) -> dict[str, Any]:
        e = report.execution
        if not e:
            return {"error": "No execution result"}
        return e.to_dict()

    def commentary_view(self, report: MarketIntelligenceReport) -> dict[str, Any]:
        c = report.commentary
        if not c:
            return {"error": "No commentary result"}
        return c.to_dict()

    def feature_view(self, report: MarketIntelligenceReport) -> dict[str, Any]:
        f = report.features
        if not f:
            return {"error": "No feature export"}
        return {
            "symbol": f.symbol,
            "timestamp": f.timestamp,
            "feature_vector": f.to_feature_vector(),
            "named_features": f.to_dict(),
        }

    def live_analysis(self, report: MarketIntelligenceReport) -> dict[str, Any]:
        """Full live analysis view — EA / Android / Website compatible."""
        t = report.tradeability
        ri = report.risk
        re = report.regime
        q = report.quality
        e = report.execution
        c = report.commentary
        li = report.liquidity
        mo = report.momentum
        tr = report.trend
        sim = report.similarity
        rel = report.reliability

        return {
            "symbol": report.symbol,
            "timestamp": report.timestamp,
            "commentary": c.summary if c else "",
            "tradeability": round(t.score, 1) if t else None,
            "trade_grade": t.grade.value if t else None,
            "risk_regime": ri.regime.value if ri else None,
            "risk_score": round(ri.risk_score, 1) if ri else None,
            "market_quality": round(q.market_quality, 1) if q else None,
            "execution_quality": round(q.execution_quality, 1) if q else None,
            "expected_move": None,  # populated by caller from inp.expected_move
            "expected_holding": None,  # populated by caller from inp.expected_holding_bars
            "momentum_state": mo.state.value if mo else None,
            "momentum_quality": round(mo.quality, 1) if mo else None,
            "trend_state": tr.state.value if tr else None,
            "trend_direction": tr.direction if tr else None,
            "trend_quality": round(tr.quality, 1) if tr else None,
            "liquidity_state": li.state.value if li else None,
            "liquidity_quality": round(li.quality, 1) if li else None,
            "prediction_reliability": round(rel.reliability_score, 1) if rel else None,
            "historical_similarity": round(sim.similarity_score, 1) if sim else None,
            "confidence": None,  # populated by caller
            "trade_recommendation": c.recommendation if c else None,
            "execution_mode": e.mode.value if e else None,
            "optimal_stop": round(e.optimal_stop, 6) if e else None,
            "optimal_target": round(e.optimal_target, 6) if e else None,
            "market_regime": re.regime.value if re else None,
        }

    def ea_view(
        self, report: MarketIntelligenceReport, inp_dict: dict | None = None
    ) -> dict[str, Any]:
        """EA-specific view with prediction, tradeability, SL/TP."""
        t = report.tradeability
        ri = report.risk
        re = report.regime
        e = report.execution
        rel = report.reliability
        inp = inp_dict or {}
        return {
            "prediction": inp.get("prediction_confidence"),
            "tradeability": round(t.score, 1) if t else None,
            "trade_grade": t.grade.value if t else None,
            "execution_mode": e.mode.value if e else None,
            "expected_move": inp.get("expected_move"),
            "holding_time": inp.get("expected_holding_bars"),
            "sl": round(e.optimal_stop, 6) if e else None,
            "tp": round(e.optimal_target, 6) if e else None,
            "market_regime": re.regime.value if re else None,
            "risk_regime": ri.regime.value if ri else None,
            "prediction_reliability": round(rel.reliability_score, 1) if rel else None,
        }

    def android_view(
        self, report: MarketIntelligenceReport, inp_dict: dict | None = None
    ) -> dict[str, Any]:
        """Android-specific view — compact for mobile display."""
        t = report.tradeability
        q = report.quality
        c = report.commentary
        e = report.execution
        inp = inp_dict or {}
        return {
            "tradeability": round(t.score, 1) if t else None,
            "trade_grade": t.grade.value if t else None,
            "market_quality": round(q.market_quality, 1) if q else None,
            "commentary": c.summary if c else "",
            "prediction_confidence": inp.get("prediction_confidence"),
            "expected_move": inp.get("expected_move"),
            "holding_time": inp.get("expected_holding_bars"),
            "execution_mode": e.mode.value if e else None,
        }

    def all_views(self, report: MarketIntelligenceReport) -> dict[str, Any]:
        return {
            "overview": self.overview(report),
            "regime": self.regime_view(report),
            "risk": self.risk_view(report),
            "tradeability": self.tradeability_view(report),
            "quality": self.quality_view(report),
            "execution": self.execution_view(report),
            "commentary": self.commentary_view(report),
            "features": self.feature_view(report),
            "live_analysis": self.live_analysis(report),
        }
