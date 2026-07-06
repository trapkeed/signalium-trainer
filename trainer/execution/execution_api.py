"""Phase 21 — Execution API.

16 endpoints wrapping the full execution intelligence stack.
Pure Python — no web framework, no broker connectivity.

Endpoints
---------
plan            — full plan for one forecast
plan_batch      — plans for multiple forecasts
entry_zones     — entry zone levels only
entry_quality   — entry quality score only
stop_loss       — stop loss recommendations only
take_profit     — take profit recommendations only
sizing          — position size only
management      — management plan only
timing          — execution timing only
risk            — risk profile only
order           — order strategy only
score           — trade score only
simulation      — execution simulation only
dashboard       — full multi-plan dashboard
stats           — database statistics
history         — stored plan history
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.execution.entry_quality import EntryQualityEngine
from trainer.execution.entry_zones import EntryZoneEngine
from trainer.execution.execution_dashboard import ExecutionDashboard
from trainer.execution.execution_database import ExecutionDatabase
from trainer.execution.execution_planner import ExecutionPlanner
from trainer.execution.execution_simulator import ExecutionSimulator
from trainer.execution.execution_timing import ExecutionTiming
from trainer.execution.execution_types import (
    BrokerConstraints,
    ExecutionForecast,
    Session,
    SizingMethod,
    default_broker,
)
from trainer.execution.order_strategy import OrderStrategyEngine
from trainer.execution.position_sizer import PositionSizer, RiskLimits
from trainer.execution.risk_engine import RiskEngine
from trainer.execution.stop_loss_engine import StopLossEngine
from trainer.execution.take_profit_engine import TakeProfitEngine
from trainer.execution.trade_manager import TradeManager
from trainer.execution.trade_scorer import TradeScorer


def _ok(data: Any, meta: Optional[Dict] = None) -> Dict:
    r: Dict = {"status": "ok", "data": data, "timestamp": round(time.time(), 3)}
    if meta:
        r["meta"] = meta
    return r


def _err(msg: str, code: str = "ERROR") -> Dict:
    return {"status": "error", "code": code, "message": msg, "timestamp": round(time.time(), 3)}


def _fc(d: Dict) -> ExecutionForecast:
    """Deserialise a forecast dict → ExecutionForecast."""
    try:
        sess = Session(d.get("session", "LONDON"))
    except ValueError:
        sess = Session.LONDON
    return ExecutionForecast(
        instrument=d["instrument"],
        direction=d["direction"],
        confidence=float(d.get("confidence", 0.65)),
        expected_return=float(d.get("expected_return", 0.08)),
        expected_risk=float(d.get("expected_risk", 0.10)),
        current_price=float(d.get("current_price", 1.0)),
        atr=float(d.get("atr", 0.001)),
        spread=float(d.get("spread", 0.0001)),
        session=sess,
        volatility=float(d.get("volatility", 0.01)),
        momentum=float(d.get("momentum", 0.0)),
        news_risk=float(d.get("news_risk", 0.0)),
        account_size=float(d.get("account_size", 10_000.0)),
    )


class ExecutionApi:
    """Pure-Python execution intelligence API.

    All methods accept plain dicts and return plain dicts.
    No HTTP layer, no broker connection.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        broker: Optional[BrokerConstraints] = None,
        risk_limits: Optional[RiskLimits] = None,
        sizing_method: str = "FIXED_RISK",
    ):
        self._broker = broker or default_broker()
        self._limits = risk_limits or RiskLimits()
        try:
            self._sizing_method = SizingMethod(sizing_method)
        except ValueError:
            self._sizing_method = SizingMethod.FIXED_RISK

        self._planner = ExecutionPlanner(self._broker, self._limits, self._sizing_method)
        self._db = ExecutionDatabase(db_path)

        # Standalone engines for partial requests
        self._zones = EntryZoneEngine()
        self._quality = EntryQualityEngine()
        self._stops = StopLossEngine()
        self._tp = TakeProfitEngine()
        self._sizer = PositionSizer(self._broker, self._limits)
        self._manager = TradeManager()
        self._timer = ExecutionTiming(self._broker)
        self._risk = RiskEngine()
        self._order = OrderStrategyEngine()
        self._scorer = TradeScorer()
        self._sim = ExecutionSimulator(self._broker)

    # ── 1. plan ───────────────────────────────────────────────────────────────

    def plan(self, forecast_dict: Dict) -> Dict:
        """Generate a full execution plan for one forecast."""
        try:
            fc = _fc(forecast_dict)
            ep = self._planner.plan(fc)
            self._db.store_plan(ep)
            return _ok(ep.to_dict(), {"instrument": fc.instrument})
        except Exception as e:
            logger.error(f"plan: {e}")
            return _err(str(e))

    # ── 2. plan_batch ─────────────────────────────────────────────────────────

    def plan_batch(self, forecast_dicts: List[Dict]) -> Dict:
        """Generate plans for multiple forecasts."""
        try:
            fcs = [_fc(d) for d in forecast_dicts]
            plans = self._planner.plan_batch(fcs)
            for p in plans:
                self._db.store_plan(p)
            return _ok(
                [p.to_dict() for p in plans],
                {"count": len(plans)},
            )
        except Exception as e:
            logger.error(f"plan_batch: {e}")
            return _err(str(e))

    # ── 3. entry_zones ────────────────────────────────────────────────────────

    def entry_zones(self, forecast_dict: Dict) -> Dict:
        try:
            fc = _fc(forecast_dict)
            z = self._zones.calculate(fc)
            return _ok(z.to_dict())
        except Exception as e:
            return _err(str(e))

    # ── 4. entry_quality ──────────────────────────────────────────────────────

    def entry_quality(self, forecast_dict: Dict) -> Dict:
        try:
            fc = _fc(forecast_dict)
            z = self._zones.calculate(fc)
            q = self._quality.evaluate(fc, z)
            return _ok(q.to_dict())
        except Exception as e:
            return _err(str(e))

    # ── 5. stop_loss ──────────────────────────────────────────────────────────

    def stop_loss(self, forecast_dict: Dict) -> Dict:
        try:
            fc = _fc(forecast_dict)
            z = self._zones.calculate(fc)
            sl = self._stops.recommend(fc, z)
            return _ok(sl.to_dict())
        except Exception as e:
            return _err(str(e))

    # ── 6. take_profit ────────────────────────────────────────────────────────

    def take_profit(self, forecast_dict: Dict) -> Dict:
        try:
            fc = _fc(forecast_dict)
            z = self._zones.calculate(fc)
            sl = self._stops.recommend(fc, z)
            tp = self._tp.recommend(fc, z, sl)
            return _ok(tp.to_dict())
        except Exception as e:
            return _err(str(e))

    # ── 7. sizing ─────────────────────────────────────────────────────────────

    def sizing(self, forecast_dict: Dict, method: str = "FIXED_RISK") -> Dict:
        try:
            fc = _fc(forecast_dict)
            z = self._zones.calculate(fc)
            sl = self._stops.recommend(fc, z)
            try:
                m = SizingMethod(method)
            except ValueError:
                m = SizingMethod.FIXED_RISK
            sz = self._sizer.calculate(fc, sl, m)
            return _ok(sz.to_dict())
        except Exception as e:
            return _err(str(e))

    # ── 8. management ─────────────────────────────────────────────────────────

    def management(self, forecast_dict: Dict) -> Dict:
        try:
            fc = _fc(forecast_dict)
            z = self._zones.calculate(fc)
            sl = self._stops.recommend(fc, z)
            tp = self._tp.recommend(fc, z, sl)
            sz = self._sizer.calculate(fc, sl)
            mp = self._manager.plan(fc, sz, sl, tp)
            return _ok(mp.to_dict())
        except Exception as e:
            return _err(str(e))

    # ── 9. timing ─────────────────────────────────────────────────────────────

    def timing(self, forecast_dict: Dict) -> Dict:
        try:
            fc = _fc(forecast_dict)
            t = self._timer.evaluate(fc)
            return _ok(t.to_dict())
        except Exception as e:
            return _err(str(e))

    # ── 10. risk ──────────────────────────────────────────────────────────────

    def risk(self, forecast_dict: Dict) -> Dict:
        try:
            fc = _fc(forecast_dict)
            z = self._zones.calculate(fc)
            sl = self._stops.recommend(fc, z)
            tp = self._tp.recommend(fc, z, sl)
            rp = self._risk.analyze(fc, sl, tp)
            return _ok(rp.to_dict())
        except Exception as e:
            return _err(str(e))

    # ── 11. order ─────────────────────────────────────────────────────────────

    def order(self, forecast_dict: Dict) -> Dict:
        try:
            fc = _fc(forecast_dict)
            z = self._zones.calculate(fc)
            t = self._timer.evaluate(fc)
            o = self._order.recommend(fc, z, t, self._broker)
            return _ok(o.to_dict())
        except Exception as e:
            return _err(str(e))

    # ── 12. score ─────────────────────────────────────────────────────────────

    def score(self, forecast_dict: Dict) -> Dict:
        try:
            fc = _fc(forecast_dict)
            z = self._zones.calculate(fc)
            q = self._quality.evaluate(fc, z)
            sl = self._stops.recommend(fc, z)
            tp = self._tp.recommend(fc, z, sl)
            rp = self._risk.analyze(fc, sl, tp)
            t = self._timer.evaluate(fc)
            s = self._scorer.score(fc, q, rp, t)
            return _ok(s.to_dict())
        except Exception as e:
            return _err(str(e))

    # ── 13. simulation ────────────────────────────────────────────────────────

    def simulation(self, forecast_dict: Dict) -> Dict:
        try:
            fc = _fc(forecast_dict)
            z = self._zones.calculate(fc)
            sl = self._stops.recommend(fc, z)
            tp = self._tp.recommend(fc, z, sl)
            sr = self._sim.simulate(fc, z, sl, tp)
            return _ok(sr.to_dict())
        except Exception as e:
            return _err(str(e))

    # ── 14. dashboard ─────────────────────────────────────────────────────────

    def dashboard(self, forecast_dicts: List[Dict]) -> Dict:
        """Generate a full 8-view dashboard for multiple forecasts."""
        try:
            fcs = [_fc(d) for d in forecast_dicts]
            plans = self._planner.plan_batch(fcs)
            dash = ExecutionDashboard(plans)
            return _ok(dash.full_dashboard(), {"plan_count": len(plans)})
        except Exception as e:
            logger.error(f"dashboard: {e}")
            return _err(str(e))

    # ── 15. stats ─────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        """Database statistics."""
        try:
            return _ok(self._db.stats())
        except Exception as e:
            return _err(str(e))

    # ── 16. history ───────────────────────────────────────────────────────────

    def history(self, instrument: Optional[str] = None, limit: int = 10) -> Dict:
        """Stored plan history, optionally filtered by instrument."""
        try:
            if instrument:
                plans = self._db.plans_by_instrument(instrument)[:limit]
            else:
                plans = self._db.latest_plans(limit)
            return _ok(plans, {"count": len(plans)})
        except Exception as e:
            return _err(str(e))
