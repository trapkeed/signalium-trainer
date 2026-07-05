"""Phase 21.5 — Lifecycle API.

Provides the public interface for the Trade Lifecycle Intelligence Engine.

Endpoints:
 1. create_lifecycle       — create a new trade lifecycle
 2. get_lifecycle          — retrieve lifecycle summary
 3. transition             — execute a state transition
 4. auto_transition        — engine-recommended transition
 5. recommend_transition   — get transition recommendations without acting
 6. position_management    — position management recommendations
 7. lifecycle_confidence   — state confidence assessment
 8. detect_events          — detect and return lifecycle events
 9. reasoning              — institutional AI reasoning for current state
10. state_history          — full state history
11. lifecycle_analytics    — analytics across a batch of lifecycles
12. optimisation           — optimisation recommendations
13. timeline_dashboard     — lifecycle timeline view
14. confidence_dashboard   — confidence timeline view
15. transition_dashboard   — transition matrix view
16. statistics_dashboard   — lifecycle statistics view
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from trainer.lifecycle.lifecycle_states import LifecycleState
from trainer.lifecycle.lifecycle_models import MarketContext, PositionMetrics
from trainer.lifecycle.trade_lifecycle import TradeLifecycle
from trainer.lifecycle.state_machine import StateMachine
from trainer.lifecycle.transition_engine import TransitionEngine
from trainer.lifecycle.position_monitor import PositionMonitor
from trainer.lifecycle.lifecycle_confidence import LifecycleConfidenceEngine
from trainer.lifecycle.event_engine import EventEngine
from trainer.lifecycle.ai_reasoning import AiReasoningEngine
from trainer.lifecycle.lifecycle_analytics import LifecycleAnalytics
from trainer.lifecycle.lifecycle_optimizer import LifecycleOptimizer
from trainer.lifecycle.lifecycle_dashboard import LifecycleDashboard
from trainer.lifecycle.lifecycle_database import LifecycleDatabase


@dataclass
class ApiResponse:
    success:  bool
    endpoint: str
    data:     Dict       = field(default_factory=dict)
    error:    str        = ""
    warnings: List[str]  = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = {
            "success":  self.success,
            "endpoint": self.endpoint,
            "data":     self.data,
        }
        if self.error:
            d["error"] = self.error
        if self.warnings:
            d["warnings"] = self.warnings
        return d


class LifecycleAPI:
    """Public API for the Trade Lifecycle Intelligence Engine."""

    def __init__(self, db: Optional[LifecycleDatabase] = None):
        self._db    = db or LifecycleDatabase()
        self._sm    = StateMachine()
        self._te    = TransitionEngine()
        self._pm    = PositionMonitor()
        self._lce   = LifecycleConfidenceEngine()
        self._ee    = EventEngine()
        self._are   = AiReasoningEngine()
        self._la    = LifecycleAnalytics()
        self._lo    = LifecycleOptimizer()
        self._dash  = LifecycleDashboard()
        # In-memory registry (supplement to DB for live objects)
        self._active: Dict[str, TradeLifecycle] = {}

    # ── 1. Create Lifecycle ───────────────────────────────────────────────────

    def create_lifecycle(
        self,
        instrument:  str,
        direction:   str,
        confidence:  float = 0.50,
        context:     Optional[MarketContext] = None,
        tags:        Optional[List[str]] = None,
    ) -> ApiResponse:
        try:
            tl = TradeLifecycle.new(
                instrument = instrument,
                direction  = direction,
                confidence = confidence,
                context    = context,
                tags       = tags,
            )
            self._active[tl.trade_id] = tl
            self._db.upsert_lifecycle(tl.summary())
            logger.info(f"LifecycleAPI.create: {tl.trade_id}")
            return ApiResponse(True, "create_lifecycle", tl.to_dict())
        except Exception as exc:
            logger.error(f"LifecycleAPI.create error: {exc}")
            return ApiResponse(False, "create_lifecycle", error=str(exc))

    # ── 2. Get Lifecycle ──────────────────────────────────────────────────────

    def get_lifecycle(self, trade_id: str) -> ApiResponse:
        tl = self._active.get(trade_id)
        if tl:
            return ApiResponse(True, "get_lifecycle", tl.to_dict())
        row = self._db.get_lifecycle(trade_id)
        if row:
            return ApiResponse(True, "get_lifecycle", row)
        return ApiResponse(False, "get_lifecycle", error=f"Trade {trade_id} not found")

    # ── 3. Transition (validated) ─────────────────────────────────────────────

    def transition(
        self,
        trade_id:   str,
        new_state:  LifecycleState,
        reason:     str,
        confidence: Optional[float]          = None,
        context:    Optional[MarketContext]  = None,
        metrics:    Optional[PositionMetrics] = None,
    ) -> ApiResponse:
        tl = self._active.get(trade_id)
        if not tl:
            return ApiResponse(False, "transition", error=f"Trade {trade_id} not in active registry")

        ok = tl.transition_to(new_state, reason, confidence, context, metrics, self._sm)
        if not ok:
            return ApiResponse(
                False, "transition",
                error=(
                    f"Transition {tl.current_state.value} → {new_state.value} "
                    f"was not permitted by state machine"
                ),
            )

        self._db.upsert_lifecycle(tl.summary())
        # Persist latest state record
        if tl.state_history:
            self._db.insert_state_record(trade_id, tl.state_history[-1].to_dict())
        return ApiResponse(True, "transition", tl.summary())

    # ── 4. Auto-Transition ────────────────────────────────────────────────────

    def auto_transition(
        self,
        trade_id:   str,
        context:    Optional[MarketContext] = None,
    ) -> ApiResponse:
        tl = self._active.get(trade_id)
        if not tl:
            return ApiResponse(False, "auto_transition", error=f"Trade {trade_id} not found")

        rec = tl.recommend_transition(context, self._te)
        if rec.recommended is None or rec.should_stay:
            return ApiResponse(True, "auto_transition", {
                "action":  "STAY",
                "reason":  rec.stay_reason or rec.reason,
                "state":   tl.current_state.value,
            })

        ok = tl.transition_to(
            rec.recommended,
            rec.reason,
            rec.confidence,
            context,
            sm=self._sm,
        )
        if not ok:
            return ApiResponse(True, "auto_transition", {
                "action":      "STAY",
                "reason":      f"Recommended {rec.recommended.value} not permitted",
                "state":       tl.current_state.value,
                "recommended": rec.recommended.value,
            })

        self._db.upsert_lifecycle(tl.summary())
        return ApiResponse(True, "auto_transition", {
            "action":    "TRANSITIONED",
            "new_state": tl.current_state.value,
            "reason":    rec.reason,
            "confidence": rec.confidence,
        })

    # ── 5. Recommend Transition (read-only) ───────────────────────────────────

    def recommend_transition(
        self,
        trade_id:   str,
        context:    Optional[MarketContext] = None,
    ) -> ApiResponse:
        tl = self._active.get(trade_id)
        if not tl:
            return ApiResponse(False, "recommend_transition", error=f"Trade {trade_id} not found")

        rec = tl.recommend_transition(context, self._te)
        return ApiResponse(True, "recommend_transition", rec.to_dict())

    # ── 6. Position Management ────────────────────────────────────────────────

    def position_management(
        self,
        trade_id:   str,
        context:    Optional[MarketContext]  = None,
        metrics:    Optional[PositionMetrics] = None,
    ) -> ApiResponse:
        tl = self._active.get(trade_id)
        if not tl:
            return ApiResponse(False, "position_management", error=f"Trade {trade_id} not found")

        plan = tl.position_management(context, metrics, self._pm)
        return ApiResponse(True, "position_management", plan.to_dict())

    # ── 7. Lifecycle Confidence ───────────────────────────────────────────────

    def lifecycle_confidence(
        self,
        trade_id:   str,
        context:    Optional[MarketContext] = None,
    ) -> ApiResponse:
        tl = self._active.get(trade_id)
        if not tl:
            return ApiResponse(False, "lifecycle_confidence", error=f"Trade {trade_id} not found")

        lc = tl.lifecycle_confidence(context, self._lce)
        return ApiResponse(True, "lifecycle_confidence", lc.to_dict())

    # ── 8. Detect Events ──────────────────────────────────────────────────────

    def detect_events(
        self,
        trade_id:    str,
        context:     Optional[MarketContext]  = None,
        metrics:     Optional[PositionMetrics] = None,
        prev_context: Optional[MarketContext] = None,
    ) -> ApiResponse:
        tl = self._active.get(trade_id)
        if not tl:
            return ApiResponse(False, "detect_events", error=f"Trade {trade_id} not found")

        new_events = tl.detect_events(context, metrics, prev_context, ee=self._ee)
        for ev in new_events:
            self._db.insert_event(ev.to_dict())
        return ApiResponse(True, "detect_events", {
            "trade_id":   trade_id,
            "new_events": len(new_events),
            "events":     [e.to_dict() for e in new_events],
        })

    # ── 9. Reasoning ──────────────────────────────────────────────────────────

    def reasoning(
        self,
        trade_id:    str,
        context:     Optional[MarketContext]  = None,
        metrics:     Optional[PositionMetrics] = None,
        prev_conf:   float = 0.65,
    ) -> ApiResponse:
        tl = self._active.get(trade_id)
        if not tl:
            return ApiResponse(False, "reasoning", error=f"Trade {trade_id} not found")

        report = tl.reasoning(context, metrics, prev_conf, self._are, self._lce)
        return ApiResponse(True, "reasoning", report.to_dict())

    # ── 10. State History ─────────────────────────────────────────────────────

    def state_history(self, trade_id: str) -> ApiResponse:
        tl = self._active.get(trade_id)
        if tl:
            return ApiResponse(True, "state_history", {
                "trade_id": trade_id,
                "history":  [r.to_dict() for r in tl.state_history],
                "count":    len(tl.state_history),
            })
        rows = self._db.get_state_history(trade_id)
        if rows:
            return ApiResponse(True, "state_history", {
                "trade_id": trade_id,
                "history":  rows,
                "count":    len(rows),
            })
        return ApiResponse(False, "state_history", error=f"No history for {trade_id}")

    # ── 11. Lifecycle Analytics ───────────────────────────────────────────────

    def lifecycle_analytics(
        self,
        trade_ids: Optional[List[str]] = None,
    ) -> ApiResponse:
        lifecycles = (
            [self._active[tid] for tid in (trade_ids or []) if tid in self._active]
            or list(self._active.values())
        )
        if not lifecycles:
            return ApiResponse(False, "lifecycle_analytics", error="No active lifecycles")

        histories    = [tl.state_history for tl in lifecycles]
        final_states = [tl.current_state for tl in lifecycles]

        report = self._la.analyse(histories, final_states)
        self._db.save_analytics_snapshot(report.to_dict())
        return ApiResponse(True, "lifecycle_analytics", report.to_dict())

    # ── 12. Optimisation ──────────────────────────────────────────────────────

    def optimisation(
        self,
        trade_ids: Optional[List[str]] = None,
    ) -> ApiResponse:
        lifecycles = (
            [self._active[tid] for tid in (trade_ids or []) if tid in self._active]
            or list(self._active.values())
        )
        if not lifecycles:
            return ApiResponse(False, "optimisation", error="No active lifecycles")

        histories    = [tl.state_history for tl in lifecycles]
        final_states = [tl.current_state for tl in lifecycles]

        report = self._lo.optimise(histories, final_states)
        self._db.save_optimisation(report.to_dict())
        return ApiResponse(True, "optimisation", report.to_dict())

    # ── 13. Timeline Dashboard ────────────────────────────────────────────────

    def timeline_dashboard(self, trade_id: str) -> ApiResponse:
        tl = self._active.get(trade_id)
        if not tl:
            return ApiResponse(False, "timeline_dashboard", error=f"Trade {trade_id} not found")

        view = self._dash.lifecycle_timeline(
            tl.trade_id, tl.instrument, tl.direction,
            tl.state_history, tl.current_state,
        )
        return ApiResponse(True, "timeline_dashboard", view)

    # ── 14. Confidence Dashboard ──────────────────────────────────────────────

    def confidence_dashboard(self, trade_id: str) -> ApiResponse:
        tl = self._active.get(trade_id)
        if not tl:
            return ApiResponse(False, "confidence_dashboard", error=f"Trade {trade_id} not found")

        view = self._dash.confidence_timeline(tl.trade_id, tl.state_history)
        return ApiResponse(True, "confidence_dashboard", view)

    # ── 15. Transition Dashboard ──────────────────────────────────────────────

    def transition_dashboard(
        self,
        trade_ids: Optional[List[str]] = None,
    ) -> ApiResponse:
        lifecycles = (
            [self._active[tid] for tid in (trade_ids or []) if tid in self._active]
            or list(self._active.values())
        )
        if not lifecycles:
            return ApiResponse(False, "transition_dashboard", error="No lifecycles")

        histories = [tl.state_history for tl in lifecycles]
        view = self._dash.transition_matrix(histories)
        return ApiResponse(True, "transition_dashboard", view)

    # ── 16. Statistics Dashboard ──────────────────────────────────────────────

    def statistics_dashboard(
        self,
        trade_ids: Optional[List[str]] = None,
    ) -> ApiResponse:
        lifecycles = (
            [self._active[tid] for tid in (trade_ids or []) if tid in self._active]
            or list(self._active.values())
        )
        histories    = [tl.state_history for tl in lifecycles]
        final_states = [tl.current_state for tl in lifecycles]
        view = self._dash.lifecycle_statistics(histories, final_states)
        return ApiResponse(True, "statistics_dashboard", view)

    # ── Database pass-through ─────────────────────────────────────────────────

    def db_stats(self) -> ApiResponse:
        return ApiResponse(True, "db_stats", self._db.stats())

    def active_trades(self) -> ApiResponse:
        return ApiResponse(True, "active_trades", {
            "count":  len(self._active),
            "trades": [tl.summary() for tl in self._active.values()],
        })
