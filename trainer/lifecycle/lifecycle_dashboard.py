"""Phase 21.5 — Lifecycle Dashboard.

Six dashboard views for the Trade Lifecycle Intelligence Engine:
1. Lifecycle Timeline — chronological state timeline per trade
2. State Transition Graph — adjacency matrix of state transitions
3. Trade Journey — human-readable narrative journey
4. Confidence Timeline — confidence evolution across states
5. Transition Matrix — frequency matrix of transitions
6. Lifecycle Statistics — aggregate statistics panel
"""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.lifecycle.lifecycle_states import LifecycleState
from trainer.lifecycle.lifecycle_models import StateRecord, LifecycleEvent


class LifecycleDashboard:
    """Generates all six lifecycle dashboard views."""

    # ── 1. Lifecycle Timeline ─────────────────────────────────────────────────

    def lifecycle_timeline(
        self,
        trade_id: str,
        instrument: str,
        direction: str,
        state_history: List[StateRecord],
        current_state: LifecycleState,
    ) -> Dict:
        """Chronological state timeline for a single trade."""

        timeline = []
        for i, record in enumerate(state_history):
            step = {
                "step": i + 1,
                "state": record.state.value,
                "timestamp": round(record.timestamp, 3),
                "confidence": round(record.confidence, 4),
                "reason": record.reason,
                "duration_h": round(record.duration_h, 3),
                "is_current": record.state == current_state and i == len(state_history) - 1,
            }
            if record.market_context:
                step["market_snapshot"] = record.market_context.to_dict()
            timeline.append(step)

        total_h = sum(r.duration_h for r in state_history)

        return {
            "view": "lifecycle_timeline",
            "trade_id": trade_id,
            "instrument": instrument,
            "direction": direction,
            "current_state": current_state.value,
            "total_steps": len(timeline),
            "total_hours": round(total_h, 3),
            "timeline": timeline,
        }

    # ── 2. State Transition Graph ──────────────────────────────────────────────

    def state_transition_graph(
        self,
        state_history: List[StateRecord],
        all_histories: Optional[List[List[StateRecord]]] = None,
    ) -> Dict:
        """Adjacency representation of all observed transitions."""

        # Count transitions in this trade
        edges: Dict[str, Dict[str, int]] = {}
        history = state_history

        def add_edge(frm: str, to: str) -> None:
            edges.setdefault(frm, {})
            edges[frm][to] = edges[frm].get(to, 0) + 1

        for i in range(len(history) - 1):
            add_edge(history[i].state.value, history[i + 1].state.value)

        # Add from portfolio if provided
        if all_histories:
            for h in all_histories:
                for i in range(len(h) - 1):
                    add_edge(h[i].state.value, h[i + 1].state.value)

        # Convert to node/edge list
        nodes = sorted({s for frm, tos in edges.items() for s in [frm, *tos.keys()]})
        edge_list = [
            {"from": frm, "to": to, "count": cnt}
            for frm, tos in edges.items()
            for to, cnt in tos.items()
        ]

        return {
            "view": "state_transition_graph",
            "nodes": nodes,
            "edges": edge_list,
            "total_nodes": len(nodes),
            "total_edges": len(edge_list),
        }

    # ── 3. Trade Journey ──────────────────────────────────────────────────────

    def trade_journey(
        self,
        trade_id: str,
        instrument: str,
        direction: str,
        state_history: List[StateRecord],
        events: List[LifecycleEvent],
    ) -> Dict:
        """Human-readable narrative journey of the trade."""

        steps = []
        for i, record in enumerate(state_history):
            narrative = self._state_narrative(record.state, record.confidence)
            step_events = [
                e.description
                for e in events
                if abs(e.timestamp - record.timestamp) < 3600 and e.state == record.state
            ]
            steps.append(
                {
                    "step": i + 1,
                    "state": record.state.value,
                    "narrative": narrative,
                    "reason": record.reason,
                    "confidence": f"{record.confidence:.0%}",
                    "duration": f"{record.duration_h:.1f}h" if record.duration_h else "current",
                    "events": step_events[:3],
                }
            )

        critical_events = [
            {"type": e.event_type, "description": e.description}
            for e in events
            if e.severity == "CRITICAL"
        ]

        return {
            "view": "trade_journey",
            "trade_id": trade_id,
            "instrument": instrument,
            "direction": direction,
            "journey_steps": steps,
            "critical_events": critical_events,
            "total_states": len(steps),
            "total_events": len(events),
        }

    def _state_narrative(self, state: LifecycleState, confidence: float) -> str:
        narratives = {
            LifecycleState.IDEA: "Opportunity identified — idea under review",
            LifecycleState.WATCHLIST: "Setup placed on watch — conditions developing",
            LifecycleState.PREPARATION: "Pre-trade preparation complete — entry zone approaching",
            LifecycleState.ENTRY_READY: "All green — ready to execute when timing is right",
            LifecycleState.PENDING_ENTRY: "Order submitted — awaiting fill",
            LifecycleState.ACTIVE: "Position live — actively managed",
            LifecycleState.POSITION_BUILDING: "Scaling into trade — adding conviction",
            LifecycleState.PROTECTED: "Breakeven secured — profit potential open",
            LifecycleState.PARTIAL_EXIT: "Partial profit banked — managing remainder",
            LifecycleState.RUNNER: "Running for full target — trail stop active",
            LifecycleState.TRAILING: "Trail stop protecting gains",
            LifecycleState.EXIT_READY: "Exit conditions met — closing imminent",
            LifecycleState.CLOSING: "Finalising trade closure",
            LifecycleState.CLOSED: "Trade completed",
            LifecycleState.INVALIDATED: "Setup invalidated — idea withdrawn cleanly",
            LifecycleState.CANCELLED: "Trade cancelled before entry",
            LifecycleState.EXPIRED: "Entry window expired",
            LifecycleState.STOPPED: "Stop loss executed — risk managed",
            LifecycleState.FORCED_EXIT: "Forced closure — external event",
            LifecycleState.EMERGENCY_EXIT: "Emergency exit — critical risk event",
        }
        return narratives.get(state, f"Trade in {state.value} state")

    # ── 4. Confidence Timeline ────────────────────────────────────────────────

    def confidence_timeline(
        self,
        trade_id: str,
        state_history: List[StateRecord],
    ) -> Dict:
        """Confidence evolution across all lifecycle states."""

        points = [
            {
                "step": i + 1,
                "state": r.state.value,
                "confidence": round(r.confidence, 4),
                "timestamp": round(r.timestamp, 3),
            }
            for i, r in enumerate(state_history)
        ]

        confidences = [r.confidence for r in state_history]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        min_conf = min(confidences) if confidences else 0.0
        max_conf = max(confidences) if confidences else 0.0

        # Detect trend
        if len(confidences) >= 2:
            trend = (
                "IMPROVING"
                if confidences[-1] > confidences[0]
                else ("DECLINING" if confidences[-1] < confidences[0] else "STABLE")
            )
        else:
            trend = "STABLE"

        return {
            "view": "confidence_timeline",
            "trade_id": trade_id,
            "points": points,
            "avg_conf": round(avg_conf, 4),
            "min_conf": round(min_conf, 4),
            "max_conf": round(max_conf, 4),
            "trend": trend,
            "data_points": len(points),
        }

    # ── 5. Transition Matrix ──────────────────────────────────────────────────

    def transition_matrix(
        self,
        all_histories: List[List[StateRecord]],
    ) -> Dict:
        """Frequency matrix of all state transitions across a portfolio."""

        # Build raw counts
        matrix: Dict[str, Dict[str, int]] = {}
        all_states = set()

        for history in all_histories:
            for i in range(len(history) - 1):
                frm = history[i].state.value
                to = history[i + 1].state.value
                all_states.update([frm, to])
                matrix.setdefault(frm, {})
                matrix[frm][to] = matrix[frm].get(to, 0) + 1

        states = sorted(all_states)

        # Normalise rows to probabilities
        prob_matrix: Dict[str, Dict[str, float]] = {}
        for frm, tos in matrix.items():
            total = sum(tos.values())
            prob_matrix[frm] = {to: round(cnt / total, 4) for to, cnt in tos.items()}

        return {
            "view": "transition_matrix",
            "states": states,
            "count_matrix": matrix,
            "prob_matrix": prob_matrix,
            "total_trades": len(all_histories),
            "total_transitions": sum(sum(v for v in tos.values()) for tos in matrix.values()),
        }

    # ── 6. Lifecycle Statistics ───────────────────────────────────────────────

    def lifecycle_statistics(
        self,
        all_histories: List[List[StateRecord]],
        final_states: Optional[List[LifecycleState]] = None,
    ) -> Dict:
        """Aggregate statistics panel."""

        n = len(all_histories)
        if n == 0:
            return {"view": "lifecycle_statistics", "total_trades": 0}

        # State visit frequencies
        state_visits: Dict[str, int] = {}
        for history in all_histories:
            for record in history:
                k = record.state.value
                state_visits[k] = state_visits.get(k, 0) + 1

        # Lifecycle lengths
        lengths = [len(h) for h in all_histories if h]
        avg_length = sum(lengths) / len(lengths) if lengths else 0.0

        # Final state distribution
        final_dist: Dict[str, int] = {}
        if final_states:
            for fs in final_states:
                k = fs.value
                final_dist[k] = final_dist.get(k, 0) + 1
        else:
            for history in all_histories:
                if history:
                    k = history[-1].state.value
                    final_dist[k] = final_dist.get(k, 0) + 1

        # Most common path length
        from collections import Counter

        common_length = Counter(lengths).most_common(1)
        modal_length = common_length[0][0] if common_length else 0

        return {
            "view": "lifecycle_statistics",
            "total_trades": n,
            "avg_path_length": round(avg_length, 2),
            "modal_path_length": modal_length,
            "state_visit_counts": state_visits,
            "final_state_dist": final_dist,
            "most_visited_state": max(state_visits, key=state_visits.get) if state_visits else None,
        }
