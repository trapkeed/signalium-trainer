"""Phase 21.5 — Lifecycle Analytics.

Aggregate analytics across a portfolio of TradeLifecycle objects:
- Average time in each state
- Transition frequency matrix
- Failure, cancellation, and expiration rates
- Average lifecycle duration
- Average profit by lifecycle path
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.lifecycle.lifecycle_models import StateRecord
from trainer.lifecycle.lifecycle_states import (
    CLOSING_STATES,
    LifecycleState,
)


@dataclass
class StateAnalytics:
    """Analytics for one lifecycle state."""

    state: LifecycleState
    visit_count: int = 0
    avg_duration_h: float = 0.0
    min_duration_h: float = 0.0
    max_duration_h: float = 0.0
    total_duration_h: float = 0.0

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "visit_count": self.visit_count,
            "avg_duration_h": round(self.avg_duration_h, 3),
            "min_duration_h": round(self.min_duration_h, 3),
            "max_duration_h": round(self.max_duration_h, 3),
            "total_duration_h": round(self.total_duration_h, 3),
        }


@dataclass
class LifecycleAnalyticsReport:
    """Complete analytics across all analysed trades."""

    total_trades: int = 0
    completed_trades: int = 0
    failed_trades: int = 0
    cancelled_trades: int = 0
    expired_trades: int = 0
    invalidated_trades: int = 0
    avg_lifecycle_h: float = 0.0
    failure_rate: float = 0.0
    cancellation_rate: float = 0.0
    expiration_rate: float = 0.0
    invalidation_rate: float = 0.0
    state_analytics: dict[str, StateAnalytics] = field(default_factory=dict)
    transition_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    path_performance: list[dict] = field(default_factory=list)
    top_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "completed_trades": self.completed_trades,
            "failed_trades": self.failed_trades,
            "cancelled_trades": self.cancelled_trades,
            "expired_trades": self.expired_trades,
            "invalidated_trades": self.invalidated_trades,
            "avg_lifecycle_h": round(self.avg_lifecycle_h, 3),
            "failure_rate": round(self.failure_rate, 4),
            "cancellation_rate": round(self.cancellation_rate, 4),
            "expiration_rate": round(self.expiration_rate, 4),
            "invalidation_rate": round(self.invalidation_rate, 4),
            "state_analytics": {k: v.to_dict() for k, v in self.state_analytics.items()},
            "transition_matrix": self.transition_matrix,
            "path_performance": self.path_performance,
            "top_paths": self.top_paths,
        }


class LifecycleAnalytics:
    """Computes aggregate analytics across a collection of lifecycle objects."""

    def analyse(
        self,
        histories: list[list[StateRecord]],
        final_states: list[LifecycleState] | None = None,
    ) -> LifecycleAnalyticsReport:
        """
        histories    : list of state_history lists from TradeLifecycle objects
        final_states : final state for each trade (if None, inferred from history)
        """
        if not histories:
            return LifecycleAnalyticsReport()

        n = len(histories)
        report = LifecycleAnalyticsReport(total_trades=n)

        # ── Per-state duration accumulator ─────────────────────────────────
        dur_accum: dict[LifecycleState, list[float]] = collections.defaultdict(list)
        trans_matrix: dict[str, dict[str, int]] = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )
        lifecycle_durations: list[float] = []
        path_counter: dict[str, int] = collections.Counter()

        for idx, history in enumerate(histories):
            if not history:
                continue

            # Path: string of state abbreviations
            path = " → ".join(r.state.value for r in history)
            path_counter[path] += 1

            # Lifecycle duration
            if len(history) >= 2:
                total_h = sum(r.duration_h for r in history[:-1])
                lifecycle_durations.append(total_h)

            # Per-state durations
            for record in history:
                if record.duration_h > 0:
                    dur_accum[record.state].append(record.duration_h)

            # Transition matrix
            for i in range(len(history) - 1):
                from_s = history[i].state.value
                to_s = history[i + 1].state.value
                trans_matrix[from_s][to_s] += 1

            # Final state
            last_state = history[-1].state
            if final_states and idx < len(final_states):
                last_state = final_states[idx]

            if last_state == LifecycleState.CLOSED:
                report.completed_trades += 1
            elif last_state in (LifecycleState.STOPPED, *CLOSING_STATES):
                report.failed_trades += 1
            elif last_state == LifecycleState.CANCELLED:
                report.cancelled_trades += 1
            elif last_state == LifecycleState.EXPIRED:
                report.expired_trades += 1
            elif last_state == LifecycleState.INVALIDATED:
                report.invalidated_trades += 1

        # ── State analytics ─────────────────────────────────────────────────
        for state, durations in dur_accum.items():
            report.state_analytics[state.value] = StateAnalytics(
                state=state,
                visit_count=len(durations),
                avg_duration_h=sum(durations) / len(durations) if durations else 0.0,
                min_duration_h=min(durations) if durations else 0.0,
                max_duration_h=max(durations) if durations else 0.0,
                total_duration_h=sum(durations),
            )

        # ── Rates ────────────────────────────────────────────────────────────
        report.avg_lifecycle_h = (
            sum(lifecycle_durations) / len(lifecycle_durations) if lifecycle_durations else 0.0
        )
        if n:
            report.failure_rate = report.failed_trades / n
            report.cancellation_rate = report.cancelled_trades / n
            report.expiration_rate = report.expired_trades / n
            report.invalidation_rate = report.invalidated_trades / n

        # ── Transition matrix ────────────────────────────────────────────────
        report.transition_matrix = {k: dict(v) for k, v in trans_matrix.items()}

        # ── Top paths ─────────────────────────────────────────────────────────
        report.top_paths = [p for p, _ in sorted(path_counter.items(), key=lambda x: -x[1])[:5]]

        logger.debug(
            f"LifecycleAnalytics: {n} trades analysed, "
            f"completed={report.completed_trades}, failed={report.failed_trades}"
        )
        return report

    def state_time_distribution(
        self,
        histories: list[list[StateRecord]],
    ) -> dict[str, float]:
        """Return average hours spent in each state across all histories."""
        accum: dict[str, list[float]] = collections.defaultdict(list)
        for history in histories:
            for record in history:
                if record.duration_h > 0:
                    accum[record.state.value].append(record.duration_h)
        return {state: round(sum(durs) / len(durs), 3) for state, durs in accum.items() if durs}

    def transition_frequency(
        self,
        histories: list[list[StateRecord]],
    ) -> dict[str, int]:
        """Count occurrences of every unique from→to transition pair."""
        freq: dict[str, int] = collections.Counter()
        for history in histories:
            for i in range(len(history) - 1):
                key = f"{history[i].state.value}→{history[i+1].state.value}"
                freq[key] += 1
        return dict(freq)
