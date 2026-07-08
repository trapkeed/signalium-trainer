"""Phase 29b — Autonomous Learning Loop Scheduler.

Everything in ``actuators.py`` is reactive: it only runs when something
calls ``LearningLoopActuator.run_cycle(...)``. That is fine for tests and
one-off analysis, but it means all 9 trigger events (drawdowns, unexpected
losses, missed opportunities, premature exits, late exits, slippage, scaling
failures, regime failures, capital inefficiency) only get noticed if a human
or another system remembers to call the actuator.

``LearningLoopScheduler`` closes that gap: it drives ``run_cycle`` on a
recurring interval (or on demand via ``run_once``) with no manual step
required to start each cycle. It does not compute any of the 9 reports
itself — a ``collector`` callable supplies whatever reports are available on
a given tick (any of them may be omitted; ``run_cycle`` already tolerates
partial input). The scheduler's only job is guaranteeing the cycle keeps
running, surviving individual failures, and keeping basic observability
(run count, last run time, last error, action counts) so operators can see
the loop is alive without digging through logs.

CRITICAL CONSTRAINT (same as actuators.py): the scheduler only ever calls
``LearningLoopActuator.run_cycle``, which only ever produces proposals
(FailureEvents, Research Experiments, Retraining Requests, Shadow
Candidates, Feature Discovery Requests). It never auto-promotes, auto-trains,
or auto-deploys anything — those remain gated by QDEPromotionEngine, shadow
testing, and human review of RetrainingAdvisor recommendations, exactly as
they are when the actuator is called manually.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

from trainer.learning_loop.actuators import ActuatorCycleResult, LearningLoopActuator

# A collector gathers whatever of the 9 report kinds are available for this
# tick and returns them as run_cycle-compatible keyword arguments, e.g.
# {"self_assessment": ..., "slippage_report": ..., ...}. Reports it cannot
# produce this tick should simply be omitted (or set to None).
CollectorFn = Callable[[], dict[str, Any]]


@dataclass
class LoopRunRecord:
    """Observability record for a single scheduled (or manual) cycle run."""

    run_index: int
    started_at: float
    duration_s: float
    total_actions: int
    by_type: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_index": self.run_index,
            "started_at": self.started_at,
            "duration_s": round(self.duration_s, 4),
            "total_actions": self.total_actions,
            "by_type": self.by_type,
            "success": self.success,
            "error": self.error,
        }


class LearningLoopScheduler:
    """Runs ``LearningLoopActuator.run_cycle`` on a recurring interval.

    Args:
        actuator: The actuator whose ``run_cycle`` should be driven.
        collector: Zero-arg callable returning a dict of run_cycle kwargs
            (any subset of the 9 report kinds). Called fresh on every tick,
            so it should pull the *latest* available reports each time.
        interval_seconds: Seconds to wait between the end of one cycle and
            the start of the next. Defaults to 5 minutes.
        max_history: Number of recent run records to retain for inspection.
    """

    def __init__(
        self,
        actuator: LearningLoopActuator,
        collector: CollectorFn,
        interval_seconds: float = 300.0,
        max_history: int = 200,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._actuator = actuator
        self._collector = collector
        self._interval = interval_seconds
        self._max_history = max_history

        self._history: list[LoopRunRecord] = []
        self._run_count = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # -- Lifecycle -------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the background loop. No-op if already running."""
        if self.is_running:
            logger.warning("LearningLoopScheduler.start: already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_forever, name="learning-loop-scheduler", daemon=True
        )
        self._thread.start()
        logger.info(f"LearningLoopScheduler.start: loop started (interval={self._interval}s)")

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the loop to stop and wait for the thread to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.info("LearningLoopScheduler.stop: loop stopped")

    def _run_forever(self) -> None:
        while not self._stop_event.is_set():
            self.run_once()
            # wait() returns early (True) if stop() was called mid-sleep.
            self._stop_event.wait(self._interval)

    # -- Cycle execution ---------------------------------------------------

    def run_once(self) -> ActuatorCycleResult:
        """Run exactly one detection-and-actuation cycle right now.

        Never raises: collector or actuator failures are caught, logged, and
        recorded in history so a single bad tick cannot kill the loop or go
        unnoticed. Useful both for the background thread and for tests /
        manual on-demand triggering.
        """
        started_at = time.time()
        result = ActuatorCycleResult()
        error: str | None = None

        try:
            kwargs = self._collector() or {}
            result = self._actuator.run_cycle(**kwargs)
        except Exception as exc:  # pragma: no cover - defensive, logged below
            error = str(exc)
            logger.error(f"LearningLoopScheduler.run_once: cycle failed — {error}")

        with self._lock:
            self._run_count += 1
            record = LoopRunRecord(
                run_index=self._run_count,
                started_at=started_at,
                duration_s=time.time() - started_at,
                total_actions=len(result.actions),
                by_type=dict(result.by_type) if result.actions else {},
                error=error,
            )
            self._history.append(record)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]

        if error is None:
            logger.info(
                f"LearningLoopScheduler.run_once: run #{record.run_index} — "
                f"{record.total_actions} actions"
            )
        return result

    # -- Observability -------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            last = self._history[-1] if self._history else None
            total_actions = sum(r.total_actions for r in self._history)
            errors = sum(1 for r in self._history if not r.success)
        return {
            "is_running": self.is_running,
            "interval_seconds": self._interval,
            "run_count": self._run_count,
            "total_actions": total_actions,
            "error_count": errors,
            "last_run": last.to_dict() if last else None,
        }

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            return [r.to_dict() for r in self._history[-limit:]]
