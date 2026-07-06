"""Phase 24.9 — Market Intelligence Worker.

Processes WorkerJob items from a queue and runs the full MIE pipeline
for each trigger type:
    - NEW_CANDLE        : triggered on every new OHLCV candle
    - ECONOMIC_EVENT    : triggered when a news/calendar event fires
    - NEW_PREDICTION    : triggered after model generates a new prediction
    - MODEL_UPDATE      : triggered after the model is retrained / updated
    - REGIME_TRANSITION : triggered when the regime engine detects a change
    - MANUAL_REFRESH    : triggered by user / system request
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.mie.mie_api import MIEAPI
from trainer.mie.mie_models import MarketInput, WorkerJob
from trainer.mie.mie_types import WorkerTrigger


class MIEWorker:
    """
    In-process job queue processor for the Market Intelligence Engine.

    The worker maintains an internal FIFO queue of WorkerJob objects.
    Call `submit()` to enqueue a job and `process_all()` to drain the queue.
    In production, a background thread / asyncio task would call `process_all()`
    on a timer or event-driven trigger.
    """

    def __init__(
        self,
        api: MIEAPI | None = None,
        max_queue: int = 100,
        on_complete: Callable[[WorkerJob], None] | None = None,
    ) -> None:
        self._api = api or MIEAPI()
        self._queue: deque[WorkerJob] = deque(maxlen=max_queue)
        self._history: list[WorkerJob] = []
        self._on_complete = on_complete
        self._processed = 0
        self._failed = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def submit(self, job: WorkerJob) -> str:
        """Enqueue a job; returns job_id."""
        self._queue.append(job)
        logger.debug(f"[Worker] Queued job {job.job_id} trigger={job.trigger.value}")
        return job.job_id

    def submit_candle(self, inp: MarketInput) -> str:
        """Convenience: enqueue a NEW_CANDLE job."""
        return self.submit(
            WorkerJob(
                trigger=WorkerTrigger.NEW_CANDLE,
                symbol=inp.symbol,
                payload={"input": inp},
            )
        )

    def submit_prediction(self, inp: MarketInput) -> str:
        """Convenience: enqueue a NEW_PREDICTION job."""
        return self.submit(
            WorkerJob(
                trigger=WorkerTrigger.NEW_PREDICTION,
                symbol=inp.symbol,
                payload={"input": inp},
            )
        )

    def submit_manual(self, inp: MarketInput) -> str:
        """Convenience: enqueue a MANUAL_REFRESH job."""
        return self.submit(
            WorkerJob(
                trigger=WorkerTrigger.MANUAL_REFRESH,
                symbol=inp.symbol,
                payload={"input": inp},
            )
        )

    def process_next(self) -> WorkerJob | None:
        """Process one job from the queue. Returns the processed job or None."""
        if not self._queue:
            return None
        job = self._queue.popleft()
        job.status = "RUNNING"
        try:
            inp: MarketInput | None = job.payload.get("input")
            if inp is None:
                raise ValueError("WorkerJob payload missing 'input' MarketInput")
            report = self._api.recalculate(inp)
            job.result = report
            job.status = "DONE"
            self._processed += 1
            t_score = f"{report.tradeability.score:.1f}" if report.tradeability else "N/A"
            logger.info(f"[Worker] ✓ {job.job_id} {job.trigger.value} → tradeability={t_score}")
        except Exception as exc:
            job.status = "FAILED"
            job.payload["error"] = str(exc)
            self._failed += 1
            logger.error(f"[Worker] ✗ {job.job_id}: {exc}")
        finally:
            self._history.append(job)
            if self._on_complete:
                self._on_complete(job)
        return job

    def process_all(self) -> list[WorkerJob]:
        """Drain the queue and process every pending job."""
        results: list[WorkerJob] = []
        while self._queue:
            job = self.process_next()
            if job:
                results.append(job)
        return results

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    @property
    def processed(self) -> int:
        return self._processed

    @property
    def failed(self) -> int:
        return self._failed

    def get_history(self, limit: int = 20) -> list[WorkerJob]:
        return self._history[-limit:]

    def stats(self) -> dict[str, Any]:
        return {
            "queue_size": self.queue_size,
            "processed": self.processed,
            "failed": self.failed,
            "history": len(self._history),
        }
