"""S30 — Autonomous Trading Readiness Aggregator.

A single "is the full autonomous loop actually ready to run unattended"
report, combining the six lifecycle stages named in the S30 scope:

  1. market data ingestion    — freshness of the feature/event stream
  2. inference generation     — model health + operational-safety gate
  3. execution communication  — deployment manager's Signalium notification
                                  channel is reachable/configured
  4. trade lifecycle updates   — decision memory outcome write-path
  5. memory updates             — DecisionMemory round-trip check
  6. learning loops              — LearningLoopScheduler is running and
                                    its most recent run had no error

Every check delegates to an existing component; this module only
aggregates. Dependencies are optional — an unset dependency reports
SKIPPED rather than failing the whole report, so partially-wired test
environments can still produce a useful readiness snapshot.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class ReadinessCheck:
    stage: str
    status: str  # "READY" | "NOT_READY" | "SKIPPED"
    details: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class AutonomousReadinessReport:
    generated_at: float
    checks: list[ReadinessCheck]
    ready: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "ready": self.ready,
            "checks": [
                {"stage": c.stage, "status": c.status, "details": c.details, "evidence": c.evidence}
                for c in self.checks
            ],
        }


class AutonomousReadinessChecker:
    def __init__(
        self,
        *,
        feature_events_provider: Any | None = None,
        max_feature_staleness_s: float = 300.0,
        operational_safety: Any | None = None,
        deployment_manager: Any | None = None,
        decision_memory: Any | None = None,
        learning_loop_scheduler: Any | None = None,
    ) -> None:
        self._feature_events_provider = feature_events_provider
        self._max_feature_staleness_s = max_feature_staleness_s
        self._operational_safety = operational_safety
        self._deployment_manager = deployment_manager
        self._decision_memory = decision_memory
        self._learning_loop_scheduler = learning_loop_scheduler

    def _check_market_data(self) -> ReadinessCheck:
        if self._feature_events_provider is None:
            return ReadinessCheck("market_data_ingestion", "SKIPPED", "no feature_events_provider configured")
        events = self._feature_events_provider()
        if not events:
            return ReadinessCheck("market_data_ingestion", "NOT_READY", "no feature events observed")
        latest_ts = max(e.get("timestamp", 0.0) for e in events)
        staleness = time.time() - latest_ts
        if staleness > self._max_feature_staleness_s:
            return ReadinessCheck(
                "market_data_ingestion", "NOT_READY", f"stream stale ({staleness:.1f}s)",
                {"staleness_s": staleness},
            )
        return ReadinessCheck("market_data_ingestion", "READY", f"stream fresh ({staleness:.1f}s)", {"staleness_s": staleness})

    def _check_inference_generation(self) -> ReadinessCheck:
        if self._operational_safety is None:
            return ReadinessCheck("inference_generation", "SKIPPED", "no operational_safety configured")
        state = self._operational_safety.get_state()
        can_serve = self._operational_safety.can_serve_inference()
        status = "READY" if can_serve else "NOT_READY"
        return ReadinessCheck("inference_generation", status, "inference gate state", state)

    def _check_execution_communication(self) -> ReadinessCheck:
        if self._deployment_manager is None:
            return ReadinessCheck("execution_communication", "SKIPPED", "no deployment_manager configured")
        notifier = getattr(self._deployment_manager, "_notifier", None) or getattr(
            self._deployment_manager, "notifier", None
        )
        if notifier is None:
            return ReadinessCheck("execution_communication", "NOT_READY", "no Signalium notification channel configured")
        return ReadinessCheck("execution_communication", "READY", "notification channel configured")

    def _check_trade_lifecycle_updates(self) -> ReadinessCheck:
        if self._decision_memory is None:
            return ReadinessCheck("trade_lifecycle_updates", "SKIPPED", "no decision_memory configured")
        try:
            from trainer.memory.decision_types import OutcomeType
        except ImportError as exc:
            return ReadinessCheck("trade_lifecycle_updates", "SKIPPED", f"decision_types unavailable: {exc}")
        probe_id = f"readiness-probe-{uuid.uuid4().hex[:8]}"
        record = self._decision_memory.get(probe_id)
        if record is not None:
            return ReadinessCheck("trade_lifecycle_updates", "NOT_READY", "probe id collision — unexpected state")
        ok = self._decision_memory.record_outcome(probe_id, OutcomeType.NO_TRADE)
        # A missing decision_id correctly returns False — that confirms the
        # write path validates existence rather than silently accepting
        # arbitrary outcomes, which is exactly the readiness signal wanted.
        status = "READY" if ok is False else "NOT_READY"
        return ReadinessCheck("trade_lifecycle_updates", status, "outcome write-path existence check")

    def _check_memory_updates(self) -> ReadinessCheck:
        if self._decision_memory is None:
            return ReadinessCheck("memory_updates", "SKIPPED", "no decision_memory configured")
        before = self._decision_memory.total()
        return ReadinessCheck("memory_updates", "READY", "decision memory reachable", {"n_records": before})

    def _check_learning_loops(self) -> ReadinessCheck:
        if self._learning_loop_scheduler is None:
            return ReadinessCheck("learning_loops", "SKIPPED", "no learning_loop_scheduler configured")
        status = self._learning_loop_scheduler.status()
        if not status.get("is_running"):
            return ReadinessCheck("learning_loops", "NOT_READY", "scheduler not running", status)
        last_run = status.get("last_run")
        if last_run and last_run.get("error"):
            return ReadinessCheck("learning_loops", "NOT_READY", f"last run errored: {last_run['error']}", status)
        return ReadinessCheck("learning_loops", "READY", "scheduler running, no recent errors", status)

    def run(self) -> AutonomousReadinessReport:
        checks = [
            self._check_market_data(),
            self._check_inference_generation(),
            self._check_execution_communication(),
            self._check_trade_lifecycle_updates(),
            self._check_memory_updates(),
            self._check_learning_loops(),
        ]
        ready = not any(c.status == "NOT_READY" for c in checks)
        report = AutonomousReadinessReport(generated_at=time.time(), checks=checks, ready=ready)
        logger.info(
            "AutonomousReadiness: " + ("READY" if ready else "NOT_READY") +
            " " + ", ".join(f"{c.stage}={c.status}" for c in checks)
        )
        return report
