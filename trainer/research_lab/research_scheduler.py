"""S24 §2 — Automated Research Scheduling.

Prior to this module, every Research Lab review (performance, regime,
failure, drift, feature discovery, ...) only ran on-demand via an explicit
API call. Nothing invoked them periodically, despite several engines'
docstrings assuming a scheduler existed (see
``FailureInvestigationEngine``).

``ResearchScheduler`` closes that gap by defining named research *cycles*
at three cadences — daily / weekly / monthly — and providing:

  - ``run_cycle(name)``      — run one named cycle immediately (manual trigger)
  - ``run_due_cycles(now)``  — run every cycle whose cadence window has
                                elapsed since its last recorded run
  - ``start_background()``  / ``stop_background()`` — a lightweight daemon
                                thread that calls ``run_due_cycles`` on a
                                fixed poll interval, so cycles fire without
                                any external cron/orchestrator.

This module only *invokes* existing engines (health monitor, failure
investigation, discovery pipelines, regime/performance/feature monitors,
composite health scorer, retraining recommendation engine, champion/
challenger tracker) — it introduces no new analysis logic and never
triggers training execution. Retraining recommendations it surfaces still
require the existing admin-approval flow.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.research_lab.discovery_models import DiscoveryQuery, DiscoverySummary
from trainer.research_lab.investigation_models import InvestigationQuery
from trainer.research_lab.rl_database import ResearchLabDatabase

CADENCE_SECONDS: dict[str, float] = {
    "daily": 24 * 3600.0,
    "weekly": 7 * 24 * 3600.0,
    "monthly": 30 * 24 * 3600.0,
}


@dataclass
class ResearchCycle:
    name: str
    cadence: str  # "daily" | "weekly" | "monthly"
    description: str
    run: Callable[[], dict[str, Any]]


class ResearchScheduler:
    """Registers and runs the named daily/weekly/monthly research cycles.

    All dependencies are injected so the scheduler never constructs its own
    engines — it always reuses the single shared instances already wired up
    by the server (or by a test), avoiding duplicate state / duplicate DB
    connections.
    """

    def __init__(
        self,
        db: ResearchLabDatabase,
        *,
        health_monitor: Any = None,
        failure_engine: Any = None,
        experiment_creator: Any = None,
        feature_pipeline: Any = None,
        label_engine: Any = None,
        rec_engine: Any = None,
        performance_monitor: Any = None,
        regime_coverage_monitor: Any = None,
        feature_health_monitor: Any = None,
        decision_quality_tracker: Any = None,
        composite_health_scorer: Any = None,
        cc_tracker: Any = None,
        poll_interval_s: float = 3600.0,
    ) -> None:
        self._db = db
        self._health_monitor = health_monitor
        self._failure_engine = failure_engine
        self._experiment_creator = experiment_creator
        self._feature_pipeline = feature_pipeline
        self._label_engine = label_engine
        self._rec_engine = rec_engine
        self._performance_monitor = performance_monitor
        self._regime_coverage_monitor = regime_coverage_monitor
        self._feature_health_monitor = feature_health_monitor
        self._decision_quality_tracker = decision_quality_tracker
        self._composite_health_scorer = composite_health_scorer
        self._cc_tracker = cc_tracker
        self._poll_interval_s = poll_interval_s

        self._cycles: dict[str, ResearchCycle] = {}
        self._register_default_cycles()

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Guards run_cycle()/is_due() so concurrent callers (the background
        # thread plus a manual API trigger, or two run_due_cycles() calls
        # racing) cannot both observe a cycle as "due" and run/persist it
        # twice. A single process-wide lock is sufficient here since the
        # scheduler is a per-process singleton (see research_lab_server.py);
        # it does not protect against multiple server processes sharing one
        # DB file, which is out of scope for this in-process scheduler.
        self._run_lock = threading.Lock()

    # ── Cycle registry ───────────────────────────────────────────────────────

    def _register_default_cycles(self) -> None:
        self._cycles = {
            # Daily: cheap, high-signal telemetry-driven checks.
            "performance_review": ResearchCycle(
                "performance_review",
                "daily",
                "Recompute the rolling performance report (Sharpe, drawdown, "
                "profit factor) from live telemetry.",
                self._run_performance_review,
            ),
            "drift_analysis": ResearchCycle(
                "drift_analysis",
                "daily",
                "Run the composite health-monitor drift check across all "
                "13 dimensions and raise alerts / recommendations.",
                self._run_drift_analysis,
            ),
            "failure_review": ResearchCycle(
                "failure_review",
                "daily",
                "Run the failure investigation engine over the last day of "
                "trades to catch new root-caused failure patterns.",
                self._run_failure_review,
            ),
            "regime_review": ResearchCycle(
                "regime_review",
                "daily",
                "Recompute regime coverage (unseen conditions, distribution "
                "shift, black swans).",
                self._run_regime_review,
            ),
            # Weekly: broader, more expensive analyses.
            "self_evaluation": ResearchCycle(
                "self_evaluation",
                "weekly",
                "Compute the composite QDE health score (grade + all "
                "sub-scores) for the active champion.",
                self._run_self_evaluation,
            ),
            "feature_discovery": ResearchCycle(
                "feature_discovery",
                "weekly",
                "Run the feature/label discovery pipelines and propose "
                "experiments from the week's failure reports.",
                self._run_feature_discovery,
            ),
            "counterfactual_review": ResearchCycle(
                "counterfactual_review",
                "weekly",
                "Recompute decision-quality / counterfactual metrics "
                "(captured MFE fraction, hindsight regret).",
                self._run_counterfactual_review,
            ),
            "shadow_promotion_review": ResearchCycle(
                "shadow_promotion_review",
                "weekly",
                "Refresh champion/challenger head-to-head status for any " "active shadow tests.",
                self._run_shadow_promotion_review,
            ),
            # Monthly: retrospective / governance-level reviews.
            "architecture_review": ResearchCycle(
                "architecture_review",
                "monthly",
                "Recompute feature-health (staleness, distribution) as a "
                "proxy for pipeline/architecture integrity.",
                self._run_architecture_review,
            ),
            "deployment_review": ResearchCycle(
                "deployment_review",
                "monthly",
                "Re-evaluate outstanding retraining recommendations across "
                "the full monthly window.",
                self._run_deployment_review,
            ),
        }

    # ── Individual cycle bodies (thin wrappers around existing engines) ─────

    def _run_performance_review(self) -> dict[str, Any]:
        if self._performance_monitor is None:
            return {"skipped": "performance_monitor not configured"}
        report = self._performance_monitor.compute_report(
            since=time.time() - CADENCE_SECONDS["daily"]
        )
        return {"report": report.to_dict() if hasattr(report, "to_dict") else report}

    def _run_drift_analysis(self) -> dict[str, Any]:
        if self._health_monitor is None:
            return {"skipped": "health_monitor not configured"}
        result = self._health_monitor.run_check()
        return result.to_dict() if hasattr(result, "to_dict") else result

    def _run_failure_review(self) -> dict[str, Any]:
        if self._failure_engine is None:
            return {"skipped": "failure_engine not configured"}
        reports = self._failure_engine.investigate(
            InvestigationQuery(since=time.time() - CADENCE_SECONDS["daily"])
        )
        return {"failure_report_count": len(reports)}

    def _run_regime_review(self) -> dict[str, Any]:
        if self._regime_coverage_monitor is None:
            return {"skipped": "regime_coverage_monitor not configured"}
        report = self._regime_coverage_monitor.compute_report(
            since=time.time() - CADENCE_SECONDS["daily"]
        )
        return {"report": report.to_dict() if hasattr(report, "to_dict") else report}

    def _run_self_evaluation(self) -> dict[str, Any]:
        if self._composite_health_scorer is None:
            return {"skipped": "composite_health_scorer not configured"}
        score = self._composite_health_scorer.compute()
        return {"score": score.to_dict() if hasattr(score, "to_dict") else score}

    def _run_feature_discovery(self) -> dict[str, Any]:
        if not (self._experiment_creator and self._feature_pipeline and self._label_engine):
            return {"skipped": "discovery pipelines not configured"}
        dq = DiscoveryQuery(since=time.time() - CADENCE_SECONDS["weekly"])
        summary = DiscoverySummary()
        self._experiment_creator.propose(dq, summary)
        self._feature_pipeline.discover(dq, summary)
        self._label_engine.discover(dq, summary)
        return {"summary": summary.to_dict()}

    def _run_counterfactual_review(self) -> dict[str, Any]:
        if self._decision_quality_tracker is None:
            return {"skipped": "decision_quality_tracker not configured"}
        report = self._decision_quality_tracker.compute_report(
            since=time.time() - CADENCE_SECONDS["weekly"]
        )
        return {"report": report.to_dict() if hasattr(report, "to_dict") else report}

    def _run_shadow_promotion_review(self) -> dict[str, Any]:
        if self._cc_tracker is None:
            return {"skipped": "cc_tracker not configured"}
        status = self._cc_tracker.get_status()
        return {"status": status}

    def _run_architecture_review(self) -> dict[str, Any]:
        if self._feature_health_monitor is None:
            return {"skipped": "feature_health_monitor not configured"}
        report = self._feature_health_monitor.compute_report(
            since=time.time() - CADENCE_SECONDS["monthly"]
        )
        return {"report": report.to_dict() if hasattr(report, "to_dict") else report}

    def _run_deployment_review(self) -> dict[str, Any]:
        if self._rec_engine is None:
            return {"skipped": "rec_engine not configured"}
        results = self._rec_engine.evaluate(since=time.time() - CADENCE_SECONDS["monthly"])
        return {"triggered_dimensions": [r.dimension.value for r in results if r.triggered]}

    # ── Public: manual + scheduled execution ────────────────────────────────

    def list_cycles(self) -> list[dict[str, Any]]:
        return [
            {"name": c.name, "cadence": c.cadence, "description": c.description}
            for c in self._cycles.values()
        ]

    def run_cycle(self, name: str) -> dict[str, Any]:
        """Run one named cycle immediately and persist a run record,
        regardless of whether it is currently 'due'.

        Serialized on ``self._run_lock`` so the background poller and a
        manual API trigger (or two overlapping ``run_due_cycles`` calls)
        can never execute the same cycle concurrently.
        """
        if name not in self._cycles:
            raise ValueError(f"Unknown research cycle {name!r}")

        with self._run_lock:
            return self._run_cycle_locked(name)

    def _run_cycle_locked(self, name: str) -> dict[str, Any]:
        """Body of run_cycle(), assumed to be called while holding
        ``self._run_lock``. Never call directly from outside the lock."""
        cycle = self._cycles[name]
        run_id = f"rcyc-{uuid.uuid4().hex[:12]}"
        started_at = time.time()
        record: dict[str, Any] = {
            "run_id": run_id,
            "cycle_name": cycle.name,
            "cadence": cycle.cadence,
            "status": "RUNNING",
            "started_at": started_at,
            "completed_at": None,
            "findings": {},
            "error": None,
        }
        self._db.upsert_research_cycle_run(record)

        try:
            findings = cycle.run()
            record["status"] = "COMPLETED"
            record["findings"] = findings
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"ResearchScheduler: cycle {name!r} failed: {exc!r}")
            record["status"] = "FAILED"
            record["error"] = str(exc)
        finally:
            record["completed_at"] = time.time()
            self._db.upsert_research_cycle_run(record)

        return record

    def is_due(self, name: str, now: float | None = None) -> bool:
        cycle = self._cycles.get(name)
        if cycle is None:
            raise ValueError(f"Unknown research cycle {name!r}")
        now = now if now is not None else time.time()
        last = self._db.get_last_research_cycle_run(name)
        if last is None:
            return True
        interval = CADENCE_SECONDS[cycle.cadence]
        return (now - last.get("started_at", 0.0)) >= interval

    def run_due_cycles(self, now: float | None = None) -> list[dict[str, Any]]:
        """Run every cycle whose cadence interval has elapsed since its last
        recorded run. Cycles that have never run are always due.

        The due-check is re-verified under the same lock ``run_cycle()``
        uses, so two concurrent ``run_due_cycles()`` calls (e.g. the
        background poller and a manual trigger racing) cannot both observe
        a cycle as due and run it twice.
        """
        now = now if now is not None else time.time()
        results = []
        for name in self._cycles:
            record = self._run_if_still_due(name, now=now)
            if record is not None:
                results.append(record)
        return results

    def _run_if_still_due(self, name: str, now: float) -> dict[str, Any] | None:
        with self._run_lock:
            if not self.is_due(name, now=now):
                return None
            return self._run_cycle_locked(name)

    def status(self) -> list[dict[str, Any]]:
        """Per-cycle last-run summary + whether it's currently due."""
        out = []
        now = time.time()
        for name, cycle in self._cycles.items():
            last = self._db.get_last_research_cycle_run(name)
            out.append(
                {
                    "name": name,
                    "cadence": cycle.cadence,
                    "description": cycle.description,
                    "last_run": last,
                    "due": self.is_due(name, now=now),
                }
            )
        return out

    # ── Background thread ────────────────────────────────────────────────────

    def start_background(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._background_loop, name="research-scheduler", daemon=True
        )
        self._thread.start()
        logger.info(
            f"ResearchScheduler: background thread started "
            f"(poll interval {self._poll_interval_s}s)"
        )

    def stop_background(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                # join() timed out — the worker is still finishing a cycle.
                # Leave the handle in place so start_background() refuses to
                # spawn a second thread until this one actually exits, and
                # log so operators can see a slow cycle blocked shutdown.
                logger.warning(
                    "ResearchScheduler: background thread did not stop within "
                    f"{timeout}s; leaving it running rather than risking two "
                    "concurrent pollers"
                )
            else:
                self._thread = None

    def _background_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.run_due_cycles()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"ResearchScheduler: run_due_cycles failed: {exc!r}")
            self._stop_event.wait(self._poll_interval_s)
