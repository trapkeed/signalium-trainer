"""S20 — Failure Investigation Engine.

Orchestrates all failure detectors, deduplicates overlapping signals,
scores severity, assembles FailureReport objects, and persists them to the
Research Lab database.

Trigger paths:
  - On-demand: POST /research/investigate  (wired in research_lab_server.py)
  - Scheduled: call engine.investigate() on a schedule from the server startup task.

Example::

    engine = FailureInvestigationEngine(warehouse, db)
    reports = engine.investigate(InvestigationQuery(model_version="v1.0", since=week_ago))
    for r in reports:
        print(r.severity_score, r.summary)
"""

from __future__ import annotations

import time
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.research_lab.failure_detectors import ALL_DETECTOR_CLASSES, BaseDetector
from trainer.research_lab.investigation_models import (
    FailureReport,
    InvestigationConfig,
    InvestigationFailureKind,
    InvestigationQuery,
)
from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.telemetry_warehouse import TelemetryWarehouse


class FailureInvestigationEngine:
    """Orchestrates failure detection across all six failure categories.

    Args:
        warehouse: TelemetryWarehouse to read telemetry from.
        db:        ResearchLabDatabase to persist FailureReport records.
        config:    Optional InvestigationConfig; defaults are used if not provided.
    """

    def __init__(
        self,
        warehouse: TelemetryWarehouse,
        db: ResearchLabDatabase,
        config: InvestigationConfig | None = None,
    ) -> None:
        self._wh = warehouse
        self._db = db
        self._cfg = config or InvestigationConfig()
        self._detectors: list[BaseDetector] = [
            cls(warehouse, self._cfg) for cls in ALL_DETECTOR_CLASSES
        ]

    # ── Primary API ───────────────────────────────────────────────────────────

    def investigate(self, query: InvestigationQuery | None = None) -> list[FailureReport]:
        """Run all applicable detectors and persist their reports.

        Args:
            query: Time-window, symbol, and model-version filters. If omitted,
                   runs against all telemetry with no time restriction.

        Returns:
            List of FailureReport objects, sorted by severity_score descending.
        """
        if query is None:
            query = InvestigationQuery()
        if query.until is None:
            # Snapshot 'now' so all detectors share the same ceiling
            query = InvestigationQuery(
                model_version=query.model_version,
                symbol=query.symbol,
                since=query.since,
                until=time.time(),
                failure_kinds=query.failure_kinds,
                limit_per_stream=query.limit_per_stream,
            )

        active = self._select_detectors(query.failure_kinds)
        logger.info(
            "FailureInvestigationEngine: running {} detectors (model_version={}, since={}, until={})",
            len(active),
            query.model_version,
            query.since,
            query.until,
        )

        all_reports: list[FailureReport] = []
        for detector in active:
            try:
                reports = detector.detect(query)
                logger.debug(
                    "{} detector produced {} report(s)",
                    detector.kind.value,
                    len(reports),
                )
                all_reports.extend(reports)
            except Exception as exc:
                logger.warning(
                    "FailureInvestigationEngine: detector {} raised {}: {}",
                    detector.kind.value,
                    type(exc).__name__,
                    exc,
                )

        deduped = self._deduplicate(all_reports)
        deduped.sort(key=lambda r: r.severity_score, reverse=True)

        for report in deduped:
            self._db.upsert_failure_report(report.to_dict())

        logger.info(
            "FailureInvestigationEngine: {} total reports persisted",
            len(deduped),
        )
        return deduped

    def get_reports(
        self,
        failure_kind: str | None = None,
        model_version: str | None = None,
        min_severity: float = 0.0,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Read persisted FailureReport records from the database.

        Args:
            failure_kind:  Filter by InvestigationFailureKind value string.
            model_version: Filter by model version.
            min_severity:  Only return reports with severity_score >= this.
            since:         Only return reports created after this timestamp.
            until:         Only return reports created before this timestamp.
            limit:         Maximum number of results.

        Returns:
            List of report dicts, sorted by severity_score descending.
        """
        return self._db.list_failure_reports(
            failure_kind=failure_kind,
            model_version=model_version,
            min_severity=min_severity,
            since=since,
            until=until,
            limit=limit,
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    def _select_detectors(self, kinds: list[InvestigationFailureKind] | None) -> list[BaseDetector]:
        if not kinds:
            return self._detectors
        kind_set = set(kinds)
        return [d for d in self._detectors if d.kind in kind_set]

    def _deduplicate(self, reports: list[FailureReport]) -> list[FailureReport]:
        """Remove duplicate reports for the same (kind, symbol, ~time) tuple.

        Two reports are considered duplicates if they share the same failure_kind,
        the same primary affected symbol, and their time ranges overlap.
        When duplicates exist, the one with the higher severity_score wins.
        """
        # Group by (kind, primary_symbol)
        groups: dict[tuple[str, str], list[FailureReport]] = {}
        for r in reports:
            primary_sym = r.affected_symbols[0] if r.affected_symbols else ""
            key = (r.failure_kind.value, primary_sym)
            groups.setdefault(key, []).append(r)

        result: list[FailureReport] = []
        for group in groups.values():
            # Within a group, merge overlapping time ranges — keep highest severity
            merged = self._merge_overlapping(group)
            result.extend(merged)
        return result

    @staticmethod
    def _merge_overlapping(reports: list[FailureReport]) -> list[FailureReport]:
        """Simple interval-merge: overlapping time ranges → keep max-severity report."""
        if not reports:
            return []
        # Sort by start time
        sorted_reports = sorted(reports, key=lambda r: r.time_range_start)
        merged: list[FailureReport] = [sorted_reports[0]]
        for current in sorted_reports[1:]:
            last = merged[-1]
            if current.time_range_start <= last.time_range_end:
                # Overlap: keep the one with higher severity
                if current.severity_score > last.severity_score:
                    merged[-1] = current
                # Extend the end time of the survivor
                merged[-1].time_range_end = max(last.time_range_end, current.time_range_end)
            else:
                merged.append(current)
        return merged
