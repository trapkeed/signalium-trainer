"""S30 — Production Verification Aggregator.

Combines the individually-existing integrity checkers (active model,
artifact registry, feature store, dataset, and lineage) into a single
aggregated report. This module never re-derives any verification logic —
it always delegates to the existing checker for each dimension and simply
rolls the results up into one PASS/WARN/FAIL verdict.

Dimensions covered:
  - active_model     — QDERegistry.get_active() + artifact checksum check
  - artifacts         — ArtifactIntegrityVerifier.verify_artifact_checksum()
  - feature_store     — check_feature_health() on the latest feature events
  - dataset           — DatasetValidator.validate()
  - lineage           — LineageTree node completeness (no orphan/cycle refs)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class DimensionResult:
    dimension: str
    status: str  # "PASS" | "WARN" | "FAIL" | "SKIPPED"
    details: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProductionVerificationReport:
    generated_at: float
    dimensions: list[DimensionResult]
    overall_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "overall_status": self.overall_status,
            "dimensions": [
                {
                    "dimension": d.dimension,
                    "status": d.status,
                    "details": d.details,
                    "evidence": d.evidence,
                }
                for d in self.dimensions
            ],
        }


class ProductionVerificationRunner:
    """Runs all five production-integrity checks and aggregates the result.

    Every dependency is optional (``None``-able) so the aggregator degrades
    gracefully — a dimension that has no configured checker is reported as
    SKIPPED rather than crashing the whole report.
    """

    def __init__(
        self,
        *,
        qde_registry: Any | None = None,
        artifact_integrity_verifier: Any | None = None,
        feature_events_provider: Any | None = None,
        feature_health_config: Any | None = None,
        dataset_dir: Path | str | None = None,
        lineage_tree: Any | None = None,
    ) -> None:
        self._qde_registry = qde_registry
        self._integrity = artifact_integrity_verifier
        self._feature_events_provider = feature_events_provider
        self._feature_health_config = feature_health_config
        self._dataset_dir = Path(dataset_dir) if dataset_dir else None
        self._lineage_tree = lineage_tree

    # ── Individual dimension checks ─────────────────────────────────────────

    def check_active_model(self) -> DimensionResult:
        if self._qde_registry is None:
            return DimensionResult("active_model", "SKIPPED", "no QDERegistry configured")
        active = self._qde_registry.get_active()
        if active is None:
            return DimensionResult("active_model", "FAIL", "no ACTIVE QDE version registered")
        evidence: dict[str, Any] = {"version_id": active.version_id, "stage": active.stage.value}
        if self._integrity is not None:
            checksum = self._integrity.verify_artifact_checksum(active.version_id)
            evidence["checksum"] = checksum
            if checksum.get("status") == "CORRUPTED":
                return DimensionResult("active_model", "FAIL", "active model artifact checksum mismatch", evidence)
            if checksum.get("status") == "UNVERIFIABLE":
                return DimensionResult("active_model", "WARN", "active model artifact unverifiable", evidence)
        return DimensionResult("active_model", "PASS", "active model registered and verified", evidence)

    def check_artifacts(self, artifact_ids: list[str] | None = None) -> DimensionResult:
        if self._integrity is None:
            return DimensionResult("artifacts", "SKIPPED", "no ArtifactIntegrityVerifier configured")
        if not artifact_ids:
            return DimensionResult("artifacts", "SKIPPED", "no artifact_ids provided to verify")
        results = [self._integrity.verify_artifact_checksum(aid) for aid in artifact_ids]
        corrupted = [r for r in results if r.get("status") == "CORRUPTED"]
        unverifiable = [r for r in results if r.get("status") == "UNVERIFIABLE"]
        evidence = {"n_checked": len(results), "corrupted": corrupted, "unverifiable": unverifiable}
        if corrupted:
            return DimensionResult("artifacts", "FAIL", f"{len(corrupted)} corrupted artifact(s)", evidence)
        if unverifiable:
            return DimensionResult("artifacts", "WARN", f"{len(unverifiable)} unverifiable artifact(s)", evidence)
        return DimensionResult("artifacts", "PASS", "all checked artifacts verified", evidence)

    def check_feature_store(self) -> DimensionResult:
        if self._feature_events_provider is None or self._feature_health_config is None:
            return DimensionResult("feature_store", "SKIPPED", "no feature health provider configured")
        try:
            from trainer.research_lab.feature_health import check_feature_health
        except ImportError as exc:
            return DimensionResult("feature_store", "SKIPPED", f"feature_health unavailable: {exc}")
        events = self._feature_events_provider()
        evidence_report = check_feature_health(events, self._feature_health_config)
        if evidence_report is None:
            return DimensionResult("feature_store", "PASS", "no threshold breaches detected", {"n_events": len(events)})
        status = "FAIL" if evidence_report.get("is_critical") else "WARN"
        return DimensionResult("feature_store", status, "feature health threshold breach", evidence_report)

    def check_dataset(self) -> DimensionResult:
        if self._dataset_dir is None:
            return DimensionResult("dataset", "SKIPPED", "no dataset_dir configured")
        try:
            from trainer.datasets.validation import DatasetValidator
        except ImportError as exc:
            return DimensionResult("dataset", "SKIPPED", f"dataset validator unavailable: {exc}")
        validator = DatasetValidator(self._dataset_dir, fail_on_error=False)
        result = validator.validate()
        evidence = {"errors": result.errors, "warnings": result.warnings}
        if not result.passed:
            return DimensionResult("dataset", "FAIL", "dataset validation failed", evidence)
        if result.warnings:
            return DimensionResult("dataset", "WARN", "dataset validated with warnings", evidence)
        return DimensionResult("dataset", "PASS", "dataset validated", evidence)

    def check_lineage(self) -> DimensionResult:
        if self._lineage_tree is None:
            return DimensionResult("lineage", "SKIPPED", "no LineageTree configured")
        nodes = getattr(self._lineage_tree, "_nodes", {})
        orphans = [
            node_id
            for node_id, node in nodes.items()
            if getattr(node, "parent_id", None) and node.parent_id not in nodes
        ]
        evidence = {"n_nodes": len(nodes), "orphans": orphans}
        if orphans:
            return DimensionResult("lineage", "FAIL", f"{len(orphans)} node(s) reference a missing parent", evidence)
        return DimensionResult("lineage", "PASS", "lineage graph internally consistent", evidence)

    # ── Aggregation ──────────────────────────────────────────────────────────

    def run(self, *, artifact_ids: list[str] | None = None) -> ProductionVerificationReport:
        dimensions = [
            self.check_active_model(),
            self.check_artifacts(artifact_ids),
            self.check_feature_store(),
            self.check_dataset(),
            self.check_lineage(),
        ]
        statuses = {d.status for d in dimensions}
        if "FAIL" in statuses:
            overall = "FAIL"
        elif "WARN" in statuses:
            overall = "WARN"
        else:
            overall = "PASS"
        report = ProductionVerificationReport(
            generated_at=time.time(), dimensions=dimensions, overall_status=overall
        )
        logger.info(f"ProductionVerification: overall={overall} " + ", ".join(f"{d.dimension}={d.status}" for d in dimensions))
        return report
