"""S20 — ExperimentCreator: maps FailureReport patterns to ExperimentRecord proposals.

Each FailureInvestigationEngine failure kind maps to a canonical experiment
hypothesis. The creator checks for existing PROPOSED/PENDING experiments with
overlapping names before creating new ones (deduplication). All proposals land
in ExperimentStatus.PROPOSED, requiring admin approval before they advance.
"""

from __future__ import annotations

import time
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.research_lab.discovery_models import DiscoveryQuery, DiscoverySummary
from trainer.research_lab.investigation_models import InvestigationFailureKind
from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.rl_models import ExperimentRecord, ExperimentStatus

# ── Hypothesis registry ────────────────────────────────────────────────────────
# Maps each failure kind to a (hypothesis_name, rationale, tags, hyperparameters)
# tuple. The name is also used as the dedup key: if an experiment with this
# exact name is already PROPOSED or PENDING, we skip creating another.

_HYPOTHESES: dict[str, dict[str, Any]] = {
    InvestigationFailureKind.HIGH_CONFIDENCE_LOSS.value: {
        "name": "Recalibrate confidence head to reduce high-confidence entry errors",
        "rationale": (
            "Failure reports show multiple losses where entry confidence ≥ 0.85. "
            "The confidence head is overestimating certainty on losing setups. "
            "Expected improvement: reduce the loss rate on high-confidence entries by "
            "retraining with harder negative examples and confidence-calibration loss."
        ),
        "tags": ["confidence_calibration", "auto-proposed", "high_confidence_loss"],
        "hyperparameters": {
            "confidence_calibration_loss_weight": 0.3,
            "hard_negative_mining_ratio": 0.4,
        },
    },
    InvestigationFailureKind.UNEXPECTED_REVERSAL.value: {
        "name": "Add short-term momentum signal to reduce unexpected reversal losses",
        "rationale": (
            "Failure reports show trades closed at a loss within a short hold window. "
            "A short-term momentum feature (e.g. rate-of-change over the first few bars) "
            "should give the model early warning of impending reversals. "
            "Expected improvement: reduce the fraction of quick-loss trades by ≥10%."
        ),
        "tags": ["momentum", "reversal_detection", "auto-proposed", "unexpected_reversal"],
        "hyperparameters": {
            "momentum_lookback_bars": 5,
            "reversal_loss_weight": 0.25,
        },
    },
    InvestigationFailureKind.LATE_EXIT.value: {
        "name": "Improve exit timing head to prevent holding past optimal exit",
        "rationale": (
            "Counterfactual analysis shows realized PnL was significantly below "
            "the available MFE. The exit-timing head needs stronger supervision "
            "from hindsight MFE labels. "
            "Expected improvement: increase captured MFE fraction by ≥15%."
        ),
        "tags": ["exit_timing", "take_profit", "auto-proposed", "late_exit"],
        "hyperparameters": {
            "mfe_label_weight": 0.4,
            "exit_timing_loss": "mfe_regression",
        },
    },
    InvestigationFailureKind.PREMATURE_EXIT.value: {
        "name": "Extend hold duration head to reduce premature exits from winning positions",
        "rationale": (
            "Counterfactual analysis shows the model exited winning trades at a "
            "fraction of the available MFE. The hold-duration head needs hindsight "
            "supervision from realized time-to-MFE labels. "
            "Expected improvement: increase average trade PnL by reducing premature exits."
        ),
        "tags": ["hold_duration", "exit_timing", "auto-proposed", "premature_exit"],
        "hyperparameters": {
            "hold_duration_label_weight": 0.35,
            "min_hold_bars_reward_discount": 2,
        },
    },
    InvestigationFailureKind.FAILED_SCALE_IN.value: {
        "name": "Add cumulative-PnL feature to prevent scale-in during drawdowns",
        "rationale": (
            "Scale-in events during negative cumulative PnL significantly worsen "
            "drawdowns. Adding a normalized cumulative-PnL-at-scale-in feature "
            "to the position management head should teach the model to avoid "
            "scaling into losing positions. "
            "Expected improvement: reduce SCALE_IN-while-negative events by ≥20%."
        ),
        "tags": ["scale_in", "position_management", "auto-proposed", "failed_scale_in"],
        "hyperparameters": {
            "cumulative_pnl_feature_weight": 0.2,
            "scale_in_loss_penalty": 0.5,
        },
    },
    InvestigationFailureKind.PORTFOLIO_CONCENTRATION.value: {
        "name": "Add cross-asset correlation feature to improve portfolio diversification",
        "rationale": (
            "Decision analysis shows over-concentration in single symbols during "
            "certain windows. A cross-asset correlation feature in the position-sizing "
            "head should enable the model to account for portfolio exposure. "
            "Expected improvement: reduce max symbol concentration by ≥15% in windows "
            "where correlation is high."
        ),
        "tags": [
            "diversification",
            "cross_asset",
            "auto-proposed",
            "portfolio_concentration",
        ],
        "hyperparameters": {
            "correlation_lookback_s": 3600,
            "concentration_penalty_weight": 0.3,
        },
    },
}


class ExperimentCreator:
    """Proposes ExperimentRecord objects from FailureReport patterns.

    Each call to `propose()` scans recent failure reports, maps them to
    hypotheses, deduplicates against open experiments, and persists new
    proposals with status=PROPOSED.
    """

    def __init__(self, db: ResearchLabDatabase) -> None:
        self._db = db

    # ── Public API ─────────────────────────────────────────────────────────────

    def propose(self, query: DiscoveryQuery, summary: DiscoverySummary) -> None:
        """Propose experiments from recent failure reports.

        Results are written to summary (mutates in place).
        """
        reports = self._db.list_failure_reports(
            min_severity=query.min_failure_severity,
            since=query.since,
            until=query.until,
            model_version=query.model_version,  # scope proposals to the requested version
            limit=500,
        )
        if not reports:
            logger.debug("ExperimentCreator.propose: no qualifying failure reports")
            return

        # Gather open experiments to use for dedup
        open_experiments = self._db.list_experiments(status=ExperimentStatus.PROPOSED.value)
        open_experiments += self._db.list_experiments(status=ExperimentStatus.PENDING.value)
        open_names: set[str] = {e.get("name", "") for e in open_experiments}

        # Aggregate: pick the highest-severity report per failure kind
        best_per_kind: dict[str, dict[str, Any]] = {}
        for report in reports:
            kind = report.get("failure_kind", "")
            if kind not in _HYPOTHESES:
                continue
            existing = best_per_kind.get(kind)
            if existing is None or report.get("severity_score", 0) > existing.get(
                "severity_score", 0
            ):
                best_per_kind[kind] = report

        proposed_count = 0
        for kind, report in sorted(
            best_per_kind.items(), key=lambda kv: -kv[1].get("severity_score", 0)
        ):
            if proposed_count >= query.max_proposed_per_run:
                break
            hypothesis = _HYPOTHESES[kind]
            name = hypothesis["name"]

            if name in open_names:
                logger.debug(f"ExperimentCreator.propose: skipping duplicate experiment '{name}'")
                summary.experiments_skipped_duplicate += 1
                continue

            # Assemble notes from the triggering failure report
            affected = report.get("affected_symbols", [])
            severity = report.get("severity_score", 0.0)
            notes = (
                f"Auto-proposed from failure report {report.get('report_id', 'unknown')}. "
                f"Failure kind: {kind}. Severity: {severity:.2f}. "
                f"Affected symbols: {', '.join(affected) if affected else 'N/A'}. "
                f"Rationale: {hypothesis['rationale']}"
            )

            mv = report.get("model_version") or query.model_version or "unknown"
            exp = ExperimentRecord(
                experiment_id=f"exp-proposed-{kind.lower()[:16]}-{int(time.time())}",
                name=name,
                qde_version=mv,
                training_dataset="pending_selection",
                status=ExperimentStatus.PROPOSED,
                created_at=time.time(),
                updated_at=time.time(),
                tags=hypothesis["tags"],
                hyperparameters=hypothesis["hyperparameters"],
                notes=notes,
                researcher="auto-discovery",
            )
            self._db.upsert_experiment(exp.to_dict())
            summary.experiment_ids.append(exp.experiment_id)
            summary.experiments_proposed += 1
            proposed_count += 1
            open_names.add(name)  # prevent duplicates within the same run
            logger.info(f"ExperimentCreator.propose: proposed {exp.experiment_id} for {kind}")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def list_proposed(self) -> list[dict[str, Any]]:
        """Return all PROPOSED experiments sorted newest first."""
        return self._db.list_experiments(status=ExperimentStatus.PROPOSED.value)

    def approve(self, experiment_id: str) -> None:
        """Advance a PROPOSED experiment to PENDING (admin approval)."""
        record = self._db.get_experiment(experiment_id)
        if not record:
            raise ValueError(f"Experiment {experiment_id!r} not found")
        if record.get("status") != ExperimentStatus.PROPOSED.value:
            raise ValueError(f"Experiment {experiment_id} is {record['status']}, expected PROPOSED")
        record["status"] = ExperimentStatus.PENDING.value
        record["updated_at"] = time.time()
        self._db.upsert_experiment(record)
        logger.info(f"ExperimentCreator.approve: {experiment_id} → PENDING")

    def reject(self, experiment_id: str) -> None:
        """Reject a PROPOSED experiment (admin decision)."""
        record = self._db.get_experiment(experiment_id)
        if not record:
            raise ValueError(f"Experiment {experiment_id!r} not found")
        if record.get("status") != ExperimentStatus.PROPOSED.value:
            raise ValueError(f"Experiment {experiment_id} is {record['status']}, expected PROPOSED")
        record["status"] = ExperimentStatus.ARCHIVED.value
        record["updated_at"] = time.time()
        record["notes"] = record.get("notes", "") + "\n[REJECTED by admin]"
        self._db.upsert_experiment(record)
        logger.info(f"ExperimentCreator.reject: {experiment_id} → ARCHIVED")
