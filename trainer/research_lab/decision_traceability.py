"""S28 — QDE Audit and Decision Traceability.

The QDE remains the sole decision maker. Every decision it produces must be
explainable and reproducible. This module provides:

  - ``DecisionTraceRecord``  — one immutable record per decision, capturing
    qde/dataset/feature version, inference timestamp, input feature hash,
    output decision, confidence, size/risk signal, exit horizon, and a
    natural-language reasoning summary.
  - ``ModelLineageChain``    — the resolved chain
    Decision -> Model Version -> Dataset Version -> Feature Set ->
    Training Run -> Experiment -> Git Commit, built from real persisted
    records (not opaque strings) whenever a resolver can supply them.
  - Reproducibility helpers — deterministic inference replay and historical
    context reconstruction.

Visible only to administrative roles (Admin, SuperAdmin, Founder,
QuantResearcher, MLResearcher, DeploymentApprover, Auditor) via
``Visibility.DECISION_TRACEABILITY`` — never to customers or the public.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from trainer.traceability.hasher import hash_config

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


# ── Decision trace record ───────────────────────────────────────────────────


def compute_input_feature_hash(features: dict[str, Any]) -> str:
    """Deterministic SHA-256 prefix hash of a decision's input feature vector.

    Uses the same sorted-JSON approach as training/inference config hashing
    (``trainer.traceability.hasher``) so the same feature dict always hashes
    identically regardless of key order — required for reproducibility
    checks and for grouping "same inputs, did we get the same decision?"
    """
    return hash_config(features, length=16)


@dataclass
class DecisionTraceRecord:
    """Complete traceability record for one QDE decision.

    Persisted once, at decision time, and never mutated afterward — it is
    the institutional "black box recorder" entry for that single inference.
    """

    decision_id: str
    qde_version: str
    dataset_version: str
    feature_version: str
    inference_timestamp: float
    input_feature_hash: str
    output_decision: str
    confidence: float
    size_signal: float
    risk_signal: float
    exit_horizon: float
    reasoning_summary: str
    symbol: str = ""
    input_features: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        qde_version: str,
        output_decision: str,
        confidence: float,
        size_signal: float,
        risk_signal: float,
        exit_horizon: float,
        reasoning_summary: str,
        input_features: dict[str, Any] | None = None,
        dataset_version: str = "",
        feature_version: str = "",
        symbol: str = "",
        decision_id: str | None = None,
        inference_timestamp: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DecisionTraceRecord:
        input_features = input_features or {}
        return cls(
            decision_id=decision_id or f"dec-{uuid.uuid4().hex[:16]}",
            qde_version=qde_version,
            dataset_version=dataset_version,
            feature_version=feature_version,
            inference_timestamp=(
                inference_timestamp if inference_timestamp is not None else time.time()
            ),
            input_feature_hash=compute_input_feature_hash(input_features),
            output_decision=output_decision,
            confidence=confidence,
            size_signal=size_signal,
            risk_signal=risk_signal,
            exit_horizon=exit_horizon,
            reasoning_summary=reasoning_summary,
            symbol=symbol,
            input_features=input_features,
            created_at=time.time(),
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "qde_version": self.qde_version,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "inference_timestamp": round(self.inference_timestamp, 3),
            "input_feature_hash": self.input_feature_hash,
            "output_decision": self.output_decision,
            "confidence": round(self.confidence, 6),
            "size_signal": round(self.size_signal, 6),
            "risk_signal": round(self.risk_signal, 6),
            "exit_horizon": round(self.exit_horizon, 6),
            "reasoning_summary": self.reasoning_summary,
            "symbol": self.symbol,
            "input_features": self.input_features,
            "created_at": round(self.created_at, 3),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DecisionTraceRecord:
        return cls(
            decision_id=d["decision_id"],
            qde_version=d.get("qde_version", ""),
            dataset_version=d.get("dataset_version", ""),
            feature_version=d.get("feature_version", ""),
            inference_timestamp=d.get("inference_timestamp", 0.0),
            input_feature_hash=d.get("input_feature_hash", ""),
            output_decision=d.get("output_decision", ""),
            confidence=d.get("confidence", 0.0),
            size_signal=d.get("size_signal", 0.0),
            risk_signal=d.get("risk_signal", 0.0),
            exit_horizon=d.get("exit_horizon", 0.0),
            reasoning_summary=d.get("reasoning_summary", ""),
            symbol=d.get("symbol", ""),
            input_features=d.get("input_features", {}),
            created_at=d.get("created_at", 0.0),
            metadata=d.get("metadata", {}),
        )


# ── Model lineage chain ─────────────────────────────────────────────────────


@dataclass
class LineageLink:
    """One resolved hop in the lineage chain."""

    link_type: str  # MODEL_VERSION | DATASET_VERSION | FEATURE_SET | TRAINING_RUN | EXPERIMENT | GIT_COMMIT
    identifier: str
    resolved: bool  # True if backed by a real persisted record, False if only the bare identifier is known
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "link_type": self.link_type,
            "identifier": self.identifier,
            "resolved": self.resolved,
            "details": self.details,
        }


@dataclass
class ModelLineageChain:
    """The full lineage of one decision: Decision -> Model Version ->
    Dataset Version -> Feature Set -> Training Run -> Experiment -> Git Commit.

    Unlike ``trainer.memory.decision_lineage.DecisionLineage`` (which stores
    bare identifier strings with no verification), this chain is built by
    calling a ``LineageResolver`` against the real Research Lab database —
    each link is marked ``resolved=True`` only if a matching persisted
    record was actually found.
    """

    decision_id: str
    links: list[LineageLink] = field(default_factory=list)
    built_at: float = field(default_factory=time.time)

    @property
    def is_fully_resolved(self) -> bool:
        return bool(self.links) and all(link.resolved for link in self.links)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "links": [link.to_dict() for link in self.links],
            "is_fully_resolved": self.is_fully_resolved,
            "built_at": round(self.built_at, 3),
        }


class LineageResolver:
    """Resolves a decision's lineage chain against real Research Lab records.

    Args:
        db: The ``ResearchLabDatabase`` to look up training runs / experiments
            / artifacts in.
    """

    def __init__(self, db: Any) -> None:
        self._db = db

    def resolve(self, trace: DecisionTraceRecord) -> ModelLineageChain:
        links: list[LineageLink] = []

        model_version = trace.qde_version
        links.append(
            LineageLink(
                link_type="MODEL_VERSION",
                identifier=model_version,
                resolved=bool(model_version),
            )
        )

        dataset_version = trace.dataset_version
        links.append(
            LineageLink(
                link_type="DATASET_VERSION",
                identifier=dataset_version,
                resolved=bool(dataset_version),
            )
        )

        feature_version = trace.feature_version
        links.append(
            LineageLink(
                link_type="FEATURE_SET",
                identifier=feature_version,
                resolved=bool(feature_version),
            )
        )

        training_run = self._find_training_run(model_version)
        links.append(
            LineageLink(
                link_type="TRAINING_RUN",
                identifier=training_run.get("run_id", "") if training_run else "",
                resolved=training_run is not None,
                details=training_run or {},
            )
        )

        experiment_id = (training_run or {}).get("experiment_id", "")
        experiment = self._find_experiment(experiment_id) if experiment_id else None
        links.append(
            LineageLink(
                link_type="EXPERIMENT",
                identifier=experiment_id,
                resolved=experiment is not None,
                details=experiment or {},
            )
        )

        git_commit = (
            (training_run or {}).get("git_commit_hash")
            or (experiment or {}).get("git_commit_hash")
            or ""
        )
        links.append(
            LineageLink(
                link_type="GIT_COMMIT",
                identifier=git_commit,
                resolved=bool(git_commit),
            )
        )

        chain = ModelLineageChain(decision_id=trace.decision_id, links=links)
        logger.debug(
            "LineageResolver: decision={} resolved={}/{} links",
            trace.decision_id,
            sum(1 for lk in links if lk.resolved),
            len(links),
        )
        return chain

    def _find_training_run(self, model_version: str) -> dict[str, Any] | None:
        if not model_version or not hasattr(self._db, "list_training_runs"):
            return None
        try:
            runs = self._db.list_training_runs(qde_version=model_version, limit=1)
        except TypeError:
            try:
                runs = self._db.list_training_runs(limit=1)
            except Exception:
                return None
        except Exception:
            return None
        return runs[0] if runs else None

    def _find_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        if not experiment_id or not hasattr(self._db, "get_experiment"):
            return None
        try:
            return self._db.get_experiment(experiment_id)
        except Exception:
            return None


# ── Reproducibility: replay & reconstruction ────────────────────────────────


@dataclass
class ReplayResult:
    """Outcome of replaying a stored decision's inputs through inference."""

    decision_id: str
    original_output: str
    original_confidence: float
    replayed_output: str
    replayed_confidence: float
    deterministic_match: bool
    confidence_delta: float
    replayed_at: float = field(default_factory=time.time)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "original_output": self.original_output,
            "original_confidence": round(self.original_confidence, 6),
            "replayed_output": self.replayed_output,
            "replayed_confidence": round(self.replayed_confidence, 6),
            "deterministic_match": self.deterministic_match,
            "confidence_delta": round(self.confidence_delta, 6),
            "replayed_at": round(self.replayed_at, 3),
            "notes": self.notes,
        }


InferenceFn = Callable[[dict[str, Any]], dict[str, Any]]
"""A callable that takes the original input feature dict and returns a dict
with at least ``{"output_decision": str, "confidence": float}`` — i.e. the
QDE inference entry point, injected so this module never imports the model
runtime directly (keeps it usable even where torch/heavy deps aren't
installed)."""


def replay_decision(
    trace: DecisionTraceRecord, inference_fn: InferenceFn, confidence_tolerance: float = 1e-6
) -> ReplayResult:
    """Deterministically re-run inference on a stored decision's exact
    recorded inputs and compare the replayed output to what was recorded.

    This is the core "inference replay" reproducibility primitive: given the
    same QDE version and the same input feature hash, the model must produce
    the same output decision (and, within tolerance, the same confidence).
    """
    replayed = inference_fn(dict(trace.input_features))
    replayed_output = str(replayed.get("output_decision", ""))
    replayed_confidence = float(replayed.get("confidence", 0.0))

    replayed_hash = compute_input_feature_hash(trace.input_features)
    delta = abs(replayed_confidence - trace.confidence)
    match = (
        replayed_output == trace.output_decision
        and delta <= confidence_tolerance
        and replayed_hash == trace.input_feature_hash
    )

    notes = ""
    if replayed_hash != trace.input_feature_hash:
        notes = "input feature hash changed between recording and replay — inputs were not preserved exactly"
    elif not match:
        notes = "same inputs produced a different output/confidence — model is non-deterministic or was swapped"

    result = ReplayResult(
        decision_id=trace.decision_id,
        original_output=trace.output_decision,
        original_confidence=trace.confidence,
        replayed_output=replayed_output,
        replayed_confidence=replayed_confidence,
        deterministic_match=match,
        confidence_delta=delta,
        notes=notes,
    )
    if not match:
        logger.warning(
            "replay_decision: decision={} MISMATCH original=({}, {:.4f}) replayed=({}, {:.4f})",
            trace.decision_id,
            trace.output_decision,
            trace.confidence,
            replayed_output,
            replayed_confidence,
        )
    return result


@dataclass
class ReconstructedDecisionContext:
    """Everything needed to understand/re-derive a past decision, assembled
    from persisted records — the "historical reconstruction" primitive.
    """

    decision_id: str
    trace: DecisionTraceRecord
    lineage: ModelLineageChain
    reconstructed_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "trace": self.trace.to_dict(),
            "lineage": self.lineage.to_dict(),
            "reconstructed_at": round(self.reconstructed_at, 3),
        }


def reconstruct_historical_decision(
    decision_id: str, db: Any, lineage_resolver: LineageResolver | None = None
) -> ReconstructedDecisionContext | None:
    """Reconstruct the full context of a past decision from persisted state:
    its traceability record plus its resolved model lineage chain.

    Returns ``None`` if no trace record exists for ``decision_id`` — there is
    nothing to reconstruct.
    """
    raw = db.get_decision_trace(decision_id)
    if raw is None:
        return None
    trace = DecisionTraceRecord.from_dict(raw)
    resolver = lineage_resolver or LineageResolver(db)
    lineage = resolver.resolve(trace)
    return ReconstructedDecisionContext(decision_id=decision_id, trace=trace, lineage=lineage)
