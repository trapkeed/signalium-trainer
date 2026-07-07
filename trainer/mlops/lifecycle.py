"""Phase 18 — Model Lifecycle Manager.

Manages all valid lifecycle transitions for registered models.

Valid transitions::

    Draft → Training → Validating → Candidate → Approved → Champion → Production
    Any → Shadow
    Any → Archived → Retired
    Production/Champion → Rolled Back → Archived

Every transition is recorded with timestamp, actor, and reason.
Invalid transitions raise LifecycleError.

Usage::

    lm = LifecycleManager()
    lm.transition(record, LifecycleState.TRAINING, actor="trainer")
    lm.transition(record, LifecycleState.VALIDATING, actor="system")
    history = lm.history(model_id)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from trainer.mlops.registry import LifecycleState, ModelRecord

# ---------------------------------------------------------------------------
# Lifecycle transition
# ---------------------------------------------------------------------------


@dataclass
class LifecycleTransition:
    """One recorded lifecycle state change.

    Attributes:
        transition_id: Unique UUID.
        model_id: Model being transitioned.
        from_state: Previous lifecycle state.
        to_state: New lifecycle state.
        actor: Username/system identifier who triggered this.
        reason: Free-form explanation.
        timestamp: UTC ISO-8601 timestamp.
        metadata: Extra context (e.g. validation scores that triggered change).
    """

    transition_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = ""
    from_state: str = ""
    to_state: str = ""
    actor: str = "system"
    reason: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "model_id": self.model_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "actor": self.actor,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Lifecycle error
# ---------------------------------------------------------------------------


class LifecycleError(Exception):
    """Raised when an invalid lifecycle transition is attempted."""

    pass


# ---------------------------------------------------------------------------
# Valid transition graph
# ---------------------------------------------------------------------------

# Maps from_state → set of valid to_states
_VALID_TRANSITIONS: dict[str, set[str]] = {
    LifecycleState.DRAFT: {LifecycleState.TRAINING, LifecycleState.ARCHIVED},
    LifecycleState.TRAINING: {
        LifecycleState.VALIDATING,
        LifecycleState.ARCHIVED,
        LifecycleState.DRAFT,
    },
    LifecycleState.VALIDATING: {
        LifecycleState.CANDIDATE,
        LifecycleState.DRAFT,
        LifecycleState.ARCHIVED,
    },
    LifecycleState.CANDIDATE: {
        LifecycleState.APPROVED,
        LifecycleState.DRAFT,
        LifecycleState.ARCHIVED,
    },
    LifecycleState.APPROVED: {
        LifecycleState.CHAMPION,
        LifecycleState.PRODUCTION,
        LifecycleState.SHADOW,
        LifecycleState.CANDIDATE,
        LifecycleState.ARCHIVED,
    },
    LifecycleState.CHAMPION: {
        LifecycleState.PRODUCTION,
        LifecycleState.SHADOW,
        LifecycleState.ROLLED_BACK,
        LifecycleState.ARCHIVED,
    },
    LifecycleState.PRODUCTION: {
        LifecycleState.CHAMPION,
        LifecycleState.SHADOW,
        LifecycleState.ROLLED_BACK,
        LifecycleState.ARCHIVED,
    },
    LifecycleState.SHADOW: {
        LifecycleState.PRODUCTION,
        LifecycleState.CHAMPION,
        LifecycleState.ARCHIVED,
        LifecycleState.CANDIDATE,
    },
    LifecycleState.ARCHIVED: {LifecycleState.RETIRED},
    LifecycleState.RETIRED: set(),  # terminal
    LifecycleState.ROLLED_BACK: {LifecycleState.ARCHIVED, LifecycleState.CANDIDATE},
}


# ---------------------------------------------------------------------------
# Lifecycle manager
# ---------------------------------------------------------------------------


class LifecycleManager:
    """Governs and records all model lifecycle state changes.

    Validates transitions, records history, and prevents invalid state changes.
    """

    def __init__(self, strict: bool = True) -> None:
        """
        Args:
            strict: If True, raise LifecycleError on invalid transitions.
                    If False, log a warning but allow the transition.
        """
        self.strict = strict
        self._history: dict[str, list[LifecycleTransition]] = {}

    def can_transition(self, from_state: str, to_state: str) -> bool:
        """Return True if the transition is valid."""
        return to_state in _VALID_TRANSITIONS.get(from_state, set())

    def transition(
        self,
        record: ModelRecord,
        to_state: str,
        actor: str = "system",
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> LifecycleTransition:
        """Execute a lifecycle state transition on a model record.

        Args:
            record: The ModelRecord to transition.
            to_state: Target LifecycleState value.
            actor: Who or what triggered this transition.
            reason: Human-readable explanation.
            metadata: Optional extra context.

        Returns:
            LifecycleTransition record.

        Raises:
            LifecycleError: If strict=True and the transition is invalid.
        """
        from_state = record.lifecycle_state
        if from_state == to_state:
            return LifecycleTransition(
                model_id=record.model_id,
                from_state=from_state,
                to_state=to_state,
                actor=actor,
                reason="no-op: same state",
                metadata=metadata or {},
            )

        if not self.can_transition(from_state, to_state):
            msg = (
                f"Invalid lifecycle transition: {from_state!r} → {to_state!r} "
                f"for model {record.model_id} ({record.full_name})"
            )
            if self.strict:
                raise LifecycleError(msg)
            logger.warning(msg)

        record.lifecycle_state = to_state
        record.updated_at = datetime.now(UTC).isoformat()

        if to_state == LifecycleState.RETIRED:
            record.retirement_status = "retired"
        if to_state in (LifecycleState.CHAMPION, LifecycleState.PRODUCTION):
            record.deployment_status = "deployed"
        if to_state == LifecycleState.ROLLED_BACK:
            record.deployment_status = "rolled_back"
        if to_state in (LifecycleState.ARCHIVED, LifecycleState.RETIRED):
            record.deployment_status = "not_deployed"

        t = LifecycleTransition(
            model_id=record.model_id,
            from_state=from_state,
            to_state=to_state,
            actor=actor,
            reason=reason,
            metadata=metadata or {},
        )
        if record.model_id not in self._history:
            self._history[record.model_id] = []
        self._history[record.model_id].append(t)

        logger.info(
            f"Lifecycle: {record.full_name} [{from_state}→{to_state}] " f"by {actor}: {reason}"
        )
        return t

    def history(self, model_id: str) -> list[LifecycleTransition]:
        """Return full transition history for a model (oldest first)."""
        return self._history.get(model_id, [])

    def history_dicts(self, model_id: str) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self.history(model_id)]

    def all_history(self) -> list[LifecycleTransition]:
        """Return all transitions across all models, sorted by timestamp."""
        all_t = []
        for transitions in self._history.values():
            all_t.extend(transitions)
        return sorted(all_t, key=lambda t: t.timestamp)

    def rollback(
        self, record: ModelRecord, actor: str = "system", reason: str = "emergency rollback"
    ) -> LifecycleTransition:
        """Emergency rollback — transition to ROLLED_BACK state."""
        return self.transition(
            record,
            LifecycleState.ROLLED_BACK,
            actor=actor,
            reason=reason,
            metadata={"rollback": True, "emergency": True},
        )

    def promote_to_candidate(
        self, record: ModelRecord, actor: str = "validator", validation_score: float = 0.0
    ) -> LifecycleTransition:
        """Transition from validating → candidate."""
        return self.transition(
            record,
            LifecycleState.CANDIDATE,
            actor=actor,
            reason=f"Validation complete. Score={validation_score:.4f}",
            metadata={"validation_score": validation_score},
        )

    def approve(
        self, record: ModelRecord, approver: str = "ops", reason: str = ""
    ) -> LifecycleTransition:
        """Approve a candidate model — transition to APPROVED."""
        t = self.transition(
            record,
            LifecycleState.APPROVED,
            actor=approver,
            reason=reason or "Manual approval",
            metadata={"approval": True},
        )
        record.add_approval(
            approver=approver,
            decision="approved",
            reason=reason,
            from_state=t.from_state,
            to_state=LifecycleState.APPROVED,
        )
        record.promotion_status = "approved"
        return t

    def reject(self, record: ModelRecord, approver: str = "ops", reason: str = "") -> None:
        """Reject a candidate model — transition back to DRAFT."""
        record.add_approval(
            approver=approver,
            decision="rejected",
            reason=reason,
            from_state=record.lifecycle_state,
            to_state=LifecycleState.DRAFT,
        )
        record.promotion_status = "rejected"
        self.transition(
            record,
            LifecycleState.DRAFT,
            actor=approver,
            reason=f"Rejected: {reason}",
            metadata={"rejection": True},
        )

    def retire(
        self, record: ModelRecord, actor: str = "system", reason: str = ""
    ) -> list[LifecycleTransition]:
        """Retire a model — transitions to ARCHIVED then RETIRED."""
        transitions = []
        if record.lifecycle_state not in (LifecycleState.ARCHIVED, LifecycleState.RETIRED):
            transitions.append(
                self.transition(
                    record,
                    LifecycleState.ARCHIVED,
                    actor=actor,
                    reason=reason or "archiving for retirement",
                )
            )
        transitions.append(
            self.transition(
                record, LifecycleState.RETIRED, actor=actor, reason=reason or "model retired"
            )
        )
        record.retirement_status = "retired"
        return transitions

    def state_summary(self) -> dict[str, Any]:
        """Return counts of models per lifecycle state."""
        counts: dict[str, int] = {}
        for transitions in self._history.values():
            if transitions:
                final_state = transitions[-1].to_state
                counts[final_state] = counts.get(final_state, 0) + 1
        return {
            "transitions_by_model": {mid: len(ts) for mid, ts in self._history.items()},
            "models_tracked": len(self._history),
            "all_transitions": sum(len(ts) for ts in self._history.values()),
        }
