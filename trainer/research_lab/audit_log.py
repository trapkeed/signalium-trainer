"""S24D §3 — complete audit logging for every research action.

Every request handled through an authenticated (or rejected) Signalium
session is recorded here with: administrator identity, role, timestamp,
originating service, action performed, and — when the handling endpoint
knows them — affected model version, datasets, and experiments.

The log is append-only (see ``ResearchLabDatabase.insert_research_audit_log``)
so a compromised or buggy handler can never quietly rewrite history.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.sso import SignaliumIdentity


class AuditLogger:
    def __init__(self, db: ResearchLabDatabase):
        self._db = db

    def log_success(
        self,
        identity: SignaliumIdentity,
        action: str,
        *,
        model_version: str | None = None,
        datasets: list[str] | None = None,
        experiments: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._insert(
            admin_id=identity.admin_id,
            role=identity.role.value,
            originating_service=identity.originating_service,
            action=action,
            outcome="SUCCESS",
            model_version=model_version,
            datasets=datasets,
            experiments=experiments,
            extra=extra,
        )

    def log_rejected(
        self,
        *,
        reason: str,
        action: str,
        admin_id: str = "UNKNOWN",
        role: str = "UNKNOWN",
        originating_service: str = "unknown",
    ) -> dict[str, Any]:
        """Records a rejected (unauthenticated/unauthorized) request.

        ``admin_id``/``role`` are frequently "UNKNOWN" here — a request can
        be rejected precisely because its identity could not be verified —
        but the rejection itself, with as much context as is safely
        available, is still logged so the audit trail is complete.
        """
        return self._insert(
            admin_id=admin_id,
            role=role,
            originating_service=originating_service,
            action=action,
            outcome="REJECTED",
            model_version=None,
            datasets=None,
            experiments=None,
            extra={"reason": reason},
        )

    def _insert(
        self,
        *,
        admin_id: str,
        role: str,
        originating_service: str,
        action: str,
        outcome: str,
        model_version: str | None,
        datasets: list[str] | None,
        experiments: list[str] | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        record = {
            "log_id": f"raud-{uuid.uuid4().hex[:16]}",
            "admin_id": admin_id,
            "role": role,
            "timestamp": time.time(),
            "originating_service": originating_service,
            "action": action,
            "outcome": outcome,
            "model_version": model_version,
            "datasets": datasets or [],
            "experiments": experiments or [],
            "extra": extra or {},
        }
        self._db.insert_research_audit_log(record)
        return record

    def list(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._db.list_research_audit_log(**kwargs)
