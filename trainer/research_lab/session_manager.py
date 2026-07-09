"""S24D §1 — internal research session establishment.

Signalium Trainer never authenticates end users itself — it only ever
*establishes an internal session* after successfully validating a
Signalium-issued JWT (see ``sso.py``). This module is that translation step:
identity in, a persisted internal session record out.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.sso import SignaliumIdentity


class SessionManager:
    """Establishes and looks up internal research sessions.

    One session record is created per distinct validated token (deduplicated
    by the token's ``jti`` claim when present, so re-presenting the same
    token across multiple requests reuses one session rather than growing the
    table unboundedly). Tokens without a ``jti`` get a fresh session per
    validation — still safe, just less deduplicated.
    """

    def __init__(self, db: ResearchLabDatabase):
        self._db = db

    def establish(self, identity: SignaliumIdentity) -> dict[str, Any]:
        if identity.jti:
            existing = self._db.get_research_session(identity.jti)
            if existing is not None and existing.get("expires_at", 0.0) > time.time():
                return existing

        session_id = identity.jti or f"rses-{uuid.uuid4().hex[:16]}"
        record = {
            "session_id": session_id,
            "admin_id": identity.admin_id,
            "role": identity.role.value,
            "originating_service": identity.originating_service,
            "created_at": time.time(),
            "expires_at": identity.expires_at,
        }
        self._db.upsert_research_session(record)
        return record

    def get(self, session_id: str) -> dict[str, Any] | None:
        return self._db.get_research_session(session_id)

    def list_active(self, admin_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return self._db.list_research_sessions(
            admin_id=admin_id, active_only=True, now=time.time(), limit=limit
        )
