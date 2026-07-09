"""S31 — Configuration sync HTTP surface.

Exposes the four endpoints required by the ownership-refactor spec:

    GET  /config/status
    POST /config/sync
    POST /config/rotate
    GET  /config/version

This is a plain ``APIRouter`` so it can be mounted onto the existing
Research Lab FastAPI app (reusing its admin auth dependency) instead of
standing up a second server/port.
"""

from __future__ import annotations

from typing import Any

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "FastAPI and Pydantic are required. Install with: pip install fastapi pydantic"
    ) from exc

from trainer.config_sync.manager import ConfigSyncError, SyncPayloadEntry, TrainerConfigManager


class SyncEntryModel(BaseModel):
    key: str
    value: str
    signature: str
    expires_at: float | None = None
    provider: str | None = None


class SyncRequest(BaseModel):
    entries: list[SyncEntryModel]


class RotateRequest(BaseModel):
    key: str


class RollbackRequest(BaseModel):
    key: str
    to_version: int


def make_config_sync_router(manager: TrainerConfigManager, auth_dep: Any = None) -> Any:
    """Create the ``/config`` router.

    Args:
        manager: The :class:`TrainerConfigManager` backing every endpoint.
        auth_dep: Optional FastAPI ``Depends(...)`` value enforcing admin
            auth on every route (mirrors the pattern used by the telemetry
            router). When omitted, the router carries no auth of its own —
            callers must mount it behind their own auth boundary.
    """
    dependencies = [auth_dep] if auth_dep is not None else []
    router = APIRouter(prefix="/config", tags=["config-sync"], dependencies=dependencies)

    @router.get("/status")
    def get_status() -> dict[str, Any]:
        return manager.status()

    @router.post("/sync")
    def post_sync(body: SyncRequest) -> dict[str, Any]:
        entries = [
            SyncPayloadEntry(
                key=e.key,
                value=e.value,
                signature=e.signature,
                expires_at=e.expires_at,
                provider=e.provider,
            )
            for e in body.entries
        ]
        results = manager.sync(entries)
        any_rejected = any(not r.accepted for r in results)
        return {
            "results": [r.to_dict() for r in results],
            "all_accepted": not any_rejected,
        }

    @router.post("/rotate")
    def post_rotate(body: RotateRequest) -> dict[str, Any]:
        try:
            return manager.rotate(body.key)
        except ConfigSyncError as exc:
            raise HTTPException(400, detail=str(exc)) from exc

    @router.post("/rollback")
    def post_rollback(body: RollbackRequest) -> dict[str, Any]:
        ok = manager.rollback(body.key, body.to_version)
        if not ok:
            raise HTTPException(
                404, detail=f"No version {body.to_version} found for key {body.key!r}"
            )
        return {"key": body.key, "rolled_back_to": body.to_version}

    @router.get("/version")
    def get_version(key: str | None = None) -> dict[str, Any]:
        return manager.current_version(key)

    return router
