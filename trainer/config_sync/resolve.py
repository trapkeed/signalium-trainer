"""S31 — Convenience resolver for consumers of Signalium-synced config.

Providers (OANDA, Tiingo, RunPod, ...) should prefer a value synced from
Signalium via :class:`TrainerConfigManager` and fall back to a raw
environment variable only if nothing has been synced yet (e.g. local dev
before the first sync, or Signalium has never been reachable). This keeps
"Signalium is source of truth, trainer stores synced copies" true without
forcing every provider to import the manager/store internals directly.
"""

from __future__ import annotations

import os

from trainer.config_sync.manager import TrainerConfigManager, get_default_manager


def resolve_synced_value(key: str, manager: TrainerConfigManager | None = None) -> str | None:
    """Return the best available value for *key*: an active synced entry
    (even if stale) takes priority since it's the Signalium-sourced value;
    an unexpired entry is preferred, but an expired one is still returned
    (better a possibly-stale credential attempt than none, for
    non-security-critical read paths — callers dealing with signed
    execution credentials, e.g. RunPod, must still call ``assert_usable``)
    over falling straight back to a raw env var meant only for local dev.
    """
    mgr = manager if manager is not None else get_default_manager()
    try:
        entry = mgr.get(key)
    except Exception:  # noqa: BLE001 — offline fallback must never raise
        entry = None
    if entry is not None:
        return entry.value
    return os.environ.get(key)
