"""S31 — Trainer Environment Ownership boundaries.

Signalium Platform is the single source of truth for platform-level
secrets (payment, SMTP, OAuth, OpenAI, platform JWT signing keys, user
management credentials). Trainer never mints, rotates, or is otherwise the
authority for these — it may *consume* a synced copy through the
:class:`~trainer.config_sync.manager.TrainerConfigManager` if a subsystem
genuinely needs one, but it must never be the last-resort owner of them.

Trainer remains the authoritative owner of model-specific configuration
(Research Lab paths, artifact signing keys, GPU/training/dataset paths,
scheduler/concurrency/cache settings) — those are never synced *from*
Signalium, only ever set locally.

Market data provider credentials (OANDA, Tiingo, future providers) and
RunPod credentials are a hybrid: Signalium remains the source of truth and
may rotate them at any time, but Trainer keeps a locally cached, versioned
copy so it keeps functioning (placing trades, launching/monitoring jobs)
even if Signalium is temporarily unreachable.

This module is intentionally pure data/classification — it has no I/O and
performs no enforcement by itself. ``TrainerConfigManager.sync`` consults it
to reject sync payloads that try to claim a category they are not permitted
to (e.g. a payload asserting a ``TRAINER_OWNED`` key came from Signalium).
"""

from __future__ import annotations

from enum import Enum


class OwnershipMode(str, Enum):
    """Who is the authoritative owner of a given configuration key."""

    #: Trainer sets/owns this itself. Never accepted via POST /config/sync.
    TRAINER_OWNED = "trainer_owned"

    #: Signalium is the source of truth; Trainer keeps a synced local copy
    #: and must keep functioning from that cached copy if Signalium is
    #: temporarily unreachable (market data providers, RunPod).
    SIGNALIUM_SYNCED = "signalium_synced"

    #: Signalium owns this outright. Trainer may consume a synced copy for
    #: a specific integration need but must never treat itself as the
    #: source of truth, must never rotate it, and should minimize how long
    #: it is retained locally (payment, SMTP, OAuth, OpenAI, platform JWT
    #: signing keys, user management credentials).
    SIGNALIUM_EXCLUSIVE = "signalium_exclusive"


class ConfigCategory(str, Enum):
    RESEARCH = "research"
    ARTIFACTS = "artifacts"
    TRAINING = "training"
    RUNTIME = "runtime"
    MARKET_DATA = "market_data"
    RUNPOD = "runpod"
    PLATFORM_MANAGED = "platform_managed"


#: Model-specific configuration Trainer owns outright. These keys are never
#: accepted from a sync payload — attempting to sync one is a validation
#: error ("trainer-owned key cannot be synced").
TRAINER_OWNED_KEYS: dict[str, ConfigCategory] = {
    "RESEARCH_LAB_DB_PATH": ConfigCategory.RESEARCH,
    "ARTIFACT_STORE_ROOT": ConfigCategory.ARTIFACTS,
    "ARTIFACT_SIGNING_PRIVATE_KEY": ConfigCategory.ARTIFACTS,
    "ARTIFACT_SIGNING_PUBLIC_KEY": ConfigCategory.ARTIFACTS,
    "GPU_DEVICE_SETTINGS": ConfigCategory.TRAINING,
    "TRAINING_CACHE_DIR": ConfigCategory.TRAINING,
    "DATASET_STORAGE_PATH": ConfigCategory.TRAINING,
    "SCHEDULER_POLL_INTERVAL_S": ConfigCategory.RUNTIME,
    "WORKER_CONCURRENCY": ConfigCategory.RUNTIME,
    "CACHE_SIZE_LIMIT": ConfigCategory.RUNTIME,
}

#: Signalium is the source of truth; Trainer stores synced copies to stay
#: operational during Signalium outages.
SIGNALIUM_SYNCED_KEYS: dict[str, ConfigCategory] = {
    "OANDA_API_KEY": ConfigCategory.MARKET_DATA,
    "OANDA_ACCOUNT_ID": ConfigCategory.MARKET_DATA,
    "TIINGO_API_KEY": ConfigCategory.MARKET_DATA,
    "EA_FEED_SHARED_SECRET": ConfigCategory.MARKET_DATA,
    "RUNPOD_API_KEY": ConfigCategory.RUNPOD,
    "RUNPOD_CREDENTIAL_SECRET": ConfigCategory.RUNPOD,
}

#: Platform-level secrets Signalium owns exclusively. Trainer must never be
#: the last-resort owner — these are only ever accepted through sync when a
#: concrete consuming subsystem requests them, and are flagged distinctly
#: in /config/status so operators can see Trainer holds a consume-only copy.
SIGNALIUM_EXCLUSIVE_KEYS: dict[str, ConfigCategory] = {
    "PAYMENT_PROVIDER_API_KEY": ConfigCategory.PLATFORM_MANAGED,
    "PAYMENT_WEBHOOK_SECRET": ConfigCategory.PLATFORM_MANAGED,
    "SMTP_HOST": ConfigCategory.PLATFORM_MANAGED,
    "SMTP_USERNAME": ConfigCategory.PLATFORM_MANAGED,
    "SMTP_PASSWORD": ConfigCategory.PLATFORM_MANAGED,
    "OAUTH_CLIENT_ID": ConfigCategory.PLATFORM_MANAGED,
    "OAUTH_CLIENT_SECRET": ConfigCategory.PLATFORM_MANAGED,
    "OPENAI_API_KEY": ConfigCategory.PLATFORM_MANAGED,
    "SIGNALIUM_JWT_SECRET": ConfigCategory.PLATFORM_MANAGED,
    "SIGNALIUM_JWT_PUBLIC_KEY": ConfigCategory.PLATFORM_MANAGED,
    "USER_MANAGEMENT_DB_CREDENTIALS": ConfigCategory.PLATFORM_MANAGED,
}

#: Providers Trainer knows how to validate synced credentials for. Any
#: MARKET_DATA/RUNPOD key naming a provider outside this set is flagged by
#: ``validate()`` as "unsupported provider configuration".
SUPPORTED_MARKET_DATA_PROVIDERS = frozenset({"oanda", "tiingo", "ea_feed"})
SUPPORTED_RUNPOD_PROVIDERS = frozenset({"runpod"})


def classify(key: str) -> tuple[OwnershipMode, ConfigCategory] | None:
    """Return ``(ownership, category)`` for a known key, or ``None``."""
    if key in TRAINER_OWNED_KEYS:
        return OwnershipMode.TRAINER_OWNED, TRAINER_OWNED_KEYS[key]
    if key in SIGNALIUM_SYNCED_KEYS:
        return OwnershipMode.SIGNALIUM_SYNCED, SIGNALIUM_SYNCED_KEYS[key]
    if key in SIGNALIUM_EXCLUSIVE_KEYS:
        return OwnershipMode.SIGNALIUM_EXCLUSIVE, SIGNALIUM_EXCLUSIVE_KEYS[key]
    return None


def is_syncable(key: str) -> bool:
    """Whether *key* may ever be accepted via ``POST /config/sync``.

    Trainer-owned keys are never syncable — they are set locally only.
    Unknown keys are rejected too (fail closed on unrecognized config).
    """
    classification = classify(key)
    if classification is None:
        return False
    ownership, _ = classification
    return ownership is not OwnershipMode.TRAINER_OWNED
