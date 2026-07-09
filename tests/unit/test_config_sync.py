"""Tests for the S31 Trainer Configuration Manager (trainer/config_sync/)."""

from __future__ import annotations

import os
import tempfile

import pytest

from trainer.config_sync.crypto import sign_sync_payload
from trainer.config_sync.manager import ConfigSyncError, SyncPayloadEntry, TrainerConfigManager
from trainer.config_sync.ownership import (
    OwnershipMode,
    classify,
    is_syncable,
)
from trainer.config_sync.store import TrainerConfigStore

SHARED_SECRET = "unit-test-shared-secret"


@pytest.fixture()
def manager() -> TrainerConfigManager:
    db_path = tempfile.mktemp(suffix=".db")
    store = TrainerConfigStore(db_path)
    mgr = TrainerConfigManager(
        store=store,
        shared_secret=SHARED_SECRET,
        required_keys=["OANDA_API_KEY"],
    )
    yield mgr
    store.close()
    if os.path.exists(db_path):
        os.remove(db_path)


def _signed_entry(key: str, value: str, version: int, expires_at=None, provider=None):
    sig = sign_sync_payload(key, value, version, expires_at, SHARED_SECRET)
    return SyncPayloadEntry(
        key=key, value=value, signature=sig, expires_at=expires_at, provider=provider
    )


class TestOwnershipClassification:
    def test_trainer_owned_not_syncable(self):
        assert classify("RESEARCH_LAB_DB_PATH")[0] is OwnershipMode.TRAINER_OWNED
        assert not is_syncable("RESEARCH_LAB_DB_PATH")

    def test_signalium_synced_is_syncable(self):
        assert classify("OANDA_API_KEY")[0] is OwnershipMode.SIGNALIUM_SYNCED
        assert is_syncable("OANDA_API_KEY")

    def test_platform_exclusive_is_syncable_but_flagged(self):
        assert classify("OPENAI_API_KEY")[0] is OwnershipMode.SIGNALIUM_EXCLUSIVE
        assert is_syncable("OPENAI_API_KEY")

    def test_unknown_key_not_syncable(self):
        assert classify("SOME_RANDOM_KEY") is None
        assert not is_syncable("SOME_RANDOM_KEY")


class TestSync:
    def test_accepts_valid_signed_entry(self, manager: TrainerConfigManager):
        entry = _signed_entry("OANDA_API_KEY", "secret-value", 1, provider="oanda")
        results = manager.sync([entry])
        assert results[0].accepted
        assert results[0].version == 1
        fetched = manager.get("OANDA_API_KEY")
        assert fetched is not None
        assert fetched.value == "secret-value"

    def test_rejects_trainer_owned_key(self, manager: TrainerConfigManager):
        sig = sign_sync_payload("RESEARCH_LAB_DB_PATH", "x", 1, None, SHARED_SECRET)
        entry = SyncPayloadEntry(key="RESEARCH_LAB_DB_PATH", value="x", signature=sig)
        results = manager.sync([entry])
        assert not results[0].accepted
        assert "trainer-owned" in results[0].reason

    def test_rejects_unknown_key(self, manager: TrainerConfigManager):
        sig = sign_sync_payload("MYSTERY_KEY", "x", 1, None, SHARED_SECRET)
        entry = SyncPayloadEntry(key="MYSTERY_KEY", value="x", signature=sig)
        results = manager.sync([entry])
        assert not results[0].accepted
        assert "unknown" in results[0].reason

    def test_rejects_invalid_signature(self, manager: TrainerConfigManager):
        entry = SyncPayloadEntry(key="OANDA_API_KEY", value="v", signature="deadbeef")
        results = manager.sync([entry])
        assert not results[0].accepted
        assert results[0].reason == "invalid signature"

    def test_rejects_unsupported_provider(self, manager: TrainerConfigManager):
        entry = _signed_entry("OANDA_API_KEY", "v", 1, provider="mystery_broker")
        results = manager.sync([entry])
        assert not results[0].accepted
        assert "unsupported provider" in results[0].reason

    def test_rejects_already_expired_secret(self, manager: TrainerConfigManager):
        entry = _signed_entry("OANDA_API_KEY", "v", 1, expires_at=1.0)
        results = manager.sync([entry])
        assert not results[0].accepted
        assert "expired" in results[0].reason

    def test_versions_increment_across_syncs(self, manager: TrainerConfigManager):
        manager.sync([_signed_entry("OANDA_API_KEY", "v1", 1)])
        manager.sync([_signed_entry("OANDA_API_KEY", "v2", 2)])
        assert manager.current_version("OANDA_API_KEY")["version"] == 2
        assert manager.get("OANDA_API_KEY").value == "v2"

    def test_one_bad_entry_does_not_block_others_in_batch(self, manager: TrainerConfigManager):
        good = _signed_entry("OANDA_API_KEY", "v1", 1)
        bad = SyncPayloadEntry(key="TIINGO_API_KEY", value="x", signature="bad")
        results = manager.sync([good, bad])
        assert results[0].accepted
        assert not results[1].accepted


class TestRollbackAndRotate:
    def test_rollback_to_prior_version(self, manager: TrainerConfigManager):
        manager.sync([_signed_entry("OANDA_API_KEY", "v1", 1)])
        manager.sync([_signed_entry("OANDA_API_KEY", "v2", 2)])
        assert manager.get("OANDA_API_KEY").value == "v2"
        assert manager.rollback("OANDA_API_KEY", 1)
        assert manager.get("OANDA_API_KEY").value == "v1"

    def test_rollback_unknown_version_fails(self, manager: TrainerConfigManager):
        manager.sync([_signed_entry("OANDA_API_KEY", "v1", 1)])
        assert not manager.rollback("OANDA_API_KEY", 99)

    def test_rotate_trainer_owned_is_noop(self, manager: TrainerConfigManager):
        result = manager.rotate("RESEARCH_LAB_DB_PATH")
        assert result["action"] == "no-op"

    def test_rotate_signalium_synced_invalidates_cache(self, manager: TrainerConfigManager):
        manager.sync([_signed_entry("OANDA_API_KEY", "v1", 1)])
        result = manager.rotate("OANDA_API_KEY")
        assert result["action"] == "invalidated_pending_resync"
        entry = manager.get("OANDA_API_KEY")
        assert entry.is_expired()

    def test_rotate_unknown_key_raises(self, manager: TrainerConfigManager):
        with pytest.raises(ConfigSyncError):
            manager.rotate("NOT_A_REAL_KEY")


class TestValidateAndStatus:
    def test_missing_required_key_flagged(self, manager: TrainerConfigManager):
        issues = manager.validate()
        assert any(i.kind == "missing" and i.key == "OANDA_API_KEY" for i in issues)

    def test_status_healthy_after_sync(self, manager: TrainerConfigManager):
        manager.sync([_signed_entry("OANDA_API_KEY", "v1", 1)])
        status = manager.status()
        assert status["healthy"] is True
        assert "OANDA_API_KEY" in status["keys"]

    def test_status_unhealthy_when_expired(self, manager: TrainerConfigManager):
        manager.sync([_signed_entry("OANDA_API_KEY", "v1", 1)])
        manager.rotate("OANDA_API_KEY")
        status = manager.status()
        assert status["healthy"] is False

    def test_expired_secret_detected_for_non_required_key(self, manager: TrainerConfigManager):
        manager.sync([_signed_entry("TIINGO_API_KEY", "v1", 1, expires_at=None)])
        manager.rotate("TIINGO_API_KEY")
        issues = manager.validate()
        assert any(i.kind == "expired" and i.key == "TIINGO_API_KEY" for i in issues)


class TestOfflineFallback:
    def test_get_returns_last_known_good_even_when_stale(self, manager: TrainerConfigManager):
        manager.sync([_signed_entry("OANDA_API_KEY", "cached-value", 1)])
        # Simulate Signalium being unreachable — no new sync happens, but
        # the trainer must keep functioning off the cached copy.
        entry = manager.get("OANDA_API_KEY")
        assert entry is not None
        assert entry.value == "cached-value"

    def test_missing_key_returns_none_not_exception(self, manager: TrainerConfigManager):
        assert manager.get("NEVER_SYNCED_KEY") is None
