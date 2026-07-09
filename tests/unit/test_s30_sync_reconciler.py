"""Tests for trainer.sync.reconciler: DLQ, retries, idempotency, audit."""

from __future__ import annotations

import pytest

from trainer.sync.reconciler import (
    DomainHandler,
    ReconciliationDomain,
    RemoteRecord,
    SyncReconciler,
)


def _handler(domain, records, apply_fn, **kwargs) -> DomainHandler:
    return DomainHandler(domain=domain, fetch_remote=lambda: records, apply_local=apply_fn, **kwargs)


class TestBasicReconciliation:
    def test_applies_new_record(self):
        applied = []
        rec = SyncReconciler()
        rec.register(_handler(
            ReconciliationDomain.IDENTITIES,
            [RemoteRecord("id-1", "v1", {"name": "alice"})],
            applied.append,
        ))
        rec.run_once()
        assert len(applied) == 1
        assert applied[0].record_id == "id-1"

    def test_idempotent_same_version_not_reapplied(self):
        applied = []
        record = RemoteRecord("id-1", "v1", {"name": "alice"})
        rec = SyncReconciler()
        rec.register(_handler(ReconciliationDomain.IDENTITIES, [record], applied.append))
        rec.run_once()
        rec.run_once()
        assert len(applied) == 1

    def test_new_version_reapplied(self):
        applied = []
        rec = SyncReconciler()
        records_box = {"records": [RemoteRecord("id-1", "v1", {})]}
        rec.register(DomainHandler(
            domain=ReconciliationDomain.IDENTITIES,
            fetch_remote=lambda: records_box["records"],
            apply_local=applied.append,
        ))
        rec.run_once()
        records_box["records"] = [RemoteRecord("id-1", "v2", {})]
        rec.run_once()
        assert len(applied) == 2
        assert applied[1].version == "v2"

    def test_multiple_domains_run_independently(self):
        applied_identities = []
        applied_experiments = []
        rec = SyncReconciler()
        rec.register(_handler(ReconciliationDomain.IDENTITIES, [RemoteRecord("i1", "v1", {})], applied_identities.append))
        rec.register(_handler(ReconciliationDomain.EXPERIMENTS, [RemoteRecord("e1", "v1", {})], applied_experiments.append))
        summary = rec.run_once()
        assert len(applied_identities) == 1
        assert len(applied_experiments) == 1
        assert set(summary["domains_run"]) == {"identities", "experiments"}


class TestRetryAndDLQ:
    def test_persistent_failure_ends_in_dlq(self):
        rec = SyncReconciler()
        rec.register(_handler(
            ReconciliationDomain.MODELS,
            [RemoteRecord("m1", "v1", {})],
            lambda r: (_ for _ in ()).throw(RuntimeError("always fails")),
            max_retries=1, retry_backoff_s=0.001,
        ))
        rec.run_once()
        dlq = rec.dead_letters()
        assert len(dlq) == 1
        assert dlq[0]["record_id"] == "m1"

    def test_eventual_success_not_dead_lettered(self):
        attempts = {"count": 0}

        def flaky(record):
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise RuntimeError("transient")

        rec = SyncReconciler()
        rec.register(_handler(
            ReconciliationDomain.MODELS, [RemoteRecord("m1", "v1", {})], flaky,
            max_retries=3, retry_backoff_s=0.001,
        ))
        rec.run_once()
        assert rec.dead_letters() == []

    def test_requeue_dead_letter_removes_entry(self):
        rec = SyncReconciler()
        rec.register(_handler(
            ReconciliationDomain.MODELS,
            [RemoteRecord("m1", "v1", {})],
            lambda r: (_ for _ in ()).throw(RuntimeError("fail")),
            max_retries=0, retry_backoff_s=0.001,
        ))
        rec.run_once()
        assert len(rec.dead_letters()) == 1
        removed = rec.requeue_dead_letter(ReconciliationDomain.MODELS, "m1")
        assert removed is True
        assert rec.dead_letters() == []

    def test_fetch_remote_failure_recorded_in_summary_errors(self):
        rec = SyncReconciler()

        def boom():
            raise ConnectionError("upstream down")

        rec.register(DomainHandler(
            domain=ReconciliationDomain.PERMISSIONS, fetch_remote=boom, apply_local=lambda r: None,
        ))
        summary = rec.run_once()
        assert len(summary["errors"]) == 1
        assert "permissions" in summary["errors"][0]


class TestAuditLog:
    def test_audit_log_records_applied_outcome(self):
        rec = SyncReconciler()
        rec.register(_handler(ReconciliationDomain.IDENTITIES, [RemoteRecord("i1", "v1", {})], lambda r: None))
        rec.run_once()
        audit = rec.audit_log()
        assert any(e["outcome"] == "applied" for e in audit)

    def test_audit_log_records_skipped_idempotent(self):
        rec = SyncReconciler()
        rec.register(_handler(ReconciliationDomain.IDENTITIES, [RemoteRecord("i1", "v1", {})], lambda r: None))
        rec.run_once()
        rec.run_once()
        audit = rec.audit_log()
        assert any(e["outcome"] == "skipped_idempotent" for e in audit)

    def test_audit_log_records_dead_lettered(self):
        rec = SyncReconciler()
        rec.register(_handler(
            ReconciliationDomain.MODELS,
            [RemoteRecord("m1", "v1", {})],
            lambda r: (_ for _ in ()).throw(RuntimeError("fail")),
            max_retries=0, retry_backoff_s=0.001,
        ))
        rec.run_once()
        audit = rec.audit_log()
        assert any(e["outcome"] == "dead_lettered" for e in audit)


class TestBackgroundLoop:
    def test_start_and_stop_does_not_raise(self):
        rec = SyncReconciler(interval_s=0.05)
        rec.register(_handler(ReconciliationDomain.IDENTITIES, [RemoteRecord("i1", "v1", {})], lambda r: None))
        rec.start()
        import time
        time.sleep(0.15)
        rec.stop(timeout=1.0)
        assert len(rec.audit_log()) >= 1
