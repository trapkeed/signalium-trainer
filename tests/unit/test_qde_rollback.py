"""Unit tests for Phase S2 additions: QDEVersion contract fields and
QDERollbackManager's schema-compatibility check.
"""

from __future__ import annotations

import unittest

from trainer.qde.qde_models import QDEStage, QDEVersion
from trainer.qde.qde_outputs import OUTPUT_SCHEMA_VERSION
from trainer.qde.qde_rollback import QDERollbackManager


class _FakeRegistry:
    """Minimal stand-in for QDERegistry's public interface used by rollback."""

    def __init__(self, versions: list[QDEVersion], active_id: str | None) -> None:
        self._versions = {v.version_id: v for v in versions}
        self._active_id = active_id

    def get(self, version_id: str) -> QDEVersion | None:
        return self._versions.get(version_id)

    def get_active(self) -> QDEVersion | None:
        return self._versions.get(self._active_id) if self._active_id else None

    def set_active(self, version_id: str) -> None:
        self._active_id = version_id
        self._versions[version_id].stage = QDEStage.ACTIVE

    def update(self, version: QDEVersion) -> None:
        self._versions[version.version_id] = version

    def list_all(self) -> list[QDEVersion]:
        return list(self._versions.values())


class TestQDEVersionContractFields(unittest.TestCase):
    def test_new_defaults_to_empty_schema_versions(self):
        v = QDEVersion.new(1)
        self.assertEqual(v.output_schema_version, "")
        self.assertEqual(v.feature_contract_version, "")

    def test_new_accepts_schema_versions(self):
        v = QDEVersion.new(
            1, output_schema_version=OUTPUT_SCHEMA_VERSION, feature_contract_version="fc_v1"
        )
        self.assertEqual(v.output_schema_version, OUTPUT_SCHEMA_VERSION)
        self.assertEqual(v.feature_contract_version, "fc_v1")

    def test_to_dict_includes_schema_fields(self):
        v = QDEVersion.new(1, output_schema_version=OUTPUT_SCHEMA_VERSION)
        d = v.to_dict()
        self.assertIn("output_schema_version", d)
        self.assertIn("feature_contract_version", d)
        self.assertEqual(d["output_schema_version"], OUTPUT_SCHEMA_VERSION)


class TestQDERollbackSchemaCompatibility(unittest.TestCase):
    def setUp(self):
        self.active = QDEVersion.new(2, output_schema_version=OUTPUT_SCHEMA_VERSION)
        self.active.stage = QDEStage.ACTIVE
        self.target_same_schema = QDEVersion.new(1, output_schema_version=OUTPUT_SCHEMA_VERSION)
        self.target_same_schema.stage = QDEStage.RETIRED
        self.target_diff_schema = QDEVersion.new(
            1, output_schema_version="legacy_buy_sell_no_trade"
        )
        self.target_diff_schema.stage = QDEStage.RETIRED
        self.target_no_schema = QDEVersion.new(1)
        self.target_no_schema.stage = QDEStage.RETIRED

    def test_rollback_allowed_without_strict_check(self):
        registry = _FakeRegistry([self.active, self.target_diff_schema], self.active.version_id)
        mgr = QDERollbackManager(registry)
        record = mgr.rollback_to(self.target_diff_schema.version_id, reason="test")
        self.assertEqual(record.to_version_id, self.target_diff_schema.version_id)

    def test_rollback_blocked_with_strict_check_on_mismatch(self):
        registry = _FakeRegistry([self.active, self.target_diff_schema], self.active.version_id)
        mgr = QDERollbackManager(registry)
        with self.assertRaises(ValueError):
            mgr.rollback_to(
                self.target_diff_schema.version_id, reason="test", require_schema_match=True
            )

    def test_rollback_allowed_with_strict_check_on_match(self):
        registry = _FakeRegistry([self.active, self.target_same_schema], self.active.version_id)
        mgr = QDERollbackManager(registry)
        record = mgr.rollback_to(
            self.target_same_schema.version_id, reason="test", require_schema_match=True
        )
        self.assertEqual(record.to_version_id, self.target_same_schema.version_id)

    def test_rollback_allowed_with_strict_check_when_target_has_no_schema(self):
        # Legacy versions with no recorded schema stay rollback-eligible even
        # under the strict flag, since there is nothing to compare against.
        registry = _FakeRegistry([self.active, self.target_no_schema], self.active.version_id)
        mgr = QDERollbackManager(registry)
        record = mgr.rollback_to(
            self.target_no_schema.version_id, reason="test", require_schema_match=True
        )
        self.assertEqual(record.to_version_id, self.target_no_schema.version_id)


if __name__ == "__main__":
    unittest.main()
