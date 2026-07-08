"""One-time migration: consolidate legacy QDE/MLOps registry data into
`UnifiedModelRegistry`'s JSON store.

Usage::

    python -m trainer.registry.migrations.migrate_legacy_registries \\
        --qde-registry /tmp/qde_registry.json \\
        --mlops-db experiments/mlops/mlops.db \\
        --output experiments/registry/unified_registry.json

Both `--qde-registry` and `--mlops-db` are optional; supply whichever legacy
stores exist. Existing entries in `--output` are preserved; migrated entries
are merged in by `entry_id` (re-running is idempotent — existing entries are
skipped, not duplicated).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from typing import Any

from trainer.mlops.registry import ModelRecord, _model_record_to_entry
from trainer.qde.qde_models import QDEStage, QDEVersion
from trainer.qde.qde_registry import _qde_version_to_entry
from trainer.registry.unified_registry import UnifiedModelRegistry


def migrate_qde_registry(qde_registry_path: str, unified: UnifiedModelRegistry) -> int:
    """Load a legacy `qde_registry.json` file and register its versions."""
    if not os.path.exists(qde_registry_path):
        print(f"[skip] no legacy QDE registry found at {qde_registry_path}")
        return 0

    with open(qde_registry_path) as fh:
        raw = json.load(fh)

    versions_raw = raw.get("versions", raw if isinstance(raw, list) else [])
    migrated = 0
    for vd in versions_raw:
        if unified.get(vd["version_id"]) is not None:
            continue  # already migrated
        version = QDEVersion(
            version_id=vd["version_id"],
            version_number=vd["version_number"],
            label=vd.get("label", ""),
            stage=QDEStage(vd.get("stage", "REGISTERED")),
            created_at=vd.get("created_at", 0.0),
            updated_at=vd.get("updated_at", 0.0),
            parent_version_id=vd.get("parent_version_id"),
            tags=vd.get("tags", []),
            description=vd.get("description", ""),
            metrics=vd.get("metrics", {}),
            stage_timestamps=vd.get("stage_timestamps", {}),
            deployed_at=vd.get("deployed_at"),
            retired_at=vd.get("retired_at"),
            artifact_uri=vd.get("artifact_uri", ""),
            checksum=vd.get("checksum", ""),
        )
        entry = _qde_version_to_entry(version)
        unified.register(entry, native_ref=version)
        if version.stage == QDEStage.ACTIVE:
            unified.set_champion(entry.entry_id, family="qde")
        migrated += 1
    print(f"[qde] migrated {migrated} version(s) from {qde_registry_path}")
    return migrated


def migrate_mlops_sqlite(mlops_db_path: str, unified: UnifiedModelRegistry) -> int:
    """Load legacy `MLOpsDatabase` model_registry rows and register them."""
    if not os.path.exists(mlops_db_path):
        print(f"[skip] no legacy MLOps database found at {mlops_db_path}")
        return 0

    conn = sqlite3.connect(mlops_db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM model_registry").fetchall()
    except sqlite3.OperationalError:
        print(f"[skip] {mlops_db_path} has no model_registry table")
        conn.close()
        return 0

    migrated = 0
    for row in rows:
        d: dict[str, Any] = dict(row)
        model_id = d["model_id"]
        if unified.get(model_id) is not None:
            continue  # already migrated

        record = ModelRecord.from_dict(
            {
                **d,
                "training_config": json.loads(d.get("training_config_json") or "{}"),
                "validation_report": json.loads(d.get("validation_report_json") or "{}"),
                "approval_history": json.loads(d.get("approval_history_json") or "[]"),
                "child_model_ids": json.loads(d.get("child_model_ids") or "[]"),
                "metadata": json.loads(d.get("metadata_json") or "{}"),
                "lifecycle_state": d.get("lifecycle_state", "draft"),
            }
        )
        entry = _model_record_to_entry(record)
        unified.register(entry, native_ref=record)
        if record.is_champion:
            unified.set_champion(entry.entry_id, family=record.name)
        migrated += 1
    conn.close()
    print(f"[mlops] migrated {migrated} model(s) from {mlops_db_path}")
    return migrated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qde-registry", default="/tmp/qde_registry.json")
    parser.add_argument("--mlops-db", default="experiments/mlops/mlops.db")
    parser.add_argument("--output", default="experiments/registry/unified_registry.json")
    args = parser.parse_args()

    unified = UnifiedModelRegistry(storage_path=args.output)
    total = 0
    total += migrate_qde_registry(args.qde_registry, unified)
    total += migrate_mlops_sqlite(args.mlops_db, unified)

    issues = unified.audit_consistency()
    if issues:
        print("[warn] consistency issues detected after migration:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("[ok] unified registry passed consistency audit")

    print(f"Done. {total} total entrie(s) migrated into {args.output}")


if __name__ == "__main__":
    main()
