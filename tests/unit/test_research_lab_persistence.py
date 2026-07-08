"""Research Lab — restart-survival persistence tests.

Proves that all 9 Research Lab categories — Experiments, Datasets, Features,
Failures, Shadow Tests, Approvals (promotion), Deployment Decisions
(promotion + rollback), Parameter Searches, and Research Notes — are written
to SQLite and remain readable after the database connection is closed and
reopened from the same file (simulating a server restart).

`trainer.research_lab` imports cleanly without torch (rl_database.py and
rl_models.py are standalone modules with a loguru/logging fallback), so this
suite uses plain unittest + tempfile without any import stubbing.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trainer.research_lab.research_lab_api import ResearchLabAPI
from trainer.research_lab.rl_database import ResearchLabDatabase


class TestResearchLabRestartSurvival(unittest.TestCase):
    """Populates all 9 categories, reopens the DB, and asserts everything persists."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmpdir.name) / "research_lab.db")

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _populate(self) -> dict[str, str]:
        """Create one record in each of the 9 categories. Returns their ids."""
        db = ResearchLabDatabase(self.db_path)
        api = ResearchLabAPI(db=db)
        ids: dict[str, str] = {}

        # 1. Experiments
        exp = api.create_experiment(
            name="restart-audit-exp",
            qde_version="v1.0.0",
            training_dataset="ds-restart",
            researcher="agent",
        )
        self.assertTrue(exp.success, exp.error)
        ids["experiment_id"] = exp.data["experiment_id"]

        # 2. Datasets
        ds = api.create_dataset(
            name="restart-audit-dataset",
            version="1",
            source="synthetic",
            instruments=["EURUSD"],
            start_date="2024-01-01",
            end_date="2024-06-01",
        )
        self.assertTrue(ds.success, ds.error)
        ids["dataset_id"] = ds.data["dataset_id"]

        # 3. Features
        feat = api.record_feature(
            name="restart-audit-feature",
            source="research",
            importance_score=0.42,
        )
        self.assertTrue(feat.success, feat.error)
        ids["feature_id"] = feat.data["feature_id"]

        # 4. Failures
        failure = api.record_failure(
            qde_version="v1.0.0",
            category="LARGE_LOSS",
            severity="HIGH",
            description="restart-audit failure",
        )
        self.assertTrue(failure.success, failure.error)
        ids["failure_id"] = failure.data["failure_id"]

        # 5. Shadow Tests
        shadow = api.create_shadow_test(
            name="restart-audit-shadow",
            challenger_version="v1.1.0",
            champion_version="v1.0.0",
        )
        self.assertTrue(shadow.success, shadow.error)
        ids["shadow_test_id"] = shadow.data["test_id"]

        # 6. Approvals (promotion approval requires a passing gate first)
        # Experiment must be COMPLETED for the gate check to run.
        api.start_experiment(ids["experiment_id"])
        api.complete_experiment(
            ids["experiment_id"], metrics={"sharpe_ratio": 2.0, "win_rate": 0.6}
        )
        gate = api.evaluate_deployment_gates(
            experiment_id=ids["experiment_id"],
            baseline_version="v0.9.0",
            min_sharpe=-100.0,
            max_drawdown_pct=1000.0,
            min_win_rate=-100.0,
            min_profit_factor=-100.0,
        )
        self.assertTrue(gate.success, gate.error)
        approval = api.approve_deployment(
            experiment_id=ids["experiment_id"], approver="agent", notes="restart-audit"
        )
        self.assertTrue(approval.success, approval.error)

        # 7. Deployment Decisions (rollback request/approval workflow)
        rollback = api.create_rollback_request(
            current_version_id="v1.0.0",
            target_version_id="v0.9.0",
            reason="restart-audit rollback",
            requested_by="agent",
        )
        self.assertTrue(rollback.success, rollback.error)
        ids["rollback_request_id"] = rollback.data["request_id"]

        # 8. Parameter Searches
        search = api.create_search(
            name="restart-audit-search",
            qde_version="v1.0.0",
            parameter_space={"lr": [0.001, 0.01]},
            strategy="RANDOM",
        )
        self.assertTrue(search.success, search.error)
        ids["search_id"] = search.data["search_id"]

        # 9. Research Notes
        note = api.create_note(
            content="restart-audit note",
            author="agent",
            linked_entity_type="experiment",
            linked_entity_id=ids["experiment_id"],
            tags=["restart-audit"],
        )
        self.assertTrue(note.success, note.error)
        ids["note_id"] = note.data["note_id"]

        db.close()
        return ids

    def test_all_nine_categories_survive_restart(self) -> None:
        ids = self._populate()

        # Simulate a server restart: brand-new ResearchLabDatabase instance
        # pointed at the same file on disk.
        db2 = ResearchLabDatabase(self.db_path)
        api2 = ResearchLabAPI(db=db2)

        # 1. Experiments
        exp = api2.get_experiment(ids["experiment_id"])
        self.assertTrue(exp.success, exp.error)
        self.assertEqual(exp.data["name"], "restart-audit-exp")

        # 2. Datasets
        ds = api2.get_dataset(ids["dataset_id"])
        self.assertTrue(ds.success, ds.error)
        self.assertEqual(ds.data["name"], "restart-audit-dataset")

        # 3. Features
        features = api2.list_features()
        self.assertTrue(features.success, features.error)
        self.assertTrue(
            any(f["feature_id"] == ids["feature_id"] for f in features.data["features"])
        )

        # 4. Failures
        failure = api2.list_failures()
        self.assertTrue(failure.success, failure.error)
        self.assertTrue(any(f["failure_id"] == ids["failure_id"] for f in failure.data["failures"]))

        # 5. Shadow Tests
        shadow = api2.get_shadow_test(ids["shadow_test_id"])
        self.assertTrue(shadow.success, shadow.error)
        self.assertEqual(shadow.data["challenger_version"], "v1.1.0")

        # 6. Approvals (promotion)
        history = api2.deployment_history()
        self.assertTrue(history.success, history.error)
        self.assertTrue(
            any(
                d.get("experiment_id") == ids["experiment_id"] and d.get("action") == "APPROVED"
                for d in history.data["deployments"]
            )
        )

        # 7. Deployment Decisions (rollback request)
        rollback = api2.get_rollback_request(ids["rollback_request_id"])
        self.assertTrue(rollback.success, rollback.error)
        self.assertEqual(rollback.data["target_version_id"], "v0.9.0")

        # 8. Parameter Searches
        search = api2.get_search(ids["search_id"])
        self.assertTrue(search.success, search.error)
        self.assertEqual(search.data["name"], "restart-audit-search")

        # 9. Research Notes
        note = api2.get_note(ids["note_id"])
        self.assertTrue(note.success, note.error)
        self.assertEqual(note.data["content"], "restart-audit note")
        self.assertEqual(note.data["linked_entity_id"], ids["experiment_id"])
        self.assertIn("restart-audit", note.data["tags"])

        # Cross-check against stats() as an independent count-based signal.
        stats = db2.stats()
        for table in (
            "experiments",
            "datasets",
            "features",
            "failures",
            "shadow_tests",
            "deployments",
            "rollback_requests",
            "parameter_searches",
            "research_notes",
        ):
            self.assertGreaterEqual(stats[table], 1, f"table {table} lost data across restart")

        db2.close()

    def test_research_notes_standalone_crud(self) -> None:
        """Notes are addressable on their own, independent of other entities."""
        db = ResearchLabDatabase(self.db_path)
        api = ResearchLabAPI(db=db)

        created = api.create_note(content="standalone note", author="researcher-1")
        self.assertTrue(created.success, created.error)
        note_id = created.data["note_id"]

        fetched = api.get_note(note_id)
        self.assertTrue(fetched.success, fetched.error)
        self.assertEqual(fetched.data["content"], "standalone note")
        self.assertEqual(fetched.data["linked_entity_type"], "")
        self.assertEqual(fetched.data["linked_entity_id"], "")

        listed = api.list_notes(author="researcher-1")
        self.assertTrue(listed.success, listed.error)
        self.assertEqual(listed.data["total"], 1)

        missing = api.get_note("note-does-not-exist")
        self.assertFalse(missing.success)

        db.close()

    def test_no_behavior_change_to_existing_categories(self) -> None:
        """Sanity check: existing entity CRUD is untouched by the notes addition."""
        db = ResearchLabDatabase(self.db_path)
        api = ResearchLabAPI(db=db)

        exp = api.create_experiment(
            name="unchanged-exp", qde_version="v2.0.0", training_dataset="ds-x"
        )
        self.assertTrue(exp.success, exp.error)
        fetched = api.get_experiment(exp.data["experiment_id"])
        self.assertEqual(fetched.data["qde_version"], "v2.0.0")

        stats = db.stats()
        self.assertIn("research_notes", stats)
        self.assertEqual(stats["research_notes"], 0)

        db.close()


if __name__ == "__main__":
    unittest.main()
