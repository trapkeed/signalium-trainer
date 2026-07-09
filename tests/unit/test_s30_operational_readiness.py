"""S30 — Operational Readiness: production verification, startup self-check,
operational safety switches, and autonomous trading readiness aggregator.
"""

from __future__ import annotations

from trainer.memory.decision_memory import DecisionMemory
from trainer.research_lab.autonomous_readiness import AutonomousReadinessChecker
from trainer.research_lab.operational_safety import (
    DeploymentsFrozenError,
    OperationalSafetySwitches,
)
from trainer.research_lab.production_verification import ProductionVerificationRunner
from trainer.research_lab.startup_self_check import StartupSelfCheck


class TestOperationalSafetySwitches:
    def test_starts_permissive(self):
        s = OperationalSafetySwitches()
        assert s.can_serve_inference() is True
        assert s.get_state() == {
            "emergency_disabled": False,
            "inference_killed": False,
            "deployments_frozen": False,
            "paused_model_versions": [],
        }

    def test_emergency_disable_blocks_inference(self):
        s = OperationalSafetySwitches()
        s.emergency_disable("halt", actor="qa")
        assert s.can_serve_inference() is False
        s.clear_emergency_disable(actor="qa")
        assert s.can_serve_inference() is True

    def test_inference_kill_switch_blocks_regardless_of_model(self):
        s = OperationalSafetySwitches()
        s.inference_kill_switch("probe", actor="qa")
        assert s.can_serve_inference(model_version="v1") is False
        s.restore_inference(actor="qa")
        assert s.can_serve_inference(model_version="v1") is True

    def test_pause_model_blocks_only_that_version(self):
        s = OperationalSafetySwitches()
        s.pause_model("v1", "bad calibration", actor="qa")
        assert s.can_serve_inference(model_version="v1") is False
        assert s.can_serve_inference(model_version="v2") is True
        s.resume_model("v1", actor="qa")
        assert s.can_serve_inference(model_version="v1") is True

    def test_freeze_deployments_blocks_rollback_shortcut(self):
        s = OperationalSafetySwitches()
        s.freeze_deployments("investigating incident", actor="qa")
        try:
            s.rollback_shortcut("attempt", actor="qa")
            assert False, "expected DeploymentsFrozenError"
        except DeploymentsFrozenError:
            pass
        s.unfreeze_deployments(actor="qa")
        try:
            s.rollback_shortcut("attempt", actor="qa")
            assert False, "expected RuntimeError (no rollback authority configured)"
        except RuntimeError:
            pass

    def test_history_records_every_action(self):
        s = OperationalSafetySwitches()
        s.emergency_disable("a", actor="qa")
        s.clear_emergency_disable(actor="qa")
        history = s.history()
        assert [e["action"] for e in history] == ["EMERGENCY_DISABLE", "CLEAR_EMERGENCY_DISABLE"]


class TestProductionVerificationRunner:
    def test_no_dependencies_skips_all_and_passes(self):
        report = ProductionVerificationRunner().run()
        assert report.overall_status == "PASS"
        assert all(d.status == "SKIPPED" for d in report.dimensions)

    def test_missing_active_model_fails(self):
        class FakeRegistry:
            def get_active(self):
                return None

        runner = ProductionVerificationRunner(qde_registry=FakeRegistry())
        report = runner.run()
        assert report.overall_status == "FAIL"


class TestStartupSelfCheck:
    def test_ok_to_serve_when_nothing_configured(self):
        report = StartupSelfCheck().run()
        assert report.ok_to_serve is True

    def test_blocked_when_operational_safety_engaged(self):
        safety = OperationalSafetySwitches()
        safety.emergency_disable("halt", actor="qa")
        report = StartupSelfCheck(operational_safety=safety).run()
        assert report.ok_to_serve is False


class TestAutonomousReadinessChecker:
    def test_all_skipped_reports_ready(self):
        report = AutonomousReadinessChecker().run()
        assert report.ready is True
        assert all(c.status == "SKIPPED" for c in report.checks)

    def test_decision_memory_wired_reports_ready_stages(self):
        dm = DecisionMemory()
        report = AutonomousReadinessChecker(decision_memory=dm).run()
        by_stage = {c.stage: c.status for c in report.checks}
        assert by_stage["memory_updates"] == "READY"
        assert by_stage["trade_lifecycle_updates"] == "READY"

    def test_inference_generation_not_ready_when_killed(self):
        safety = OperationalSafetySwitches()
        safety.inference_kill_switch("probe", actor="qa")
        report = AutonomousReadinessChecker(operational_safety=safety).run()
        by_stage = {c.stage: c.status for c in report.checks}
        assert by_stage["inference_generation"] == "NOT_READY"
        assert report.ready is False

    def test_learning_loop_error_marks_not_ready(self):
        class FakeScheduler:
            def status(self):
                return {"is_running": True, "last_run": {"error": "boom"}}

        report = AutonomousReadinessChecker(learning_loop_scheduler=FakeScheduler()).run()
        by_stage = {c.stage: c.status for c in report.checks}
        assert by_stage["learning_loops"] == "NOT_READY"
