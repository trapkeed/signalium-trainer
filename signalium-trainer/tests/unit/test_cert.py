"""Phase 25 — Certification Platform tests (~230 tests)."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import unittest

# ── Test isolation (same pattern as MIE) ──────────────────────────────────────

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "trainer", "cert")
ROOT = os.path.normpath(ROOT)

_LOAD_ORDER = [
    "cert_types",
    "cert_models",
    "cert_database",
    "validation_suite",
    "stress_engine",
    "monte_carlo_engine",
    "benchmark_engine",
    "reproducibility_engine",
    "security_engine",
    "operational_engine",
    "checklist_engine",
    "scorecard_engine",
    "cert_dashboard",
    "cert_api",
]


def _preload(short_name: str):
    path = os.path.join(ROOT, short_name + ".py")
    dotted = "trainer.cert." + short_name
    spec = importlib.util.spec_from_file_location(dotted, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


def _load(short_name: str):
    return sys.modules["trainer.cert." + short_name]


# Pre-register package stub so sub-imports resolve correctly
if "trainer" not in sys.modules:
    import types
    sys.modules["trainer"] = types.ModuleType("trainer")
if "trainer.cert" not in sys.modules:
    import types
    pkg = types.ModuleType("trainer.cert")
    sys.modules["trainer.cert"] = pkg

for _name in _LOAD_ORDER:
    _preload(_name)


# ── Convenience helpers ───────────────────────────────────────────────────────

def _types():    return _load("cert_types")
def _models():   return _load("cert_models")
def _db_mod():   return _load("cert_database")
def _vs():       return _load("validation_suite")
def _se():       return _load("stress_engine")
def _mc():       return _load("monte_carlo_engine")
def _be():       return _load("benchmark_engine")
def _re():       return _load("reproducibility_engine")
def _sec():      return _load("security_engine")
def _oe():       return _load("operational_engine")
def _cl():       return _load("checklist_engine")
def _sc():       return _load("scorecard_engine")
def _dash():     return _load("cert_dashboard")
def _api_mod():  return _load("cert_api")


def _good_input(**kwargs):
    """Return a CertificationInput that passes all checks by default."""
    M = _models()
    defaults = dict(
        model_id="mdl-001",
        model_name="Signalium-Quant-v1",
        model_version="1.0.0",
        git_commit="abc1234def",
        research_quality_score=80.0,
        oos_sharpe=1.8,
        oos_win_rate=0.60,
        oos_max_drawdown=0.12,
        walk_forward_results=[{"sharpe": 1.5, "win_rate": 0.58}] * 5,
        cross_regime_results=[{"score": 70.0}] * 4,
        cross_instrument_results=[{"score": 68.0}] * 3,
        cross_timeframe_results=[{"score": 65.0}] * 2,
        generalization_score=72.0,
        stability_score=75.0,
        calibration_score=0.75,
        brier_score=0.18,
        ece=0.08,
        dataset_version="v2.3.1",
        feature_version="v1.4.0",
        label_version="v1.2.0",
        training_config={"lr": 1e-3, "epochs": 100, "batch": 64},
        random_seed=42,
        git_commit_hash="abc1234def",
        environment="python3.11-linux",
        package_versions={"torch": "2.0.0", "numpy": "1.24", "sklearn": "1.2.0"},
        checkpoint_hash="deadbeef01",
        model_hash="cafebabe02",
        security_checks={
            "AUTHENTICATION": True, "AUTHORIZATION": True, "AUDIT_LOGGING": True,
            "API_SECURITY": True, "SECRETS_MANAGEMENT": True,
            "INPUT_VALIDATION": True, "DEPENDENCY_REVIEW": True, "ENCRYPTION": True,
        },
        operational_checks={
            "MONITORING_ENABLED": True, "ALERTS_CONFIGURED": True,
            "HEALTH_CHECKS_PASSING": True, "ROLLBACK_AVAILABLE": True,
            "BACKUPS_VERIFIED": True, "LOGGING_FUNCTIONAL": True,
            "DOCUMENTATION_COMPLETE": True, "RECOVERY_PROCEDURES_VERIFIED": True,
        },
        benchmark_values={
            "INFERENCE_LATENCY": 20.0, "TRAINING_TIME": 1200.0,
            "GPU_USAGE": 70.0, "CPU_USAGE": 55.0, "MEMORY_USAGE": 3000.0,
            "DISK_USAGE": 400.0, "MODEL_SIZE": 128.0,
            "API_THROUGHPUT": 250.0, "RESPONSE_TIME": 60.0,
        },
        notes="Fully validated model for production.",
        tags=["production", "certified"],
    )
    defaults.update(kwargs)
    return M.CertificationInput(**{k: v for k, v in defaults.items()
                                    if hasattr(M.CertificationInput, '__dataclass_fields__') and
                                    k in M.CertificationInput.__dataclass_fields__})


# ── cert_types ────────────────────────────────────────────────────────────────

class TestCertTypes(unittest.TestCase):

    def test_certification_status_values(self):
        T = _types()
        self.assertEqual(T.CertificationStatus.CERTIFIED.value, "CERTIFIED")
        self.assertEqual(T.CertificationStatus.REJECTED.value, "REJECTED")
        self.assertIn(T.CertificationStatus.PILOT_ONLY, list(T.CertificationStatus))

    def test_certification_status_count(self):
        T = _types()
        self.assertEqual(len(T.CertificationStatus), 5)

    def test_certification_stage_values(self):
        T = _types()
        stages = list(T.CertificationStage)
        self.assertEqual(len(stages), 13)
        self.assertIn(T.CertificationStage.CERTIFICATION_DECISION, stages)

    def test_check_status_values(self):
        T = _types()
        self.assertIn(T.CheckStatus.PASS, list(T.CheckStatus))
        self.assertIn(T.CheckStatus.NOT_APPLICABLE, list(T.CheckStatus))

    def test_stress_scenario_count(self):
        T = _types()
        self.assertEqual(len(T.StressScenario), 10)

    def test_benchmark_metric_count(self):
        T = _types()
        self.assertEqual(len(T.BenchmarkMetric), 9)

    def test_validation_check_count(self):
        T = _types()
        self.assertEqual(len(T.ValidationCheck), 8)

    def test_scorecard_category_count(self):
        T = _types()
        self.assertEqual(len(T.ScorecardCategory), 9)

    def test_dashboard_type_count(self):
        T = _types()
        self.assertEqual(len(T.DashboardType), 5)

    def test_clamp(self):
        T = _types()
        self.assertEqual(T.clamp(150.0), 100.0)
        self.assertEqual(T.clamp(-5.0), 0.0)
        self.assertEqual(T.clamp(50.0), 50.0)
        self.assertEqual(T.clamp(3.0, lo=1.0, hi=2.0), 2.0)

    def test_cert_status_label_certified(self):
        T = _types()
        label = T.cert_status_label(T.CertificationStatus.CERTIFIED)
        self.assertIn("Certified", label)

    def test_cert_status_label_rejected(self):
        T = _types()
        label = T.cert_status_label(T.CertificationStatus.REJECTED)
        self.assertIn("Rejected", label)

    def test_cert_thresholds_present(self):
        T = _types()
        self.assertIn("min_overall_score", T.CERT_THRESHOLDS)
        self.assertIn("min_security", T.CERT_THRESHOLDS)
        self.assertIn("pilot_overall", T.CERT_THRESHOLDS)

    def test_scorecard_weights_sum_to_one(self):
        T = _types()
        total = sum(T.SCORECARD_WEIGHTS.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_risk_level_values(self):
        T = _types()
        self.assertIn(T.RiskLevel.CRITICAL, list(T.RiskLevel))
        self.assertIn(T.RiskLevel.INFO, list(T.RiskLevel))


# ── cert_models ───────────────────────────────────────────────────────────────

class TestCertModels(unittest.TestCase):

    def _inp(self):
        M = _models()
        return M.CertificationInput(model_id="m1", model_name="TestModel")

    def test_certification_input_defaults(self):
        inp = self._inp()
        self.assertEqual(inp.model_id, "m1")
        self.assertEqual(inp.model_version, "1.0.0")
        self.assertEqual(inp.oos_win_rate, 0.5)

    def test_certification_input_to_dict(self):
        inp = self._inp()
        d = inp.to_dict()
        self.assertIn("model_id", d)
        self.assertNotIn("walk_forward_results", d)  # lists excluded

    def test_stage_result_defaults(self):
        M = _models()
        T = _types()
        sr = M.StageResult()
        self.assertEqual(sr.stage, T.CertificationStage.RESEARCH)
        self.assertEqual(sr.status, T.CheckStatus.PASS)

    def test_stage_result_to_dict(self):
        M = _models()
        T = _types()
        sr = M.StageResult(stage=T.CertificationStage.BENCHMARK, score=88.5)
        d = sr.to_dict()
        self.assertEqual(d["stage"], "BENCHMARK")
        self.assertEqual(d["score"], 88.5)

    def test_checklist_item_to_dict(self):
        M = _models()
        T = _types()
        item = M.ChecklistItem(check_id="VL-001", status=T.CheckStatus.PASS)
        d = item.to_dict()
        self.assertEqual(d["check_id"], "VL-001")
        self.assertEqual(d["status"], "PASS")

    def test_checklist_report_defaults(self):
        M = _models()
        report = M.ChecklistReport()
        self.assertTrue(report.can_proceed)
        self.assertEqual(report.mandatory_failed, 0)

    def test_validation_result_to_dict(self):
        M = _models()
        T = _types()
        vr = M.ValidationResult(check=T.ValidationCheck.OUT_OF_SAMPLE, score=72.0)
        d = vr.to_dict()
        self.assertEqual(d["check"], "OUT_OF_SAMPLE")
        self.assertEqual(d["score"], 72.0)

    def test_stress_test_result_to_dict(self):
        M = _models()
        T = _types()
        sr = M.StressTestResult(scenario=T.StressScenario.FLASH_CRASH, degradation_pct=35.0)
        d = sr.to_dict()
        self.assertEqual(d["scenario"], "FLASH_CRASH")
        self.assertEqual(d["degradation_pct"], 35.0)

    def test_monte_carlo_result_to_dict(self):
        M = _models()
        T = _types()
        mr = M.MonteCarloResult(technique="BOOTSTRAP_SAMPLING", prob_profitable=0.62)
        d = mr.to_dict()
        self.assertEqual(d["technique"], "BOOTSTRAP_SAMPLING")
        self.assertAlmostEqual(d["prob_profitable"], 0.62, places=4)

    def test_benchmark_result_to_dict(self):
        M = _models()
        T = _types()
        br = M.BenchmarkResult(metric=T.BenchmarkMetric.INFERENCE_LATENCY, value=25.0, unit="ms")
        d = br.to_dict()
        self.assertEqual(d["metric"], "INFERENCE_LATENCY")
        self.assertEqual(d["unit"], "ms")

    def test_reproducibility_result_to_dict(self):
        M = _models()
        T = _types()
        rr = M.ReproducibilityResult(check_name="RANDOM_SEED", status=T.CheckStatus.PASS)
        d = rr.to_dict()
        self.assertEqual(d["check_name"], "RANDOM_SEED")

    def test_security_result_to_dict(self):
        M = _models()
        T = _types()
        sr = M.SecurityResult(check_name="AUTHENTICATION", status=T.CheckStatus.FAIL,
                               risk_level=T.RiskLevel.CRITICAL)
        d = sr.to_dict()
        self.assertEqual(d["risk_level"], "CRITICAL")

    def test_operational_result_to_dict(self):
        M = _models()
        T = _types()
        or_ = M.OperationalResult(check_name="MONITORING_ENABLED", status=T.CheckStatus.PASS)
        d = or_.to_dict()
        self.assertEqual(d["check_name"], "MONITORING_ENABLED")

    def test_scorecard_result_to_dict(self):
        M = _models()
        sc = M.ScorecardResult(overall=82.5, grade="A")
        d = sc.to_dict()
        self.assertEqual(d["overall"], 82.5)
        self.assertEqual(d["grade"], "A")

    def test_scorecard_category_scores(self):
        M = _models()
        sc = M.ScorecardResult(validation=75.0, security=80.0)
        cats = sc.category_scores()
        self.assertIn("VALIDATION", cats)
        self.assertEqual(cats["VALIDATION"], 75.0)

    def test_certification_decision_to_dict(self):
        M = _models()
        T = _types()
        dec = M.CertificationDecision(
            status=T.CertificationStatus.CERTIFIED,
            rationale="All checks passed.",
        )
        d = dec.to_dict()
        self.assertEqual(d["status"], "CERTIFIED")

    def test_audit_package_to_dict(self):
        M = _models()
        ap = M.AuditPackage(certification_report="Test report")
        d = ap.to_dict()
        self.assertIn("run_id", d)
        self.assertEqual(d["certification_report"], "Test report")

    def test_certification_run_to_dict(self):
        M = _models()
        run = M.CertificationRun(model_id="m1", model_name="Test")
        d = run.to_dict()
        self.assertIn("run_id", d)
        self.assertIn("model_id", d)

    def test_certification_run_run_id_generated(self):
        M = _models()
        r1 = M.CertificationRun()
        r2 = M.CertificationRun()
        self.assertNotEqual(r1.run_id, r2.run_id)


# ── cert_database ─────────────────────────────────────────────────────────────

class TestCertDatabase(unittest.TestCase):

    def setUp(self):
        DB = _db_mod()
        self.db = DB.CertDatabase()  # :memory:

    def test_tables_created(self):
        for tbl in ["certification_runs", "certification_results",
                     "certification_criteria", "benchmark_results",
                     "audit_packages", "production_approvals"]:
            count = self.db.table_count(tbl)
            self.assertGreaterEqual(count, 0, f"table {tbl} not accessible")

    def test_save_and_get_run(self):
        self.db.save_run({
            "run_id": "run-001", "model_id": "m1", "model_name": "TestModel",
            "started_at": time.time(),
        })
        rec = self.db.get_run("run-001")
        self.assertIsNotNone(rec)
        self.assertEqual(rec["run_id"], "run-001")

    def test_save_run_upsert(self):
        self.db.save_run({"run_id": "run-u1", "model_id": "m1", "started_at": time.time()})
        self.db.save_run({"run_id": "run-u1", "model_id": "m1", "started_at": time.time(),
                          "decision_status": "CERTIFIED", "overall_score": 85.0})
        rec = self.db.get_run("run-u1")
        self.assertEqual(rec["decision_status"], "CERTIFIED")

    def test_get_runs_for_model(self):
        for i in range(3):
            self.db.save_run({"run_id": f"run-m{i}", "model_id": "model-X",
                               "started_at": time.time()})
        runs = self.db.get_runs_for_model("model-X")
        self.assertEqual(len(runs), 3)

    def test_save_stage_result(self):
        self.db.save_run({"run_id": "run-s1", "model_id": "m1", "started_at": time.time()})
        row_id = self.db.save_stage_result({
            "run_id": "run-s1", "stage": "VALIDATION", "status": "PASS", "score": 90.0,
        })
        self.assertGreater(row_id, 0)

    def test_get_stage_results(self):
        self.db.save_run({"run_id": "run-s2", "model_id": "m1", "started_at": time.time()})
        for stage in ["RESEARCH", "VALIDATION", "BENCHMARK"]:
            self.db.save_stage_result({"run_id": "run-s2", "stage": stage,
                                        "status": "PASS", "score": 80.0})
        stages = self.db.get_stage_results("run-s2")
        self.assertEqual(len(stages), 3)

    def test_save_checklist_item(self):
        self.db.save_run({"run_id": "run-c1", "model_id": "m1", "started_at": time.time()})
        row_id = self.db.save_checklist_item("run-c1", {
            "check_id": "VL-001", "category": "VALIDATION",
            "description": "OOS Sharpe > 0", "mandatory": True,
            "status": "PASS", "risk_level": "HIGH",
        })
        self.assertGreater(row_id, 0)

    def test_get_checklist(self):
        self.db.save_run({"run_id": "run-c2", "model_id": "m1", "started_at": time.time()})
        for i in range(5):
            self.db.save_checklist_item("run-c2", {
                "check_id": f"VL-{i:03d}", "status": "PASS",
            })
        items = self.db.get_checklist("run-c2")
        self.assertEqual(len(items), 5)

    def test_save_benchmark(self):
        self.db.save_run({"run_id": "run-b1", "model_id": "m1", "started_at": time.time()})
        row_id = self.db.save_benchmark({
            "run_id": "run-b1", "metric": "INFERENCE_LATENCY",
            "value": 25.0, "unit": "ms", "threshold": 50.0, "status": "PASS",
        })
        self.assertGreater(row_id, 0)

    def test_get_benchmarks(self):
        self.db.save_run({"run_id": "run-b2", "model_id": "m1", "started_at": time.time()})
        for m in ["INFERENCE_LATENCY", "MEMORY_USAGE", "API_THROUGHPUT"]:
            self.db.save_benchmark({"run_id": "run-b2", "metric": m, "value": 1.0,
                                     "unit": "ms", "status": "PASS"})
        bm = self.db.get_benchmarks("run-b2")
        self.assertEqual(len(bm), 3)

    def test_save_audit_package(self):
        self.db.save_run({"run_id": "run-a1", "model_id": "m1", "started_at": time.time()})
        row_id = self.db.save_audit_package({
            "run_id": "run-a1", "executive_summary": "All good.",
            "generated_at": time.time(),
        })
        self.assertGreater(row_id, 0)

    def test_get_audit_package(self):
        self.db.save_run({"run_id": "run-a2", "model_id": "m1", "started_at": time.time()})
        self.db.save_audit_package({
            "run_id": "run-a2", "certification_report": "Certified.",
            "generated_at": time.time(),
        })
        pkg = self.db.get_audit_package("run-a2")
        self.assertIsNotNone(pkg)
        self.assertEqual(pkg["certification_report"], "Certified.")

    def test_save_approval(self):
        self.db.save_run({"run_id": "run-p1", "model_id": "m1", "started_at": time.time()})
        row_id = self.db.save_approval({
            "run_id": "run-p1", "model_id": "m1",
            "approved_by": "Head of Risk", "approval_status": "APPROVED",
        })
        self.assertGreater(row_id, 0)

    def test_get_approvals(self):
        self.db.save_run({"run_id": "run-p2", "model_id": "m-prod", "started_at": time.time()})
        self.db.save_approval({"run_id": "run-p2", "model_id": "m-prod",
                                "approved_by": "CTO", "approval_status": "APPROVED"})
        approvals = self.db.get_approvals("m-prod")
        self.assertEqual(len(approvals), 1)

    def test_get_run_not_found(self):
        rec = self.db.get_run("nonexistent")
        self.assertIsNone(rec)

    def test_get_latest_certified(self):
        for i, status in enumerate(["REJECTED", "CERTIFIED"]):
            self.db.save_run({
                "run_id": f"run-lc{i}", "model_id": "m-lc",
                "started_at": time.time() + i,
                "decision_status": status, "overall_score": 80.0,
            })
        rec = self.db.get_latest_certified("m-lc")
        self.assertIsNotNone(rec)
        self.assertEqual(rec["decision_status"], "CERTIFIED")

    def test_table_count(self):
        count = self.db.table_count("certification_runs")
        self.assertIsInstance(count, int)


# ── validation_suite ──────────────────────────────────────────────────────────

class TestValidationSuite(unittest.TestCase):

    def setUp(self):
        VS = _vs()
        self.suite = VS.ValidationSuite()

    def _inp(self, **kw):
        M = _models()
        defaults = dict(
            oos_sharpe=1.5, oos_win_rate=0.58, oos_max_drawdown=0.15,
            walk_forward_results=[{"sharpe": 1.2, "win_rate": 0.56}] * 4,
            cross_regime_results=[{"score": 65.0}] * 3,
            cross_instrument_results=[{"score": 60.0}] * 2,
            cross_timeframe_results=[{"score": 62.0}] * 2,
            generalization_score=68.0, stability_score=70.0,
            calibration_score=0.70, brier_score=0.20, ece=0.10,
        )
        defaults.update(kw)
        return M.CertificationInput(**defaults)

    def test_run_all_returns_8_results(self):
        results = self.suite.run_all(self._inp())
        self.assertEqual(len(results), 8)

    def test_all_checks_present(self):
        T = _types()
        results = self.suite.run_all(self._inp())
        checks = {r.check for r in results}
        for c in T.ValidationCheck:
            self.assertIn(c, checks)

    def test_oos_pass_good_metrics(self):
        results = self.suite.run_all(self._inp())
        oos = next(r for r in results if r.check.value == "OUT_OF_SAMPLE")
        self.assertIn(oos.status.value, ["PASS", "WARNING"])
        self.assertGreater(oos.score, 50.0)

    def test_oos_fail_negative_sharpe(self):
        results = self.suite.run_all(self._inp(oos_sharpe=-0.5, oos_win_rate=0.42))
        oos = next(r for r in results if r.check.value == "OUT_OF_SAMPLE")
        self.assertEqual(oos.status.value, "FAIL")

    def test_walk_forward_warning_no_data(self):
        results = self.suite.run_all(self._inp(walk_forward_results=[]))
        wf = next(r for r in results if r.check.value == "WALK_FORWARD")
        self.assertEqual(wf.status.value, "WARNING")

    def test_walk_forward_pass_with_data(self):
        results = self.suite.run_all(self._inp(
            walk_forward_results=[{"sharpe": 2.0, "win_rate": 0.65}] * 6
        ))
        wf = next(r for r in results if r.check.value == "WALK_FORWARD")
        self.assertIn(wf.status.value, ["PASS", "WARNING"])

    def test_cross_regime_na_no_data(self):
        results = self.suite.run_all(self._inp(cross_regime_results=[]))
        cr = next(r for r in results if r.check.value == "CROSS_REGIME")
        self.assertEqual(cr.status.value, "WARNING")

    def test_cross_instrument_na_no_data(self):
        results = self.suite.run_all(self._inp(cross_instrument_results=[]))
        ci = next(r for r in results if r.check.value == "CROSS_INSTRUMENT")
        self.assertEqual(ci.status.value, "NOT_APPLICABLE")

    def test_calibration_pass(self):
        results = self.suite.run_all(self._inp(
            calibration_score=0.90, brier_score=0.10, ece=0.05
        ))
        cal = next(r for r in results if r.check.value == "CALIBRATION")
        self.assertIn(cal.status.value, ["PASS", "WARNING"])

    def test_generalization_na_zero(self):
        results = self.suite.run_all(self._inp(generalization_score=0.0))
        gen = next(r for r in results if r.check.value == "GENERALIZATION")
        self.assertEqual(gen.status.value, "NOT_APPLICABLE")

    def test_stability_pass(self):
        results = self.suite.run_all(self._inp(stability_score=80.0))
        stab = next(r for r in results if r.check.value == "STABILITY")
        self.assertIn(stab.status.value, ["PASS", "WARNING"])

    def test_validation_results_have_scores(self):
        results = self.suite.run_all(self._inp())
        for r in results:
            self.assertGreaterEqual(r.score, 0.0)
            self.assertLessEqual(r.score, 100.0)


# ── stress_engine ─────────────────────────────────────────────────────────────

class TestStressEngine(unittest.TestCase):

    def setUp(self):
        SE = _se()
        self.engine = SE.StressEngine()

    def _inp(self, stress_results=None):
        M = _models()
        return M.CertificationInput(
            oos_sharpe=1.5, oos_win_rate=0.58, oos_max_drawdown=0.15,
            stress_results=stress_results or {},
        )

    def test_run_all_returns_10_results(self):
        results = self.engine.run_all(self._inp())
        self.assertEqual(len(results), 10)

    def test_all_scenarios_present(self):
        T = _types()
        results = self.engine.run_all(self._inp())
        scenarios = {r.scenario for r in results}
        for s in T.StressScenario:
            self.assertIn(s, scenarios)

    def test_simulation_produces_valid_degradation(self):
        results = self.engine.run_all(self._inp())
        for r in results:
            self.assertGreaterEqual(r.degradation_pct, 0.0)

    def test_provided_result_used_over_simulation(self):
        results = self.engine.run_all(self._inp(stress_results={
            "HIGH_VOLATILITY": {"degradation_pct": 5.0, "max_drawdown": 0.10,
                                 "sharpe_under_stress": 1.2},
        }))
        hv = next(r for r in results if r.scenario.value == "HIGH_VOLATILITY")
        self.assertAlmostEqual(hv.degradation_pct, 5.0)
        self.assertEqual(hv.status.value, "PASS")

    def test_flash_crash_high_degradation_fails(self):
        results = self.engine.run_all(self._inp(stress_results={
            "FLASH_CRASH": {"degradation_pct": 80.0, "max_drawdown": 0.50,
                             "sharpe_under_stress": -0.2},
        }))
        fc = next(r for r in results if r.scenario.value == "FLASH_CRASH")
        self.assertEqual(fc.status.value, "FAIL")

    def test_results_have_to_dict(self):
        results = self.engine.run_all(self._inp())
        for r in results:
            d = r.to_dict()
            self.assertIn("scenario", d)
            self.assertIn("status", d)
            self.assertIn("degradation_pct", d)

    def test_server_restart_low_degradation(self):
        results = self.engine.run_all(self._inp())
        sr = next(r for r in results if r.scenario.value == "SERVER_RESTART")
        # SERVER_RESTART has 2% degradation in simulation
        self.assertLess(sr.degradation_pct, 10.0)


# ── monte_carlo_engine ────────────────────────────────────────────────────────

class TestMonteCarloEngine(unittest.TestCase):

    def setUp(self):
        MCE = _mc()
        self.engine = MCE.MonteCarloEngine(iterations=100, seed=42)

    def _inp(self, monte_carlo_results=None):
        M = _models()
        return M.CertificationInput(
            oos_sharpe=1.5, oos_win_rate=0.58, oos_max_drawdown=0.15,
            monte_carlo_results=monte_carlo_results or {},
        )

    def test_run_all_returns_6_results(self):
        results = self.engine.run_all(self._inp())
        self.assertEqual(len(results), 6)

    def test_technique_names_correct(self):
        expected = {
            "RANDOMIZED_TRADE_ORDER", "BOOTSTRAP_SAMPLING",
            "SEQUENCE_PERTURBATION", "NOISE_INJECTION",
            "PARAMETER_SENSITIVITY", "CONFIDENCE_INTERVAL_ESTIMATION",
        }
        results = self.engine.run_all(self._inp())
        got = {r.technique for r in results}
        self.assertEqual(got, expected)

    def test_ci_low_lt_ci_high(self):
        results = self.engine.run_all(self._inp())
        for r in results:
            self.assertLessEqual(r.ci_low, r.ci_high)

    def test_prob_profitable_in_range(self):
        results = self.engine.run_all(self._inp())
        for r in results:
            self.assertGreaterEqual(r.prob_profitable, 0.0)
            self.assertLessEqual(r.prob_profitable, 1.0)

    def test_provided_result_overrides_simulation(self):
        provided = {
            "BOOTSTRAP_SAMPLING": {
                "mean_return": 0.05, "std_return": 0.02,
                "ci_low": 0.01, "ci_high": 0.09, "prob_profitable": 0.80,
            }
        }
        results = self.engine.run_all(self._inp(monte_carlo_results=provided))
        bs = next(r for r in results if r.technique == "BOOTSTRAP_SAMPLING")
        self.assertAlmostEqual(bs.prob_profitable, 0.80, places=3)
        self.assertEqual(bs.status.value, "PASS")

    def test_low_prob_profitable_fails(self):
        provided = {
            "NOISE_INJECTION": {
                "mean_return": -0.02, "std_return": 0.05,
                "ci_low": -0.30, "ci_high": -0.01, "prob_profitable": 0.20,
            }
        }
        results = self.engine.run_all(self._inp(monte_carlo_results=provided))
        ni = next(r for r in results if r.technique == "NOISE_INJECTION")
        self.assertEqual(ni.status.value, "FAIL")

    def test_to_dict_structure(self):
        results = self.engine.run_all(self._inp())
        for r in results:
            d = r.to_dict()
            self.assertIn("technique", d)
            self.assertIn("prob_profitable", d)
            self.assertIn("status", d)

    def test_seed_reproducibility(self):
        MCE = _mc()
        e1 = MCE.MonteCarloEngine(iterations=100, seed=7)
        e2 = MCE.MonteCarloEngine(iterations=100, seed=7)
        inp = self._inp()
        r1 = e1.run_all(inp)
        r2 = e2.run_all(inp)
        for a, b in zip(r1, r2):
            self.assertAlmostEqual(a.mean_return, b.mean_return, places=8)


# ── benchmark_engine ──────────────────────────────────────────────────────────

class TestBenchmarkEngine(unittest.TestCase):

    def setUp(self):
        BE = _be()
        self.engine = BE.BenchmarkEngine()

    def _inp(self, benchmark_values=None):
        M = _models()
        return M.CertificationInput(benchmark_values=benchmark_values or {})

    def test_run_all_returns_9_results(self):
        results = self.engine.run_all(self._inp())
        self.assertEqual(len(results), 9)

    def test_all_metrics_present(self):
        T = _types()
        results = self.engine.run_all(self._inp())
        metrics = {r.metric for r in results}
        for m in T.BenchmarkMetric:
            self.assertIn(m, metrics)

    def test_default_values_produce_pass(self):
        results = self.engine.run_all(self._inp())
        statuses = {r.status.value for r in results}
        # Default values should be good
        self.assertIn("PASS", statuses)

    def test_high_latency_fails(self):
        results = self.engine.run_all(self._inp({
            "INFERENCE_LATENCY": 200.0,
        }))
        lat = next(r for r in results if r.metric.value == "INFERENCE_LATENCY")
        self.assertEqual(lat.status.value, "FAIL")

    def test_low_latency_passes(self):
        results = self.engine.run_all(self._inp({
            "INFERENCE_LATENCY": 10.0,
        }))
        lat = next(r for r in results if r.metric.value == "INFERENCE_LATENCY")
        self.assertEqual(lat.status.value, "PASS")

    def test_low_api_throughput_fails(self):
        results = self.engine.run_all(self._inp({
            "API_THROUGHPUT": 20.0,   # need > 100 rps
        }))
        thr = next(r for r in results if r.metric.value == "API_THROUGHPUT")
        self.assertEqual(thr.status.value, "FAIL")

    def test_high_api_throughput_passes(self):
        results = self.engine.run_all(self._inp({
            "API_THROUGHPUT": 500.0,
        }))
        thr = next(r for r in results if r.metric.value == "API_THROUGHPUT")
        self.assertEqual(thr.status.value, "PASS")

    def test_overall_score_full_pass(self):
        bv = {
            "INFERENCE_LATENCY": 10.0, "TRAINING_TIME": 600.0,
            "GPU_USAGE": 60.0, "CPU_USAGE": 40.0, "MEMORY_USAGE": 2000.0,
            "DISK_USAGE": 200.0, "MODEL_SIZE": 64.0,
            "API_THROUGHPUT": 500.0, "RESPONSE_TIME": 30.0,
        }
        results = self.engine.run_all(self._inp(bv))
        score = self.engine.overall_score(results)
        self.assertGreater(score, 80.0)

    def test_overall_score_empty(self):
        score = self.engine.overall_score([])
        self.assertEqual(score, 50.0)


# ── reproducibility_engine ────────────────────────────────────────────────────

class TestReproducibilityEngine(unittest.TestCase):

    def setUp(self):
        RE = _re()
        self.engine = RE.ReproducibilityEngine()

    def _inp(self, **kw):
        M = _models()
        defaults = dict(
            dataset_version="v1.0", feature_version="v1.0", label_version="v1.0",
            training_config={"lr": 1e-3}, random_seed=42, git_commit="abc1234",
            environment="python3.11", package_versions={"numpy": "1.24", "torch": "2.0", "sklearn": "1.2"},
            checkpoint_hash="deadbeef01", model_hash="cafebabe02",
        )
        defaults.update(kw)
        return M.CertificationInput(**defaults)

    def test_run_all_returns_10_results(self):
        results = self.engine.run_all(self._inp())
        self.assertEqual(len(results), 10)

    def test_all_pass_with_good_input(self):
        results = self.engine.run_all(self._inp())
        failed = [r.check_name for r in results if r.status.value == "FAIL"]
        self.assertEqual(failed, [])

    def test_missing_dataset_version_fails(self):
        results = self.engine.run_all(self._inp(dataset_version=""))
        ds = next(r for r in results if r.check_name == "DATASET_VERSION")
        self.assertEqual(ds.status.value, "FAIL")

    def test_missing_random_seed_fails(self):
        results = self.engine.run_all(self._inp(random_seed=None))
        rs = next(r for r in results if r.check_name == "RANDOM_SEED")
        self.assertEqual(rs.status.value, "FAIL")

    def test_short_git_commit_fails(self):
        results = self.engine.run_all(self._inp(git_commit="abc"))
        gc = next(r for r in results if r.check_name == "GIT_COMMIT")
        self.assertEqual(gc.status.value, "FAIL")

    def test_short_model_hash_fails(self):
        results = self.engine.run_all(self._inp(model_hash="ab"))
        mh = next(r for r in results if r.check_name == "MODEL_HASH")
        self.assertEqual(mh.status.value, "FAIL")

    def test_short_checkpoint_hash_fails(self):
        results = self.engine.run_all(self._inp(checkpoint_hash="ab"))
        ch = next(r for r in results if r.check_name == "CHECKPOINT_INTEGRITY")
        self.assertEqual(ch.status.value, "FAIL")

    def test_missing_training_config_warning(self):
        results = self.engine.run_all(self._inp(training_config={}))
        tc = next(r for r in results if r.check_name == "TRAINING_CONFIG")
        self.assertEqual(tc.status.value, "WARNING")

    def test_few_packages_warning(self):
        results = self.engine.run_all(self._inp(package_versions={"torch": "2.0"}))
        pv = next(r for r in results if r.check_name == "PACKAGE_VERSIONS")
        self.assertEqual(pv.status.value, "WARNING")

    def test_overall_score_range(self):
        results = self.engine.run_all(self._inp())
        score = self.engine.overall_score(results)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)

    def test_overall_score_full_pass_high(self):
        results = self.engine.run_all(self._inp())
        score = self.engine.overall_score(results)
        self.assertGreater(score, 70.0)


# ── security_engine ───────────────────────────────────────────────────────────

class TestSecurityEngine(unittest.TestCase):

    def setUp(self):
        SE = _sec()
        self.engine = SE.SecurityEngine()

    def _inp(self, security_checks=None):
        M = _models()
        return M.CertificationInput(security_checks=security_checks or {})

    def _all_pass(self):
        return {
            "AUTHENTICATION": True, "AUTHORIZATION": True, "AUDIT_LOGGING": True,
            "API_SECURITY": True, "SECRETS_MANAGEMENT": True,
            "INPUT_VALIDATION": True, "DEPENDENCY_REVIEW": True, "ENCRYPTION": True,
        }

    def test_run_all_returns_8_results(self):
        results = self.engine.run_all(self._inp(self._all_pass()))
        self.assertEqual(len(results), 8)

    def test_all_pass_good_checks(self):
        results = self.engine.run_all(self._inp(self._all_pass()))
        failed = [r.check_name for r in results if r.status.value == "FAIL"]
        self.assertEqual(failed, [])

    def test_missing_check_is_warning(self):
        results = self.engine.run_all(self._inp({}))
        for r in results:
            self.assertEqual(r.status.value, "WARNING")

    def test_failed_authentication_fails(self):
        sc = self._all_pass()
        sc["AUTHENTICATION"] = False
        results = self.engine.run_all(self._inp(sc))
        auth = next(r for r in results if r.check_name == "AUTHENTICATION")
        self.assertEqual(auth.status.value, "FAIL")
        self.assertEqual(auth.risk_level.value, "CRITICAL")

    def test_overall_score_full_pass_high(self):
        results = self.engine.run_all(self._inp(self._all_pass()))
        score = self.engine.overall_score(results)
        self.assertGreater(score, 80.0)

    def test_overall_score_all_fail_low(self):
        sc = {k: False for k in self._all_pass()}
        results = self.engine.run_all(self._inp(sc))
        score = self.engine.overall_score(results)
        self.assertLess(score, 20.0)

    def test_remediation_provided_on_fail(self):
        sc = self._all_pass()
        sc["ENCRYPTION"] = False
        results = self.engine.run_all(self._inp(sc))
        enc = next(r for r in results if r.check_name == "ENCRYPTION")
        self.assertTrue(len(enc.remediation) > 0)


# ── operational_engine ────────────────────────────────────────────────────────

class TestOperationalEngine(unittest.TestCase):

    def setUp(self):
        OE = _oe()
        self.engine = OE.OperationalEngine()

    def _inp(self, operational_checks=None):
        M = _models()
        return M.CertificationInput(operational_checks=operational_checks or {})

    def _all_pass(self):
        return {
            "MONITORING_ENABLED": True, "ALERTS_CONFIGURED": True,
            "HEALTH_CHECKS_PASSING": True, "ROLLBACK_AVAILABLE": True,
            "BACKUPS_VERIFIED": True, "LOGGING_FUNCTIONAL": True,
            "DOCUMENTATION_COMPLETE": True, "RECOVERY_PROCEDURES_VERIFIED": True,
        }

    def test_run_all_returns_8_results(self):
        results = self.engine.run_all(self._inp(self._all_pass()))
        self.assertEqual(len(results), 8)

    def test_all_pass_good_checks(self):
        results = self.engine.run_all(self._inp(self._all_pass()))
        failed = [r.check_name for r in results if r.status.value == "FAIL"]
        self.assertEqual(failed, [])

    def test_missing_monitoring_warning(self):
        results = self.engine.run_all(self._inp({}))
        mon = next(r for r in results if r.check_name == "MONITORING_ENABLED")
        self.assertEqual(mon.status.value, "WARNING")

    def test_health_check_fail_is_fail(self):
        oc = self._all_pass()
        oc["HEALTH_CHECKS_PASSING"] = False
        results = self.engine.run_all(self._inp(oc))
        hc = next(r for r in results if r.check_name == "HEALTH_CHECKS_PASSING")
        self.assertEqual(hc.status.value, "FAIL")

    def test_overall_score_all_pass_high(self):
        results = self.engine.run_all(self._inp(self._all_pass()))
        score = self.engine.overall_score(results)
        self.assertGreater(score, 80.0)

    def test_overall_score_mandatory_fail_low(self):
        oc = {k: False for k in self._all_pass()}
        results = self.engine.run_all(self._inp(oc))
        score = self.engine.overall_score(results)
        # Mandatory failures → weight 0; non-mandatory → WARNING weight 55.
        # Score = (0*5 + 55*3) / 8 = 20.625 — still much lower than all-pass.
        self.assertLessEqual(score, 30.0)
        # All-pass should be significantly higher
        good = self.engine.run_all(self._inp(self._all_pass()))
        self.assertGreater(self.engine.overall_score(good), score)

    def test_action_provided_on_fail(self):
        oc = self._all_pass()
        oc["ROLLBACK_AVAILABLE"] = False
        results = self.engine.run_all(self._inp(oc))
        rb = next(r for r in results if r.check_name == "ROLLBACK_AVAILABLE")
        self.assertTrue(len(rb.action) > 0)


# ── checklist_engine ──────────────────────────────────────────────────────────

class TestChecklistEngine(unittest.TestCase):

    def setUp(self):
        CLE = _cl()
        self.engine = CLE.ChecklistEngine()

    def _build(self, **kw):
        M = _models()
        inp = M.CertificationInput(**kw) if kw else M.CertificationInput(
            model_id="m1", git_commit="abc1234",
            research_quality_score=80.0,
            oos_sharpe=1.5, oos_win_rate=0.60, oos_max_drawdown=0.15,
            calibration_score=0.70, brier_score=0.20, ece=0.10,
            dataset_version="v1", feature_version="v1", label_version="v1",
            random_seed=42, model_hash="deadbeef01", checkpoint_hash="cafebabe02",
            security_checks={"AUTHENTICATION": True, "AUTHORIZATION": True,
                              "SECRETS_MANAGEMENT": True, "API_SECURITY": True, "ENCRYPTION": True},
            operational_checks={"MONITORING_ENABLED": True, "ALERTS_CONFIGURED": True,
                                  "HEALTH_CHECKS_PASSING": True, "ROLLBACK_AVAILABLE": True,
                                  "LOGGING_FUNCTIONAL": True},
        )
        return self.engine.build(inp)

    def test_build_returns_report(self):
        report = self._build()
        M = _models()
        self.assertIsInstance(report, M.ChecklistReport)

    def test_can_proceed_good_input(self):
        report = self._build()
        self.assertTrue(report.can_proceed)

    def test_mandatory_failed_blocked(self):
        report = self.engine.build(_models().CertificationInput(
            model_id="",      # missing → RQ-002 FAIL
            git_commit="",    # missing → RQ-003 FAIL
            oos_sharpe=-1.0,  # negative → VL-001 FAIL
        ))
        self.assertFalse(report.can_proceed)
        self.assertGreater(report.mandatory_failed, 0)

    def test_pass_count_positive(self):
        report = self._build()
        self.assertGreater(report.pass_count, 0)

    def test_to_dict_structure(self):
        report = self._build()
        d = report.to_dict()
        self.assertIn("items", d)
        self.assertIn("can_proceed", d)
        self.assertIn("mandatory_failed", d)

    def test_items_have_check_id(self):
        report = self._build()
        for item in report.items:
            self.assertTrue(len(item.check_id) > 0)

    def test_security_mandatory_items_present(self):
        report = self._build()
        sec_items = [i for i in report.items if i.category == "SECURITY"]
        self.assertGreater(len(sec_items), 0)

    def test_reproducibility_items_present(self):
        report = self._build()
        repro_items = [i for i in report.items if i.category == "REPRODUCIBILITY"]
        self.assertGreater(len(repro_items), 0)

    def test_benchmark_item_na_when_no_data(self):
        report = self.engine.build(_models().CertificationInput(benchmark_values={}))
        bm_items = [i for i in report.items if i.category == "BENCHMARK"]
        self.assertTrue(any(i.status.value == "NOT_APPLICABLE" for i in bm_items))


# ── scorecard_engine ──────────────────────────────────────────────────────────

class TestScorecardEngine(unittest.TestCase):

    def setUp(self):
        SCE = _sc()
        self.engine = SCE.ScorecardEngine()

    def _compute(self, inp=None):
        M  = _models()
        T  = _types()
        VS = _vs()
        SE = _se()
        MCE= _mc()
        BE = _be()
        RE = _re()
        SEC= _sec()
        OE = _oe()
        if inp is None:
            inp = M.CertificationInput(
                model_id="m1", git_commit="abc1234",
                research_quality_score=80.0,
                oos_sharpe=1.5, oos_win_rate=0.60, oos_max_drawdown=0.15,
                walk_forward_results=[{"sharpe": 1.2, "win_rate": 0.56}] * 4,
                calibration_score=0.75, brier_score=0.18, ece=0.08,
                generalization_score=70.0, stability_score=72.0,
                notes="good", tags=["prod"], training_config={"lr": 1e-3},
                package_versions={"np": "1.24"}, environment="py3.11",
                security_checks={k: True for k in ["AUTHENTICATION","AUTHORIZATION",
                                                    "SECRETS_MANAGEMENT","API_SECURITY","ENCRYPTION"]},
                operational_checks={k: True for k in ["MONITORING_ENABLED","ALERTS_CONFIGURED",
                                                       "HEALTH_CHECKS_PASSING","ROLLBACK_AVAILABLE",
                                                       "LOGGING_FUNCTIONAL"]},
            )
        return self.engine.compute(
            inp,
            VS.ValidationSuite().run_all(inp),
            SE.StressEngine().run_all(inp),
            MCE.MonteCarloEngine(iterations=50, seed=1).run_all(inp),
            BE.BenchmarkEngine().run_all(inp),
            RE.ReproducibilityEngine().run_all(inp),
            SEC.SecurityEngine().run_all(inp),
            OE.OperationalEngine().run_all(inp),
        )

    def test_compute_returns_scorecard(self):
        sc = self._compute()
        M = _models()
        self.assertIsInstance(sc, M.ScorecardResult)

    def test_overall_score_in_range(self):
        sc = self._compute()
        self.assertGreaterEqual(sc.overall, 0.0)
        self.assertLessEqual(sc.overall, 100.0)

    def test_grade_assigned(self):
        sc = self._compute()
        self.assertIn(sc.grade, ["A+", "A", "B", "C", "D", "F"])

    def test_good_input_gives_decent_score(self):
        sc = self._compute()
        self.assertGreater(sc.overall, 50.0)

    def test_all_categories_set(self):
        sc = self._compute()
        for attr in ["research_quality", "validation", "performance", "calibration",
                     "robustness", "security", "operational", "documentation", "reproducibility"]:
            val = getattr(sc, attr)
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 100.0)

    def test_to_dict_has_all_keys(self):
        sc = self._compute()
        d = sc.to_dict()
        for k in ["research_quality", "validation", "performance", "overall", "grade"]:
            self.assertIn(k, d)

    def test_grade_a_plus_high_score(self):
        SCE = _sc()
        eng = SCE.ScorecardEngine()
        self.assertEqual(SCE._grade(95.0), "A+")
        self.assertEqual(SCE._grade(85.0), "A")
        self.assertEqual(SCE._grade(75.0), "B")
        self.assertEqual(SCE._grade(55.0), "D")
        self.assertEqual(SCE._grade(30.0), "F")


# ── cert_dashboard ────────────────────────────────────────────────────────────

class TestCertDashboard(unittest.TestCase):

    def setUp(self):
        DASH = _dash()
        self.dashboard = DASH.CertDashboard()

    def _run(self):
        M = _models()
        T = _types()
        run = M.CertificationRun(model_id="m1", model_name="Test", model_version="1.0.0")
        run.scorecard = M.ScorecardResult(overall=82.0, grade="A")
        run.decision  = M.CertificationDecision(
            status=T.CertificationStatus.CERTIFIED, rationale="All passed."
        )
        run.validations = [M.ValidationResult(score=80.0)]
        run.stress_tests = [M.StressTestResult(degradation_pct=10.0)]
        run.monte_carlos = [M.MonteCarloResult(prob_profitable=0.65)]
        run.benchmarks   = [M.BenchmarkResult(value=25.0, unit="ms")]
        run.security     = [M.SecurityResult(check_name="AUTH", status=T.CheckStatus.PASS)]
        run.operational  = [M.OperationalResult(check_name="MONITORING", status=T.CheckStatus.PASS)]
        run.reproducibility = [M.ReproducibilityResult(check_name="MODEL_HASH", status=T.CheckStatus.PASS)]
        run.audit = M.AuditPackage(executive_summary="All clear.")
        run.checklist = M.ChecklistReport(pass_count=10, can_proceed=True)
        return run

    def test_generate_all_returns_5_dashboards(self):
        dashboards = self.dashboard.generate_all(self._run())
        self.assertEqual(len(dashboards), 5)

    def test_certification_dashboard_has_status(self):
        d = self.dashboard.certification(self._run())
        self.assertIn("status", d)
        self.assertEqual(d["status"], "CERTIFIED")

    def test_certification_dashboard_has_score(self):
        d = self.dashboard.certification(self._run())
        self.assertAlmostEqual(d["overall_score"], 82.0)

    def test_readiness_dashboard_can_proceed(self):
        d = self.dashboard.readiness(self._run())
        self.assertTrue(d["can_proceed"])

    def test_validation_dashboard_summary(self):
        d = self.dashboard.validation(self._run())
        self.assertIn("validation_summary", d)
        self.assertIn("mc_summary", d)

    def test_benchmark_dashboard_has_stress(self):
        d = self.dashboard.benchmark(self._run())
        self.assertIn("stress_tests", d)
        self.assertIn("benchmarks", d)

    def test_audit_dashboard_has_repro(self):
        d = self.dashboard.audit(self._run())
        self.assertIn("reproducibility", d)
        self.assertIn("stage_audit", d)

    def test_all_dashboards_have_generated_at(self):
        dashboards = self.dashboard.generate_all(self._run())
        for key, d in dashboards.items():
            self.assertIn("generated_at", d, f"Missing generated_at in {key}")


# ── cert_api ──────────────────────────────────────────────────────────────────

class TestCertAPI(unittest.TestCase):

    def _api(self):
        DB = _db_mod()
        API = _api_mod()
        return API.CertificationAPI(db=DB.CertDatabase())

    def _good_inp(self):
        M = _models()
        return M.CertificationInput(
            model_id="mdl-good", model_name="GoodModel", model_version="1.0.0",
            git_commit="abc1234def567",
            research_quality_score=85.0,
            oos_sharpe=2.0, oos_win_rate=0.62, oos_max_drawdown=0.10,
            walk_forward_results=[{"sharpe": 1.8, "win_rate": 0.60}] * 5,
            cross_regime_results=[{"score": 72.0}] * 4,
            calibration_score=0.80, brier_score=0.15, ece=0.07,
            generalization_score=75.0, stability_score=78.0,
            dataset_version="v3.0", feature_version="v2.0", label_version="v1.5",
            training_config={"lr": 1e-3, "epochs": 200},
            random_seed=42,
            environment="python3.11-cuda",
            package_versions={"torch": "2.1", "numpy": "1.25", "sklearn": "1.3"},
            checkpoint_hash="deadbeef0102",
            model_hash="cafebabe0304",
            security_checks={k: True for k in [
                "AUTHENTICATION","AUTHORIZATION","AUDIT_LOGGING","API_SECURITY",
                "SECRETS_MANAGEMENT","INPUT_VALIDATION","DEPENDENCY_REVIEW","ENCRYPTION"
            ]},
            operational_checks={k: True for k in [
                "MONITORING_ENABLED","ALERTS_CONFIGURED","HEALTH_CHECKS_PASSING",
                "ROLLBACK_AVAILABLE","BACKUPS_VERIFIED","LOGGING_FUNCTIONAL",
                "DOCUMENTATION_COMPLETE","RECOVERY_PROCEDURES_VERIFIED"
            ]},
            benchmark_values={
                "INFERENCE_LATENCY": 15.0, "TRAINING_TIME": 900.0,
                "GPU_USAGE": 65.0, "CPU_USAGE": 50.0, "MEMORY_USAGE": 2048.0,
                "DISK_USAGE": 256.0, "MODEL_SIZE": 64.0,
                "API_THROUGHPUT": 400.0, "RESPONSE_TIME": 40.0,
            },
            notes="Full production candidate.", tags=["production"],
        )

    def _poor_inp(self):
        M = _models()
        return M.CertificationInput(
            model_id="",         # missing
            git_commit="",       # missing
            research_quality_score=20.0,
            oos_sharpe=-0.5, oos_win_rate=0.40, oos_max_drawdown=0.60,
        )

    def test_certify_returns_run(self):
        api = self._api()
        M   = _models()
        run = api.certify(self._good_inp())
        self.assertIsInstance(run, M.CertificationRun)

    def test_certify_good_model_certified(self):
        api = self._api()
        T   = _types()
        run = api.certify(self._good_inp())
        self.assertIsNotNone(run.decision)
        self.assertIn(run.decision.status, [
            T.CertificationStatus.CERTIFIED,
            T.CertificationStatus.CERTIFIED_WITH_CONDITIONS,
        ])

    def test_certify_poor_model_rejected(self):
        api = self._api()
        T   = _types()
        run = api.certify(self._poor_inp())
        self.assertIsNotNone(run.decision)
        self.assertEqual(run.decision.status, T.CertificationStatus.REJECTED)

    def test_certify_has_13_stages(self):
        api = self._api()
        run = api.certify(self._good_inp())
        self.assertEqual(len(run.stages), 13)

    def test_certify_has_scorecard(self):
        api = self._api()
        run = api.certify(self._good_inp())
        self.assertIsNotNone(run.scorecard)

    def test_certify_has_checklist(self):
        api = self._api()
        run = api.certify(self._good_inp())
        self.assertIsNotNone(run.checklist)

    def test_certify_has_audit(self):
        api = self._api()
        run = api.certify(self._good_inp())
        self.assertIsNotNone(run.audit)
        self.assertTrue(len(run.audit.executive_summary) > 0)

    def test_certify_has_validations(self):
        api = self._api()
        run = api.certify(self._good_inp())
        self.assertEqual(len(run.validations), 8)

    def test_certify_has_stress_tests(self):
        api = self._api()
        run = api.certify(self._good_inp())
        self.assertEqual(len(run.stress_tests), 10)

    def test_certify_has_monte_carlos(self):
        api = self._api()
        run = api.certify(self._good_inp())
        self.assertEqual(len(run.monte_carlos), 6)

    def test_certify_has_benchmarks(self):
        api = self._api()
        run = api.certify(self._good_inp())
        self.assertEqual(len(run.benchmarks), 9)

    def test_certify_has_reproducibility(self):
        api = self._api()
        run = api.certify(self._good_inp())
        self.assertEqual(len(run.reproducibility), 10)

    def test_certify_has_security(self):
        api = self._api()
        run = api.certify(self._good_inp())
        self.assertEqual(len(run.security), 8)

    def test_certify_has_operational(self):
        api = self._api()
        run = api.certify(self._good_inp())
        self.assertEqual(len(run.operational), 8)

    def test_certify_run_persisted(self):
        api = self._api()
        run = api.certify(self._good_inp())
        rec = api._db.get_run(run.run_id)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["run_id"], run.run_id)

    def test_certify_stages_persisted(self):
        api = self._api()
        run = api.certify(self._good_inp())
        stages = api._db.get_stage_results(run.run_id)
        self.assertEqual(len(stages), 13)

    def test_certify_benchmarks_persisted(self):
        api = self._api()
        run = api.certify(self._good_inp())
        bm = api._db.get_benchmarks(run.run_id)
        self.assertEqual(len(bm), 9)

    def test_certify_audit_persisted(self):
        api = self._api()
        run = api.certify(self._good_inp())
        pkg = api._db.get_audit_package(run.run_id)
        self.assertIsNotNone(pkg)

    def test_get_certification_status_found(self):
        api = self._api()
        run = api.certify(self._good_inp())
        status = api.get_certification_status(run.run_id)
        self.assertIn("decision_status", status)
        self.assertEqual(status["run_id"], run.run_id)

    def test_get_certification_status_not_found(self):
        api = self._api()
        status = api.get_certification_status("nonexistent")
        self.assertIn("error", status)

    def test_get_certification_report_found(self):
        api = self._api()
        run = api.certify(self._good_inp())
        report = api.get_certification_report(run.run_id)
        self.assertIn("run", report)
        self.assertIn("stages", report)

    def test_get_audit_package_found(self):
        api = self._api()
        run = api.certify(self._good_inp())
        pkg = api.get_audit_package(run.run_id)
        self.assertNotIn("error", pkg)

    def test_get_audit_package_not_found(self):
        api = self._api()
        pkg = api.get_audit_package("bad-run")
        self.assertIn("error", pkg)

    def test_get_benchmark_results_found(self):
        api = self._api()
        run = api.certify(self._good_inp())
        bm = api.get_benchmark_results(run.run_id)
        self.assertIn("benchmarks", bm)
        self.assertEqual(len(bm["benchmarks"]), 9)

    def test_get_validation_results_found(self):
        api = self._api()
        run = api.certify(self._good_inp())
        val = api.get_validation_results(run.run_id)
        self.assertIn("validation_stages", val)

    def test_to_dict_structure(self):
        api = self._api()
        run = api.certify(self._good_inp())
        d = run.to_dict()
        self.assertIn("run_id", d)
        self.assertIn("decision", d)
        self.assertIn("scorecard", d)

    def test_non_certified_status_on_medium_score(self):
        T = _types()
        api = self._api()
        M  = _models()
        inp = M.CertificationInput(
            model_id="mdl-med", git_commit="abc1234",
            research_quality_score=55.0,
            oos_sharpe=0.8, oos_win_rate=0.52, oos_max_drawdown=0.25,
            calibration_score=0.55, brier_score=0.22, ece=0.12,
            dataset_version="v1", feature_version="v1", label_version="v1",
            random_seed=1, checkpoint_hash="aabbccdd", model_hash="11223344",
            security_checks={k: True for k in [
                "AUTHENTICATION","AUTHORIZATION","SECRETS_MANAGEMENT",
                "API_SECURITY","ENCRYPTION","AUDIT_LOGGING","INPUT_VALIDATION","DEPENDENCY_REVIEW"
            ]},
            operational_checks={k: True for k in [
                "MONITORING_ENABLED","ALERTS_CONFIGURED","HEALTH_CHECKS_PASSING",
                "ROLLBACK_AVAILABLE","LOGGING_FUNCTIONAL"
            ]},
        )
        run = api.certify(inp)
        # A medium-quality model should not receive full CERTIFIED status.
        # Any other outcome (PILOT, RESEARCH, REJECTED, WITH_CONDITIONS) is valid.
        self.assertIsNotNone(run.decision)
        self.assertNotEqual(run.decision.status, T.CertificationStatus.CERTIFIED)

    def test_completed_at_set(self):
        api = self._api()
        run = api.certify(self._good_inp())
        self.assertIsNotNone(run.completed_at)
        self.assertGreater(run.completed_at, run.started_at)


if __name__ == "__main__":
    unittest.main()
