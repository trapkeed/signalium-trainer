"""Phase 29 — Learning Loop Actuators Test Suite.

Wires the six observation-only modules (SelfEvaluationEngine, PatternDiscovery,
CounterfactualEngine, LifecycleOptimizer, PortfolioOptimizer, OutcomeTracker)
into concrete downstream actions and verifies:

  - Each action kind (feature discovery, retraining recommendation, dataset
    enrichment, experiment proposal, parameter search, investigation) is
    reachable from the appropriate observation module.
  - Nothing in the actuator layer ever mutates trading logic directly — all
    actions land as records in ResearchLabAPI or as advisory
    RetrainingRecommendation objects.

`trainer.research` cannot be imported normally in this environment because
its package __init__ eagerly imports trainer.research.benchmark, which in
turn imports torch (broken native lib in this sandbox). We pre-register a
stub `trainer.research` package and standalone-load retraining_advisor.py
directly from its file so trainer.learning_loop.actuators's normal
`from trainer.research.retraining_advisor import ...` resolves against the
cached module instead of re-triggering the broken package __init__.
"""

from __future__ import annotations

import importlib.util
import sys
import time
import types
import unittest
from pathlib import Path

# ── Stub loguru before any imports ──────────────────────────────────────────

_loguru = types.ModuleType("loguru")


class _Logger:
    def info(self, *a, **k):
        pass

    def debug(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def warn(self, *a, **k):
        pass

    def success(self, *a, **k):
        pass

    def trace(self, *a, **k):
        pass

    def critical(self, *a, **k):
        pass

    def exception(self, *a, **k):
        pass

    def bind(self, **k):
        return self


_loguru.logger = _Logger()
sys.modules.setdefault("loguru", _loguru)

ROOT = Path(__file__).resolve().parents[2]


def _torch_is_broken() -> bool:
    """Detect whether torch's native extension is actually unusable here.

    Only bypass trainer.research's real package __init__ when torch is
    genuinely broken. If a real, working torch is installed, importing
    trainer.research normally works fine — and installing a bare stub
    ModuleType in sys.modules["trainer.research"] here would otherwise
    poison every later test in the same pytest session that needs the
    real package (e.g. `from trainer.research import BenchmarkConditions`),
    since a stub has no proper module contents.
    """
    if "torch" in sys.modules and not isinstance(sys.modules["torch"], types.ModuleType):
        return True
    try:
        import torch as _t

        _t.tensor([0])
        return False
    except Exception:
        return True


def _load_standalone(mod_key: str, path: Path):
    spec = importlib.util.spec_from_file_location(mod_key, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    spec.loader.exec_module(mod)
    return mod


if _torch_is_broken():
    # ── Bypass trainer.research's torch-dependent __init__.py ──────────────
    _research_pkg = types.ModuleType("trainer.research")
    _research_pkg.__path__ = [str(ROOT / "trainer" / "research")]
    sys.modules.setdefault("trainer.research", _research_pkg)

    _retraining_advisor = _load_standalone(
        "trainer.research.retraining_advisor",
        ROOT / "trainer" / "research" / "retraining_advisor.py",
    )
else:
    from trainer.research import retraining_advisor as _retraining_advisor

# trainer.memory / trainer.lifecycle / trainer.portfolio / trainer.research_lab
# all import cleanly without torch in this environment, so normal imports
# (which will also satisfy trainer.learning_loop.actuators's own imports via
# sys.modules caching) are used from here on.

from trainer.learning_loop.actuators import (  # noqa: E402
    ActionType,
    ActuatorAction,
    ActuatorCycleResult,
    CapitalUtilizationReport,
    LearningLoopActuator,
    ScalingOutcome,
    ScalingReport,
    SlippageRecord,
    SlippageReport,
)
from trainer.learning_loop.scheduler import LearningLoopScheduler  # noqa: E402
from trainer.lifecycle.lifecycle_optimizer import (  # noqa: E402
    OptimisationFinding,
    OptimisationReport,
)
from trainer.memory.decision_models import CounterfactualResult  # noqa: E402
from trainer.memory.outcome_tracker import OutcomeSummary  # noqa: E402
from trainer.memory.pattern_discovery import DiscoveredPattern, PatternReport  # noqa: E402
from trainer.memory.self_evaluation import SelfAssessment  # noqa: E402
from trainer.portfolio.optimizer import OptimizationResult  # noqa: E402
from trainer.portfolio.portfolio_scorer import PortfolioMetrics  # noqa: E402
from trainer.research_lab.research_lab_api import ResearchLabAPI  # noqa: E402


def _make_actuator() -> LearningLoopActuator:
    from trainer.research_lab.rl_database import ResearchLabDatabase

    lab = ResearchLabAPI(db=ResearchLabDatabase(db_path=":memory:"))
    advisor = _retraining_advisor.build_default_advisor()
    return LearningLoopActuator(
        research_lab=lab, retraining_advisor=advisor, model_id="qde-test", qde_version="v1.0.0"
    )


def _metrics(diversification_score: float = 0.8) -> PortfolioMetrics:
    return PortfolioMetrics(
        expected_return=0.10,
        expected_volatility=0.15,
        expected_drawdown=0.08,
        expected_sharpe=1.2,
        expected_sortino=1.5,
        diversification_score=diversification_score,
        correlation_risk=0.2,
        concentration_score=0.3,
        instrument_count=4,
        long_count=3,
        short_count=1,
        neutral_count=0,
        weighted_confidence=0.7,
    )


# ══════════════════════════════════════════════════════════════════════════
# 1. SelfEvaluationEngine → SelfAssessment
# ══════════════════════════════════════════════════════════════════════════
class TestSelfAssessmentActuator(unittest.TestCase):

    def setUp(self):
        self.actuator = _make_actuator()

    def test_low_quality_triggers_retraining_recommendation(self):
        assessment = SelfAssessment(
            period="2026-06",
            total_decisions=50,
            decisions_with_outcome=50,
            overall_win_rate=0.55,
            overall_quality_score=0.30,
            grade="POOR",
        )
        actions = self.actuator.act_on_self_assessment(assessment)
        retraining = [a for a in actions if a.action_type == ActionType.RETRAINING_RECOMMENDATION]
        self.assertTrue(retraining, "Expected a retraining recommendation for low quality score")
        self.assertTrue(all(a.success for a in retraining))

    def test_low_win_rate_triggers_retraining_recommendation(self):
        assessment = SelfAssessment(
            period="2026-06",
            overall_win_rate=0.20,
            overall_quality_score=0.75,
            grade="ACCEPTABLE",
        )
        actions = self.actuator.act_on_self_assessment(assessment)
        self.assertTrue(any(a.action_type == ActionType.RETRAINING_RECOMMENDATION for a in actions))

    def test_overconfidence_weakness_triggers_confidence_degradation_signal(self):
        assessment = SelfAssessment(
            period="2026-06",
            overall_win_rate=0.70,
            overall_quality_score=0.80,
            weaknesses=["Shows signs of overconfidence in trending regimes"],
        )
        actions = self.actuator.act_on_self_assessment(assessment)
        retraining = [a for a in actions if a.action_type == ActionType.RETRAINING_RECOMMENDATION]
        self.assertTrue(retraining)

    def test_failure_modes_open_investigations(self):
        assessment = SelfAssessment(
            period="2026-06",
            overall_win_rate=0.70,
            overall_quality_score=0.80,
            failure_modes=["Repeated missed opportunities during London session opens"],
        )
        actions = self.actuator.act_on_self_assessment(assessment)
        investigations = [a for a in actions if a.action_type == ActionType.INVESTIGATION]
        self.assertEqual(len(investigations), 1)
        self.assertTrue(investigations[0].success)

    def test_placeholder_failure_mode_is_ignored(self):
        assessment = SelfAssessment(
            period="2026-06",
            overall_win_rate=0.70,
            overall_quality_score=0.80,
            failure_modes=["No recurring failure modes identified"],
        )
        actions = self.actuator.act_on_self_assessment(assessment)
        self.assertFalse([a for a in actions if a.action_type == ActionType.INVESTIGATION])

    def test_healthy_assessment_produces_no_actions(self):
        assessment = SelfAssessment(
            period="2026-06", overall_win_rate=0.70, overall_quality_score=0.85, grade="GOOD"
        )
        actions = self.actuator.act_on_self_assessment(assessment)
        self.assertEqual(actions, [])


# ══════════════════════════════════════════════════════════════════════════
# 2. PatternDiscovery → PatternReport
# ══════════════════════════════════════════════════════════════════════════
class TestPatternReportActuator(unittest.TestCase):

    def setUp(self):
        self.actuator = _make_actuator()

    def test_strong_positive_pattern_triggers_feature_discovery(self):
        pattern = DiscoveredPattern(
            pattern_id="p1",
            category="SESSION",
            description="London open longs outperform",
            evidence="72% win rate over 40 trades",
            win_rate=0.72,
            avg_pnl=0.02,
            sample_size=40,
            strength="STRONG",
            direction="POSITIVE",
        )
        report = PatternReport(total_decisions=200, patterns=[pattern])
        actions = self.actuator.act_on_pattern_report(report)
        features = [a for a in actions if a.action_type == ActionType.FEATURE_DISCOVERY]
        self.assertEqual(len(features), 1)
        self.assertTrue(features[0].success)

    def test_strong_negative_regime_pattern_triggers_retraining_signal(self):
        pattern = DiscoveredPattern(
            pattern_id="p2",
            category="REGIME",
            description="High volatility regime underperforms",
            evidence="25% win rate over 30 trades",
            win_rate=0.25,
            avg_pnl=-0.015,
            sample_size=30,
            strength="STRONG",
            direction="NEGATIVE",
        )
        report = PatternReport(total_decisions=200, patterns=[pattern])
        actions = self.actuator.act_on_pattern_report(report)
        self.assertTrue(any(a.action_type == ActionType.RETRAINING_RECOMMENDATION for a in actions))

    def test_weak_pattern_with_small_sample_triggers_dataset_enrichment(self):
        pattern = DiscoveredPattern(
            pattern_id="p3",
            category="INSTRUMENT",
            description="XAUUSD shorts show mild edge",
            evidence="55% win rate over 6 trades",
            win_rate=0.55,
            avg_pnl=0.005,
            sample_size=6,
            strength="WEAK",
            direction="POSITIVE",
        )
        report = PatternReport(total_decisions=200, patterns=[pattern])
        actions = self.actuator.act_on_pattern_report(report)
        enrichment = [a for a in actions if a.action_type == ActionType.DATASET_ENRICHMENT]
        self.assertEqual(len(enrichment), 1)
        self.assertTrue(enrichment[0].success)

    def test_moderate_pattern_produces_no_action(self):
        pattern = DiscoveredPattern(
            pattern_id="p4",
            category="TIMING",
            description="Mid-session entries slightly better",
            evidence="60% win rate over 15 trades",
            win_rate=0.60,
            avg_pnl=0.01,
            sample_size=15,
            strength="MODERATE",
            direction="POSITIVE",
        )
        report = PatternReport(total_decisions=200, patterns=[pattern])
        actions = self.actuator.act_on_pattern_report(report)
        self.assertEqual(actions, [])


# ══════════════════════════════════════════════════════════════════════════
# 3. CounterfactualEngine → list[CounterfactualResult]
# ══════════════════════════════════════════════════════════════════════════
class TestCounterfactualActuator(unittest.TestCase):

    def setUp(self):
        self.actuator = _make_actuator()

    def _batch(self, alt_action: str, n: int, better_ratio: float):
        n_better = int(round(n * better_ratio))
        batches = []
        for i in range(n):
            better = i < n_better
            batches.append(
                [
                    CounterfactualResult(
                        scenario=f"{alt_action} instead of actual",
                        alternative_action=alt_action,
                        better_than_actual=better,
                        delta_pnl=0.01 if better else -0.005,
                    )
                ]
            )
        return batches

    def test_systematic_tighter_sl_edge_triggers_parameter_search(self):
        batches = self._batch("TIGHTER_SL", n=10, better_ratio=0.7)
        actions = self.actuator.act_on_counterfactual_batch(batches)
        searches = [a for a in actions if a.action_type == ActionType.PARAMETER_SEARCH]
        self.assertEqual(len(searches), 1)
        self.assertTrue(searches[0].success)

    def test_insufficient_sample_produces_no_action(self):
        batches = self._batch("LARGER_TP", n=3, better_ratio=1.0)
        actions = self.actuator.act_on_counterfactual_batch(batches)
        self.assertEqual(actions, [])

    def test_below_threshold_ratio_produces_no_action(self):
        batches = self._batch("EARLY_TRAIL", n=10, better_ratio=0.3)
        actions = self.actuator.act_on_counterfactual_batch(batches)
        self.assertEqual(actions, [])

    def test_systematic_wait_edge_triggers_retraining_signal(self):
        batches = self._batch("WAIT", n=10, better_ratio=0.8)
        actions = self.actuator.act_on_counterfactual_batch(batches)
        self.assertTrue(any(a.action_type == ActionType.RETRAINING_RECOMMENDATION for a in actions))


# ══════════════════════════════════════════════════════════════════════════
# 4. LifecycleOptimizer → OptimisationReport
# ══════════════════════════════════════════════════════════════════════════
class TestLifecycleReportActuator(unittest.TestCase):

    def setUp(self):
        self.actuator = _make_actuator()

    def test_risk_finding_triggers_investigation(self):
        finding = OptimisationFinding(
            finding_id="f1",
            category="RISK",
            recommendation="Emergency exits cluster around news events",
            evidence="8 of 10 emergency exits within 5m of high-impact news",
            expected_impact="NEGATIVE",
            confidence=0.8,
            priority=1,
        )
        report = OptimisationReport(total_analysed=50, findings=[finding])
        actions = self.actuator.act_on_lifecycle_report(report)
        self.assertTrue(any(a.action_type == ActionType.INVESTIGATION for a in actions))

    def test_confident_positive_finding_triggers_parameter_search(self):
        finding = OptimisationFinding(
            finding_id="f2",
            category="TRAILING",
            recommendation="Activate trailing earlier",
            evidence="Backtests show +12% on trades with early trailing",
            expected_impact="POSITIVE",
            confidence=0.75,
            priority=2,
        )
        report = OptimisationReport(total_analysed=50, findings=[finding])
        actions = self.actuator.act_on_lifecycle_report(report)
        self.assertTrue(any(a.action_type == ActionType.PARAMETER_SEARCH for a in actions))

    def test_top_priority_with_enough_samples_triggers_experiment(self):
        finding = OptimisationFinding(
            finding_id="f3",
            category="BREAKEVEN",
            recommendation="Move to breakeven sooner",
            evidence="Reduces avg loss by 15%",
            expected_impact="POSITIVE",
            confidence=0.9,
            priority=1,
        )
        report = OptimisationReport(
            total_analysed=40, findings=[finding], top_priority="f3", summary="Breakeven tweak"
        )
        actions = self.actuator.act_on_lifecycle_report(report)
        self.assertTrue(any(a.action_type == ActionType.EXPERIMENT_PROPOSAL for a in actions))

    def test_low_confidence_finding_produces_no_search(self):
        finding = OptimisationFinding(
            finding_id="f4",
            category="SCALING",
            recommendation="Maybe scale in more aggressively",
            evidence="Thin sample",
            expected_impact="POSITIVE",
            confidence=0.30,
            priority=3,
        )
        report = OptimisationReport(total_analysed=50, findings=[finding])
        actions = self.actuator.act_on_lifecycle_report(report)
        self.assertEqual(actions, [])


# ══════════════════════════════════════════════════════════════════════════
# 5. PortfolioOptimizer → OptimizationResult
# ══════════════════════════════════════════════════════════════════════════
class TestPortfolioResultActuator(unittest.TestCase):

    def setUp(self):
        self.actuator = _make_actuator()

    def test_improved_objective_triggers_experiment_proposal(self):
        result = OptimizationResult(
            objective="SHARPE",
            optimal_weights={"EURUSD": 0.5, "XAUUSD": 0.5},
            optimal_metrics=_metrics(),
            iterations=40,
            converged=True,
            objective_value=1.5,
        )
        actions = self.actuator.act_on_portfolio_result(result, baseline_objective_value=1.0)
        self.assertTrue(any(a.action_type == ActionType.EXPERIMENT_PROPOSAL for a in actions))

    def test_non_convergence_triggers_investigation(self):
        result = OptimizationResult(
            objective="SHARPE",
            optimal_weights={"EURUSD": 1.0},
            optimal_metrics=_metrics(),
            iterations=200,
            converged=False,
            objective_value=0.5,
        )
        actions = self.actuator.act_on_portfolio_result(result)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, ActionType.INVESTIGATION)

    def test_low_diversification_triggers_dataset_enrichment(self):
        result = OptimizationResult(
            objective="SHARPE",
            optimal_weights={"EURUSD": 1.0},
            optimal_metrics=_metrics(diversification_score=0.1),
            iterations=40,
            converged=True,
            objective_value=1.0,
        )
        actions = self.actuator.act_on_portfolio_result(result)
        self.assertTrue(any(a.action_type == ActionType.DATASET_ENRICHMENT for a in actions))

    def test_no_baseline_and_healthy_diversification_produces_no_action(self):
        result = OptimizationResult(
            objective="SHARPE",
            optimal_weights={"EURUSD": 0.5, "XAUUSD": 0.5},
            optimal_metrics=_metrics(diversification_score=0.8),
            iterations=40,
            converged=True,
            objective_value=1.2,
        )
        actions = self.actuator.act_on_portfolio_result(result)
        self.assertEqual(actions, [])


# ══════════════════════════════════════════════════════════════════════════
# 6. OutcomeTracker → OutcomeSummary
# ══════════════════════════════════════════════════════════════════════════
class TestOutcomeSummaryActuator(unittest.TestCase):

    def setUp(self):
        self.actuator = _make_actuator()

    def test_low_win_rate_triggers_retraining_recommendation(self):
        summary = OutcomeSummary(total=100, with_outcome=100, win_rate=0.20, hit_sl_count=10)
        actions = self.actuator.act_on_outcome_summary(summary)
        self.assertTrue(any(a.action_type == ActionType.RETRAINING_RECOMMENDATION for a in actions))

    def test_high_sl_hit_rate_triggers_investigation(self):
        summary = OutcomeSummary(total=100, with_outcome=100, win_rate=0.55, hit_sl_count=55)
        actions = self.actuator.act_on_outcome_summary(summary)
        self.assertTrue(any(a.action_type == ActionType.INVESTIGATION for a in actions))

    def test_undersampled_instrument_triggers_dataset_enrichment(self):
        summary = OutcomeSummary(
            total=100,
            with_outcome=100,
            win_rate=0.60,
            hit_sl_count=10,
            by_instrument={"XAUUSD": {"count": 5, "win_rate": 0.6}},
        )
        actions = self.actuator.act_on_outcome_summary(summary)
        enrichment = [a for a in actions if a.action_type == ActionType.DATASET_ENRICHMENT]
        self.assertEqual(len(enrichment), 1)

    def test_no_outcomes_produces_no_action(self):
        summary = OutcomeSummary(total=10, with_outcome=0)
        actions = self.actuator.act_on_outcome_summary(summary)
        self.assertEqual(actions, [])

    def test_healthy_summary_produces_no_action(self):
        summary = OutcomeSummary(
            total=100,
            with_outcome=100,
            win_rate=0.65,
            hit_sl_count=10,
            by_instrument={"EURUSD": {"count": 100, "win_rate": 0.65}},
        )
        actions = self.actuator.act_on_outcome_summary(summary)
        self.assertEqual(actions, [])


# ══════════════════════════════════════════════════════════════════════════
# 7. Orchestration / run_cycle
# ══════════════════════════════════════════════════════════════════════════
class TestRunCycle(unittest.TestCase):

    def setUp(self):
        self.actuator = _make_actuator()

    def test_run_cycle_with_no_reports_is_empty(self):
        result = self.actuator.run_cycle()
        self.assertIsInstance(result, ActuatorCycleResult)
        self.assertEqual(result.actions, [])

    def test_run_cycle_aggregates_multiple_reports(self):
        assessment = SelfAssessment(
            period="2026-06", overall_win_rate=0.20, overall_quality_score=0.2
        )
        summary = OutcomeSummary(total=50, with_outcome=50, win_rate=0.55, hit_sl_count=30)
        result = self.actuator.run_cycle(self_assessment=assessment, outcome_summary=summary)
        self.assertGreaterEqual(len(result.actions), 2)
        self.assertIn(ActionType.RETRAINING_RECOMMENDATION, result.by_type)
        d = result.to_dict()
        self.assertEqual(d["total_actions"], len(result.actions))

    def test_actuator_history_accumulates_across_calls(self):
        assessment = SelfAssessment(
            period="2026-06",
            overall_win_rate=0.20,
            overall_quality_score=0.2,
            failure_modes=["Repeated missed opportunities during London session opens"],
        )
        summary = OutcomeSummary(total=50, with_outcome=50, win_rate=0.55, hit_sl_count=30)
        self.actuator.act_on_self_assessment(assessment)
        self.actuator.act_on_outcome_summary(summary)
        hist = self.actuator.history()
        self.assertGreaterEqual(len(hist), 2)


# ══════════════════════════════════════════════════════════════════════════
# 8. No direct trading-logic mutation guarantee
# ══════════════════════════════════════════════════════════════════════════
class TestNoDirectMutationGuarantee(unittest.TestCase):
    """Every action produced must be one of the six advisory action kinds —
    never a direct write to trading logic."""

    def setUp(self):
        self.actuator = _make_actuator()

    def test_all_action_types_are_advisory_kinds(self):
        assessment = SelfAssessment(
            period="2026-06",
            overall_win_rate=0.20,
            overall_quality_score=0.2,
            failure_modes=["Missed opportunities during Tokyo session"],
        )
        actions = self.actuator.act_on_self_assessment(assessment)
        for action in actions:
            self.assertIn(action.action_type, ActionType.ALL)
            self.assertIsInstance(action, ActuatorAction)

    def test_retraining_advisor_never_auto_retrains(self):
        # auto_retrain is hard-pinned False by RetrainingAdvisor itself —
        # evaluate() only ever returns advisory recommendations.
        self.assertFalse(self.actuator.retraining_advisor.auto_retrain)
        assessment = SelfAssessment(period="p", overall_win_rate=0.1, overall_quality_score=0.1)
        actions = self.actuator.act_on_self_assessment(assessment)
        for action in actions:
            if action.action_type == ActionType.RETRAINING_RECOMMENDATION:
                self.assertIn("ADVISORY ONLY", str(action.detail))


# ══════════════════════════════════════════════════════════════════════════
# 9. ExecutionFills → SlippageReport
# ══════════════════════════════════════════════════════════════════════════
class TestSlippageReportActuator(unittest.TestCase):

    def setUp(self):
        self.actuator = _make_actuator()

    def _record(self, adverse_atr: float, instrument: str = "EURUSD") -> SlippageRecord:
        # BUY where actual > expected by adverse_atr * atr is adverse slippage.
        atr = 0.0010
        return SlippageRecord(
            instrument=instrument,
            direction="BUY",
            expected_price=1.1000,
            actual_price=1.1000 + adverse_atr * atr,
            atr=atr,
        )

    def test_systematic_adverse_slippage_triggers_investigation_and_search(self):
        records = [self._record(0.30) for _ in range(6)] + [self._record(0.01) for _ in range(2)]
        report = SlippageReport(records=records)
        actions = self.actuator.act_on_slippage_report(report)
        investigations = [a for a in actions if a.action_type == ActionType.INVESTIGATION]
        searches = [a for a in actions if a.action_type == ActionType.PARAMETER_SEARCH]
        self.assertEqual(len(investigations), 1)
        self.assertEqual(len(searches), 1)
        self.assertTrue(all(a.success for a in actions))

    def test_below_sample_threshold_produces_no_action(self):
        report = SlippageReport(records=[self._record(0.5) for _ in range(3)])
        actions = self.actuator.act_on_slippage_report(report)
        self.assertEqual(actions, [])

    def test_mostly_clean_fills_produce_no_action(self):
        records = [self._record(0.01) for _ in range(9)] + [self._record(0.5) for _ in range(1)]
        report = SlippageReport(records=records)
        actions = self.actuator.act_on_slippage_report(report)
        self.assertEqual(actions, [])

    def test_slippage_atr_direction_handling(self):
        buy_adverse = SlippageRecord(
            instrument="EURUSD",
            direction="BUY",
            expected_price=1.1000,
            actual_price=1.1010,
            atr=0.0010,
        )
        sell_adverse = SlippageRecord(
            instrument="EURUSD",
            direction="SELL",
            expected_price=1.1000,
            actual_price=1.0990,
            atr=0.0010,
        )
        self.assertAlmostEqual(buy_adverse.slippage_atr(), 1.0)
        self.assertAlmostEqual(sell_adverse.slippage_atr(), 1.0)

    def test_zero_atr_is_safe(self):
        rec = SlippageRecord(
            instrument="EURUSD", direction="BUY", expected_price=1.1, actual_price=1.11, atr=0.0
        )
        self.assertEqual(rec.slippage_atr(), 0.0)


# ══════════════════════════════════════════════════════════════════════════
# 10. TradeManager → ScalingReport
# ══════════════════════════════════════════════════════════════════════════
class TestScalingReportActuator(unittest.TestCase):

    def setUp(self):
        self.actuator = _make_actuator()

    def test_failing_scale_in_triggers_investigation_and_shadow_candidate(self):
        outcomes = [
            ScalingOutcome(
                instrument="EURUSD", action="SCALE_IN", confidence_at_action=0.8, pnl_impact=-0.002
            )
            for _ in range(6)
        ] + [
            ScalingOutcome(
                instrument="EURUSD", action="SCALE_IN", confidence_at_action=0.8, pnl_impact=0.001
            )
            for _ in range(2)
        ]
        report = ScalingReport(outcomes=outcomes)
        actions = self.actuator.act_on_scaling_report(report)
        investigations = [a for a in actions if a.action_type == ActionType.INVESTIGATION]
        shadow = [a for a in actions if a.action_type == ActionType.SHADOW_CANDIDATE]
        self.assertEqual(len(investigations), 1)
        self.assertEqual(len(shadow), 1)
        self.assertTrue(all(a.success for a in actions))

    def test_below_sample_threshold_produces_no_action(self):
        outcomes = [
            ScalingOutcome(
                instrument="EURUSD", action="SCALE_OUT", confidence_at_action=0.8, pnl_impact=-0.01
            )
            for _ in range(3)
        ]
        report = ScalingReport(outcomes=outcomes)
        actions = self.actuator.act_on_scaling_report(report)
        self.assertEqual(actions, [])

    def test_profitable_scaling_produces_no_action(self):
        outcomes = [
            ScalingOutcome(
                instrument="EURUSD", action="SCALE_IN", confidence_at_action=0.8, pnl_impact=0.005
            )
            for _ in range(8)
        ]
        report = ScalingReport(outcomes=outcomes)
        actions = self.actuator.act_on_scaling_report(report)
        self.assertEqual(actions, [])

    def test_action_types_are_evaluated_independently(self):
        good = [
            ScalingOutcome(
                instrument="EURUSD", action="SCALE_OUT", confidence_at_action=0.8, pnl_impact=0.003
            )
            for _ in range(6)
        ]
        bad = [
            ScalingOutcome(
                instrument="EURUSD", action="SCALE_IN", confidence_at_action=0.8, pnl_impact=-0.003
            )
            for _ in range(6)
        ]
        report = ScalingReport(outcomes=good + bad)
        actions = self.actuator.act_on_scaling_report(report)
        self.assertTrue(any(a.action_type == ActionType.INVESTIGATION for a in actions))
        self.assertEqual(len([a for a in actions if a.action_type == ActionType.INVESTIGATION]), 1)


# ══════════════════════════════════════════════════════════════════════════
# 11. PortfolioAllocator → CapitalUtilizationReport
# ══════════════════════════════════════════════════════════════════════════
class TestCapitalUtilizationActuator(unittest.TestCase):

    def setUp(self):
        self.actuator = _make_actuator()

    def test_high_idle_capital_with_skipped_signals_triggers_actions(self):
        report = CapitalUtilizationReport(
            period="2026-06",
            deployed_capital_pct=0.40,
            idle_capital_pct=0.60,
            high_confidence_signals_skipped=5,
            avg_skipped_confidence=0.82,
        )
        actions = self.actuator.act_on_capital_utilization(report)
        investigations = [a for a in actions if a.action_type == ActionType.INVESTIGATION]
        searches = [a for a in actions if a.action_type == ActionType.PARAMETER_SEARCH]
        self.assertEqual(len(investigations), 1)
        self.assertEqual(len(searches), 1)
        self.assertTrue(all(a.success for a in actions))

    def test_idle_capital_without_skipped_signals_is_not_flagged(self):
        report = CapitalUtilizationReport(
            period="2026-06",
            deployed_capital_pct=0.30,
            idle_capital_pct=0.70,
            high_confidence_signals_skipped=0,
        )
        actions = self.actuator.act_on_capital_utilization(report)
        self.assertEqual(actions, [])

    def test_low_idle_capital_is_not_flagged(self):
        report = CapitalUtilizationReport(
            period="2026-06",
            deployed_capital_pct=0.80,
            idle_capital_pct=0.20,
            high_confidence_signals_skipped=3,
        )
        actions = self.actuator.act_on_capital_utilization(report)
        self.assertEqual(actions, [])


# ══════════════════════════════════════════════════════════════════════════
# 12. run_cycle wiring for the 3 new trigger kinds
# ══════════════════════════════════════════════════════════════════════════
class TestRunCycleNewTriggers(unittest.TestCase):

    def setUp(self):
        self.actuator = _make_actuator()

    def test_run_cycle_accepts_all_three_new_reports(self):
        slippage = SlippageReport(
            records=[
                SlippageRecord(
                    instrument="EURUSD",
                    direction="BUY",
                    expected_price=1.1,
                    actual_price=1.1005,
                    atr=0.001,
                )
                for _ in range(6)
            ]
        )
        scaling = ScalingReport(
            outcomes=[
                ScalingOutcome(
                    instrument="EURUSD",
                    action="SCALE_IN",
                    confidence_at_action=0.8,
                    pnl_impact=-0.003,
                )
                for _ in range(6)
            ]
        )
        capital = CapitalUtilizationReport(
            period="2026-06",
            deployed_capital_pct=0.30,
            idle_capital_pct=0.70,
            high_confidence_signals_skipped=4,
        )
        result = self.actuator.run_cycle(
            slippage_report=slippage, scaling_report=scaling, capital_utilization=capital
        )
        self.assertGreaterEqual(len(result.actions), 3)
        self.assertIn(ActionType.INVESTIGATION, result.by_type)


# ══════════════════════════════════════════════════════════════════════════
# 13. LearningLoopScheduler — autonomous trigger/scheduler loop
# ══════════════════════════════════════════════════════════════════════════
class TestLearningLoopScheduler(unittest.TestCase):

    def setUp(self):
        self.actuator = _make_actuator()

    def test_run_once_invokes_collector_and_returns_result(self):
        assessment = SelfAssessment(
            period="2026-06", overall_win_rate=0.20, overall_quality_score=0.2
        )
        calls = []

        def collector():
            calls.append(1)
            return {"self_assessment": assessment}

        scheduler = LearningLoopScheduler(self.actuator, collector, interval_seconds=60)
        result = scheduler.run_once()
        self.assertEqual(len(calls), 1)
        self.assertGreaterEqual(len(result.actions), 1)
        status = scheduler.status()
        self.assertEqual(status["run_count"], 1)
        self.assertEqual(status["error_count"], 0)

    def test_run_once_survives_collector_exception(self):
        def bad_collector():
            raise RuntimeError("boom")

        scheduler = LearningLoopScheduler(self.actuator, bad_collector, interval_seconds=60)
        result = scheduler.run_once()
        self.assertEqual(result.actions, [])
        status = scheduler.status()
        self.assertEqual(status["run_count"], 1)
        self.assertEqual(status["error_count"], 1)
        self.assertIsNotNone(status["last_run"]["error"])

    def test_start_stop_runs_loop_in_background(self):
        counter = {"n": 0}

        def collector():
            counter["n"] += 1
            return {}

        scheduler = LearningLoopScheduler(self.actuator, collector, interval_seconds=0.05)
        scheduler.start()
        try:
            deadline = time.time() + 2.0
            while counter["n"] < 2 and time.time() < deadline:
                time.sleep(0.02)
        finally:
            scheduler.stop(timeout=2.0)
        self.assertFalse(scheduler.is_running)
        self.assertGreaterEqual(counter["n"], 2)

    def test_double_start_is_a_no_op(self):
        scheduler = LearningLoopScheduler(self.actuator, lambda: {}, interval_seconds=10)
        scheduler.start()
        try:
            scheduler.start()
            self.assertTrue(scheduler.is_running)
        finally:
            scheduler.stop(timeout=2.0)

    def test_invalid_interval_raises(self):
        with self.assertRaises(ValueError):
            LearningLoopScheduler(self.actuator, lambda: {}, interval_seconds=0)

    def test_history_respects_limit(self):
        scheduler = LearningLoopScheduler(self.actuator, lambda: {}, interval_seconds=60)
        for _ in range(5):
            scheduler.run_once()
        hist = scheduler.history(limit=2)
        self.assertEqual(len(hist), 2)


if __name__ == "__main__":
    unittest.main()
