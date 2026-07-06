"""Phase 24 — ICARI Unit Tests (160+ tests).

Uses spec_from_file_location to avoid importing trainer via importlib.import_module.
"""

import importlib.util
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load(rel_path: str):
    full = os.path.join(ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(
        rel_path.replace("/", ".").replace(".py", ""), full
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-load all modules in dependency order
_types = _load("trainer/icari/icari_types.py")
_models = _load("trainer/icari/icari_models.py")
_alloc = _load("trainer/icari/allocation_engine.py")
_budget = _load("trainer/icari/risk_budget_engine.py")
_sizing = _load("trainer/icari/position_sizing_engine.py")
_dyn = _load("trainer/icari/dynamic_sizer.py")
_effic = _load("trainer/icari/capital_efficiency.py")
_risk = _load("trainer/icari/portfolio_risk_engine.py")
_dd = _load("trainer/icari/drawdown_manager.py")
_corr = _load("trainer/icari/correlation_engine.py")
_scen = _load("trainer/icari/scenario_engine.py")
_optim = _load("trainer/icari/allocation_optimizer.py")
_health = _load("trainer/icari/capital_health.py")
_db = _load("trainer/icari/icari_database.py")
_dash = _load("trainer/icari/icari_dashboard.py")
_api = _load("trainer/icari/icari_api.py")
_init = _load("trainer/icari/__init__.py")

# Convenience aliases
SizingMethod = _types.SizingMethod
RiskBudgetType = _types.RiskBudgetType
DrawdownMode = _types.DrawdownMode
ScenarioType = _types.ScenarioType
OptimizationObjective = _types.OptimizationObjective
AllocationRecommendation = _types.AllocationRecommendation
AllocationStatus = _types.AllocationStatus
CapitalHealthGrade = _types.CapitalHealthGrade
health_grade = _types.health_grade
SCENARIO_FACTORS = _types.SCENARIO_FACTORS
DEFAULT_RISK_BUDGETS = _types.DEFAULT_RISK_BUDGETS

AllocationRequest = _models.AllocationRequest
AllocationResult = _models.AllocationResult
PositionSize = _models.PositionSize
RiskBudget = _models.RiskBudget
PortfolioRisk = _models.PortfolioRisk
CapitalHealth = _models.CapitalHealth
ScenarioResult = _models.ScenarioResult
DrawdownEvent = _models.DrawdownEvent
OptimizationResult = _models.OptimizationResult

AllocationEngine = _alloc.AllocationEngine
RiskBudgetEngine = _budget.RiskBudgetEngine
PositionSizingEngine = _sizing.PositionSizingEngine
DynamicSizer = _dyn.DynamicSizer
CapitalEfficiencyEngine = _effic.CapitalEfficiencyEngine
PortfolioRiskEngine = _risk.PortfolioRiskEngine
DrawdownManager = _dd.DrawdownManager
CorrelationEngine = _corr.CorrelationEngine
ScenarioEngine = _scen.ScenarioEngine
AllocationOptimizer = _optim.AllocationOptimizer
CapitalHealthEngine = _health.CapitalHealthEngine
IcariDatabase = _db.IcariDatabase
IcariDashboard = _dash.IcariDashboard
IcariAPI = _api.IcariAPI


def make_request(**kwargs) -> AllocationRequest:
    defaults = dict(
        instrument="EURUSD",
        direction="LONG",
        forecast_confidence=0.75,
        research_score=0.70,
        execution_score=0.65,
        expected_value=0.01,
        expected_return=0.02,
        expected_drawdown=0.05,
        portfolio_exposure=0.30,
        correlation=0.20,
        market_regime="NORMAL",
        volatility=0.008,
        liquidity=0.90,
        model_confidence=0.72,
        ensemble_confidence=0.68,
        decision_quality=0.70,
        atr=0.00080,
        capital=100_000.0,
    )
    defaults.update(kwargs)
    return AllocationRequest(**defaults)


def make_returns(n: int = 60) -> list:
    """Alternating +1% / -0.5% sequence."""
    return [(0.01 if i % 2 == 0 else -0.005) for i in range(n)]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Types & Enums
# ══════════════════════════════════════════════════════════════════════════════


class TestIcariTypes(unittest.TestCase):

    def test_sizing_method_count(self):
        self.assertEqual(len(SizingMethod), 10)

    def test_sizing_method_values(self):
        self.assertIn(SizingMethod.KELLY, SizingMethod)
        self.assertIn(SizingMethod.FRACTIONAL_KELLY, SizingMethod)
        self.assertIn(SizingMethod.CUSTOM, SizingMethod)

    def test_risk_budget_type_count(self):
        self.assertEqual(len(RiskBudgetType), 12)

    def test_risk_budget_includes_daily_weekly_monthly(self):
        for t in ["DAILY", "WEEKLY", "MONTHLY"]:
            self.assertIn(t, [b.value for b in RiskBudgetType])

    def test_drawdown_mode_count(self):
        self.assertEqual(len(DrawdownMode), 6)

    def test_drawdown_mode_lockdown(self):
        self.assertIn(DrawdownMode.EMERGENCY_LOCKDOWN, DrawdownMode)

    def test_scenario_type_count(self):
        self.assertEqual(len(ScenarioType), 11)

    def test_optimization_objective_count(self):
        self.assertEqual(len(OptimizationObjective), 8)

    def test_allocation_recommendation_count(self):
        self.assertEqual(len(AllocationRecommendation), 8)

    def test_allocation_status_count(self):
        self.assertEqual(len(AllocationStatus), 4)

    def test_capital_health_grade_count(self):
        self.assertEqual(len(CapitalHealthGrade), 5)

    def test_health_grade_excellent(self):
        self.assertEqual(health_grade(0.90), CapitalHealthGrade.EXCELLENT)

    def test_health_grade_good(self):
        self.assertEqual(health_grade(0.72), CapitalHealthGrade.GOOD)

    def test_health_grade_fair(self):
        self.assertEqual(health_grade(0.60), CapitalHealthGrade.FAIR)

    def test_health_grade_poor(self):
        self.assertEqual(health_grade(0.45), CapitalHealthGrade.POOR)

    def test_health_grade_critical(self):
        self.assertEqual(health_grade(0.30), CapitalHealthGrade.CRITICAL)

    def test_scenario_factors_count(self):
        self.assertEqual(len(SCENARIO_FACTORS), 11)

    def test_scenario_factors_flash_crash(self):
        f = SCENARIO_FACTORS[ScenarioType.FLASH_CRASH]
        self.assertLess(f[0], 0)  # negative return
        self.assertGreater(f[1], 1.0)  # high vol

    def test_default_risk_budgets_count(self):
        self.assertEqual(len(DEFAULT_RISK_BUDGETS), 12)

    def test_default_risk_budgets_values_in_range(self):
        for v in DEFAULT_RISK_BUDGETS.values():
            self.assertGreater(v, 0)
            self.assertLessEqual(v, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Models
# ══════════════════════════════════════════════════════════════════════════════


class TestIcariModels(unittest.TestCase):

    def test_allocation_request_defaults(self):
        req = AllocationRequest(instrument="EURUSD")
        self.assertEqual(req.instrument, "EURUSD")
        self.assertEqual(req.capital, 100_000.0)

    def test_allocation_request_to_dict(self):
        req = make_request()
        d = req.to_dict()
        self.assertIn("instrument", d)
        self.assertIn("forecast_confidence", d)

    def test_allocation_result_to_dict(self):
        r = AllocationResult(
            instrument="GBPUSD", status=AllocationStatus.APPROVED, allocation_frac=0.1
        )
        d = r.to_dict()
        self.assertEqual(d["status"], "APPROVED")
        self.assertIn("allocation_frac", d)

    def test_risk_budget_remaining(self):
        b = RiskBudget(budget_type=RiskBudgetType.DAILY, limit=0.02, used=0.01)
        self.assertAlmostEqual(b.remaining, 0.01, places=5)

    def test_risk_budget_utilisation(self):
        b = RiskBudget(budget_type=RiskBudgetType.DAILY, limit=0.02, used=0.01)
        self.assertAlmostEqual(b.utilisation, 0.50, places=5)

    def test_risk_budget_is_breached(self):
        b = RiskBudget(budget_type=RiskBudgetType.DAILY, limit=0.02, used=0.03)
        self.assertTrue(b.is_breached)

    def test_risk_budget_not_breached(self):
        b = RiskBudget(budget_type=RiskBudgetType.DAILY, limit=0.02, used=0.01)
        self.assertFalse(b.is_breached)

    def test_risk_budget_to_dict(self):
        b = RiskBudget(budget_type=RiskBudgetType.PORTFOLIO, limit=0.20, used=0.08)
        d = b.to_dict()
        self.assertIn("utilisation", d)
        self.assertIn("breached", d)

    def test_portfolio_risk_to_dict(self):
        pr = PortfolioRisk(var_95=0.015, var_99=0.025)
        d = pr.to_dict()
        self.assertIn("var_95", d)
        self.assertIn("var_99", d)

    def test_capital_health_to_dict(self):
        h = CapitalHealth(overall_score=0.80, grade=CapitalHealthGrade.GOOD)
        d = h.to_dict()
        self.assertEqual(d["grade"], "GOOD")

    def test_scenario_result_to_dict(self):
        sr = ScenarioResult(scenario=ScenarioType.FLASH_CRASH, resilience_score=0.30)
        d = sr.to_dict()
        self.assertEqual(d["scenario"], "FLASH_CRASH")

    def test_drawdown_event_to_dict(self):
        e = DrawdownEvent(peak=100_000.0, trough=95_000.0, drawdown=0.05)
        d = e.to_dict()
        self.assertIn("drawdown", d)
        self.assertIn("mode", d)

    def test_drawdown_event_duration_none_when_not_closed(self):
        e = DrawdownEvent(peak=100_000.0, trough=95_000.0)
        self.assertIsNone(e.duration_seconds)

    def test_optimization_result_to_dict(self):
        r = OptimizationResult(
            objective=OptimizationObjective.MAX_SHARPE,
            weights={"EURUSD": 0.5, "GBPUSD": 0.5},
            converged=True,
        )
        d = r.to_dict()
        self.assertTrue(d["converged"])


# ══════════════════════════════════════════════════════════════════════════════
# 3. Allocation Engine
# ══════════════════════════════════════════════════════════════════════════════


class TestAllocationEngine(unittest.TestCase):

    def setUp(self):
        self.engine = AllocationEngine()

    def test_weights_sum_to_one(self):
        from trainer.icari.allocation_engine import _WEIGHTS

        self.assertAlmostEqual(sum(_WEIGHTS.values()), 1.0, places=9)

    def test_high_confidence_approved(self):
        req = make_request(
            forecast_confidence=0.90,
            research_score=0.88,
            execution_score=0.85,
            model_confidence=0.90,
            ensemble_confidence=0.88,
            decision_quality=0.85,
            liquidity=0.95,
        )
        result = self.engine.allocate(req)
        self.assertEqual(result.status, AllocationStatus.APPROVED)
        self.assertGreater(result.allocation_frac, 0.0)

    def test_low_confidence_denied(self):
        # Force negative EV/return so those factors contribute 0 to composite
        req = make_request(
            forecast_confidence=0.10,
            research_score=0.10,
            execution_score=0.10,
            model_confidence=0.10,
            ensemble_confidence=0.10,
            decision_quality=0.10,
            liquidity=0.10,
            expected_value=-0.01,
            expected_return=-0.02,
        )
        result = self.engine.allocate(req)
        self.assertEqual(result.status, AllocationStatus.DENIED)
        self.assertEqual(result.allocation_frac, 0.0)
        self.assertEqual(result.allocation_usd, 0.0)

    def test_denied_has_skip_trade_recommendation(self):
        req = make_request(
            forecast_confidence=0.05,
            research_score=0.05,
            execution_score=0.05,
            model_confidence=0.05,
            ensemble_confidence=0.05,
            decision_quality=0.05,
            liquidity=0.05,
            expected_value=-0.01,
            expected_return=-0.02,
        )
        result = self.engine.allocate(req)
        self.assertIn(AllocationRecommendation.SKIP_TRADE, result.recommendations)

    def test_trending_regime_gives_higher_allocation(self):
        req_normal = make_request(market_regime="NORMAL")
        req_trending = make_request(market_regime="TRENDING")
        r_normal = self.engine.allocate(req_normal)
        r_trending = self.engine.allocate(req_trending)
        if (
            r_normal.status == AllocationStatus.APPROVED
            and r_trending.status == AllocationStatus.APPROVED
        ):
            self.assertGreaterEqual(r_trending.allocation_frac, r_normal.allocation_frac)

    def test_volatile_regime_lower_allocation(self):
        req_normal = make_request(market_regime="NORMAL")
        req_volatile = make_request(market_regime="VOLATILE")
        r_n = self.engine.allocate(req_normal)
        r_v = self.engine.allocate(req_volatile)
        if r_n.status == AllocationStatus.APPROVED and r_v.status == AllocationStatus.APPROVED:
            self.assertLessEqual(r_v.allocation_frac, r_n.allocation_frac)

    def test_allocation_never_exceeds_max(self):
        req = make_request(
            forecast_confidence=1.0,
            research_score=1.0,
            execution_score=1.0,
            model_confidence=1.0,
            ensemble_confidence=1.0,
            decision_quality=1.0,
            liquidity=1.0,
        )
        result = self.engine.allocate(req)
        self.assertLessEqual(result.allocation_frac, 0.25)

    def test_high_exposure_reduces_allocation(self):
        req_low = make_request(portfolio_exposure=0.20)
        req_high = make_request(portfolio_exposure=0.85)
        r_l = self.engine.allocate(req_low)
        r_h = self.engine.allocate(req_high)
        if r_l.status == AllocationStatus.APPROVED and r_h.status == AllocationStatus.APPROVED:
            self.assertLessEqual(r_h.allocation_frac, r_l.allocation_frac)

    def test_high_correlation_adds_recommendation(self):
        req = make_request(correlation=0.80)
        result = self.engine.allocate(req)
        if result.status != AllocationStatus.DENIED:
            self.assertIn(AllocationRecommendation.REDUCE_CORRELATED, result.recommendations)

    def test_allocation_usd_matches_frac(self):
        req = make_request()
        result = self.engine.allocate(req)
        # Both values are independently rounded, so check within $2
        self.assertAlmostEqual(
            result.allocation_usd, result.allocation_frac * req.capital, delta=2.0
        )

    def test_batch_allocation(self):
        reqs = [make_request(instrument=f"INS{i}") for i in range(5)]
        results = self.engine.allocate_batch(reqs)
        self.assertEqual(len(results), 5)

    def test_history_populated(self):
        req = make_request()
        self.engine.allocate(req)
        self.assertGreater(len(self.engine.history()), 0)

    def test_factor_weights_returns_dict(self):
        w = self.engine.factor_weights()
        self.assertIsInstance(w, dict)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=9)

    def test_average_confidence_nonzero(self):
        self.engine.allocate(make_request())
        self.assertGreater(self.engine.average_confidence(), 0.0)

    def test_result_has_factors(self):
        result = self.engine.allocate(make_request())
        self.assertIsInstance(result.factors, dict)
        self.assertGreater(len(result.factors), 0)

    def test_result_has_rationale(self):
        result = self.engine.allocate(make_request())
        self.assertIsInstance(result.rationale, str)
        self.assertGreater(len(result.rationale), 0)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Risk Budget Engine
# ══════════════════════════════════════════════════════════════════════════════


class TestRiskBudgetEngine(unittest.TestCase):

    def setUp(self):
        self.engine = RiskBudgetEngine(capital=100_000.0)

    def test_defaults_loaded(self):
        self.assertEqual(len(self.engine.all_budgets()), 12)

    def test_set_budget(self):
        b = self.engine.set_budget(RiskBudgetType.DAILY, 0.03, "Custom Daily")
        self.assertAlmostEqual(b.limit, 0.03, places=4)

    def test_consume_within_limit(self):
        result = self.engine.consume(RiskBudgetType.DAILY, 0.01)
        self.assertFalse(result["breached"])

    def test_consume_breaches(self):
        self.engine.consume(RiskBudgetType.DAILY, 0.015)
        result = self.engine.consume(RiskBudgetType.DAILY, 0.015)
        self.assertTrue(result["breached"])

    def test_release(self):
        self.engine.consume(RiskBudgetType.DAILY, 0.015)
        self.engine.release(RiskBudgetType.DAILY, 0.010)
        b = self.engine.get_budget(RiskBudgetType.DAILY)
        self.assertAlmostEqual(b.used, 0.005, places=4)

    def test_reset_single(self):
        self.engine.consume(RiskBudgetType.DAILY, 0.015)
        self.engine.reset(RiskBudgetType.DAILY)
        b = self.engine.get_budget(RiskBudgetType.DAILY)
        self.assertEqual(b.used, 0.0)

    def test_reset_all(self):
        for bt in list(RiskBudgetType)[:3]:
            self.engine.consume(bt, 0.001)
        self.engine.reset_all()
        for b in self.engine.all_budgets():
            self.assertEqual(b.used, 0.0)

    def test_check_allowed(self):
        result = self.engine.check(RiskBudgetType.DAILY, 0.005)
        self.assertTrue(result["allowed"])

    def test_check_denied(self):
        result = self.engine.check(RiskBudgetType.DAILY, 0.99)
        self.assertFalse(result["allowed"])

    def test_breached_budgets_empty_initially(self):
        self.assertEqual(len(self.engine.breached_budgets()), 0)

    def test_breached_budgets_after_breach(self):
        self.engine.consume(RiskBudgetType.DAILY, 1.0)
        self.assertGreater(len(self.engine.breached_budgets()), 0)

    def test_near_limit_budgets(self):
        self.engine.consume(RiskBudgetType.DAILY, 0.019)
        near = self.engine.near_limit_budgets(threshold=0.80)
        self.assertGreater(len(near), 0)

    def test_limit_usd(self):
        usd = self.engine.limit_usd(RiskBudgetType.PORTFOLIO)
        self.assertGreater(usd, 0.0)

    def test_summary_keys(self):
        s = self.engine.summary()
        self.assertIn("total_budgets", s)
        self.assertIn("breached", s)
        self.assertIn("all", s)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Position Sizing Engine
# ══════════════════════════════════════════════════════════════════════════════


class TestPositionSizingEngine(unittest.TestCase):

    def setUp(self):
        self.req = make_request()

    def _engine(self, method=SizingMethod.FIXED_RISK):
        return PositionSizingEngine(method=method, risk_frac=0.01, atr_mult=2.0)

    def test_fixed_fractional_units_positive(self):
        e = self._engine(SizingMethod.FIXED_FRACTIONAL)
        r = e.size(self.req)
        self.assertGreater(r.units, 0.0)

    def test_fixed_risk_units_positive(self):
        e = self._engine(SizingMethod.FIXED_RISK)
        r = e.size(self.req)
        self.assertGreater(r.units, 0.0)

    def test_atr_adjusted(self):
        e = self._engine(SizingMethod.ATR_ADJUSTED)
        r = e.size(self.req)
        self.assertGreater(r.units, 0.0)
        self.assertEqual(r.method, SizingMethod.ATR_ADJUSTED)

    def test_volatility_adjusted_smaller_in_high_vol(self):
        req_low = make_request(volatility=0.005)
        req_high = make_request(volatility=0.05)
        e = self._engine(SizingMethod.VOLATILITY_ADJUSTED)
        r_l = e.size(req_low)
        r_h = e.size(req_high)
        self.assertGreater(r_l.units, r_h.units)

    def test_expected_value_adjusted(self):
        e = self._engine(SizingMethod.EXPECTED_VALUE)
        r = e.size(self.req)
        self.assertGreater(r.units, 0.0)

    def test_confidence_adjusted(self):
        e = self._engine(SizingMethod.CONFIDENCE_ADJUSTED)
        r = e.size(self.req)
        self.assertGreater(r.units, 0.0)

    def test_risk_budget_adjusted(self):
        e = self._engine(SizingMethod.RISK_BUDGET_ADJUSTED)
        r = e.size(make_request(portfolio_exposure=0.80))
        self.assertGreater(r.units, 0.0)

    def test_kelly_positive(self):
        req = make_request(forecast_confidence=0.60, expected_return=0.03, expected_drawdown=0.02)
        e = self._engine(SizingMethod.KELLY)
        r = e.size(req)
        self.assertGreater(r.units, 0.0)

    def test_kelly_risk_capped(self):
        req = make_request(forecast_confidence=0.99, expected_return=0.10, expected_drawdown=0.001)
        e = self._engine(SizingMethod.KELLY)
        r = e.size(req)
        self.assertLessEqual(r.risk_frac, 0.05)

    def test_fractional_kelly_smaller_than_full(self):
        req = make_request(forecast_confidence=0.60, expected_return=0.03, expected_drawdown=0.02)
        e_full = PositionSizingEngine(method=SizingMethod.KELLY, risk_frac=0.02, kelly_frac=0.5)
        e_frac = PositionSizingEngine(
            method=SizingMethod.FRACTIONAL_KELLY, risk_frac=0.02, kelly_frac=0.5
        )
        r_full = e_full.size(req)
        r_frac = e_frac.size(req)
        self.assertLessEqual(r_frac.units, r_full.units + 0.001)

    def test_custom_falls_back_to_fixed_risk(self):
        e = PositionSizingEngine(method=SizingMethod.CUSTOM)
        r = e.size(self.req)
        self.assertGreater(r.units, 0.0)

    def test_custom_fn_called(self):
        def my_fn(req):
            return PositionSize(
                instrument=req.instrument,
                method=SizingMethod.CUSTOM,
                units=999.0,
                risk_amount=1.0,
                risk_frac=0.01,
            )

        e = PositionSizingEngine(method=SizingMethod.CUSTOM, custom_fn=my_fn)
        r = e.size(self.req)
        self.assertAlmostEqual(r.units, 999.0)

    def test_compare_methods_returns_all(self):
        e = self._engine()
        results = e.compare_methods(self.req)
        # 9 methods (excluding CUSTOM)
        self.assertEqual(len(results), 9)

    def test_position_size_to_dict(self):
        e = self._engine()
        r = e.size(self.req)
        d = r.to_dict()
        self.assertIn("units", d)
        self.assertIn("risk_frac", d)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Dynamic Sizer
# ══════════════════════════════════════════════════════════════════════════════


class TestDynamicSizer(unittest.TestCase):

    def setUp(self):
        self.sizer = DynamicSizer(drawdown_soft_limit=0.05, drawdown_hard_limit=0.10)
        self.req = make_request()

    def test_basic_size_returns_result(self):
        r = self.sizer.size(self.req)
        self.assertGreater(r.units, 0.0)
        self.assertTrue(r.adjusted)

    def test_adjustments_populated(self):
        r = self.sizer.size(self.req)
        self.assertGreater(len(r.adjustments), 0)

    def test_drawdown_reduces_size(self):
        r_none = self.sizer.size(self.req, current_drawdown=0.0)
        r_soft = self.sizer.size(self.req, current_drawdown=0.07)
        self.assertGreater(r_none.units, r_soft.units)

    def test_high_drawdown_reduces_more(self):
        r_soft = self.sizer.size(self.req, current_drawdown=0.06)
        r_hard = self.sizer.size(self.req, current_drawdown=0.12)
        self.assertGreater(r_soft.units, r_hard.units)

    def test_session_tokyo_smaller_than_london(self):
        r_london = self.sizer.size(self.req, session="LONDON")
        r_tokyo = self.sizer.size(self.req, session="TOKYO")
        self.assertGreater(r_london.units, r_tokyo.units)

    def test_low_liquidity_reduces_size(self):
        req_hi = make_request(liquidity=0.95)
        req_lo = make_request(liquidity=0.10)
        r_hi = self.sizer.size(req_hi)
        r_lo = self.sizer.size(req_lo)
        self.assertGreater(r_hi.units, r_lo.units)

    def test_high_volatility_reduces_size(self):
        req_lo = make_request(volatility=0.005)
        req_hi = make_request(volatility=0.04)
        r_lo = self.sizer.size(req_lo)
        r_hi = self.sizer.size(req_hi)
        self.assertGreater(r_lo.units, r_hi.units)

    def test_drawdown_mode_normal(self):
        mode = self.sizer.drawdown_mode(0.02)
        self.assertEqual(mode, "NORMAL")

    def test_drawdown_mode_soft(self):
        mode = self.sizer.drawdown_mode(0.06)
        self.assertEqual(mode, "SOFT_LIMIT")

    def test_drawdown_mode_hard(self):
        mode = self.sizer.drawdown_mode(0.11)
        self.assertEqual(mode, "HARD_LIMIT")

    def test_drawdown_mode_lockdown(self):
        mode = self.sizer.drawdown_mode(0.25)
        self.assertEqual(mode, "EMERGENCY_LOCKDOWN")

    def test_units_always_positive(self):
        r = self.sizer.size(self.req, current_drawdown=0.19)
        self.assertGreater(r.units, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Capital Efficiency
# ══════════════════════════════════════════════════════════════════════════════


class TestCapitalEfficiency(unittest.TestCase):

    def setUp(self):
        self.engine = CapitalEfficiencyEngine(capital=100_000.0)

    def test_analyse_returns_dict(self):
        m = self.engine.analyse(60_000, 5_000, 1_200, 800, 5)
        self.assertIn("utilisation", m)

    def test_utilisation_correct(self):
        m = self.engine.analyse(70_000, 5_000, 1_000, 500, 4)
        self.assertAlmostEqual(m["utilisation"], 0.70, places=2)

    def test_idle_fraction(self):
        m = self.engine.analyse(30_000, 3_000, 500, 200, 2)
        self.assertAlmostEqual(m["idle_fraction"], 0.70, places=2)

    def test_risk_efficiency_positive(self):
        m = self.engine.analyse(50_000, 5_000, 500, 200, 3)
        self.assertGreater(m["risk_efficiency"], 0.0)

    def test_risk_efficiency_zero_at_risk(self):
        m = self.engine.analyse(50_000, 0, 500, 200, 3)
        self.assertEqual(m["risk_efficiency"], 0.0)

    def test_efficiency_score_range(self):
        m = self.engine.analyse(70_000, 5_000, 1_000, 500, 4)
        s = self.engine.efficiency_score(m)
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)

    def test_utilisation_score_peaks_at_70pct(self):
        s_70 = self.engine.utilisation_score(70_000)
        s_30 = self.engine.utilisation_score(30_000)
        self.assertGreater(s_70, s_30)

    def test_history_populated(self):
        self.engine.analyse(60_000, 5_000, 1_000, 400, 4)
        self.assertEqual(len(self.engine.history()), 1)

    def test_trend_returns_direction(self):
        self.engine.analyse(40_000, 4_000, 400, 100, 2)
        self.engine.analyse(70_000, 6_000, 600, 200, 4)
        t = self.engine.trend()
        self.assertIn("direction", t)


# ══════════════════════════════════════════════════════════════════════════════
# 8. Portfolio Risk Engine
# ══════════════════════════════════════════════════════════════════════════════


class TestPortfolioRiskEngine(unittest.TestCase):

    def setUp(self):
        self.engine = PortfolioRiskEngine()
        self.returns = {
            "EURUSD": make_returns(60),
            "GBPUSD": make_returns(60),
        }
        self.weights = {"EURUSD": 0.50, "GBPUSD": 0.50}

    def test_calculate_returns_portfolio_risk(self):
        pr = self.engine.calculate(self.returns, self.weights)
        self.assertIsInstance(pr, PortfolioRisk)

    def test_var_95_positive(self):
        pr = self.engine.calculate(self.returns, self.weights)
        self.assertGreaterEqual(pr.var_95, 0.0)

    def test_var_99_gte_var_95(self):
        pr = self.engine.calculate(self.returns, self.weights)
        self.assertGreaterEqual(pr.var_99, pr.var_95)

    def test_cvar_gte_var(self):
        pr = self.engine.calculate(self.returns, self.weights)
        self.assertGreaterEqual(pr.cvar_95, pr.var_95)

    def test_portfolio_volatility_positive(self):
        pr = self.engine.calculate(self.returns, self.weights)
        self.assertGreater(pr.portfolio_volatility, 0.0)

    def test_diversification_ratio_gte_one(self):
        pr = self.engine.calculate(self.returns, self.weights)
        self.assertGreaterEqual(pr.diversification_ratio, 1.0)

    def test_risk_contribution_sums_to_one(self):
        pr = self.engine.calculate(self.returns, self.weights)
        total = sum(pr.risk_contribution.values())
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_max_expected_drawdown_range(self):
        pr = self.engine.calculate(self.returns, self.weights)
        self.assertGreaterEqual(pr.max_expected_drawdown, 0.0)
        self.assertLessEqual(pr.max_expected_drawdown, 1.0)

    def test_empty_returns_returns_default(self):
        pr = self.engine.calculate({}, {})
        self.assertEqual(pr.var_95, 0.0)

    def test_var_parametric(self):
        var = self.engine.var_parametric(0.001, 0.01)
        self.assertGreater(var, 0.0)

    def test_stress_var(self):
        base = 0.02
        stressed = self.engine.stress_var(base, 2.0)
        self.assertAlmostEqual(stressed, 0.04, places=4)

    def test_to_dict(self):
        pr = self.engine.calculate(self.returns, self.weights)
        d = pr.to_dict()
        self.assertIn("var_95", d)
        self.assertIn("risk_contribution", d)


# ══════════════════════════════════════════════════════════════════════════════
# 9. Drawdown Manager
# ══════════════════════════════════════════════════════════════════════════════


class TestDrawdownManager(unittest.TestCase):

    def setUp(self):
        self.dm = DrawdownManager(
            capital=100_000.0, soft_limit=0.05, hard_limit=0.10, emergency_limit=0.20
        )

    def test_initial_mode_normal(self):
        self.assertEqual(self.dm.mode, DrawdownMode.NORMAL)

    def test_no_drawdown_normal(self):
        self.dm.update(100_000.0)
        self.assertEqual(self.dm.mode, DrawdownMode.NORMAL)

    def test_soft_limit_triggered(self):
        self.dm.update(94_500.0)  # 5.5% drawdown
        self.assertEqual(self.dm.mode, DrawdownMode.SOFT_LIMIT)

    def test_hard_limit_triggered(self):
        self.dm.update(89_000.0)  # 11% drawdown
        self.assertEqual(self.dm.mode, DrawdownMode.HARD_LIMIT)

    def test_emergency_lockdown_triggered(self):
        self.dm.update(79_000.0)  # 21% drawdown
        self.assertEqual(self.dm.mode, DrawdownMode.EMERGENCY_LOCKDOWN)
        self.assertTrue(self.dm.is_locked())

    def test_can_trade_normal(self):
        self.assertTrue(self.dm.can_trade())

    def test_cannot_trade_lockdown(self):
        self.dm.update(79_000.0)
        self.assertFalse(self.dm.can_trade())

    def test_risk_multiplier_decreases_with_dd(self):
        self.dm.update(94_500.0)  # soft
        m_soft = self.dm.risk_multiplier
        self.dm.update(89_000.0)  # hard
        m_hard = self.dm.risk_multiplier
        self.assertGreater(m_soft, m_hard)

    def test_risk_multiplier_zero_in_lockdown(self):
        self.dm.update(79_000.0)
        self.assertEqual(self.dm.risk_multiplier, 0.0)

    def test_new_high_resets_event(self):
        self.dm.update(94_000.0)
        self.assertIsNotNone(self.dm._active_event)
        self.dm.update(101_000.0)  # new peak
        self.assertIsNone(self.dm._active_event)

    def test_events_recorded(self):
        self.dm.update(92_000.0)
        self.assertGreater(len(self.dm.events()), 0)

    def test_activate_preservation(self):
        self.dm.activate_preservation()
        self.assertEqual(self.dm.mode, DrawdownMode.PRESERVATION)

    def test_reset_peak(self):
        self.dm.update(88_000.0)
        self.dm.reset_peak(120_000.0)
        self.assertEqual(self.dm.mode, DrawdownMode.NORMAL)
        self.assertEqual(self.dm._peak, 120_000.0)

    def test_stats_keys(self):
        s = self.dm.stats()
        self.assertIn("peak", s)
        self.assertIn("mode", s)
        self.assertIn("can_trade", s)

    def test_update_returns_dict(self):
        result = self.dm.update(95_000.0)
        self.assertIn("drawdown", result)
        self.assertIn("risk_multiplier", result)


# ══════════════════════════════════════════════════════════════════════════════
# 10. Correlation Engine
# ══════════════════════════════════════════════════════════════════════════════


class TestCorrelationEngine(unittest.TestCase):

    def setUp(self):
        self.engine = CorrelationEngine(rolling_window=10)
        self.r1 = [
            0.01,
            -0.005,
            0.008,
            -0.003,
            0.012,
            -0.004,
            0.009,
            -0.002,
            0.007,
            -0.001,
            0.01,
            -0.005,
            0.008,
            -0.003,
            0.012,
            -0.004,
            0.009,
            -0.002,
            0.007,
            -0.001,
        ]
        self.r2 = [
            0.009,
            -0.004,
            0.007,
            -0.002,
            0.011,
            -0.003,
            0.008,
            -0.001,
            0.006,
            -0.001,
            0.009,
            -0.004,
            0.007,
            -0.002,
            0.011,
            -0.003,
            0.008,
            -0.001,
            0.006,
            -0.001,
        ]
        self.r3 = [
            -0.01,
            0.005,
            -0.008,
            0.003,
            -0.012,
            0.004,
            -0.009,
            0.002,
            -0.007,
            0.001,
            -0.01,
            0.005,
            -0.008,
            0.003,
            -0.012,
            0.004,
            -0.009,
            0.002,
            -0.007,
            0.001,
        ]

    def test_current_correlation_identical(self):
        c = self.engine.current_correlation(self.r1, self.r1)
        self.assertAlmostEqual(c, 1.0, places=3)

    def test_current_correlation_perfectly_neg(self):
        c = self.engine.current_correlation(self.r1, self.r3)
        self.assertAlmostEqual(c, -1.0, places=2)

    def test_current_correlation_similar(self):
        c = self.engine.current_correlation(self.r1, self.r2)
        self.assertGreater(c, 0.80)

    def test_rolling_correlation_length(self):
        rc = self.engine.rolling_correlation(self.r1, self.r2)
        expected_len = len(self.r1) - 10 + 1
        self.assertEqual(len(rc), expected_len)

    def test_correlation_matrix_diagonal_ones(self):
        returns = {"A": self.r1, "B": self.r2}
        m = self.engine.correlation_matrix(returns)
        self.assertAlmostEqual(m["A"]["A"], 1.0, places=3)
        self.assertAlmostEqual(m["B"]["B"], 1.0, places=3)

    def test_correlation_matrix_symmetric(self):
        returns = {"A": self.r1, "B": self.r2}
        m = self.engine.correlation_matrix(returns)
        self.assertAlmostEqual(m["A"]["B"], m["B"]["A"], places=4)

    def test_dynamic_correlation_returns_dict(self):
        returns = {"A": self.r1, "B": self.r2, "C": self.r3}
        dc = self.engine.dynamic_correlation(returns)
        self.assertIn("A", dc)

    def test_correlation_clusters_single_cluster(self):
        # r1 and r2 are highly correlated → same cluster
        returns = {"A": self.r1, "B": self.r2}
        clusters = self.engine.correlation_clusters(returns)
        # With threshold=0.70, A and B should be in same cluster
        all_in_one = any(len(c) == 2 for c in clusters)
        self.assertTrue(all_in_one)

    def test_hidden_exposure_keys(self):
        returns = {"A": self.r1, "B": self.r2}
        weights = {"A": 0.5, "B": 0.5}
        he = self.engine.hidden_exposure(weights, returns)
        self.assertIn("A", he)

    def test_currency_concentration(self):
        weights = {"EURUSD": 0.4, "GBPUSD": 0.3, "USDJPY": 0.3}
        currencies = {"EURUSD": "EUR", "GBPUSD": "GBP", "USDJPY": "USD"}
        cc = self.engine.currency_concentration(weights, currencies)
        self.assertAlmostEqual(sum(cc.values()), 1.0, places=4)

    def test_tail_correlations_returns_matrix(self):
        returns = {"A": self.r1, "B": self.r2}
        tc = self.engine.tail_correlations(returns)
        self.assertIn("A", tc)
        self.assertIn("B", tc["A"])

    def test_average_correlation_in_range(self):
        returns = {"A": self.r1, "B": self.r2, "C": self.r3}
        avg = self.engine.average_correlation(returns)
        self.assertGreaterEqual(avg, -1.0)
        self.assertLessEqual(avg, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 11. Scenario Engine
# ══════════════════════════════════════════════════════════════════════════════


class TestScenarioEngine(unittest.TestCase):

    def setUp(self):
        self.engine = ScenarioEngine(capital=100_000.0)
        self.weights = {"EURUSD": 0.5, "XAUUSD": 0.3, "BTCUSD": 0.2}
        self.vols = {"EURUSD": 0.008, "XAUUSD": 0.012, "BTCUSD": 0.04}

    def test_flash_crash_negative_return(self):
        r = self.engine.run(ScenarioType.FLASH_CRASH, self.weights, self.vols)
        self.assertLess(r.portfolio_return, 0.0)

    def test_risk_on_positive_return(self):
        r = self.engine.run(ScenarioType.RISK_ON, self.weights, self.vols)
        self.assertGreater(r.portfolio_return, 0.0)

    def test_resilience_range(self):
        r = self.engine.run(ScenarioType.HIGH_VOLATILITY, self.weights, self.vols)
        self.assertGreaterEqual(r.resilience_score, 0.0)
        self.assertLessEqual(r.resilience_score, 1.0)

    def test_run_all_returns_11(self):
        results = self.engine.run_all(self.weights, self.vols)
        self.assertEqual(len(results), 11)

    def test_run_all_sorted_by_resilience(self):
        results = self.engine.run_all(self.weights, self.vols)
        for i in range(len(results) - 1):
            self.assertLessEqual(results[i].resilience_score, results[i + 1].resilience_score)

    def test_worst_case_lowest_resilience(self):
        all_r = self.engine.run_all(self.weights, self.vols)
        worst = self.engine.worst_case(self.weights, self.vols)
        self.assertAlmostEqual(worst.resilience_score, all_r[0].resilience_score, places=4)

    def test_resilience_summary_keys(self):
        results = self.engine.run_all(self.weights, self.vols)
        s = self.engine.resilience_summary(results)
        self.assertIn("average_resilience", s)
        self.assertIn("worst_scenario", s)

    def test_instrument_impacts_populated(self):
        r = self.engine.run(ScenarioType.FLASH_CRASH, self.weights, self.vols)
        self.assertEqual(len(r.instrument_impacts), 3)

    def test_recommendations_non_empty(self):
        r = self.engine.run(ScenarioType.FLASH_CRASH, self.weights, self.vols)
        self.assertGreater(len(r.recommendations), 0)

    def test_to_dict_keys(self):
        r = self.engine.run(ScenarioType.CRYPTO_CRASH, self.weights, self.vols)
        d = r.to_dict()
        self.assertIn("scenario", d)
        self.assertIn("resilience_score", d)


# ══════════════════════════════════════════════════════════════════════════════
# 12. Allocation Optimizer
# ══════════════════════════════════════════════════════════════════════════════


class TestAllocationOptimizer(unittest.TestCase):

    def setUp(self):
        self.optimizer = AllocationOptimizer()
        self.returns = {
            "EURUSD": make_returns(60),
            "GBPUSD": [r * 0.9 for r in make_returns(60)],
        }

    def test_max_sharpe_converges(self):
        r = self.optimizer.optimize(OptimizationObjective.MAX_SHARPE, self.returns)
        self.assertTrue(r.converged)

    def test_min_drawdown_converges(self):
        r = self.optimizer.optimize(OptimizationObjective.MIN_DRAWDOWN, self.returns)
        self.assertTrue(r.converged)

    def test_weights_sum_to_one(self):
        r = self.optimizer.optimize(OptimizationObjective.MAX_SHARPE, self.returns)
        self.assertAlmostEqual(sum(r.weights.values()), 1.0, places=3)

    def test_weights_all_non_negative(self):
        r = self.optimizer.optimize(OptimizationObjective.RISK_PARITY, self.returns)
        for w in r.weights.values():
            self.assertGreaterEqual(w, -0.001)

    def test_weights_max_respected(self):
        # Need 3+ instruments so projection can redistribute excess off the capped instrument
        returns3 = {
            "A": make_returns(60),
            "B": [r * 0.9 for r in make_returns(60)],
            "C": [r * 0.8 for r in make_returns(60)],
        }
        opt = AllocationOptimizer(max_weight=0.60)
        r = opt.optimize(OptimizationObjective.MAX_SHARPE, returns3)
        for w in r.weights.values():
            self.assertLessEqual(w, 0.61)

    def test_metrics_populated(self):
        r = self.optimizer.optimize(OptimizationObjective.MAX_SHARPE, self.returns)
        self.assertIn("sharpe", r.metrics)
        self.assertIn("max_drawdown", r.metrics)

    def test_all_objectives(self):
        results = self.optimizer.compare_objectives(self.returns)
        self.assertEqual(len(results), 8)

    def test_history_populated(self):
        self.optimizer.optimize(OptimizationObjective.MAX_SHARPE, self.returns)
        self.assertGreater(len(self.optimizer.history()), 0)

    def test_to_dict_keys(self):
        r = self.optimizer.optimize(OptimizationObjective.BALANCED, self.returns)
        d = r.to_dict()
        self.assertIn("weights", d)
        self.assertIn("converged", d)

    def test_empty_returns(self):
        r = self.optimizer.optimize(OptimizationObjective.MAX_SHARPE, {})
        self.assertFalse(r.converged)


# ══════════════════════════════════════════════════════════════════════════════
# 13. Capital Health Engine
# ══════════════════════════════════════════════════════════════════════════════


class TestCapitalHealthEngine(unittest.TestCase):

    def setUp(self):
        self.engine = CapitalHealthEngine()
        self.pr = PortfolioRisk(
            var_95=0.012,
            var_99=0.020,
            cvar_95=0.016,
            max_expected_drawdown=0.08,
            portfolio_volatility=0.08,
            diversification_ratio=1.8,
            risk_concentration=0.25,
            risk_contribution={"A": 0.5, "B": 0.5},
        )

    def test_score_returns_health_object(self):
        h = self.engine.score(self.pr, utilisation=0.60)
        self.assertIsInstance(h, CapitalHealth)

    def test_overall_score_in_range(self):
        h = self.engine.score(self.pr, utilisation=0.60)
        self.assertGreaterEqual(h.overall_score, 0.0)
        self.assertLessEqual(h.overall_score, 1.0)

    def test_grade_set(self):
        h = self.engine.score(self.pr, utilisation=0.70)
        self.assertIsInstance(h.grade, CapitalHealthGrade)

    def test_lockdown_lowers_risk_score(self):
        h_normal = self.engine.score(self.pr, utilisation=0.60, drawdown_mode=DrawdownMode.NORMAL)
        h_lock = self.engine.score(
            self.pr, utilisation=0.60, drawdown_mode=DrawdownMode.EMERGENCY_LOCKDOWN
        )
        self.assertGreater(h_normal.risk_score, h_lock.risk_score)

    def test_more_instruments_better_diversification(self):
        h1 = self.engine.score(self.pr, utilisation=0.60, num_instruments=2)
        h5 = self.engine.score(self.pr, utilisation=0.60, num_instruments=10)
        self.assertGreaterEqual(h5.diversification, h1.diversification)

    def test_history_grows(self):
        self.engine.score(self.pr, utilisation=0.60)
        self.engine.score(self.pr, utilisation=0.70)
        self.assertEqual(len(self.engine.history()), 2)

    def test_latest_returns_most_recent(self):
        self.engine.score(self.pr, utilisation=0.60)
        self.engine.score(self.pr, utilisation=0.80)
        h = self.engine.latest()
        self.assertIsNotNone(h)

    def test_trend_direction(self):
        low_pr = PortfolioRisk(var_95=0.03, portfolio_volatility=0.15)
        high_pr = PortfolioRisk(var_95=0.01, portfolio_volatility=0.05)
        self.engine.score(low_pr, utilisation=0.30)
        self.engine.score(high_pr, utilisation=0.70)
        t = self.engine.trend()
        self.assertIn("direction", t)

    def test_allocation_quality_with_data(self):
        allocs = [
            {"status": "APPROVED", "confidence": 0.80},
            {"status": "APPROVED", "confidence": 0.75},
            {"status": "DENIED", "confidence": 0.30},
        ]
        h = self.engine.score(self.pr, utilisation=0.60, allocation_results=allocs)
        self.assertGreater(h.allocation_quality, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# 14. ICARI Database
# ══════════════════════════════════════════════════════════════════════════════


class TestIcariDatabase(unittest.TestCase):

    def setUp(self):
        self.db = IcariDatabase(db_path=":memory:")

    def _make_allocation_result(self):
        return AllocationResult(
            instrument="EURUSD",
            status=AllocationStatus.APPROVED,
            allocation_frac=0.10,
            allocation_usd=10_000.0,
            confidence=0.75,
            rationale="Test allocation",
        )

    def test_save_allocation_increments_count(self):
        self.db.save_allocation(self._make_allocation_result())
        self.assertEqual(self.db.count_allocations(), 1)

    def test_save_multiple_allocations(self):
        for _ in range(5):
            self.db.save_allocation(self._make_allocation_result())
        self.assertEqual(self.db.count_allocations(), 5)

    def test_recent_allocations_returns_list(self):
        self.db.save_allocation(self._make_allocation_result())
        rows = self.db.recent_allocations()
        self.assertIsInstance(rows, list)

    def test_allocation_stats_has_status_key(self):
        self.db.save_allocation(self._make_allocation_result())
        stats = self.db.allocation_stats()
        self.assertIn("APPROVED", stats)

    def test_save_budget(self):
        b = RiskBudget(budget_type=RiskBudgetType.DAILY, limit=0.02, used=0.005)
        self.db.save_budget(b)
        self.assertEqual(self.db.count_budgets(), 1)

    def test_save_portfolio_risk(self):
        pr = PortfolioRisk(var_95=0.012, var_99=0.020)
        self.db.save_portfolio_risk(pr)
        self.assertEqual(self.db.count_risk_snapshots(), 1)

    def test_latest_risk_populated(self):
        pr = PortfolioRisk(var_95=0.012)
        self.db.save_portfolio_risk(pr)
        row = self.db.latest_risk()
        self.assertIsNotNone(row)
        self.assertAlmostEqual(row["var_95"], 0.012, places=4)

    def test_save_scenario(self):
        sr = ScenarioResult(scenario=ScenarioType.FLASH_CRASH, resilience_score=0.40)
        self.db.save_scenario(sr)
        self.assertEqual(self.db.count_scenarios(), 1)

    def test_worst_scenario(self):
        self.db.save_scenario(ScenarioResult(scenario=ScenarioType.RISK_ON, resilience_score=0.90))
        self.db.save_scenario(
            ScenarioResult(scenario=ScenarioType.FLASH_CRASH, resilience_score=0.20)
        )
        worst = self.db.worst_scenario()
        self.assertIsNotNone(worst)
        self.assertAlmostEqual(worst["resilience"], 0.20, places=3)

    def test_save_health(self):
        h = CapitalHealth(overall_score=0.75, grade=CapitalHealthGrade.GOOD)
        self.db.save_health(h)
        self.assertEqual(self.db.count_health_snapshots(), 1)

    def test_latest_health(self):
        h = CapitalHealth(overall_score=0.80)
        self.db.save_health(h)
        row = self.db.latest_health()
        self.assertIsNotNone(row)

    def test_save_optimization(self):
        r = OptimizationResult(
            objective=OptimizationObjective.MAX_SHARPE,
            weights={"A": 0.5, "B": 0.5},
            converged=True,
        )
        self.db.save_optimization(r)
        self.assertEqual(self.db.count_optimizations(), 1)

    def test_save_drawdown_event(self):
        e = DrawdownEvent(
            peak=100_000.0, trough=93_000.0, drawdown=0.07, mode=DrawdownMode.SOFT_LIMIT
        )
        self.db.save_drawdown_event(e)
        self.assertEqual(self.db.count_drawdown_events(), 1)

    def test_unrecovered_events(self):
        e = DrawdownEvent(peak=100_000.0, trough=93_000.0, drawdown=0.07, recovered=False)
        self.db.save_drawdown_event(e)
        rows = self.db.unrecovered_events()
        self.assertEqual(len(rows), 1)

    def test_stats_all_tables(self):
        s = self.db.stats()
        for key in [
            "allocations",
            "risk_snapshots",
            "health_snapshots",
            "scenarios",
            "optimizations",
            "drawdown_events",
        ]:
            self.assertIn(key, s)


# ══════════════════════════════════════════════════════════════════════════════
# 15. ICARI Dashboard
# ══════════════════════════════════════════════════════════════════════════════


class TestIcariDashboard(unittest.TestCase):

    def setUp(self):
        self.dash = IcariDashboard()
        self.result = AllocationResult(
            instrument="EURUSD",
            status=AllocationStatus.APPROVED,
            allocation_frac=0.10,
            allocation_usd=10_000.0,
            confidence=0.75,
        )
        self.pr = PortfolioRisk(
            var_95=0.012, var_99=0.020, diversification_ratio=1.8, risk_concentration=0.2
        )
        self.budgets = [RiskBudget(budget_type=RiskBudgetType.DAILY, limit=0.02, used=0.01)]
        self.health = CapitalHealth(overall_score=0.75, grade=CapitalHealthGrade.GOOD)

    def test_capital_allocation_view(self):
        v = self.dash.capital_allocation([self.result])
        self.assertEqual(v["view"], "CAPITAL_ALLOCATION")
        self.assertEqual(v["approved"], 1)

    def test_capital_allocation_empty(self):
        v = self.dash.capital_allocation([])
        self.assertEqual(v["total"], 0)

    def test_risk_dashboard_view(self):
        v = self.dash.risk_dashboard(self.pr, self.budgets)
        self.assertEqual(v["view"], "RISK_DASHBOARD")
        self.assertIn("var_95", v)

    def test_exposure_dashboard_view(self):
        weights = {"EURUSD": 0.6, "GBPUSD": 0.4}
        rc = {"EURUSD": 0.6, "GBPUSD": 0.4}
        v = self.dash.exposure_dashboard(weights, rc)
        self.assertEqual(v["view"], "EXPOSURE_DASHBOARD")

    def test_drawdown_dashboard_view(self):
        events = [DrawdownEvent(peak=100_000.0, trough=95_000.0, drawdown=0.05)]
        v = self.dash.drawdown_dashboard(events, 0.05, DrawdownMode.SOFT_LIMIT, 0.75)
        self.assertEqual(v["view"], "DRAWDOWN_DASHBOARD")
        self.assertIn("mode", v)

    def test_capital_health_dashboard(self):
        v = self.dash.capital_health_dashboard(self.health)
        self.assertEqual(v["view"], "CAPITAL_HEALTH")
        self.assertEqual(v["grade"], "GOOD")

    def test_optimization_dashboard_empty(self):
        v = self.dash.optimization_dashboard([])
        self.assertEqual(v["runs"], 0)

    def test_optimization_dashboard_with_results(self):
        opt_r = OptimizationResult(
            objective=OptimizationObjective.MAX_SHARPE,
            weights={"A": 0.5, "B": 0.5},
            metrics={"sharpe": 1.2},
            converged=True,
        )
        v = self.dash.optimization_dashboard([opt_r])
        self.assertEqual(v["runs"], 1)
        self.assertIn("latest_objective", v)

    def test_scenario_dashboard_empty(self):
        v = self.dash.scenario_dashboard([])
        self.assertEqual(v["scenarios"], 0)

    def test_scenario_dashboard_with_results(self):
        sr = ScenarioResult(scenario=ScenarioType.FLASH_CRASH, resilience_score=0.30)
        v = self.dash.scenario_dashboard([sr])
        self.assertIn("avg_resilience", v)

    def test_health_alerts_critical(self):
        h = CapitalHealth(overall_score=0.20, grade=CapitalHealthGrade.CRITICAL)
        v = self.dash.capital_health_dashboard(h)
        self.assertTrue(any("CRITICAL" in a for a in v["alerts"]))

    def test_all_dashboards_returns_7(self):
        v = self.dash.all_dashboards(
            results=[self.result],
            portfolio_risk=self.pr,
            budgets=self.budgets,
            weights={"EURUSD": 1.0},
            risk_contrib={"EURUSD": 1.0},
            dd_events=[],
            current_dd=0.0,
            dd_mode=DrawdownMode.NORMAL,
            risk_mult=1.0,
            health=self.health,
            opt_results=[],
            scenario_results=[],
        )
        self.assertEqual(len(v), 7)


# ══════════════════════════════════════════════════════════════════════════════
# 16. ICARI API (Integration)
# ══════════════════════════════════════════════════════════════════════════════


class TestIcariAPI(unittest.TestCase):

    def setUp(self):
        self.api = IcariAPI(capital=100_000.0)
        self.req = make_request()

    def test_allocate_returns_dict(self):
        d = self.api.allocate(self.req)
        self.assertIn("status", d)

    def test_allocate_approved_has_usd(self):
        req = make_request(
            forecast_confidence=0.90,
            research_score=0.88,
            model_confidence=0.88,
            ensemble_confidence=0.85,
            decision_quality=0.87,
        )
        d = self.api.allocate(req)
        if d["status"] == "APPROVED":
            self.assertGreater(d["allocation_usd"], 0.0)

    def test_allocate_denied_zero_usd(self):
        req = make_request(
            forecast_confidence=0.01,
            research_score=0.01,
            execution_score=0.01,
            model_confidence=0.01,
            ensemble_confidence=0.01,
            decision_quality=0.01,
            liquidity=0.01,
        )
        d = self.api.allocate(req)
        self.assertEqual(d["status"], "DENIED")
        self.assertEqual(d["allocation_usd"], 0.0)

    def test_allocate_lockdown_blocks(self):
        self.api._dd_manager._mode = DrawdownMode.EMERGENCY_LOCKDOWN
        d = self.api.allocate(self.req)
        self.assertEqual(d["status"], "DENIED")
        self.api._dd_manager._mode = DrawdownMode.NORMAL

    def test_allocate_batch(self):
        reqs = [make_request(instrument=f"I{i}") for i in range(3)]
        results = self.api.allocate_batch(reqs)
        self.assertEqual(len(results), 3)

    def test_size_position_static(self):
        d = self.api.size_position(self.req, use_dynamic=False)
        self.assertIn("units", d)
        self.assertGreater(d["units"], 0.0)

    def test_size_position_dynamic(self):
        d = self.api.size_position(self.req, use_dynamic=True)
        self.assertTrue(d["adjusted"])

    def test_compare_sizing_methods(self):
        d = self.api.compare_sizing_methods(self.req)
        self.assertEqual(len(d), 9)

    def test_portfolio_risk(self):
        returns = {"EURUSD": make_returns(60), "GBPUSD": make_returns(60)}
        weights = {"EURUSD": 0.5, "GBPUSD": 0.5}
        d = self.api.portfolio_risk(returns, weights)
        self.assertIn("var_95", d)

    def test_correlation_matrix(self):
        returns = {"EURUSD": make_returns(30), "GBPUSD": make_returns(30)}
        m = self.api.correlation_matrix(returns)
        self.assertIn("EURUSD", m)

    def test_correlation_clusters(self):
        returns = {"EURUSD": make_returns(30), "GBPUSD": make_returns(30)}
        c = self.api.correlation_clusters(returns)
        self.assertIsInstance(c, list)

    def test_capital_health(self):
        d = self.api.capital_health(utilisation=0.60)
        self.assertIn("overall_score", d)

    def test_get_budgets(self):
        s = self.api.get_budgets()
        self.assertIn("total_budgets", s)
        self.assertEqual(s["total_budgets"], 12)

    def test_set_budget(self):
        d = self.api.set_budget(RiskBudgetType.DAILY, 0.03, "Test")
        self.assertAlmostEqual(d["limit"], 0.03, places=4)

    def test_check_budget(self):
        d = self.api.check_budget(RiskBudgetType.DAILY, 0.005)
        self.assertTrue(d["allowed"])

    def test_reset_budget(self):
        self.api.get_budgets()
        ok = self.api.reset_budget(RiskBudgetType.DAILY)
        self.assertTrue(ok)

    def test_optimize(self):
        returns = {"A": make_returns(60), "B": make_returns(60)}
        d = self.api.optimize(OptimizationObjective.MAX_SHARPE, returns)
        self.assertIn("weights", d)

    def test_optimize_all(self):
        returns = {"A": make_returns(60), "B": make_returns(60)}
        d = self.api.optimize_all(returns)
        self.assertEqual(len(d), 8)

    def test_scenario(self):
        weights = {"EURUSD": 0.6, "BTCUSD": 0.4}
        vols = {"EURUSD": 0.008, "BTCUSD": 0.04}
        d = self.api.scenario(ScenarioType.FLASH_CRASH, weights, vols)
        self.assertIn("resilience_score", d)

    def test_scenario_all(self):
        weights = {"EURUSD": 0.6, "BTCUSD": 0.4}
        vols = {"EURUSD": 0.008, "BTCUSD": 0.04}
        d = self.api.scenario_all(weights, vols)
        self.assertIn("scenarios", d)
        self.assertEqual(len(d["scenarios"]), 11)

    def test_update_equity(self):
        d = self.api.update_equity(95_000.0)
        self.assertIn("drawdown", d)

    def test_drawdown_stats(self):
        s = self.api.drawdown_stats()
        self.assertIn("mode", s)

    def test_capital_efficiency(self):
        d = self.api.capital_efficiency(60_000, 5_000, 1_000, 500, 4)
        self.assertIn("utilisation", d)

    def test_get_dashboard_all(self):
        d = self.api.get_dashboard("all")
        self.assertEqual(len(d), 7)

    def test_get_dashboard_single(self):
        d = self.api.get_dashboard("capital_allocation")
        self.assertEqual(d["view"], "CAPITAL_ALLOCATION")

    def test_get_dashboard_unknown(self):
        d = self.api.get_dashboard("nonexistent_view")
        self.assertIn("error", d)

    def test_system_stats_keys(self):
        s = self.api.system_stats()
        self.assertIn("capital", s)
        self.assertIn("database", s)
        self.assertIn("drawdown", s)

    def test_reset_all_budgets(self):
        self.api.reset_all_budgets()
        s = self.api.get_budgets()
        self.assertEqual(s["breached"], 0)


if __name__ == "__main__":
    unittest.main()
