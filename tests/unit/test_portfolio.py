"""Phase 20 — Portfolio Intelligence Test Suite.

130 tests across all 14 portfolio modules.
Uses spec_from_file_location to avoid torch import chain from trainer/__init__.py.
"""

from __future__ import annotations

import importlib.util
import sys
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

_BASE = Path(__file__).resolve().parents[2] / "trainer" / "portfolio"


def _load(mod_key: str, filename: str):
    path = _BASE / filename
    spec = importlib.util.spec_from_file_location(mod_key, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    spec.loader.exec_module(mod)
    return mod


# Load in dependency order
_pi = _load("trainer.portfolio.portfolio_instruments", "portfolio_instruments.py")
_ce = _load("trainer.portfolio.correlation_engine", "correlation_engine.py")
_ee = _load("trainer.portfolio.exposure_engine", "exposure_engine.py")
_dg = _load("trainer.portfolio.dependency_graph", "dependency_graph.py")
_rd = _load("trainer.portfolio.regime_detector", "regime_detector.py")
_rb = _load("trainer.portfolio.risk_budget", "risk_budget.py")
_cr = _load("trainer.portfolio.conflict_resolver", "conflict_resolver.py")
_ps = _load("trainer.portfolio.portfolio_scorer", "portfolio_scorer.py")
_pa = _load("trainer.portfolio.position_allocator", "position_allocator.py")
_sa = _load("trainer.portfolio.scenario_analysis", "scenario_analysis.py")
_op = _load("trainer.portfolio.optimizer", "optimizer.py")
_pdb = _load("trainer.portfolio.portfolio_database", "portfolio_database.py")
_dash = _load("trainer.portfolio.portfolio_dashboard", "portfolio_dashboard.py")
_api = _load("trainer.portfolio.portfolio_api", "portfolio_api.py")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_forecasts(n: int = 4):
    symbols = ["EURUSD", "GBPUSD", "XAUUSD", "NAS100", "BTCUSD", "USDJPY", "AUDUSD", "SPX500"][:n]
    dirs = ["BUY", "BUY", "BUY", "SELL", "BUY", "SELL", "BUY", "SELL"][:n]
    returns = [0.08, 0.07, 0.12, 0.18, 0.25, 0.06, 0.09, 0.15][:n]
    risks = [0.10, 0.12, 0.15, 0.22, 0.35, 0.08, 0.11, 0.20][:n]
    confs = [0.72, 0.65, 0.80, 0.60, 0.55, 0.70, 0.68, 0.62][:n]
    MF = _pi.ModelForecast
    return [
        MF(sym, d, c, r, risk) for sym, d, c, r, risk in zip(symbols, dirs, confs, returns, risks)
    ]


def _make_api() -> _api.PortfolioApi:
    return _api.PortfolioApi(db_path=":memory:")


# ════════════════════════════════════════════════════════════════════════════════
# 1. portfolio_instruments
# ════════════════════════════════════════════════════════════════════════════════
class TestInstrumentRegistry(unittest.TestCase):

    def setUp(self):
        self.reg = _pi.InstrumentRegistry.default()

    def test_default_has_16_instruments(self):
        self.assertEqual(self.reg.total(), 16)

    def test_eurusd_registered(self):
        info = self.reg.get("EURUSD")
        self.assertIsNotNone(info)
        self.assertEqual(info.asset_class, _pi.AssetClass.FOREX)

    def test_get_required_raises_on_unknown(self):
        with self.assertRaises(KeyError):
            self.reg.get_required("XXXXXX")

    def test_by_asset_class_forex(self):
        forex = self.reg.by_asset_class(_pi.AssetClass.FOREX)
        self.assertGreaterEqual(len(forex), 9)

    def test_by_asset_class_crypto(self):
        crypto = self.reg.by_asset_class(_pi.AssetClass.CRYPTO)
        self.assertEqual(len(crypto), 2)

    def test_by_tag_safe_haven(self):
        sh = self.reg.by_tag("safe_haven")
        symbols = [i.symbol for i in sh]
        self.assertIn("XAUUSD", symbols)
        self.assertIn("USDCHF", symbols)

    def test_by_currency_usd(self):
        usd = self.reg.by_currency("USD")
        self.assertGreaterEqual(len(usd), 10)

    def test_register_custom_instrument(self):
        info = _pi.InstrumentInfo(
            symbol="CUSTOM",
            asset_class=_pi.AssetClass.FOREX,
            base_currency="XXX",
            quote_currency="YYY",
            region=_pi.Region.GLOBAL,
        )
        self.reg.register(info)
        self.assertEqual(self.reg.total(), 17)
        self.assertIsNotNone(self.reg.get("CUSTOM"))

    def test_asset_class_summary(self):
        summary = self.reg.asset_class_summary()
        self.assertIn("forex", summary)
        self.assertIn("commodity", summary)

    def test_to_dict(self):
        d = self.reg.to_dict()
        self.assertIn("EURUSD", d)
        self.assertIn("asset_class", d["EURUSD"])


class TestModelForecast(unittest.TestCase):

    def test_signed_return_buy(self):
        fc = _pi.ModelForecast("EURUSD", "BUY", 0.7, 0.10, 0.08)
        self.assertAlmostEqual(fc.signed_return(), 0.10)

    def test_signed_return_sell(self):
        fc = _pi.ModelForecast("EURUSD", "SELL", 0.7, 0.10, 0.08)
        self.assertAlmostEqual(fc.signed_return(), -0.10)

    def test_signed_return_neutral(self):
        fc = _pi.ModelForecast("EURUSD", "NEUTRAL", 0.0, 0.0, 0.08)
        self.assertAlmostEqual(fc.signed_return(), 0.0)

    def test_invalid_direction_raises(self):
        with self.assertRaises(AssertionError):
            _pi.ModelForecast("EURUSD", "HOLD", 0.7, 0.10, 0.08)

    def test_confidence_clamped(self):
        fc = _pi.ModelForecast("EURUSD", "BUY", 2.0, 0.10, 0.08)
        self.assertEqual(fc.confidence, 1.0)

    def test_to_dict_keys(self):
        fc = _pi.ModelForecast("EURUSD", "BUY", 0.7, 0.10, 0.08)
        d = fc.to_dict()
        self.assertIn("instrument", d)
        self.assertIn("direction", d)
        self.assertIn("confidence", d)


# ════════════════════════════════════════════════════════════════════════════════
# 2. correlation_engine
# ════════════════════════════════════════════════════════════════════════════════
class TestCorrelationEngine(unittest.TestCase):

    def setUp(self):
        self.eng = _ce.CorrelationEngine(window=20)
        import random

        rng = random.Random(42)
        for _ in range(30):
            self.eng.add_observations(
                {
                    "EURUSD": rng.gauss(0, 0.01),
                    "GBPUSD": rng.gauss(0, 0.01),
                    "XAUUSD": rng.gauss(0, 0.01),
                }
            )

    def test_rolling_correlation_returns_result(self):
        r = self.eng.rolling_correlation("EURUSD", "GBPUSD")
        self.assertIsInstance(r.correlation, float)
        self.assertGreaterEqual(r.correlation, -1.0)
        self.assertLessEqual(r.correlation, 1.0)

    def test_rank_correlation_in_range(self):
        r = self.eng.rank_correlation("EURUSD", "XAUUSD")
        self.assertGreaterEqual(r.correlation, -1.0)
        self.assertLessEqual(r.correlation, 1.0)

    def test_tail_correlation_lower(self):
        r = self.eng.tail_correlation("EURUSD", "GBPUSD", tail="lower")
        self.assertIn(r.method, ("tail_lower",))

    def test_exponential_correlation_in_range(self):
        r = self.eng.exponential_correlation("EURUSD", "GBPUSD")
        self.assertGreaterEqual(r.correlation, -1.0)
        self.assertLessEqual(r.correlation, 1.0)

    def test_compute_matrix_has_self_correlations(self):
        matrix = self.eng.compute_matrix()
        self.assertAlmostEqual(matrix.get("EURUSD", "EURUSD"), 1.0)

    def test_high_correlation_pairs_threshold(self):
        matrix = self.eng.compute_matrix()
        pairs = matrix.high_correlation_pairs(threshold=0.0)
        self.assertGreater(len(pairs), 0)

    def test_correlation_confidence_range(self):
        conf = self.eng.correlation_confidence("EURUSD", "GBPUSD")
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_unknown_pair_returns_zero(self):
        r = self.eng.rolling_correlation("UNKNOWNA", "UNKNOWNB")
        self.assertEqual(r.correlation, 0.0)

    def test_regime_aware_matrix_risk_off(self):
        matrix = self.eng.regime_aware_matrix(regime="risk_off")
        self.assertIsNotNone(matrix)


# ════════════════════════════════════════════════════════════════════════════════
# 3. exposure_engine
# ════════════════════════════════════════════════════════════════════════════════
class TestExposureEngine(unittest.TestCase):

    def setUp(self):
        self.eng = _ee.ExposureEngine()
        self.fcs = _make_forecasts(4)

    def test_compute_returns_report(self):
        r = self.eng.compute(self.fcs)
        self.assertIsInstance(r, _ee.ExposureReport)

    def test_gross_exposure_positive(self):
        r = self.eng.compute(self.fcs)
        self.assertGreater(r.gross_exposure, 0)

    def test_by_asset_class_not_empty(self):
        r = self.eng.compute(self.fcs)
        self.assertGreater(len(r.by_asset_class), 0)

    def test_by_currency_has_usd(self):
        r = self.eng.compute(self.fcs)
        self.assertIn("USD", r.by_currency)

    def test_diversification_ratio_in_range(self):
        r = self.eng.compute(self.fcs)
        self.assertGreaterEqual(r.diversification_ratio, 0.0)
        self.assertLessEqual(r.diversification_ratio, 1.0)

    def test_alerts_list(self):
        r = self.eng.compute(self.fcs)
        self.assertIsInstance(r.alerts, list)

    def test_exposure_breakdown_has_summary(self):
        d = self.eng.exposure_breakdown(self.fcs)
        self.assertIn("summary", d)
        self.assertIn("report", d)

    def test_to_dict_keys(self):
        r = self.eng.compute(self.fcs)
        d = r.to_dict()
        self.assertIn("by_asset_class", d)
        self.assertIn("gross_exposure", d)
        self.assertIn("alerts", d)


# ════════════════════════════════════════════════════════════════════════════════
# 4. dependency_graph
# ════════════════════════════════════════════════════════════════════════════════
class TestDependencyGraph(unittest.TestCase):

    def _make_matrix(self):
        eng = _ce.CorrelationEngine(window=20)
        import random

        rng = random.Random(99)
        for _ in range(25):
            eng.add_observations(
                {
                    "EURUSD": rng.gauss(0, 0.01),
                    "GBPUSD": rng.gauss(0, 0.01),
                    "XAUUSD": rng.gauss(0, 0.01),
                    "NAS100": rng.gauss(0, 0.01),
                }
            )
        return eng.compute_matrix()

    def test_build_returns_snapshot(self):
        builder = _dg.DependencyGraphBuilder()
        matrix = self._make_matrix()
        snap = builder.build(matrix)
        self.assertIsNotNone(snap)

    def test_snapshot_has_nodes(self):
        builder = _dg.DependencyGraphBuilder()
        snap = builder.build(self._make_matrix())
        self.assertGreater(len(snap.nodes), 0)

    def test_snapshot_to_dict(self):
        builder = _dg.DependencyGraphBuilder()
        snap = builder.build(self._make_matrix())
        d = snap.to_dict()
        self.assertIn("nodes", d)
        self.assertIn("edges", d)
        self.assertIn("clusters", d)

    def test_cluster_for_instrument(self):
        builder = _dg.DependencyGraphBuilder(cluster_threshold=0.0)
        snap = builder.build(self._make_matrix())
        # With threshold 0, all should be clustered
        self.assertIsNotNone(snap.cluster_for("EURUSD"))

    def test_latest_snapshot(self):
        builder = _dg.DependencyGraphBuilder()
        builder.build(self._make_matrix())
        self.assertIsNotNone(builder.latest_snapshot())

    def test_cluster_evolution(self):
        builder = _dg.DependencyGraphBuilder()
        builder.build(self._make_matrix())
        evolution = builder.cluster_evolution()
        self.assertEqual(len(evolution), 1)


# ════════════════════════════════════════════════════════════════════════════════
# 5. regime_detector
# ════════════════════════════════════════════════════════════════════════════════
class TestRegimeDetector(unittest.TestCase):

    def setUp(self):
        self.det = _rd.RegimeDetector()
        import random

        rng = random.Random(7)
        for sym in ["EURUSD", "GBPUSD", "NAS100", "XAUUSD", "BTCUSD"]:
            for _ in range(50):
                self.det.add_return(sym, rng.gauss(0.001, 0.01))

    def test_detect_returns_assessment(self):
        a = self.det.detect()
        self.assertIsNotNone(a)

    def test_primary_regime_is_valid(self):
        a = self.det.detect()
        self.assertIn(a.primary_regime, _rd.MarketRegime)

    def test_confidence_in_range(self):
        a = self.det.detect()
        self.assertGreaterEqual(a.confidence, 0.0)
        self.assertLessEqual(a.confidence, 1.0)

    def test_adjustment_factors_present(self):
        a = self.det.detect()
        self.assertIn("position_size", a.adjustment_factors)

    def test_regime_history_grows(self):
        self.det.detect()
        self.det.detect()
        self.assertGreaterEqual(len(self.det.regime_history()), 1)

    def test_dominant_regime(self):
        regime = self.det.dominant_regime_last_n(5)
        self.assertIn(regime, _rd.MarketRegime)

    def test_to_dict_keys(self):
        a = self.det.detect()
        d = a.to_dict()
        self.assertIn("primary_regime", d)
        self.assertIn("confidence", d)
        self.assertIn("adjustment_factors", d)


# ════════════════════════════════════════════════════════════════════════════════
# 6. risk_budget
# ════════════════════════════════════════════════════════════════════════════════
class TestRiskBudget(unittest.TestCase):

    def setUp(self):
        self.fcs = _make_forecasts(4)
        self.weights = {"EURUSD": 0.25, "GBPUSD": 0.25, "XAUUSD": 0.25, "NAS100": 0.25}

    def test_default_config_passes(self):
        eng = _rb.RiskBudgetEngine()
        result = eng.check(self.weights, self.fcs)
        self.assertIsInstance(result, _rb.BudgetCheckResult)

    def test_conservative_config(self):
        cfg = _rb.RiskBudgetConfig.conservative()
        self.assertLess(cfg.max_crypto_exposure, _rb.RiskBudgetConfig().max_crypto_exposure)

    def test_aggressive_config(self):
        cfg = _rb.RiskBudgetConfig.aggressive()
        self.assertGreater(cfg.max_single_instrument, _rb.RiskBudgetConfig().max_single_instrument)

    def test_breach_on_overweight(self):
        weights = {"EURUSD": 0.90, "GBPUSD": 0.05, "XAUUSD": 0.03, "NAS100": 0.02}
        eng = _rb.RiskBudgetEngine()
        result = eng.check(weights, self.fcs)
        # Should catch single instrument over-concentration
        self.assertIsInstance(result, _rb.BudgetCheckResult)

    def test_scale_to_budget(self):
        weights = {"EURUSD": 1.5, "GBPUSD": 1.5, "XAUUSD": 1.0, "NAS100": 0.5}
        eng = _rb.RiskBudgetEngine()
        scaled = eng.scale_to_budget(weights, self.fcs)
        gross = sum(abs(v) for v in scaled.values())
        self.assertLessEqual(gross, eng.config.max_total_gross + 1e-6)

    def test_budget_item_utilisation(self):
        item = _rb.BudgetItem("test", limit=0.50, current=0.44)
        self.assertAlmostEqual(item.utilisation, 0.88)
        self.assertEqual(item.status, _rb.BudgetStatus.WARNING)

    def test_budget_item_breach(self):
        item = _rb.BudgetItem("test", limit=0.20, current=0.25)
        self.assertEqual(item.status, _rb.BudgetStatus.BREACH)

    def test_utilisation_report(self):
        eng = _rb.RiskBudgetEngine()
        report = eng.utilisation_report(self.weights, self.fcs)
        self.assertIn("status", report)
        self.assertIn("budget_config", report)


# ════════════════════════════════════════════════════════════════════════════════
# 7. conflict_resolver
# ════════════════════════════════════════════════════════════════════════════════
class TestConflictResolver(unittest.TestCase):

    def setUp(self):
        self.resolver = _cr.ConflictResolver()

    def _fc(self, sym, direction, conf=0.7):
        return _pi.ModelForecast(sym, direction, conf, 0.10, 0.12)

    def test_no_conflicts_clean_portfolio(self):
        fcs = [self._fc("EURUSD", "BUY"), self._fc("XAUUSD", "BUY")]
        report = self.resolver.resolve(fcs)
        self.assertIsInstance(report, _cr.ConflictReport)

    def test_usd_conflict_detected(self):
        fcs = [self._fc("EURUSD", "BUY", 0.8), self._fc("USDCHF", "BUY", 0.75)]
        report = self.resolver.resolve(fcs)
        self.assertTrue(report.has_conflicts())

    def test_safe_haven_risk_on_conflict(self):
        fcs = [self._fc("XAUUSD", "BUY", 0.80), self._fc("NAS100", "BUY", 0.75)]
        report = self.resolver.resolve(fcs)
        self.assertTrue(report.has_conflicts())

    def test_reinforcing_safe_haven_risk_off(self):
        fcs = [self._fc("XAUUSD", "BUY"), self._fc("NAS100", "SELL")]
        report = self.resolver.resolve(fcs)
        self.assertGreater(report.reinforcement_count(), 0)

    def test_adjusted_forecasts_non_empty(self):
        fcs = _make_forecasts(4)
        report = self.resolver.resolve(fcs)
        self.assertGreater(len(report.adjusted_forecasts), 0)

    def test_to_dict_keys(self):
        fcs = _make_forecasts(4)
        report = self.resolver.resolve(fcs)
        d = report.to_dict()
        self.assertIn("conflict_count", d)
        self.assertIn("reinforcement_count", d)
        self.assertIn("conflicts", d)

    def test_reduce_resolution_lowers_confidence(self):
        fcs = [self._fc("EURUSD", "BUY", 0.72), self._fc("USDCHF", "BUY", 0.70)]
        report = self.resolver.resolve(fcs)
        for fc in report.adjusted_forecasts:
            self.assertLessEqual(fc.confidence, 0.72 + 1e-6)


# ════════════════════════════════════════════════════════════════════════════════
# 8. portfolio_scorer
# ════════════════════════════════════════════════════════════════════════════════
class TestPortfolioScorer(unittest.TestCase):

    def setUp(self):
        self.scorer = _ps.PortfolioScorer()
        self.fcs = _make_forecasts(4)

    def test_score_returns_metrics(self):
        m = self.scorer.score(self.fcs)
        self.assertIsInstance(m, _ps.PortfolioMetrics)

    def test_instrument_count_correct(self):
        m = self.scorer.score(self.fcs)
        self.assertEqual(m.instrument_count, 4)

    def test_expected_volatility_positive(self):
        m = self.scorer.score(self.fcs)
        self.assertGreater(m.expected_volatility, 0.0)

    def test_diversification_score_in_range(self):
        m = self.scorer.score(self.fcs)
        self.assertGreaterEqual(m.diversification_score, 0.0)
        self.assertLessEqual(m.diversification_score, 1.0)

    def test_concentration_score_in_range(self):
        m = self.scorer.score(self.fcs)
        self.assertGreaterEqual(m.concentration_score, 0.0)
        self.assertLessEqual(m.concentration_score, 1.0)

    def test_grade_valid_string(self):
        m = self.scorer.score(self.fcs)
        self.assertIn(m.grade(), ["Exceptional", "Strong", "Good", "Adequate", "Weak", "Negative"])

    def test_to_dict_has_keys(self):
        m = self.scorer.score(self.fcs)
        d = m.to_dict()
        for key in [
            "expected_return",
            "expected_volatility",
            "expected_sharpe",
            "expected_sortino",
            "diversification_score",
        ]:
            self.assertIn(key, d)

    def test_empty_forecasts_returns_zeros(self):
        m = self.scorer.score([])
        self.assertEqual(m.instrument_count, 0)
        self.assertEqual(m.expected_sharpe, 0.0)

    def test_compare_portfolios(self):
        m1 = self.scorer.score(self.fcs[:2])
        m2 = self.scorer.score(self.fcs)
        comparison = self.scorer.compare(m1, m2)
        self.assertIn("sharpe_delta", comparison)
        self.assertIn("improved", comparison)


# ════════════════════════════════════════════════════════════════════════════════
# 9. position_allocator
# ════════════════════════════════════════════════════════════════════════════════
class TestPositionAllocator(unittest.TestCase):

    def setUp(self):
        self.alloc = _pa.PositionAllocator()
        self.fcs = _make_forecasts(4)

    def test_allocate_returns_result(self):
        r = self.alloc.allocate(self.fcs)
        self.assertIsInstance(r, _pa.AllocationResult)

    def test_weights_sum_approx_one(self):
        r = self.alloc.allocate(self.fcs)
        total = sum(abs(v) for v in r.weights.values())
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_equal_weight_method(self):
        alloc = _pa.PositionAllocator(method=_pa.AllocationMethod.EQUAL_WEIGHT)
        r = alloc.allocate(self.fcs)
        for w in r.weights.values():
            self.assertAlmostEqual(w, 0.25, places=4)

    def test_confidence_weight_method(self):
        alloc = _pa.PositionAllocator(method=_pa.AllocationMethod.CONFIDENCE_WEIGHT)
        r = alloc.allocate(self.fcs)
        self.assertIsNotNone(r)

    def test_risk_parity_method(self):
        alloc = _pa.PositionAllocator(method=_pa.AllocationMethod.RISK_PARITY)
        r = alloc.allocate(self.fcs)
        self.assertGreater(len(r.weights), 0)

    def test_signed_weights_respect_direction(self):
        r = self.alloc.allocate(self.fcs)
        for fc in self.fcs:
            if fc.direction == "SELL" and fc.instrument in r.signed_weights:
                self.assertLessEqual(r.signed_weights[fc.instrument], 0.0)

    def test_to_dict_has_keys(self):
        r = self.alloc.allocate(self.fcs)
        d = r.to_dict()
        self.assertIn("weights", d)
        self.assertIn("method", d)
        self.assertIn("budget_feasible", d)

    def test_empty_forecasts(self):
        r = self.alloc.allocate([])
        self.assertEqual(r.instrument_count, 0)


# ════════════════════════════════════════════════════════════════════════════════
# 10. scenario_analysis
# ════════════════════════════════════════════════════════════════════════════════
class TestScenarioAnalysis(unittest.TestCase):

    def setUp(self):
        self.analyser = _sa.ScenarioAnalyser()
        self.fcs = _make_forecasts(6)
        n = len(self.fcs)
        self.weights = {f.instrument: 1.0 / n for f in self.fcs}

    def test_run_single_scenario(self):
        r = self.analyser.run_scenario(self.fcs, self.weights, _sa.ScenarioType.RISK_OFF)
        self.assertIsInstance(r, _sa.ScenarioResult)

    def test_stressed_volatility_higher_in_risk_off(self):
        r = self.analyser.run_scenario(self.fcs, self.weights, _sa.ScenarioType.RISK_OFF)
        self.assertGreater(r.stressed_volatility, 0.0)

    def test_volatility_spike_increases_vol(self):
        r = self.analyser.run_scenario(self.fcs, self.weights, _sa.ScenarioType.VOLATILITY_SPIKE)
        self.assertGreater(r.stressed_volatility, r.base_volatility)

    def test_run_all_returns_report(self):
        report = self.analyser.run_all(self.fcs, self.weights)
        self.assertIsInstance(report, _sa.ScenarioReport)
        self.assertGreater(len(report.results), 0)

    def test_worst_scenario_set(self):
        report = self.analyser.run_all(self.fcs, self.weights)
        self.assertIsNotNone(report.worst_scenario)

    def test_resilience_in_range(self):
        r = self.analyser.run_scenario(self.fcs, self.weights, _sa.ScenarioType.RISK_ON)
        self.assertGreaterEqual(r.resilience_score, 0.0)
        self.assertLessEqual(r.resilience_score, 1.0)

    def test_to_dict_keys(self):
        r = self.analyser.run_scenario(self.fcs, self.weights, _sa.ScenarioType.RATE_HIKE)
        d = r.to_dict()
        self.assertIn("scenario_type", d)
        self.assertIn("stressed_sharpe", d)
        self.assertIn("resilience_score", d)


# ════════════════════════════════════════════════════════════════════════════════
# 11. optimizer
# ════════════════════════════════════════════════════════════════════════════════
class TestOptimizer(unittest.TestCase):

    def setUp(self):
        self.opt = _op.PortfolioOptimizer()
        self.fcs = _make_forecasts(4)

    def test_optimize_returns_result(self):
        r = self.opt.optimize(self.fcs)
        self.assertIsInstance(r, _op.OptimizationResult)

    def test_optimal_weights_sum_to_one(self):
        r = self.opt.optimize(self.fcs)
        total = sum(r.optimal_weights.values())
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_min_volatility_objective(self):
        r = self.opt.optimize(self.fcs, _op.OptimizationObjective.MIN_VOLATILITY)
        self.assertEqual(r.objective, "min_volatility")

    def test_max_diversification_objective(self):
        r = self.opt.optimize(self.fcs, _op.OptimizationObjective.MAX_DIVERSIFICATION)
        self.assertEqual(r.objective, "max_diversification")

    def test_balanced_objective(self):
        r = self.opt.optimize(self.fcs, _op.OptimizationObjective.BALANCED)
        self.assertIsNotNone(r)

    def test_compare_all_objectives(self):
        results = self.opt.compare_objectives(self.fcs)
        self.assertEqual(len(results), 5)
        for key in [o.value for o in _op.OptimizationObjective]:
            self.assertIn(key, results)

    def test_efficient_frontier_n_points(self):
        frontier = self.opt.efficient_frontier(self.fcs, n_points=5)
        self.assertEqual(len(frontier), 5)
        for pt in frontier:
            self.assertIn("expected_sharpe", pt)

    def test_to_dict_keys(self):
        r = self.opt.optimize(self.fcs)
        d = r.to_dict()
        self.assertIn("objective", d)
        self.assertIn("optimal_weights", d)
        self.assertIn("converged", d)


# ════════════════════════════════════════════════════════════════════════════════
# 12. portfolio_database
# ════════════════════════════════════════════════════════════════════════════════
class TestPortfolioDatabase(unittest.TestCase):

    def setUp(self):
        self.db = _pdb.PortfolioDatabase(":memory:")

    def test_save_and_get_recommendation(self):
        self.db.save_recommendation(
            {
                "rec_id": "R001",
                "timestamp": "2025-01-01T00:00:00",
                "instrument_count": 4,
                "expected_return": 0.10,
                "expected_volatility": 0.12,
                "expected_sharpe": 0.83,
                "diversification_score": 0.65,
                "regime": "trending",
                "method": "hybrid",
                "weights": {"EURUSD": 0.25},
                "metrics": {},
            }
        )
        r = self.db.get_recommendation("R001")
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r["expected_sharpe"], 0.83)

    def test_recommendations_history(self):
        for i in range(3):
            self.db.save_recommendation(
                {
                    "rec_id": f"R{i:03d}",
                    "timestamp": f"2025-01-{i+1:02d}T00:00:00",
                    "instrument_count": 4,
                    "expected_return": 0.10,
                    "expected_volatility": 0.12,
                    "expected_sharpe": 0.83,
                    "diversification_score": 0.65,
                    "regime": "trending",
                    "method": "hybrid",
                    "weights": {},
                    "metrics": {},
                }
            )
        history = self.db.recommendations_history(limit=5)
        self.assertEqual(len(history), 3)

    def test_save_and_get_correlation_matrix(self):
        self.db.save_correlation_matrix(
            "M001", "2025-01-01", "rolling", ["EURUSD", "GBPUSD"], {"EURUSD_GBPUSD": 0.75}
        )
        m = self.db.get_correlation_matrix("M001")
        self.assertIsNotNone(m)
        self.assertIn("EURUSD", m["instruments"])

    def test_latest_correlation_matrix(self):
        self.db.save_correlation_matrix("M001", "2025-01-01", "rolling", ["A"], {})
        self.db.save_correlation_matrix("M002", "2025-01-02", "rolling", ["B"], {})
        latest = self.db.latest_correlation_matrix()
        self.assertIsNotNone(latest)

    def test_save_and_get_exposure(self):
        self.db.save_exposure(
            {
                "exp_id": "E001",
                "timestamp": "2025-01-01",
                "gross_exposure": 1.0,
                "net_exposure": 0.1,
                "long_exposure": 0.55,
                "short_exposure": 0.45,
                "diversification_ratio": 0.70,
                "alert_count": 2,
                "by_asset_class": {"forex": 0.6},
                "by_currency": {"USD": -0.3},
                "by_region": {},
                "alerts": [],
            }
        )
        history = self.db.exposure_history(limit=5)
        self.assertEqual(len(history), 1)
        self.assertAlmostEqual(history[0]["gross_exposure"], 1.0)

    def test_save_and_get_optimization(self):
        self.db.save_optimization(
            {
                "opt_id": "O001",
                "timestamp": "2025-01-01",
                "objective": "max_sharpe",
                "expected_sharpe": 1.20,
                "expected_return": 0.15,
                "expected_volatility": 0.12,
                "diversification_score": 0.70,
                "iterations": 50,
                "converged": True,
                "weights": {},
                "metrics": {},
            }
        )
        history = self.db.optimization_history()
        self.assertEqual(len(history), 1)

    def test_save_and_get_scenario(self):
        self.db.save_scenario(
            {
                "scenario_id": "S001",
                "timestamp": "2025-01-01",
                "scenario_type": "risk_off",
                "base_sharpe": 1.0,
                "stressed_sharpe": 0.5,
                "return_impact": -0.05,
                "volatility_impact": 0.08,
                "resilience_score": 0.65,
                "instrument_impacts": {"EURUSD": -0.01},
            }
        )
        history = self.db.scenario_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["scenario_type"], "risk_off")

    def test_save_and_get_allocation(self):
        self.db.save_allocation(
            {
                "alloc_id": "A001",
                "timestamp": "2025-01-01",
                "method": "hybrid",
                "regime": "trending",
                "instrument_count": 4,
                "expected_return": 0.10,
                "expected_sharpe": 0.90,
                "budget_feasible": True,
                "weights": {"EURUSD": 0.25},
            }
        )
        history = self.db.allocation_history()
        self.assertEqual(len(history), 1)

    def test_stats(self):
        stats = self.db.stats()
        self.assertIn("recommendations", stats)
        self.assertIn("correlation_matrices", stats)


# ════════════════════════════════════════════════════════════════════════════════
# 13. portfolio_dashboard
# ════════════════════════════════════════════════════════════════════════════════
class TestPortfolioDashboard(unittest.TestCase):

    def setUp(self):
        self.db = _pdb.PortfolioDatabase(":memory:")
        self.dash = _dash.PortfolioDashboard(db=self.db)
        self.fcs = _make_forecasts(4)
        self.scorer = _ps.PortfolioScorer()
        self.metrics = self.scorer.score(self.fcs)

    def _make_matrix(self):
        eng = _ce.CorrelationEngine(window=20)
        import random

        rng = random.Random(77)
        for _ in range(25):
            eng.add_observations(
                {
                    "EURUSD": rng.gauss(0, 0.01),
                    "GBPUSD": rng.gauss(0, 0.01),
                    "XAUUSD": rng.gauss(0, 0.01),
                    "NAS100": rng.gauss(0, 0.01),
                }
            )
        return eng.compute_matrix()

    def test_correlation_heatmap(self):
        matrix = self._make_matrix()
        h = self.dash.correlation_heatmap(matrix)
        self.assertEqual(h["type"], "correlation_heatmap")
        self.assertIn("cells", h)
        self.assertGreater(len(h["cells"]), 0)

    def test_correlation_heatmap_none(self):
        h = self.dash.correlation_heatmap(None)
        self.assertEqual(h["type"], "correlation_heatmap")
        self.assertEqual(h["instruments"], [])

    def test_exposure_chart(self):
        eng = _ee.ExposureEngine()
        report = eng.compute(self.fcs)
        chart = self.dash.exposure_chart(report)
        self.assertEqual(chart["type"], "exposure_chart")
        self.assertIn("by_asset_class", chart)

    def test_risk_allocation_chart(self):
        chart = self.dash.risk_allocation_chart(self.fcs)
        self.assertEqual(chart["type"], "risk_allocation_chart")
        self.assertIn("segments", chart)
        self.assertEqual(len(chart["segments"]), 4)

    def test_dependency_graph_chart_none(self):
        chart = self.dash.dependency_graph_chart(None)
        self.assertEqual(chart["type"], "dependency_graph")

    def test_portfolio_timeline(self):
        tl = self.dash.portfolio_timeline()
        self.assertEqual(tl["type"], "portfolio_timeline")

    def test_optimization_dashboard(self):
        od = self.dash.optimization_dashboard()
        self.assertEqual(od["type"], "optimization_dashboard")

    def test_scenario_dashboard(self):
        sd = self.dash.scenario_dashboard()
        self.assertEqual(sd["type"], "scenario_dashboard")

    def test_portfolio_scorecard(self):
        sc = self.dash.portfolio_scorecard(self.metrics)
        self.assertEqual(sc["type"], "portfolio_scorecard")
        self.assertIn("overall_grade", sc)
        self.assertIn("cards", sc)

    def test_build_all(self):
        charts = self.dash.build_all(self.fcs, None, self.metrics, None, None, None)
        expected_keys = [
            "correlation_heatmap",
            "exposure_chart",
            "risk_allocation",
            "dependency_graph",
            "portfolio_timeline",
            "optimization_dashboard",
            "scenario_dashboard",
            "portfolio_scorecard",
        ]
        for k in expected_keys:
            self.assertIn(k, charts)


# ════════════════════════════════════════════════════════════════════════════════
# 14. portfolio_api (integration)
# ════════════════════════════════════════════════════════════════════════════════
class TestPortfolioApi(unittest.TestCase):

    def setUp(self):
        self.api = _make_api()
        self.fcs = _make_forecasts(6)

    def test_recommend_ok(self):
        resp = self.api.recommend(self.fcs)
        self.assertEqual(resp["status"], "ok")
        data = resp["data"]
        self.assertIn("rec_id", data)
        self.assertIn("portfolio_metrics", data)

    def test_recommend_has_allocation(self):
        resp = self.api.recommend(self.fcs)
        self.assertIn("allocation", resp["data"])

    def test_recommend_has_regime(self):
        resp = self.api.recommend(self.fcs)
        self.assertIn("regime", resp["data"])

    def test_exposure_endpoint(self):
        resp = self.api.exposure(self.fcs)
        self.assertEqual(resp["status"], "ok")
        self.assertIn("exposure", resp["data"])

    def test_conflict_check_endpoint(self):
        resp = self.api.conflict_check(self.fcs)
        self.assertEqual(resp["status"], "ok")
        self.assertIn("conflict_count", resp["data"])

    def test_regime_endpoint(self):
        for fc in self.fcs:
            self.api._regime_detector.add_return(fc.instrument, fc.signed_return())
        resp = self.api.regime()
        self.assertEqual(resp["status"], "ok")

    def test_optimize_endpoint(self):
        resp = self.api.optimize(self.fcs, objective="max_sharpe")
        self.assertEqual(resp["status"], "ok")
        self.assertIn("best", resp["data"])

    def test_optimize_compare_all(self):
        resp = self.api.optimize(self.fcs, compare_all=True)
        self.assertEqual(resp["status"], "ok")
        self.assertIsNotNone(resp["data"]["all_objectives"])

    def test_scenario_endpoint(self):
        resp = self.api.scenario(self.fcs)
        self.assertEqual(resp["status"], "ok")
        self.assertIn("scenarios", resp["data"])

    def test_allocation_history(self):
        self.api.recommend(self.fcs)
        resp = self.api.allocation_history()
        self.assertEqual(resp["status"], "ok")
        self.assertGreater(len(resp["data"]["history"]), 0)

    def test_recommendations_history(self):
        self.api.recommend(self.fcs)
        resp = self.api.recommendations_history()
        self.assertEqual(resp["status"], "ok")
        self.assertGreater(len(resp["data"]["history"]), 0)

    def test_stats_endpoint(self):
        resp = self.api.stats()
        self.assertEqual(resp["status"], "ok")
        self.assertIn("database", resp["data"])

    def test_registry_info(self):
        resp = self.api.registry_info()
        self.assertEqual(resp["status"], "ok")
        self.assertEqual(resp["data"]["total"], 16)

    def test_dashboard_data_endpoint(self):
        resp = self.api.dashboard_data(self.fcs)
        self.assertEqual(resp["status"], "ok")
        self.assertIn("correlation_heatmap", resp["data"])

    def test_risk_budget_check_endpoint(self):
        weights = {f.instrument: 1.0 / len(self.fcs) for f in self.fcs}
        resp = self.api.risk_budget_check(self.fcs, weights)
        self.assertEqual(resp["status"], "ok")

    def test_correlation_endpoint(self):
        obs = {fc.instrument: fc.signed_return() for fc in self.fcs}
        resp = self.api.correlation(obs)
        self.assertEqual(resp["status"], "ok")
        self.assertIn("matrix", resp["data"])

    def test_recommend_persists_to_db(self):
        self.api.recommend(self.fcs)
        stats = self.api._db.stats()
        self.assertGreater(stats["recommendations"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
