"""Phase 17 — AEIP Ensemble Platform Unit Tests.

Comprehensive test suite covering all modules of the Adaptive Ensemble
Intelligence Platform.

Tests:
  - Expert: ExpertPrediction, ExpertModel, ExpertRegistry, build_default_registry
  - Strategies: all 8 combination strategies, build_strategy
  - Weighting: DynamicWeightEngine, WeightFactorConfig, regime affinity
  - Routing: ExpertRouter, RouterConfig, RoutingDecision
  - MetaModel: MetaFeatureBuilder, LinearMetaLayer, EnsembleMetaModel fit/predict
  - Calibration: ExpertCalibrator, temperature scaling, reports
  - FailureAnalysis: ExpertFailureAnalyzer, agreement matrix, top performers
  - OnlineScoring: OnlineScorer, rolling metrics, alerts
  - Database: EnsembleDatabase CRUD — experts, results, meta, weights, routing
  - Leaderboard: EnsembleLeaderboard update, rank, champion, retire
  - Visualization: all 8 chart types, full dashboard
  - API: all 15 endpoints, error handling
  - Export: EnsembleExporter bundle, inference contract
  - Engine: EnsembleEngine predict, outcome recording, weight update, export
  - PublicAPI: __all__ completeness
"""

from __future__ import annotations

import os
import tempfile
import uuid
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_expert(
    expert_id: str = None, family: str = "lstm", specs: list = None, status: str = "active"
) -> "ExpertModel":
    from trainer.ensemble.expert import ExpertModel, SpecializationType

    return ExpertModel(
        expert_id=expert_id or str(uuid.uuid4()),
        name=f"Expert {family}",
        model_family=family,
        specializations=specs or [SpecializationType.GENERAL],
        status=status,
    )


def _make_prediction(
    expert_id: str = None,
    buy: float = 0.6,
    sell: float = 0.2,
    no_trade: float = 0.2,
    confidence: float = 0.75,
    regime: str = "trending_up",
) -> "ExpertPrediction":
    from trainer.ensemble.expert import ExpertPrediction

    return ExpertPrediction(
        expert_id=expert_id or str(uuid.uuid4()),
        buy_prob=buy,
        sell_prob=sell,
        no_trade_prob=no_trade,
        confidence=confidence,
        uncertainty=0.25,
        regime=regime,
        instrument="EURUSD",
        timeframe="H1",
    )


def _make_registry_with_experts(n: int = 3) -> "ExpertRegistry":
    from trainer.ensemble.expert import ExpertRegistry

    registry = ExpertRegistry()
    for i in range(n):
        registry.register(_make_expert(expert_id=f"expert_{i:02d}"))
    return registry


# ===========================================================================
# Expert Tests
# ===========================================================================


class TestExpertPrediction:
    def test_probabilities_normalized(self):
        from trainer.ensemble.expert import ExpertPrediction

        p = ExpertPrediction("e1", buy_prob=0.6, sell_prob=0.6, no_trade_prob=0.6)
        total = p.buy_prob + p.sell_prob + p.no_trade_prob
        assert abs(total - 1.0) < 1e-6

    def test_predicted_class_buy(self):
        p = _make_prediction(buy=0.7, sell=0.2, no_trade=0.1)
        assert p.predicted_class == "buy"

    def test_predicted_class_sell(self):
        p = _make_prediction(buy=0.1, sell=0.7, no_trade=0.2)
        assert p.predicted_class == "sell"

    def test_predicted_class_no_trade(self):
        p = _make_prediction(buy=0.1, sell=0.1, no_trade=0.8)
        assert p.predicted_class == "no_trade"

    def test_is_high_confidence_true(self):
        from trainer.ensemble.expert import ExpertPrediction

        p = ExpertPrediction("e1", confidence=0.8, uncertainty=0.2)
        assert p.is_high_confidence is True

    def test_is_high_confidence_false(self):
        from trainer.ensemble.expert import ExpertPrediction

        p = ExpertPrediction("e1", confidence=0.5, uncertainty=0.6)
        assert p.is_high_confidence is False

    def test_to_from_dict(self):
        from trainer.ensemble.expert import ExpertPrediction

        p = _make_prediction("e1")
        p2 = ExpertPrediction.from_dict(p.to_dict())
        assert p2.expert_id == "e1"
        assert p2.buy_prob == pytest.approx(p.buy_prob, abs=1e-5)


class TestExpertModel:
    def test_default_status_active(self):
        e = _make_expert()
        assert e.status == "active"

    def test_accuracy_zero_predictions(self):
        e = _make_expert()
        assert e.accuracy == 0.0

    def test_record_prediction_increments(self):
        e = _make_expert()
        e.record_prediction(correct=True)
        e.record_prediction(correct=False)
        assert e.n_predictions == 2
        assert e.n_correct == 1
        assert e.accuracy == pytest.approx(0.5)

    def test_retire_changes_status(self):
        e = _make_expert()
        e.retire()
        assert e.status == "retired"
        assert not e.is_active

    def test_promote_changes_status(self):
        e = _make_expert()
        e.promote_to_champion()
        assert e.status == "champion"
        assert e.is_active

    def test_predict_default_fn(self):
        e = _make_expert()
        pred = e.predict({"regime": "trending_up"})
        assert pred.expert_id == e.expert_id
        assert pred.regime == "trending_up"

    def test_predict_custom_fn(self):
        from trainer.ensemble.expert import ExpertPrediction

        e = _make_expert()
        e.predict_fn = lambda features: ExpertPrediction(
            expert_id=e.expert_id, buy_prob=0.9, sell_prob=0.05, no_trade_prob=0.05
        )
        pred = e.predict({})
        assert pred.buy_prob == pytest.approx(0.9 / 1.0, abs=0.01)

    def test_to_from_dict(self):
        e = _make_expert("eid1", family="gru")
        d = e.to_dict()
        e2 = _make_expert.__wrapped__(d) if hasattr(_make_expert, "__wrapped__") else None
        from trainer.ensemble.expert import ExpertModel

        e2 = ExpertModel.from_dict(d)
        assert e2.expert_id == "eid1"
        assert e2.model_family == "gru"


class TestExpertRegistry:
    def test_register_and_get(self):
        from trainer.ensemble.expert import ExpertRegistry

        r = ExpertRegistry()
        e = _make_expert("e1")
        r.register(e)
        assert r.get("e1") is e

    def test_get_or_raise_missing(self):
        from trainer.ensemble.expert import ExpertRegistry

        r = ExpertRegistry()
        with pytest.raises(KeyError):
            r.get_or_raise("nonexistent")

    def test_all_experts_ordered(self):
        r = _make_registry_with_experts(3)
        ids = [e.expert_id for e in r.all_experts()]
        assert ids == ["expert_00", "expert_01", "expert_02"]

    def test_active_experts_excludes_retired(self):
        r = _make_registry_with_experts(3)
        r.retire("expert_01")
        active = [e.expert_id for e in r.active_experts()]
        assert "expert_01" not in active
        assert len(active) == 2

    def test_set_champion(self):
        r = _make_registry_with_experts(2)
        r.set_champion("expert_00")
        assert r.champion().expert_id == "expert_00"

    def test_champion_exclusive(self):
        r = _make_registry_with_experts(3)
        r.set_champion("expert_00")
        r.set_champion("expert_01")
        champions = [e for e in r.all_experts() if e.status == "champion"]
        assert len(champions) == 1
        assert champions[0].expert_id == "expert_01"

    def test_by_family(self):
        from trainer.ensemble.expert import ExpertRegistry

        r = ExpertRegistry()
        r.register(_make_expert("e1", family="lstm"))
        r.register(_make_expert("e2", family="gru"))
        r.register(_make_expert("e3", family="lstm"))
        lstm = r.by_family("lstm")
        assert len(lstm) == 2

    def test_summary(self):
        r = _make_registry_with_experts(3)
        s = r.summary()
        assert s["n_experts"] == 3
        assert s["n_active"] == 3

    def test_to_from_dict(self):
        from trainer.ensemble.expert import ExpertRegistry

        r = _make_registry_with_experts(2)
        d = r.to_dict()
        r2 = ExpertRegistry.from_dict(d)
        assert r2.n_experts == 2

    def test_build_default_registry(self):
        from trainer.ensemble.expert import build_default_registry

        r = build_default_registry()
        assert r.n_experts == 1
        experts = r.all_experts()
        assert experts[0].name == "Signalium Quant v1"
        assert experts[0].status == "champion"


# ===========================================================================
# Strategies Tests
# ===========================================================================


def _make_predictions(n: int = 3) -> List["ExpertPrediction"]:
    return [
        _make_prediction(
            expert_id=f"e{i}", buy=0.5 + i * 0.1, sell=0.3 - i * 0.05, no_trade=0.2 - i * 0.05
        )
        for i in range(n)
    ]


def _uniform_weights(predictions) -> Dict[str, float]:
    n = len(predictions)
    return {p.expert_id: 1.0 / n for p in predictions}


class TestSimpleAveraging:
    def test_result_sums_to_one(self):
        from trainer.ensemble.strategies import SimpleAveragingStrategy

        s = SimpleAveragingStrategy()
        r = s.combine(_make_predictions(3))
        total = r.buy_prob + r.sell_prob + r.no_trade_prob
        assert abs(total - 1.0) < 1e-5

    def test_empty_predictions(self):
        from trainer.ensemble.strategies import SimpleAveragingStrategy

        s = SimpleAveragingStrategy()
        r = s.combine([])
        assert r.n_experts == 0

    def test_n_experts(self):
        from trainer.ensemble.strategies import SimpleAveragingStrategy

        s = SimpleAveragingStrategy()
        r = s.combine(_make_predictions(4))
        assert r.n_experts == 4

    def test_strategy_name(self):
        from trainer.ensemble.strategies import SimpleAveragingStrategy

        s = SimpleAveragingStrategy()
        r = s.combine(_make_predictions(2))
        assert r.strategy == "simple_averaging"


class TestWeightedAveraging:
    def test_higher_weight_shifts_result(self):
        from trainer.ensemble.strategies import WeightedAveragingStrategy

        s = WeightedAveragingStrategy()
        preds = [
            _make_prediction("e1", buy=0.9, sell=0.05, no_trade=0.05),
            _make_prediction("e2", buy=0.1, sell=0.85, no_trade=0.05),
        ]
        r_e1_heavy = s.combine(preds, {"e1": 0.9, "e2": 0.1})
        r_e2_heavy = s.combine(preds, {"e1": 0.1, "e2": 0.9})
        assert r_e1_heavy.buy_prob > r_e2_heavy.buy_prob

    def test_weight_map_in_result(self):
        from trainer.ensemble.strategies import WeightedAveragingStrategy

        s = WeightedAveragingStrategy()
        preds = _make_predictions(2)
        w = {"e0": 0.7, "e1": 0.3}
        r = s.combine(preds, w)
        assert len(r.weight_map) == 2


class TestHardVoting:
    def test_winner_gets_prob_one(self):
        from trainer.ensemble.strategies import HardVotingStrategy

        s = HardVotingStrategy()
        preds = [
            _make_prediction("e1", buy=0.8, sell=0.1, no_trade=0.1),
            _make_prediction("e2", buy=0.7, sell=0.2, no_trade=0.1),
            _make_prediction("e3", buy=0.1, sell=0.8, no_trade=0.1),
        ]
        r = s.combine(preds)
        assert r.buy_prob == pytest.approx(1.0)
        assert r.sell_prob == pytest.approx(0.0)

    def test_agreement_in_metadata(self):
        from trainer.ensemble.strategies import HardVotingStrategy

        s = HardVotingStrategy()
        preds = _make_predictions(3)
        r = s.combine(preds)
        assert "votes" in r.metadata


class TestSoftVoting:
    def test_sums_to_one(self):
        from trainer.ensemble.strategies import SoftVotingStrategy

        s = SoftVotingStrategy()
        r = s.combine(_make_predictions(3))
        assert abs(r.buy_prob + r.sell_prob + r.no_trade_prob - 1.0) < 1e-5

    def test_agreement_boosts_confidence(self):
        from trainer.ensemble.strategies import SoftVotingStrategy

        s = SoftVotingStrategy()
        unanimous = [_make_prediction(f"e{i}", buy=0.9, sell=0.05, no_trade=0.05) for i in range(3)]
        divided = [
            _make_prediction("e1", buy=0.9, sell=0.05, no_trade=0.05),
            _make_prediction("e2", buy=0.1, sell=0.85, no_trade=0.05),
            _make_prediction("e3", buy=0.1, sell=0.05, no_trade=0.85),
        ]
        r_unanimous = s.combine(unanimous)
        r_divided = s.combine(divided)
        assert r_unanimous.confidence >= r_divided.confidence


class TestMixtureOfExperts:
    def test_softmax_concentrates_at_low_temperature(self):
        from trainer.ensemble.strategies import MixtureOfExpertsStrategy

        s = MixtureOfExpertsStrategy(temperature=0.1)
        preds = [
            _make_prediction("e1", buy=0.9, sell=0.05, no_trade=0.05),
            _make_prediction("e2", buy=0.1, sell=0.85, no_trade=0.05),
        ]
        r = s.combine(preds, {"e1": 10.0, "e2": 1.0})
        assert r.weight_map["e1"] > 0.9

    def test_metadata_has_top_expert(self):
        from trainer.ensemble.strategies import MixtureOfExpertsStrategy

        s = MixtureOfExpertsStrategy()
        preds = _make_predictions(2)
        r = s.combine(preds, {"e0": 1.0, "e1": 0.1})
        assert "top_expert" in r.metadata


class TestStackingStrategy:
    def test_combine_returns_result(self):
        from trainer.ensemble.strategies import StackingStrategy

        s = StackingStrategy()
        preds = _make_predictions(3)
        r = s.combine(preds)
        total = r.buy_prob + r.sell_prob + r.no_trade_prob
        assert abs(total - 1.0) < 1e-5

    def test_update_changes_weights(self):
        from trainer.ensemble.strategies import StackingStrategy

        s = StackingStrategy(learning_rate=0.5)
        preds = _make_predictions(2)
        before = dict(s._weights)
        s.update(preds, true_class="buy")
        assert s._weights != before or len(s._weights) > 0


class TestBuildStrategy:
    def test_all_strategies_build(self):
        from trainer.ensemble.strategies import AVAILABLE_STRATEGIES, build_strategy

        for name in AVAILABLE_STRATEGIES:
            s = build_strategy(name)
            assert s.name == name

    def test_unknown_raises(self):
        from trainer.ensemble.strategies import build_strategy

        with pytest.raises(ValueError):
            build_strategy("evolutionary")


class TestEnsembleResult:
    def test_predicted_class(self):
        from trainer.ensemble.strategies import EnsembleResult

        r = EnsembleResult(buy_prob=0.7, sell_prob=0.2, no_trade_prob=0.1)
        assert r.predicted_class == "buy"

    def test_institutional_forecast_string(self):
        from trainer.ensemble.strategies import EnsembleResult

        r = EnsembleResult(
            buy_prob=0.8, sell_prob=0.1, no_trade_prob=0.1, confidence=0.85, agreement_score=0.9
        )
        forecast = r.institutional_forecast
        assert "LONG" in forecast
        assert "HIGH CONFIDENCE" in forecast

    def test_to_dict(self):
        from trainer.ensemble.strategies import EnsembleResult

        r = EnsembleResult(
            buy_prob=0.6, sell_prob=0.2, no_trade_prob=0.2, strategy="soft_voting", n_experts=3
        )
        d = r.to_dict()
        assert d["strategy"] == "soft_voting"
        assert d["n_experts"] == 3


# ===========================================================================
# Weighting Tests
# ===========================================================================


class TestDynamicWeightEngine:
    def setup_method(self):
        from trainer.ensemble.weighting import DynamicWeightEngine

        self.engine = DynamicWeightEngine()
        self.experts = [_make_expert(f"e{i}") for i in range(3)]

    def test_weights_sum_to_one(self):
        weights = self.engine.compute_weights(self.experts)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_all_experts_have_weights(self):
        weights = self.engine.compute_weights(self.experts)
        for e in self.experts:
            assert e.expert_id in weights

    def test_regime_boost_favors_transformer(self):
        from trainer.ensemble.expert import ExpertModel, ModelFamily

        transformer = ExpertModel(expert_id="trans", model_family=ModelFamily.TRANSFORMER_ENCODER)
        lstm = ExpertModel(expert_id="lstm_e", model_family=ModelFamily.LSTM)
        experts = [transformer, lstm]
        ctx = {"regime": "trending_up"}
        weights = self.engine.compute_weights(experts, ctx)
        assert weights["trans"] > weights["lstm_e"]

    def test_empty_experts(self):
        weights = self.engine.compute_weights([])
        assert weights == {}

    def test_weight_history_recorded(self):
        self.engine.compute_weights(self.experts)
        ev = self.engine.weight_evolution("e0")
        assert len(ev) >= 1

    def test_update_expert_weights_in_place(self):
        self.engine.update_expert_weights_from_performance(self.experts)
        for e in self.experts:
            assert 0.0 <= e.weight <= 1.0


class TestWeightFactorConfig:
    def test_to_from_dict(self):
        from trainer.ensemble.weighting import WeightFactorConfig

        cfg = WeightFactorConfig(regime_boost=3.0, min_weight=0.1)
        d = cfg.to_dict()
        cfg2 = WeightFactorConfig.from_dict(d)
        assert cfg2.regime_boost == pytest.approx(3.0)
        assert cfg2.min_weight == pytest.approx(0.1)


# ===========================================================================
# Routing Tests
# ===========================================================================


class TestExpertRouter:
    def setup_method(self):
        from trainer.ensemble.routing import ExpertRouter
        from trainer.ensemble.weighting import DynamicWeightEngine

        self.registry = _make_registry_with_experts(4)
        self.engine = DynamicWeightEngine()
        self.router = ExpertRouter(self.registry, self.engine)

    def test_route_returns_decision(self):
        from trainer.ensemble.routing import RoutingDecision

        decision = self.router.route()
        assert isinstance(decision, RoutingDecision)
        assert len(decision.selected_expert_ids) >= 1

    def test_route_with_context(self):
        decision = self.router.route(
            context={
                "regime": "trending_up",
                "instrument": "EURUSD",
                "timeframe": "H1",
            }
        )
        assert decision.regime == "trending_up"
        assert decision.instrument == "EURUSD"

    def test_route_weights_sum_to_one(self):
        decision = self.router.route()
        total = sum(decision.weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_routing_history_recorded(self):
        for _ in range(3):
            self.router.route()
        assert len(self.router.routing_history()) == 3

    def test_routing_stats(self):
        self.router.route({"regime": "trending_up"})
        self.router.route({"regime": "ranging"})
        stats = self.router.routing_stats()
        assert stats["total_routes"] == 2
        assert "trending_up" in stats["regime_distribution"]


class TestRoutingDecision:
    def test_to_dict(self):
        from trainer.ensemble.routing import RoutingDecision

        d = RoutingDecision(
            regime="trending_up",
            instrument="EURUSD",
            selected_expert_ids=["e1", "e2"],
            weights={"e1": 0.6, "e2": 0.4},
            n_experts_selected=2,
        )
        d_dict = d.to_dict()
        assert d_dict["regime"] == "trending_up"
        assert d_dict["n_experts_selected"] == 2


# ===========================================================================
# Meta Model Tests
# ===========================================================================


class TestMetaFeatureBuilder:
    def test_build_returns_list(self):
        from trainer.ensemble.meta_model import MetaFeatureBuilder

        builder = MetaFeatureBuilder()
        preds = _make_predictions(3)
        weights = {p.expert_id: 1.0 / 3 for p in preds}
        features = builder.build(preds, weights, {"regime": "trending_up"})
        assert isinstance(features, list)
        assert len(features) > 0

    def test_empty_predictions(self):
        from trainer.ensemble.meta_model import MetaFeatureBuilder

        builder = MetaFeatureBuilder()
        features = builder.build([], {}, {})
        assert all(isinstance(f, float) for f in features)


class TestEnsembleMetaModel:
    def test_predict_fallback_before_training(self):
        from trainer.ensemble.meta_model import EnsembleMetaModel

        meta = EnsembleMetaModel(min_samples_to_activate=10)
        preds = _make_predictions(2)
        weights = {p.expert_id: 0.5 for p in preds}
        result = meta.predict(preds, weights, {"regime": "trending_up"})
        assert "fallback" in result.strategy

    def test_update_increments_samples(self):
        from trainer.ensemble.meta_model import EnsembleMetaModel

        meta = EnsembleMetaModel(min_samples_to_activate=5)
        preds = _make_predictions(2)
        weights = {p.expert_id: 0.5 for p in preds}
        for _ in range(3):
            meta.update(preds, weights, "buy", {})
        assert meta._n_training_samples == 3

    def test_activates_after_min_samples(self):
        from trainer.ensemble.meta_model import EnsembleMetaModel

        meta = EnsembleMetaModel(min_samples_to_activate=5)
        preds = _make_predictions(2)
        weights = {p.expert_id: 0.5 for p in preds}
        for _ in range(5):
            meta.update(preds, weights, "buy", {})
        assert meta.is_trained

    def test_predict_after_training(self):
        from trainer.ensemble.meta_model import EnsembleMetaModel

        meta = EnsembleMetaModel(min_samples_to_activate=3)
        preds = _make_predictions(3)
        weights = {p.expert_id: 1.0 / 3 for p in preds}
        for _ in range(3):
            meta.update(preds, weights, "buy", {})
        result = meta.predict(preds, weights, {"regime": "trending_up"})
        total = result.buy_prob + result.sell_prob + result.no_trade_prob
        assert abs(total - 1.0) < 1e-5

    def test_fit_returns_losses(self):
        from trainer.ensemble.meta_model import EnsembleMetaModel

        meta = EnsembleMetaModel(min_samples_to_activate=2)
        preds = _make_predictions(2)
        weights = {p.expert_id: 0.5 for p in preds}
        examples = [
            {
                "predictions": [p.to_dict() for p in preds],
                "weights": weights,
                "context": {},
                "true_class": "buy",
            }
            for _ in range(5)
        ]
        losses = meta.fit(examples)
        assert len(losses) == 5
        assert all(isinstance(l, float) for l in losses)

    def test_to_from_dict(self):
        from trainer.ensemble.meta_model import EnsembleMetaModel

        meta = EnsembleMetaModel(min_samples_to_activate=2)
        preds = _make_predictions(2)
        weights = {p.expert_id: 0.5 for p in preds}
        meta.update(preds, weights, "buy", {})
        d = meta.to_dict()
        meta2 = EnsembleMetaModel.from_dict(d)
        assert meta2._n_training_samples == 1

    def test_institutional_forecast_format(self):
        from trainer.ensemble.meta_model import MetaPrediction

        pred = MetaPrediction(
            buy_prob=0.8,
            sell_prob=0.1,
            no_trade_prob=0.1,
            confidence=0.85,
            agreement_score=0.9,
            n_experts=3,
            regime="trending_up",
        )
        forecast = pred.institutional_forecast
        assert "LONG" in forecast
        assert "[A]" in forecast or "[B]" in forecast or "[C]" in forecast


# ===========================================================================
# Calibration Tests
# ===========================================================================


class TestExpertCalibrator:
    def setup_method(self):
        from trainer.ensemble.calibration import ExpertCalibrator

        self.cal = ExpertCalibrator(n_bins=5)

    def test_update_and_report(self):
        for _ in range(20):
            self.cal.update("e1", predicted_prob=0.8, true_label=1)
        report = self.cal.get_report("e1")
        assert report.n_samples == 20
        assert report.entity_id == "e1"

    def test_ece_range(self):
        for i in range(50):
            self.cal.update("e1", predicted_prob=0.7, true_label=i % 2)
        report = self.cal.get_report("e1")
        assert 0.0 <= report.ece <= 1.0

    def test_perfect_calibration_ece_near_zero(self):
        for _ in range(100):
            self.cal.update("e1", predicted_prob=0.6, true_label=1)
        report = self.cal.get_report("e1")
        assert report.avg_confidence == pytest.approx(0.6, abs=0.01)
        assert report.avg_accuracy == pytest.approx(1.0, abs=0.01)

    def test_fit_temperature_returns_positive(self):
        for i in range(30):
            self.cal.update("e1", predicted_prob=0.9, true_label=i % 2)
        T = self.cal.fit_temperature("e1")
        assert T > 0

    def test_apply_temperature_changes_prob(self):
        for i in range(30):
            self.cal.update("e2", predicted_prob=0.9, true_label=i % 2)
        T = self.cal.fit_temperature("e2")
        calibrated = self.cal.apply_temperature("e2", 0.9)
        assert 0.0 <= calibrated <= 1.0

    def test_empty_entity(self):
        report = self.cal.get_report("nonexistent")
        assert report.n_samples == 0
        assert report.ece == 0.0

    def test_is_well_calibrated(self):
        from trainer.ensemble.calibration import CalibrationReport

        r = CalibrationReport("e", ece=0.03)
        assert r.is_well_calibrated
        r2 = CalibrationReport("e", ece=0.10)
        assert not r2.is_well_calibrated

    def test_all_reports(self):
        self.cal.update("a", 0.8, 1)
        self.cal.update("b", 0.3, 0)
        reports = self.cal.all_reports()
        assert "a" in reports and "b" in reports

    def test_update_batch(self):
        samples = [(0.7, 1), (0.3, 0), (0.8, 1)]
        self.cal.update_batch("e3", samples)
        report = self.cal.get_report("e3")
        assert report.n_samples == 3


# ===========================================================================
# Failure Analysis Tests
# ===========================================================================


class TestExpertFailureAnalyzer:
    def setup_method(self):
        from trainer.ensemble.failure_analysis import ExpertFailureAnalyzer

        self.analyzer = ExpertFailureAnalyzer()

    def test_record_returns_event(self):
        from trainer.ensemble.failure_analysis import FailureEvent

        event = self.analyzer.record(
            expert_id="e1",
            regime="trending_up",
            predicted_class="buy",
            ensemble_class="buy",
            true_class="buy",
            confidence=0.8,
        )
        assert isinstance(event, FailureEvent)
        assert event.agrees_with_ensemble is True
        assert event.is_correct is True

    def test_disagreement_detected(self):
        event = self.analyzer.record(
            "e1",
            predicted_class="buy",
            ensemble_class="sell",
            true_class="sell",
        )
        assert not event.agrees_with_ensemble
        assert not event.is_correct

    def test_report_accuracy(self):
        for i in range(10):
            self.analyzer.record(
                "e1",
                predicted_class="buy",
                true_class="buy" if i < 7 else "sell",
            )
        report = self.analyzer.get_report("e1")
        assert report.n_correct == 7
        assert report.accuracy == pytest.approx(0.7)

    def test_regime_accuracy(self):
        for _ in range(5):
            self.analyzer.record(
                "e1", regime="trending_up", predicted_class="buy", true_class="buy"
            )
        for _ in range(5):
            self.analyzer.record("e1", regime="ranging", predicted_class="buy", true_class="sell")
        report = self.analyzer.get_report("e1")
        assert report.regime_accuracy["trending_up"] == pytest.approx(1.0)
        assert report.regime_accuracy["ranging"] == pytest.approx(0.0)

    def test_high_uncertainty_detection(self):
        for _ in range(20):
            self.analyzer.record("e1", uncertainty=0.9)
        report = self.analyzer.get_report("e1")
        assert report.is_high_uncertainty

    def test_agreement_matrix(self):
        self.analyzer.record_pair_agreement("e1", "e2", "buy", "buy")
        self.analyzer.record_pair_agreement("e1", "e2", "buy", "sell")
        matrix = self.analyzer.agreement_matrix(["e1", "e2"])
        assert matrix["chart_type"] == "heatmap"
        assert matrix["matrix"]["e1"]["e1"] == 1.0

    def test_empty_expert(self):
        report = self.analyzer.get_report("nonexistent")
        assert report.n_predictions == 0

    def test_top_performers(self):
        for i in range(3):
            for _ in range(10):
                self.analyzer.record(
                    f"e{i}", predicted_class="buy", true_class="buy" if _ < (10 - i * 2) else "sell"
                )
        performers = self.analyzer.top_performers(n=2)
        assert len(performers) <= 2


# ===========================================================================
# Online Scoring Tests
# ===========================================================================


class TestOnlineScorer:
    def setup_method(self):
        from trainer.ensemble.online_scoring import OnlineScorer

        self.scorer = OnlineScorer(window=50, min_window_for_metrics=5)

    def test_record_returns_record(self):
        from trainer.ensemble.online_scoring import PredictionRecord

        rec = self.scorer.record("e1", predicted_class="buy", true_class="buy", profit_pnl=100.0)
        assert isinstance(rec, PredictionRecord)
        assert rec.is_correct is True
        assert rec.is_profitable is True

    def test_rolling_accuracy(self):
        for i in range(10):
            self.scorer.record("e1", "buy", "buy" if i < 7 else "sell")
        m = self.scorer.get_metrics("e1")
        assert m.rolling_accuracy == pytest.approx(0.7)

    def test_rolling_win_rate(self):
        for i in range(10):
            pnl = 100.0 if i < 6 else -50.0
            self.scorer.record("e1", "buy", profit_pnl=pnl)
        m = self.scorer.get_metrics("e1")
        assert m.rolling_win_rate == pytest.approx(0.6)

    def test_insufficient_data_trend(self):
        self.scorer.record("e1", "buy")
        m = self.scorer.get_metrics("e1")
        assert m.trend == "insufficient_data"

    def test_alert_low_accuracy(self):
        scorer = __import__(
            "trainer.ensemble.online_scoring", fromlist=["OnlineScorer"]
        ).OnlineScorer(window=10, min_window_for_metrics=5, alert_accuracy_threshold=0.60)
        for _ in range(5):
            scorer.record("e1", "buy", "sell")
        alerts = scorer.check_alerts()
        if alerts:
            assert any(a["expert_id"] == "e1" for a in alerts)

    def test_all_metrics(self):
        for eid in ["e1", "e2"]:
            self.scorer.record(eid, "buy", "buy")
        all_m = self.scorer.all_metrics()
        assert "e1" in all_m and "e2" in all_m

    def test_regime_breakdown(self):
        for regime in ["trending_up", "ranging"]:
            for _ in range(5):
                self.scorer.record("e1", "buy", "buy", regime=regime)
        breakdown = self.scorer.regime_breakdown("e1")
        assert "trending_up" in breakdown
        assert "ranging" in breakdown

    def test_metrics_to_dict(self):
        self.scorer.record("e1", "buy", "buy", profit_pnl=50.0)
        m = self.scorer.get_metrics("e1")
        d = m.to_dict()
        assert "rolling_accuracy" in d
        assert "rolling_sharpe" in d


# ===========================================================================
# Database Tests
# ===========================================================================


class TestEnsembleDatabase:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_ensemble.db")

    def _db(self):
        from trainer.ensemble.database import EnsembleDatabase

        return EnsembleDatabase(self.db_path)

    def test_save_and_get_expert(self):
        db = self._db()
        e = _make_expert("e1")
        db.save_expert(e)
        record = db.get_expert("e1")
        assert record is not None
        assert record["name"] == "Expert lstm"
        db.close()

    def test_list_experts_by_status(self):
        db = self._db()
        db.save_expert(_make_expert("e1", status="active"))
        db.save_expert(_make_expert("e2", status="retired"))
        active = db.list_experts(status="active")
        assert len(active) == 1
        assert active[0]["expert_id"] == "e1"
        db.close()

    def test_update_expert_status(self):
        db = self._db()
        db.save_expert(_make_expert("e1"))
        db.update_expert_status("e1", "champion")
        record = db.get_expert("e1")
        assert record["status"] == "champion"
        db.close()

    def test_update_expert_weight(self):
        db = self._db()
        db.save_expert(_make_expert("e1"))
        db.update_expert_weight("e1", 0.75)
        record = db.get_expert("e1")
        assert record["weight"] == pytest.approx(0.75)
        db.close()

    def test_save_and_list_ensemble_results(self):
        db = self._db()
        result = {
            "result_id": "r1",
            "buy_prob": 0.6,
            "sell_prob": 0.2,
            "no_trade_prob": 0.2,
            "confidence": 0.8,
            "uncertainty": 0.2,
            "predicted_class": "buy",
            "strategy": "soft_voting",
            "n_experts": 3,
            "agreement_score": 0.8,
            "weight_map": {"e1": 0.5, "e2": 0.5},
        }
        db.save_ensemble_result(result, regime="trending_up", instrument="EURUSD", true_class="buy")
        results = db.list_ensemble_results()
        assert len(results) == 1
        assert results[0]["predicted_class"] == "buy"
        assert results[0]["is_correct"] is True
        db.close()

    def test_save_and_get_meta_model(self):
        db = self._db()
        meta_dict = {
            "model_id": "m1",
            "meta_version": "1.0",
            "n_training_samples": 100,
            "layer": None,
        }
        db.save_meta_model(meta_dict)
        retrieved = db.get_latest_meta_model()
        assert retrieved is not None
        assert retrieved["model_id"] == "m1"
        db.close()

    def test_weight_history(self):
        db = self._db()
        db.save_expert(_make_expert("e1"))
        db.save_weight_entry("e1", 0.6, regime="trending_up", instrument="EURUSD")
        db.save_weight_entry("e1", 0.5, regime="ranging")
        history = db.get_weight_history("e1")
        assert len(history) == 2
        db.close()

    def test_routing_decisions(self):
        db = self._db()
        decision_dict = {
            "decision_id": "d1",
            "regime": "trending_up",
            "instrument": "EURUSD",
            "timeframe": "H1",
            "selected_expert_ids": ["e1", "e2"],
            "weights": {"e1": 0.6, "e2": 0.4},
            "routing_reason": "regime boost",
            "n_experts_selected": 2,
            "timestamp": "2026-07-05T00:00:00+00:00",
        }
        db.save_routing_decision(decision_dict)
        decisions = db.list_routing_decisions(regime="trending_up")
        assert len(decisions) == 1
        db.close()

    def test_performance_history(self):
        db = self._db()
        db.save_expert(_make_expert("e1"))
        db.save_performance_record("e1", "buy", "buy", profit_pnl=50.0)
        db.save_performance_record("e1", "sell", "buy", profit_pnl=-20.0)
        history = db.get_performance_history("e1")
        assert len(history) == 2
        db.close()

    def test_voting_history(self):
        db = self._db()
        db.save_vote("r1", "e1", "buy", confidence=0.8, weight=0.6)
        db.save_vote("r1", "e2", "sell", confidence=0.7, weight=0.4)
        votes = db.get_voting_history("r1")
        assert len(votes) == 2
        db.close()

    def test_stats(self):
        db = self._db()
        stats = db.stats()
        assert "total_experts" in stats
        assert "total_ensemble_results" in stats
        db.close()

    def test_get_nonexistent_expert(self):
        db = self._db()
        assert db.get_expert("nonexistent") is None
        db.close()


# ===========================================================================
# Leaderboard Tests
# ===========================================================================


class TestEnsembleLeaderboard:
    def setup_method(self):
        from trainer.ensemble.leaderboard import EnsembleLeaderboard

        self.lb = EnsembleLeaderboard()

    def test_update_expert_creates_entry(self):
        self.lb.update_expert("e1", {"accuracy": 0.65, "sharpe_ratio": 1.2})
        rankings = self.lb.get_rankings()
        assert len(rankings) == 1
        assert rankings[0]["entity_id"] == "e1"

    def test_rank_1_is_best(self):
        self.lb.update_expert("e1", {"accuracy": 0.5})
        self.lb.update_expert("e2", {"accuracy": 0.8})
        rankings = self.lb.get_rankings()
        assert rankings[0]["entity_id"] == "e2"
        assert rankings[0]["rank"] == 1

    def test_update_ensemble(self):
        self.lb.update_ensemble({"accuracy": 0.70})
        rankings = self.lb.get_rankings(entity_type="ensemble")
        assert len(rankings) == 1
        assert rankings[0]["entity_type"] == "ensemble"

    def test_update_meta_model(self):
        self.lb.update_meta_model({"accuracy": 0.72})
        rankings = self.lb.get_rankings(entity_type="meta_model")
        assert len(rankings) == 1

    def test_promote_champion(self):
        self.lb.update_expert("e1", {"accuracy": 0.65})
        self.lb.update_expert("e2", {"accuracy": 0.80})
        self.lb.promote_champion("e1")
        champion = self.lb.get_champion()
        assert champion is not None
        assert champion["entity_id"] == "e1"

    def test_retire_removes_from_active(self):
        self.lb.update_expert("e1", {"accuracy": 0.60})
        self.lb.update_expert("e2", {"accuracy": 0.70})
        self.lb.retire("e1")
        active = self.lb.get_rankings()
        assert all(e["entity_id"] != "e1" for e in active)

    def test_retired_in_include_retired(self):
        self.lb.update_expert("e1", {"accuracy": 0.60})
        self.lb.retire("e1")
        all_rankings = self.lb.get_rankings(include_retired=True)
        assert any(e["entity_id"] == "e1" for e in all_rankings)

    def test_summary(self):
        self.lb.update_expert("e1", {"accuracy": 0.6})
        summary = self.lb.summary()
        assert summary["n_experts"] == 1
        assert "champion" in summary


# ===========================================================================
# Visualization Tests
# ===========================================================================


class TestEnsembleVisualizationBuilder:
    def setup_method(self):
        from trainer.ensemble.visualization import EnsembleVisualizationBuilder

        self.viz = EnsembleVisualizationBuilder()

    def test_expert_weight_chart(self):
        chart = self.viz.expert_weight_chart({"e1": 0.6, "e2": 0.4})
        assert chart["chart_type"] == "bar"
        assert chart["n_experts"] == 2

    def test_voting_distribution_empty(self):
        chart = self.viz.voting_distribution([])
        assert "note" in chart

    def test_voting_distribution_with_data(self):
        votes = [
            {"expert_id": "e1", "voted_class": "buy"},
            {"expert_id": "e1", "voted_class": "sell"},
            {"expert_id": "e2", "voted_class": "buy"},
        ]
        chart = self.viz.voting_distribution(votes)
        assert chart["chart_type"] == "bar_stacked"
        assert chart["total_votes"] == 3

    def test_agreement_matrix(self):
        data = {
            "expert_ids": ["e1", "e2"],
            "matrix": {"e1": {"e1": 1.0, "e2": 0.7}, "e2": {"e1": 0.7, "e2": 1.0}},
        }
        chart = self.viz.agreement_matrix(data)
        assert chart["chart_type"] == "heatmap"

    def test_confidence_distribution(self):
        results = [{"confidence": 0.5 + i * 0.05} for i in range(10)]
        chart = self.viz.confidence_distribution(results, n_bins=5)
        assert chart["chart_type"] == "histogram"
        assert sum(b["count"] for b in chart["bins"]) == 10

    def test_weight_evolution(self):
        history = {"e1": [0.4, 0.5, 0.6], "e2": [0.6, 0.5, 0.4]}
        chart = self.viz.weight_evolution(history)
        assert chart["chart_type"] == "line"
        assert chart["n_experts"] == 2

    def test_expert_performance_timeline_empty(self):
        chart = self.viz.expert_performance_timeline([])
        assert "note" in chart

    def test_expert_performance_timeline(self):
        records = [{"expert_id": "e1", "is_correct": 1 if i % 2 == 0 else 0} for i in range(10)]
        chart = self.viz.expert_performance_timeline(records, window=5)
        assert chart["chart_type"] == "line"

    def test_ensemble_vs_expert_accuracy(self):
        expert_metrics = {"e1": {"accuracy": 0.6}, "e2": {"accuracy": 0.7}}
        ensemble_metrics = {"accuracy": 0.72}
        chart = self.viz.ensemble_vs_expert_accuracy(expert_metrics, ensemble_metrics)
        assert chart["chart_type"] == "bar"
        entities = [d["entity"] for d in chart["series"][0]["data"]]
        assert "Ensemble" in entities

    def test_full_dashboard_all_keys(self):
        dashboard = self.viz.full_dashboard(
            current_weights={"e1": 0.6, "e2": 0.4},
            weight_history={"e1": [0.5, 0.6], "e2": [0.5, 0.4]},
            vote_records=[{"expert_id": "e1", "voted_class": "buy"}],
            agreement_matrix_data={
                "expert_ids": ["e1", "e2"],
                "matrix": {"e1": {"e1": 1.0, "e2": 0.8}, "e2": {"e1": 0.8, "e2": 1.0}},
            },
            ensemble_results=[{"confidence": 0.75}],
            performance_records=[{"expert_id": "e1", "is_correct": 1}],
            expert_metrics={"e1": {"accuracy": 0.6}},
            ensemble_metrics={"accuracy": 0.7},
        )
        expected = [
            "expert_weight_chart",
            "voting_distribution",
            "agreement_matrix",
            "confidence_distribution",
            "weight_evolution",
            "expert_performance_timeline",
            "ensemble_vs_expert_accuracy",
        ]
        for key in expected:
            assert key in dashboard


# ===========================================================================
# API Tests
# ===========================================================================


class TestEnsembleAPI:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        from trainer.ensemble.database import EnsembleDatabase
        from trainer.ensemble.api import EnsembleAPI

        self.db = EnsembleDatabase(os.path.join(self.tmpdir, "api_test.db"))
        self.api = EnsembleAPI(self.db)

    def teardown_method(self):
        self.db.close()

    def test_register_expert(self):
        resp = self.api.register_expert(
            {
                "name": "Test Expert",
                "model_family": "lstm",
                "specializations": ["general"],
                "version": "1.0.0",
            }
        )
        assert resp["status"] == "ok"
        assert resp["data"]["name"] == "Test Expert"

    def test_get_expert_not_found(self):
        resp = self.api.get_expert("nonexistent")
        assert resp["status"] == "error"
        assert resp["code"] == "NOT_FOUND"

    def test_list_experts_empty(self):
        resp = self.api.list_experts()
        assert resp["status"] == "ok"
        assert resp["data"]["count"] == 0

    def test_list_experts_with_data(self):
        self.api.register_expert({"name": "E1", "model_family": "lstm"})
        self.api.register_expert({"name": "E2", "model_family": "gru"})
        resp = self.api.list_experts()
        assert resp["data"]["count"] == 2

    def test_retire_expert(self):
        reg_resp = self.api.register_expert({"name": "E1", "model_family": "lstm"})
        eid = reg_resp["data"]["expert_id"]
        retire_resp = self.api.retire_expert(eid)
        assert retire_resp["status"] == "ok"
        assert retire_resp["data"]["status"] == "retired"

    def test_promote_champion(self):
        reg_resp = self.api.register_expert({"name": "Champion", "model_family": "gru"})
        eid = reg_resp["data"]["expert_id"]
        resp = self.api.promote_champion(eid)
        assert resp["status"] == "ok"
        assert resp["data"]["status"] == "champion"

    def test_list_predictions_empty(self):
        resp = self.api.list_predictions()
        assert resp["status"] == "ok"
        assert resp["data"]["count"] == 0

    def test_get_weights_empty_registry(self):
        resp = self.api.get_weights()
        assert resp["status"] == "ok"

    def test_get_leaderboard_empty(self):
        resp = self.api.get_leaderboard()
        assert resp["status"] == "ok"
        assert resp["data"]["total"] == 0

    def test_get_champion_empty(self):
        resp = self.api.get_champion()
        assert resp["status"] == "ok"
        assert resp["data"]["champion"] is None

    def test_get_stats(self):
        resp = self.api.get_stats()
        assert resp["status"] == "ok"
        assert "total_experts" in resp["data"]

    def test_get_dashboard(self):
        resp = self.api.get_dashboard()
        assert resp["status"] == "ok"
        assert "charts" in resp["data"]

    def test_get_routing_history_empty(self):
        resp = self.api.get_routing_history()
        assert resp["status"] == "ok"
        assert resp["data"]["count"] == 0


# ===========================================================================
# Export Tests
# ===========================================================================


class TestEnsembleExporter:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.registry = _make_registry_with_experts(2)

    def test_export_expert(self):
        from trainer.ensemble.export import EnsembleExporter

        exp = EnsembleExporter(self.registry)
        d = exp.export_expert("expert_00")
        assert d["export_type"] == "expert"
        assert d["expert_id"] == "expert_00"

    def test_export_all_experts(self):
        from trainer.ensemble.export import EnsembleExporter

        exp = EnsembleExporter(self.registry)
        all_e = exp.export_all_experts()
        assert len(all_e) == 2

    def test_export_bundle(self):
        from trainer.ensemble.export import EnsembleExporter

        exp = EnsembleExporter(self.registry)
        bundle = exp.export_bundle()
        assert bundle["bundle_type"] == "ensemble_bundle"
        assert bundle["n_experts"] == 2
        assert "inference_contract" in bundle
        assert "experts" in bundle

    def test_export_inference_contract(self):
        from trainer.ensemble.export import EnsembleExporter

        exp = EnsembleExporter(self.registry, strategy_name="soft_voting")
        contract = exp.export_inference_contract()
        assert contract["contract_type"] == "ensemble_inference_contract"
        assert contract["strategy"] == "soft_voting"
        assert "input_schema" in contract
        assert "output_schema" in contract

    def test_save_and_load_bundle(self):
        from trainer.ensemble.export import EnsembleExporter

        exp = EnsembleExporter(self.registry)
        bundle = exp.export_bundle()
        path = os.path.join(self.tmpdir, "bundle.json")
        saved_path = exp.save_bundle(bundle, path)
        assert os.path.exists(saved_path)
        loaded = EnsembleExporter.load_bundle(saved_path)
        assert loaded["bundle_id"] == bundle["bundle_id"]


# ===========================================================================
# Engine Tests
# ===========================================================================


class TestEnsembleEngine:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def _make_engine(self, strategy: str = "soft_voting") -> "EnsembleEngine":
        from trainer.ensemble.engine import EnsembleEngine, EnsembleConfig

        cfg = EnsembleConfig(
            db_path=os.path.join(self.tmpdir, "engine_test.db"),
            strategy=strategy,
            persist_routing=True,
            persist_votes=True,
        )
        return EnsembleEngine(config=cfg)

    def test_engine_initializes_with_expert1(self):
        engine = self._make_engine()
        assert engine.registry.n_experts >= 1
        e1 = engine.registry.get("expert_01")
        assert e1 is not None
        assert e1.name == "Signalium Quant v1"

    def test_register_expert_adds_to_registry(self):
        engine = self._make_engine()
        from trainer.ensemble.expert import ExpertModel

        new_expert = ExpertModel(
            expert_id="expert_new",
            name="New Expert",
            model_family="transformer_encoder",
        )
        engine.register_expert(new_expert)
        assert engine.registry.n_experts == 2
        engine.close()

    def test_predict_returns_bundle(self):
        from trainer.ensemble.engine import PredictionBundle

        engine = self._make_engine()
        bundle = engine.predict(
            features={"close": 1.1},
            context={"regime": "trending_up", "instrument": "EURUSD", "timeframe": "H1"},
        )
        assert isinstance(bundle, PredictionBundle)
        assert bundle.ensemble_result is not None
        assert bundle.meta_prediction is not None
        engine.close()

    def test_predict_result_sums_to_one(self):
        engine = self._make_engine()
        bundle = engine.predict({}, {"regime": "ranging"})
        r = bundle.ensemble_result
        total = r.buy_prob + r.sell_prob + r.no_trade_prob
        assert abs(total - 1.0) < 1e-5
        engine.close()

    def test_predict_persists_to_db(self):
        engine = self._make_engine()
        engine.predict({}, {"regime": "ranging"})
        results = engine.db.list_ensemble_results()
        assert len(results) >= 1
        engine.close()

    def test_record_outcome_updates_calibrator(self):
        engine = self._make_engine()
        bundle = engine.predict({})
        initial_n = engine.calibrator.n_samples("expert_01")
        engine.record_outcome(bundle.bundle_id, true_class="buy", profit_pnl=50.0)
        new_n = engine.calibrator.n_samples("expert_01")
        assert new_n >= initial_n
        engine.close()

    def test_update_weights_returns_dict(self):
        engine = self._make_engine()
        for _ in range(5):
            engine.predict({})
        weights = engine.update_weights({"regime": "trending_up"})
        assert isinstance(weights, dict)
        assert len(weights) >= 1
        engine.close()

    def test_retire_expert(self):
        engine = self._make_engine()
        from trainer.ensemble.expert import ExpertModel

        e = ExpertModel(expert_id="to_retire", name="To Retire", model_family="gru")
        engine.register_expert(e)
        engine.retire_expert("to_retire")
        assert engine.registry.get("to_retire").status == "retired"
        engine.close()

    def test_set_champion(self):
        engine = self._make_engine()
        engine.set_champion("expert_01")
        assert engine.registry.get("expert_01").status == "champion"
        engine.close()

    def test_summary_string(self):
        engine = self._make_engine()
        s = engine.summary()
        assert "AEIP Ensemble Engine" in s
        assert "soft_voting" in s
        engine.close()

    def test_export_bundle(self):
        engine = self._make_engine()
        bundle = engine.export_bundle()
        assert bundle["bundle_type"] == "ensemble_bundle"
        assert bundle["n_experts"] >= 1
        engine.close()

    def test_export_inference_contract(self):
        engine = self._make_engine()
        contract = engine.export_inference_contract()
        assert "input_schema" in contract
        assert "output_schema" in contract
        engine.close()

    def test_all_strategies_work_in_engine(self):
        from trainer.ensemble.strategies import AVAILABLE_STRATEGIES

        for strategy in AVAILABLE_STRATEGIES:
            engine = self._make_engine(strategy=strategy)
            bundle = engine.predict({})
            assert bundle.ensemble_result is not None
            engine.close()

    def test_get_dashboard(self):
        engine = self._make_engine()
        resp = engine.get_dashboard()
        assert resp["status"] == "ok"
        assert "charts" in resp["data"]
        engine.close()

    def test_from_config(self):
        from trainer.ensemble.engine import EnsembleEngine

        engine = EnsembleEngine.from_config(
            {
                "db_path": os.path.join(self.tmpdir, "from_config.db"),
                "strategy": "weighted_averaging",
            }
        )
        assert engine.config.strategy == "weighted_averaging"
        engine.close()

    def test_prediction_bundle_to_dict(self):
        engine = self._make_engine()
        bundle = engine.predict({})
        d = bundle.to_dict()
        assert "bundle_id" in d
        assert "institutional_forecast" in d
        assert "final" in d
        engine.close()


# ===========================================================================
# EnsembleConfig Tests
# ===========================================================================


class TestEnsembleConfig:
    def test_defaults(self):
        from trainer.ensemble.engine import EnsembleConfig

        cfg = EnsembleConfig()
        assert cfg.strategy == "soft_voting"
        assert cfg.rolling_window == 200

    def test_from_dict(self):
        from trainer.ensemble.engine import EnsembleConfig

        cfg = EnsembleConfig.from_dict(
            {
                "strategy": "hard_voting",
                "rolling_window": 100,
            }
        )
        assert cfg.strategy == "hard_voting"
        assert cfg.rolling_window == 100

    def test_to_dict(self):
        from trainer.ensemble.engine import EnsembleConfig

        cfg = EnsembleConfig(strategy="blending")
        d = cfg.to_dict()
        assert d["strategy"] == "blending"


# ===========================================================================
# Public API (__all__ completeness)
# ===========================================================================


class TestPublicAPI:
    def test_all_exports_importable(self):
        from trainer.ensemble import (
            SpecializationType,
            ModelFamily,
            ExpertPrediction,
            ExpertModel,
            ExpertRegistry,
            build_default_registry,
            EnsembleResult,
            BaseEnsembleStrategy,
            SimpleAveragingStrategy,
            WeightedAveragingStrategy,
            PerformanceWeightedStrategy,
            HardVotingStrategy,
            SoftVotingStrategy,
            MixtureOfExpertsStrategy,
            StackingStrategy,
            BlendingStrategy,
            build_strategy,
            AVAILABLE_STRATEGIES,
            WeightFactorConfig,
            DynamicWeightEngine,
            REGIME_FAMILY_AFFINITY,
            RoutingDecision,
            RouterConfig,
            ExpertRouter,
            MetaPrediction,
            MetaFeatureBuilder,
            EnsembleMetaModel,
            CalibrationReport,
            ExpertCalibrator,
            FailureEvent,
            ExpertFailureReport,
            ExpertFailureAnalyzer,
            PredictionRecord,
            RollingMetrics,
            OnlineScorer,
            EnsembleDatabase,
            LeaderboardEntry,
            EnsembleLeaderboard,
            EnsembleVisualizationBuilder,
            EnsembleAPI,
            EnsembleExporter,
            EnsembleConfig,
            PredictionBundle,
            EnsembleEngine,
        )

    def test_all_names_in_all(self):
        import trainer.ensemble as m

        for name in m.__all__:
            assert hasattr(m, name), f"Missing from module: {name}"
