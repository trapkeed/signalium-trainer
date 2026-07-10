"""S33 — Opportunity Quality Scoring tests."""

from __future__ import annotations

import tempfile

import pytest

from trainer.qde.opportunity_scoring.confidence_stability import confidence_stability_score
from trainer.qde.opportunity_scoring.engine import OpportunityScoringEngine
from trainer.qde.opportunity_scoring.knowledge_bridge import record_to_knowledge_base
from trainer.qde.opportunity_scoring.models import (
    OpportunityBand,
    OpportunityScoreInputs,
    OpportunityScoreWeights,
    band_for_score,
)
from trainer.qde.opportunity_scoring.store import OpportunityScoreStore
from trainer.setup_memory.engine import SetupMemoryEngine
from trainer.setup_memory.store import SetupMemoryStore


def _inputs(**overrides) -> OpportunityScoreInputs:
    base = dict(
        symbol="EURUSD",
        timeframe="1h",
        historical_similarity=70.0,
        spread_quality=70.0,
        volatility_quality=70.0,
        session_quality=70.0,
        structure_quality=70.0,
        expected_move_quality=70.0,
        holding_period_quality=70.0,
        confidence_stability=70.0,
        cross_asset_confirmation=70.0,
        historical_performance=70.0,
    )
    base.update(overrides)
    return OpportunityScoreInputs(**base)


@pytest.fixture()
def engine() -> OpportunityScoringEngine:
    return OpportunityScoringEngine(store=OpportunityScoreStore(tempfile.mktemp(suffix=".db")))


def test_band_boundaries() -> None:
    assert band_for_score(0) == OpportunityBand.IGNORE
    assert band_for_score(39.9) == OpportunityBand.IGNORE
    assert band_for_score(40) == OpportunityBand.OBSERVE
    assert band_for_score(59.9) == OpportunityBand.OBSERVE
    assert band_for_score(60) == OpportunityBand.WATCH_CLOSELY
    assert band_for_score(74.9) == OpportunityBand.WATCH_CLOSELY
    assert band_for_score(75) == OpportunityBand.CANDIDATE_TRADE
    assert band_for_score(84.9) == OpportunityBand.CANDIDATE_TRADE
    assert band_for_score(85) == OpportunityBand.INSTITUTIONAL_QUALITY
    assert band_for_score(100) == OpportunityBand.INSTITUTIONAL_QUALITY


def test_equal_weight_average(engine: OpportunityScoringEngine) -> None:
    result = engine.score(_inputs())
    assert result.score == pytest.approx(70.0)
    assert result.band == OpportunityBand.WATCH_CLOSELY


def test_score_does_not_take_or_expose_direction(engine: OpportunityScoringEngine) -> None:
    result = engine.score(_inputs())
    assert not hasattr(result, "direction")
    assert "direction" not in result.to_dict()


def test_worth_capital_allocation_threshold(engine: OpportunityScoringEngine) -> None:
    low = engine.score(
        _inputs(
            historical_similarity=10,
            spread_quality=10,
            volatility_quality=10,
            session_quality=10,
            structure_quality=10,
            expected_move_quality=10,
            holding_period_quality=10,
            confidence_stability=10,
            cross_asset_confirmation=10,
            historical_performance=10,
        )
    )
    assert low.worth_capital_allocation() is False
    high = engine.score(_inputs())
    assert high.worth_capital_allocation() is True


def test_out_of_range_subscores_are_clamped(engine: OpportunityScoringEngine) -> None:
    result = engine.score(_inputs(historical_similarity=250, spread_quality=-50))
    assert result.sub_scores["historical_similarity"] == 100.0
    assert result.sub_scores["spread_quality"] == 0.0


def test_custom_weights_must_normalize() -> None:
    weights = OpportunityScoreWeights(
        historical_similarity=2.0,
        spread_quality=0,
        volatility_quality=0,
        session_quality=0,
        structure_quality=0,
        expected_move_quality=0,
        holding_period_quality=0,
        confidence_stability=0,
        cross_asset_confirmation=0,
        historical_performance=0,
    )
    engine = OpportunityScoringEngine(
        store=OpportunityScoreStore(tempfile.mktemp(suffix=".db")), weights=weights
    )
    result = engine.score(_inputs(historical_similarity=80.0))
    assert result.score == pytest.approx(80.0)


def test_history_persists_and_orders_desc(engine: OpportunityScoringEngine) -> None:
    engine.score(_inputs(historical_similarity=10))
    engine.score(_inputs(historical_similarity=90))
    hist = engine.history("EURUSD", "1h")
    assert len(hist) == 2
    assert hist[0].sub_scores["historical_similarity"] == 90


def test_history_persists_across_restarts() -> None:
    db_path = tempfile.mktemp(suffix=".db")
    e1 = OpportunityScoringEngine(store=OpportunityScoreStore(db_path))
    e1.score(_inputs())
    e2 = OpportunityScoringEngine(store=OpportunityScoreStore(db_path))
    assert len(e2.history("EURUSD", "1h")) == 1


def test_history_for_decision(engine: OpportunityScoringEngine) -> None:
    engine.score(_inputs(decision_id="dec-1"))
    engine.score(_inputs(decision_id="dec-2"))
    only_dec1 = engine.history_for_decision("dec-1")
    assert len(only_dec1) == 1
    assert only_dec1[0].decision_id == "dec-1"


def test_on_score_hook_invoked() -> None:
    captured = []
    engine = OpportunityScoringEngine(
        store=OpportunityScoreStore(tempfile.mktemp(suffix=".db")),
        on_score=captured.append,
    )
    engine.score(_inputs())
    assert len(captured) == 1


def test_on_score_hook_failure_does_not_break_scoring() -> None:
    def broken(_result):
        raise RuntimeError("boom")

    engine = OpportunityScoringEngine(
        store=OpportunityScoreStore(tempfile.mktemp(suffix=".db")), on_score=broken
    )
    result = engine.score(_inputs())
    assert result.score == pytest.approx(70.0)


def test_confidence_stability_bridge_neutral_with_no_history() -> None:
    setup_engine = SetupMemoryEngine(store=SetupMemoryStore(tempfile.mktemp(suffix=".db")))
    score = confidence_stability_score(setup_engine, "EURUSD", "1h")
    assert score == 50.0


def test_confidence_stability_bridge_rewards_steady_strengthening() -> None:
    setup_engine = SetupMemoryEngine(store=SetupMemoryStore(tempfile.mktemp(suffix=".db")))
    setup_engine.advance("EURUSD", "1h")
    for _ in range(5):
        setup_engine.strengthen("EURUSD", "1h", 0.05)
    score = confidence_stability_score(setup_engine, "EURUSD", "1h")
    assert score > 50.0


class _FakeKnowledgeBase:
    def __init__(self) -> None:
        self.entries: list[dict] = []

    def create_entry(
        self,
        entry_type,
        title,
        description,
        tags=None,
        links=None,
        decision_ids=None,
        metadata=None,
    ):
        entry = {
            "entry_type": entry_type,
            "title": title,
            "description": description,
            "tags": tags,
            "decision_ids": decision_ids,
            "metadata": metadata,
        }
        self.entries.append(entry)
        return entry


def test_knowledge_bridge_indexes_high_scores_only(engine: OpportunityScoringEngine) -> None:
    kb = _FakeKnowledgeBase()
    low = engine.score(
        _inputs(
            historical_similarity=10,
            spread_quality=10,
            volatility_quality=10,
            session_quality=10,
            structure_quality=10,
            expected_move_quality=10,
            holding_period_quality=10,
            confidence_stability=10,
            cross_asset_confirmation=10,
            historical_performance=10,
        )
    )
    high = engine.score(_inputs(decision_id="dec-9"))
    assert record_to_knowledge_base(kb, low) is None
    assert record_to_knowledge_base(kb, high) is not None
    assert len(kb.entries) == 1
    assert kb.entries[0]["decision_ids"] == ["dec-9"]
