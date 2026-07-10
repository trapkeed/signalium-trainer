"""QDE Brain — internal market consciousness subsystem tests.

Covers continuous ingestion (`on_market_update`), hypothesis formation /
reinforcement / invalidation, attention reallocation, belief tracking,
reflection, experience accumulation, and cross-session persistence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trainer.lifecycle.lifecycle_models import MarketContext
from trainer.qde.brain.brain import QDEBrain
from trainer.qde.brain.brain_api import QDEBrainAPI
from trainer.qde.brain.brain_types import BeliefDirection, HypothesisStatus
from trainer.qde.brain.persistence import QDEBrainDatabase
from trainer.qde.qde_observer_models import ObservationResult, SymbolObservationInput
from trainer.qde.qde_observer_types import ObserverState

SYM = "EURUSD"


def _tick(
    price: float = 1.1000,
    momentum: float = 0.0,
    volatility: float = 0.01,
    regime: str = "RANGE",
    instrument: str = SYM,
) -> SymbolObservationInput:
    return SymbolObservationInput(
        instrument=instrument,
        market=MarketContext(price=price, momentum=momentum, volatility=volatility),
        market_regime=regime,
    )


def _observation(readiness_score: float = 0.7, instrument: str = SYM) -> ObservationResult:
    return ObservationResult(
        instrument=instrument,
        state=ObserverState.OBSERVING,
        previous_state=ObserverState.OBSERVING,
        readiness_score=readiness_score,
        confidence=readiness_score,
    )


@pytest.fixture()
def db(tmp_path: Path) -> QDEBrainDatabase:
    return QDEBrainDatabase(str(tmp_path / "test_qde_brain.db"))


@pytest.fixture()
def brain(db: QDEBrainDatabase) -> QDEBrain:
    return QDEBrain(db=db)


# ── Continuous ingestion ────────────────────────────────────────────────────


def test_on_market_update_creates_state_for_new_symbol(brain: QDEBrain) -> None:
    thought = brain.on_market_update(_tick())
    assert thought.instrument == SYM
    state = brain.get_state(SYM)
    assert state.instrument == SYM
    assert len(state.working_memory) == 1


def test_on_market_update_ticks_regardless_of_connection_state(brain: QDEBrain) -> None:
    """Brain has no concept of user-connection/autonomous-trading gating —
    every tick that reaches it updates internal state."""
    for i in range(5):
        brain.on_market_update(_tick(price=1.1000 + i * 0.0005))
    state = brain.get_state(SYM)
    assert len(state.working_memory) == 5


def test_multiple_symbols_tracked_independently(brain: QDEBrain) -> None:
    brain.on_market_update(_tick(instrument="EURUSD"))
    brain.on_market_update(_tick(instrument="GBPUSD", price=1.27))
    states = brain.all_states()
    assert set(states.keys()) == {"EURUSD", "GBPUSD"}


# ── Hypothesis formation / reinforcement / invalidation ─────────────────────


def test_sustained_trend_forms_hypothesis(brain: QDEBrain) -> None:
    for i in range(8):
        brain.on_market_update(
            _tick(price=1.1000 + i * 0.0010, momentum=0.6, regime="TREND_UP"),
            observation_result=_observation(),
        )
    state = brain.get_state(SYM)
    assert len(state.hypotheses) >= 1
    assert any(not h.is_terminal() for h in state.hypotheses.values())


def test_reversal_invalidates_hypothesis(brain: QDEBrain) -> None:
    for i in range(8):
        brain.on_market_update(
            _tick(price=1.1000 + i * 0.0010, momentum=0.6, regime="TREND_UP"),
            observation_result=_observation(),
        )
    state = brain.get_state(SYM)
    active_before = [h for h in state.hypotheses.values() if not h.is_terminal()]
    assert active_before

    for i in range(8):
        brain.on_market_update(
            _tick(price=1.1080 - i * 0.0015, momentum=-0.6, regime="RANGE"),
            observation_result=_observation(readiness_score=0.1),
        )
    state = brain.get_state(SYM)
    invalidated_or_weakened = [
        h
        for h in state.hypotheses.values()
        if h.status in (HypothesisStatus.INVALIDATED, HypothesisStatus.WEAKENING)
    ]
    assert invalidated_or_weakened


# ── Belief tracking ──────────────────────────────────────────────────────────


def test_belief_direction_tracks_dominant_hypothesis(brain: QDEBrain) -> None:
    for i in range(8):
        brain.on_market_update(
            _tick(price=1.1000 + i * 0.0010, momentum=0.6, regime="TREND_UP"),
            observation_result=_observation(),
        )
    state = brain.get_state(SYM)
    assert state.belief is not None
    assert state.belief.direction in (BeliefDirection.BULLISH, BeliefDirection.NEUTRAL)


def test_belief_id_stable_across_updates(brain: QDEBrain) -> None:
    brain.on_market_update(_tick())
    first_id = brain.get_state(SYM).belief.belief_id
    brain.on_market_update(_tick(price=1.1010))
    second_id = brain.get_state(SYM).belief.belief_id
    assert first_id == second_id
    assert first_id is not None


# ── Attention allocation ─────────────────────────────────────────────────────


def test_reallocate_attention_normalizes_weights(brain: QDEBrain) -> None:
    brain.on_market_update(_tick(instrument="EURUSD", volatility=0.02))
    brain.on_market_update(_tick(instrument="GBPUSD", volatility=0.005))
    weights = brain.reallocate_attention()
    assert set(weights.keys()) == {"EURUSD", "GBPUSD"}
    assert pytest.approx(sum(weights.values()), abs=1e-6) == 1.0
    assert all(w >= 0 for w in weights.values())


# ── Experience accumulation & reflection ────────────────────────────────────


def test_record_experience_persists(brain: QDEBrain) -> None:
    brain.on_market_update(_tick())
    exp = brain.record_experience(SYM, "trade_closed", "closed a winning long", outcome_quality=0.8)
    assert exp.instrument == SYM
    assert exp.outcome_quality == 0.8


def test_reflect_produces_entry_and_respects_cooldown(brain: QDEBrain) -> None:
    brain.on_market_update(_tick())
    brain.record_experience(SYM, "trade_closed", "closed a winning long", outcome_quality=0.8)
    entry = brain.reflect(SYM, force=True)
    assert entry is not None
    assert entry.instrument == SYM

    again = brain.reflect(SYM, force=False)
    assert again is None  # cooldown should suppress an immediate re-reflection


# ── Persistence / cross-session continuity ──────────────────────────────────


def test_state_persists_and_restores_across_brain_instances(tmp_path: Path) -> None:
    db_path = str(tmp_path / "cross_session.db")

    db1 = QDEBrainDatabase(db_path)
    brain1 = QDEBrain(db=db1)
    for i in range(6):
        brain1.on_market_update(
            _tick(price=1.1000 + i * 0.0010, momentum=0.6, regime="TREND_UP"),
            observation_result=_observation(),
        )
    brain1.record_experience(SYM, "trade_closed", "closed a winning long", outcome_quality=0.8)
    brain1.save_all()

    db2 = QDEBrainDatabase(db_path)
    brain2 = QDEBrain(db=db2)
    restored = brain2.get_state(SYM)

    assert restored.instrument == SYM
    assert len(restored.hypotheses) >= 1
    assert restored.belief is not None


# ── enrich_inference_input ───────────────────────────────────────────────────


def test_enrich_inference_input_attaches_brain_context(brain: QDEBrain) -> None:
    from trainer.execution.execution_types import ExecutionForecast
    from trainer.lifecycle.lifecycle_models import PositionMetrics
    from trainer.portfolio.portfolio_scorer import PortfolioMetrics
    from trainer.qde.qde_inference import QDEAccountSnapshot, QDEInferenceInput, QDEMemorySnapshot

    brain.on_market_update(_tick())
    inp = QDEInferenceInput(
        forecast=ExecutionForecast(
            instrument=SYM,
            direction="BUY",
            confidence=0.6,
            expected_return=0.05,
            expected_risk=0.03,
            current_price=1.1,
            atr=0.001,
            spread=0.0001,
        ),
        market=MarketContext(),
        portfolio=PortfolioMetrics(
            expected_return=0.0,
            expected_volatility=0.0,
            expected_drawdown=0.0,
            expected_sharpe=0.0,
            expected_sortino=0.0,
            diversification_score=0.0,
            correlation_risk=0.0,
            concentration_score=0.0,
            instrument_count=0,
            long_count=0,
            short_count=0,
            neutral_count=0,
            weighted_confidence=0.0,
        ),
        position=PositionMetrics(),
        account=QDEAccountSnapshot(),
        memory=QDEMemorySnapshot(instrument=SYM),
    )
    enriched = brain.enrich_inference_input(inp, SYM)
    assert enriched.brain_context
    assert enriched.brain_context.get("instrument") == SYM


# ── QDEBrainAPI envelope ─────────────────────────────────────────────────────


def test_brain_api_on_market_update_success_envelope(tmp_path: Path) -> None:
    api = QDEBrainAPI(db_path=str(tmp_path / "api.db"))
    resp = api.on_market_update(_tick())
    d = resp.to_dict()
    assert d["success"] is True
    assert d["data"]["instrument"] == SYM


def test_brain_api_reflect_and_experience_endpoints(tmp_path: Path) -> None:
    api = QDEBrainAPI(db_path=str(tmp_path / "api2.db"))
    api.on_market_update(_tick())
    exp_resp = api.record_experience(SYM, "trade_closed", "took a small loss", outcome_quality=0.2)
    assert exp_resp.to_dict()["success"] is True

    reflect_resp = api.reflect(SYM, force=True)
    assert reflect_resp.to_dict()["success"] is True

    history_resp = api.experience_history(SYM)
    assert history_resp.to_dict()["data"]["count"] == 1
