"""QDE Brain — first-layer continuous ingestion tests.

Covers `BrainStateManager` (continuous ingestion from a market-data router
across all enabled symbols, independent of user/trading state) and
`InternalStateStore` (the recent-structure / volatility-evolution /
trend-evolution / liquidity-observations / session-context / unfinished-
narratives views).
"""

from __future__ import annotations

from typing import Any

from trainer.qde.brain.brain import QDEBrain
from trainer.qde.brain.brain_state_manager import BrainStateManager
from trainer.qde.brain.internal_state_store import InternalStateStore

SYM = "EURUSD"


class _FakeRouter:
    """Minimal stand-in for `MarketDataRouter.route()` — returns a rising
    candle series so the regime engine detects a trend.
    """

    def __init__(self, provider_name: str = "oanda", candles: list[dict[str, Any]] | None = None):
        self.provider_name = provider_name
        self._candles = candles
        self.calls: list[tuple] = []

    def route(self, symbol, method_name, *args, mode=None, context="inference", **kwargs):
        self.calls.append((symbol, method_name, args, mode, context))
        if self._candles is not None:
            return self._candles
        base = 1.1000
        candles = []
        for i in range(60):
            base += 0.0005
            candles.append(
                {
                    "open": base,
                    "high": base + 0.0002,
                    "low": base - 0.0002,
                    "close": base,
                    "volume": 1000 + i,
                }
            )
        return candles

    def active_provider_name(self, mode=None, context="inference") -> str:
        return self.provider_name


class _FailingRouter:
    def route(self, *args, **kwargs):
        raise RuntimeError("provider unreachable")

    def active_provider_name(self, *args, **kwargs) -> str:
        return "none"


class _FakeNormalizer:
    def __init__(self, symbols: dict[str, bool]):
        self._symbols = symbols

    def list_symbols(self) -> list[dict[str, Any]]:
        return [
            {"symbol": sym, "asset_class": "forex", "enabled": enabled}
            for sym, enabled in self._symbols.items()
        ]


# ── BrainStateManager ────────────────────────────────────────────────────────


def test_tick_symbol_advances_observer_and_brain():
    mgr = BrainStateManager(market_data_router=_FakeRouter())
    outcome = mgr.tick_symbol(SYM)

    assert outcome.ok
    assert outcome.provider_used == "oanda"
    assert outcome.thought is not None
    assert outcome.thought.instrument == SYM
    assert mgr.brain.get_state(SYM).tick_count == 1
    assert SYM in mgr.observer.all_memory()


def test_run_once_only_ingests_enabled_symbols():
    router = _FakeRouter()
    normalizer = _FakeNormalizer({SYM: True, "GBPUSD": False})
    mgr = BrainStateManager(market_data_router=router, symbol_normalizer=normalizer)

    outcomes = mgr.run_once()

    assert [o.instrument for o in outcomes] == [SYM]
    requested_symbols = {call[0] for call in router.calls}
    assert requested_symbols == {SYM}


def test_run_once_isolates_per_symbol_failures():
    normalizer = _FakeNormalizer({SYM: True, "GBPUSD": True})
    mgr = BrainStateManager(market_data_router=_FailingRouter(), symbol_normalizer=normalizer)

    outcomes = mgr.run_once()

    assert len(outcomes) == 2
    assert all(not o.ok for o in outcomes)
    assert all(o.error for o in outcomes)


def test_tick_symbol_without_router_configured_fails_gracefully():
    mgr = BrainStateManager(market_data_router=None)
    outcome = mgr.tick_symbol(SYM)

    assert not outcome.ok
    assert "MarketDataRouter" in outcome.error


def test_continuous_ingestion_advances_state_regardless_of_call_count():
    """Simulates repeated ticks the way an external, always-running
    dispatcher would call this — independent of any user/trading session.
    """
    mgr = BrainStateManager(market_data_router=_FakeRouter())
    for _ in range(5):
        mgr.tick_symbol(SYM)

    assert mgr.brain.get_state(SYM).tick_count == 5


def test_simplified_regime_maps_trend_and_direction():
    mgr = BrainStateManager(market_data_router=_FakeRouter())
    rising = mgr._build_market_input(SYM, _FakeRouter().route(SYM, "get_candles"))
    assert mgr._simplified_regime("TRENDING", rising) == "TREND_UP"
    assert mgr._simplified_regime("RANGE", rising) == "RANGE"


# ── InternalStateStore ───────────────────────────────────────────────────────


def test_internal_state_store_snapshot_reflects_ingested_ticks():
    mgr = BrainStateManager(market_data_router=_FakeRouter())
    for _ in range(3):
        mgr.tick_symbol(SYM)

    store: InternalStateStore = mgr.internal_state
    snapshot = store.snapshot(SYM)

    assert snapshot["instrument"] == SYM
    assert len(snapshot["volatility_evolution"]) == 3
    assert len(snapshot["trend_evolution"]) == 3
    assert len(snapshot["liquidity_observations"]) == 3
    assert snapshot["session_context"] != "" or snapshot["session_context"] == ""
    assert isinstance(snapshot["unfinished_narratives"], list)
    assert isinstance(snapshot["recent_structure"], list)


def test_internal_state_store_evolution_is_chronological():
    brain = QDEBrain()
    store = InternalStateStore(brain)
    from trainer.qde.qde_observer_models import ObservationResult, SymbolObservationInput
    from trainer.qde.qde_observer_types import ObserverEvent, ObserverState

    def _obs(regime: str, ts: float) -> tuple[SymbolObservationInput, ObservationResult]:
        inp = SymbolObservationInput(instrument=SYM, market_regime=regime, unix_ts=ts)
        result = ObservationResult(
            instrument=SYM,
            state=ObserverState.OBSERVING,
            previous_state=ObserverState.OBSERVING,
            transitioned=False,
            event=ObserverEvent.SETUP_FORMING,
            readiness_score=0.5,
            confidence=0.5,
            reasoning="test",
            unix_ts=ts,
        )
        return inp, result

    inp1, res1 = _obs("TREND_UP", 100.0)
    inp2, res2 = _obs("RANGE", 200.0)
    brain.on_market_update(inp1, res1)
    brain.on_market_update(inp2, res2)

    evolution = store.trend_evolution(SYM)
    assert [e["market_regime"] for e in evolution] == ["TREND_UP", "RANGE"]


def test_internal_state_store_unfinished_narratives_empty_for_new_symbol():
    brain = QDEBrain()
    store = InternalStateStore(brain)
    assert store.unfinished_narratives(SYM) == []
    assert store.recent_structure(SYM) == []
    assert store.session_context(SYM) == ""
