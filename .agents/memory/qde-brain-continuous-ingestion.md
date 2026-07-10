---
name: Brain continuous-ingestion driver
description: How BrainStateManager unifies MarketDataRouter → MIE engines → QDEMarketObserver → QDEBrain into one per-symbol tick, and how InternalStateStore exposes read views over that state.
---

## Design

`trainer/qde/brain/brain_state_manager.py::BrainStateManager` is the "first layer"
driver that makes QDE's internal consciousness advance continuously for every
enabled symbol, independent of whether a user is connected or autonomous
trading is on.

Per-symbol tick pipeline:
1. `MarketDataRouter.route(symbol, "get_candles", mode=TradingMode.AUTONOMOUS, context="inference")` — fetches candles from whichever provider (OANDA/Tiingo/EA) is active. Never re-implements provider selection.
2. Existing MIE engines (`MarketRegimeEngine`, `VolatilityIntelligenceEngine`, `LiquidityIntelligenceEngine`, `SessionIntelligenceEngine`) classify the candles. Never reimplement this math — always call the existing engines.
3. MIE's 16-way regime taxonomy is mapped down to Brain's 3-way vocabulary (`TREND_UP` / `TREND_DOWN` / `RANGE`) via a small `_simplified_regime` helper — this mapping lives only in `BrainStateManager`, not duplicated elsewhere.
4. `QDEMarketObserver.observe()` is ticked first (owns the lifecycle FSM), then `QDEBrain.on_market_update()` consumes its `ObservationResult` — Brain never drives the FSM itself.

**Why a driver, not a scheduler owner:** the codebase convention (seen across observer/lifecycle managers) is that these classes expose `tick_symbol()` / `run_once()` and enumerate `enabled_instruments()` via `SymbolNormalizer.list_symbols()`, but do not own their own thread/loop — an external always-on scheduler (or test harness) calls them repeatedly. Per-symbol failures (e.g. `ProviderNotConfigured`) are caught and reported via a `SymbolTickOutcome` result object so one broken symbol/provider never blocks others.

**How to apply:** if you need to add ingestion of a new signal type or a new provider into the brain, extend the single pipeline inside `BrainStateManager.tick_symbol`, not a parallel path. If you need a scheduler that actually loops, wrap `BrainStateManager.run_once()` externally rather than adding loop ownership to the manager itself.

## InternalStateStore

`trainer/qde/brain/internal_state_store.py::InternalStateStore` is a read-only
facade over `BrainState`/`QDEBrain` — it adds no new persistence schema, just
convenience views: `recent_structure()`, `volatility_evolution()`,
`trend_evolution()`, `liquidity_observations()`, `session_context()`,
`unfinished_narratives()`, and a combined `snapshot()`. Always reuse
`QDEBrainDatabase`/`BrainState` as the source of truth; do not add a second
store for brain state.
