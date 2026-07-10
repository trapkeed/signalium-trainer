"""QDE Brain — brain state manager (first layer, continuous ingestion).

This is the piece that makes the Brain's "always thinking" promise actually
happen end-to-end: it drives raw market updates from **OANDA / Tiingo / EA
feed** (via the existing `MarketDataRouter`) through the existing MIE
classification engines (`MarketRegimeEngine`, `VolatilityIntelligenceEngine`,
`LiquidityIntelligenceEngine`, `SessionIntelligenceEngine`) into a
`SymbolObservationInput`, ticks the existing `QDEMarketObserver`, and feeds
the result into `QDEBrain.on_market_update()` — for every *enabled* symbol,
regardless of whether a user is connected or autonomous trading is active.

It deliberately does not:
  - reimplement regime/volatility/liquidity/session classification (reuses
    the MIE engines as-is);
  - reimplement lifecycle staging (reuses `QDEMarketObserver` as-is);
  - reimplement narrative/hypothesis/belief reasoning (reuses `QDEBrain`
    as-is);
  - own a scheduler/background loop — like every other "always-on" QDE
    component (`QDEMarketObserver`, `QDEBrain`), it exposes `run_once()` /
    `tick_symbol()` for an external dispatcher (a feed-driven loop, cron, or
    the `/qde/observer/{instrument}/observe` HTTP tick) to call on its own
    cadence. That dispatcher is what must not gate calls on user-connection
    or autonomous-trading-enabled state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.market_data.provider_interface import ProviderNotConfigured
from trainer.market_data.provider_priority import TradingMode
from trainer.mie.liquidity_intelligence import LiquidityIntelligenceEngine
from trainer.mie.market_regime_engine import MarketRegimeEngine
from trainer.mie.mie_models import MarketInput
from trainer.mie.session_intelligence import SessionIntelligenceEngine
from trainer.mie.volatility_intelligence import VolatilityIntelligenceEngine
from trainer.qde.brain.brain import QDEBrain
from trainer.qde.brain.internal_state_store import InternalStateStore
from trainer.qde.brain.persistence import QDEBrainDatabase
from trainer.qde.qde_observer import QDEMarketObserver
from trainer.qde.qde_observer_models import ObservationResult, SymbolObservationInput

if TYPE_CHECKING:
    from trainer.market_data.market_data_router import MarketDataRouter
    from trainer.market_data.symbol_normalizer import SymbolNormalizer
    from trainer.qde.brain.brain_models import BrainThought

DEFAULT_CANDLE_TIMEFRAME = "M15"
DEFAULT_CANDLE_COUNT = 100

# Regimes the MIE engine reports as directional continuation, mapped onto the
# simplified "TREND_UP" / "TREND_DOWN" / "RANGE" vocabulary `QDEBrain`'s
# hypothesis formation already understands (see `brain.py::_update_hypotheses`).
_TREND_LIKE_REGIMES = {"TRENDING", "STRONG_TREND", "WEAK_TREND", "EXPANSION", "BREAKOUT"}


@dataclass
class SymbolTickOutcome:
    """Result of one continuous-ingestion pass for a single symbol."""

    instrument: str
    ok: bool
    thought: "BrainThought | None" = None
    error: str = ""
    provider_used: str = ""
    unix_ts: float = field(default_factory=time.time)


class BrainStateManager:
    """Orchestrates continuous, session-independent ingestion into the Brain.

    One instance is meant to be long-lived for the process: it holds the
    `QDEMarketObserver` (lifecycle) and `QDEBrain` (consciousness) instances,
    and per-symbol rolling OHLCV history used to build `MarketInput` for the
    MIE engines, since providers return recent candles rather than a running
    series.
    """

    def __init__(
        self,
        market_data_router: "MarketDataRouter | None" = None,
        symbol_normalizer: "SymbolNormalizer | None" = None,
        brain: QDEBrain | None = None,
        observer: QDEMarketObserver | None = None,
        db: QDEBrainDatabase | None = None,
        candle_timeframe: str = DEFAULT_CANDLE_TIMEFRAME,
        candle_count: int = DEFAULT_CANDLE_COUNT,
    ) -> None:
        self._router = market_data_router
        self._normalizer = symbol_normalizer
        self._db = db if db is not None else (brain._db if brain is not None else None)
        self._brain = brain if brain is not None else QDEBrain(db=self._db)
        self._observer = observer if observer is not None else QDEMarketObserver()
        self.internal_state = InternalStateStore(self._brain)

        self._regime_engine = MarketRegimeEngine()
        self._volatility_engine = VolatilityIntelligenceEngine()
        self._liquidity_engine = LiquidityIntelligenceEngine()
        self._session_engine = SessionIntelligenceEngine()

        self._candle_timeframe = candle_timeframe
        self._candle_count = candle_count

    # ── Continuous ingestion ──────────────────────────────────────────────

    def enabled_instruments(self) -> list[str]:
        """Every symbol currently enabled for observation — the Brain must
        keep ingesting all of these continuously, independent of whatever a
        connected user happens to be looking at.
        """
        if self._normalizer is None:
            return list(self._brain.all_states().keys())
        return [s["symbol"] for s in self._normalizer.list_symbols() if s["enabled"]]

    def run_once(self, instruments: list[str] | None = None) -> list[SymbolTickOutcome]:
        """One continuous-ingestion pass over every enabled symbol (or an
        explicit override list). A failure on one symbol (e.g. a provider
        not yet configured) is isolated and logged — it must never stop the
        rest of the enabled symbols from being ingested this pass.
        """
        targets = instruments if instruments is not None else self.enabled_instruments()
        outcomes: list[SymbolTickOutcome] = []
        for instrument in targets:
            outcomes.append(self.tick_symbol(instrument))
        return outcomes

    def tick_symbol(self, instrument: str) -> SymbolTickOutcome:
        """Pull the latest market data for one symbol, classify it via the
        existing MIE engines, and advance both the Observer's lifecycle
        state and the Brain's internal consciousness for it.
        """
        try:
            candles, provider_used = self._fetch_candles(instrument)
            market_input = self._build_market_input(instrument, candles)
            obs_input = self._to_observation_input(instrument, market_input)

            observer_result: ObservationResult = self._observer.observe(obs_input)
            thought = self._brain.on_market_update(obs_input, observer_result)

            return SymbolTickOutcome(
                instrument=instrument, ok=True, thought=thought, provider_used=provider_used
            )
        except ProviderNotConfigured as exc:
            logger.warning(f"BrainStateManager: no provider configured for {instrument}: {exc}")
            return SymbolTickOutcome(instrument=instrument, ok=False, error=str(exc))
        except Exception as exc:  # noqa: BLE001 — one symbol's failure must not break the pass
            logger.warning(f"BrainStateManager: tick failed for {instrument}: {exc}")
            return SymbolTickOutcome(instrument=instrument, ok=False, error=str(exc))

    # ── Market data → MIE input ───────────────────────────────────────────

    def _fetch_candles(self, instrument: str) -> tuple[list[dict[str, Any]], str]:
        if self._router is None:
            raise ProviderNotConfigured(
                "BrainStateManager has no MarketDataRouter configured — cannot pull "
                "OANDA/Tiingo/EA data for continuous ingestion"
            )
        candles = self._router.route(
            instrument,
            "get_candles",
            self._candle_timeframe,
            self._candle_count,
            mode=TradingMode.AUTONOMOUS,
            context="inference",
        )
        provider_used = self._router.active_provider_name(
            mode=TradingMode.AUTONOMOUS, context="inference"
        )
        if not candles:
            raise ValueError(f"no candles returned for {instrument}")
        return candles, provider_used

    def _build_market_input(self, instrument: str, candles: list[dict[str, Any]]) -> MarketInput:
        closes = [float(c.get("close", 0.0)) for c in candles]
        highs = [float(c.get("high", c.get("close", 0.0))) for c in candles]
        lows = [float(c.get("low", c.get("close", 0.0))) for c in candles]
        volumes = [float(c.get("volume", 0.0)) for c in candles]
        returns = [
            (closes[i] - closes[i - 1]) / closes[i - 1] if closes[i - 1] else 0.0
            for i in range(1, len(closes))
        ]
        latest = candles[-1]
        now = time.time()

        return MarketInput(
            symbol=instrument,
            timestamp=now,
            open=float(latest.get("open", closes[-1] if closes else 0.0)),
            high=highs[-1] if highs else 0.0,
            low=lows[-1] if lows else 0.0,
            close=closes[-1] if closes else 1.0,
            volume=volumes[-1] if volumes else 1_000.0,
            spread=float(latest.get("spread", 0.0001)),
            atr=float(latest.get("atr", 0.001)),
            adx=float(latest.get("adx", 25.0)),
            rsi=float(latest.get("rsi", 50.0)),
            closes=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
            returns=returns,
            session_hour=time.gmtime(now).tm_hour,
        )

    def _to_observation_input(
        self, instrument: str, market_input: MarketInput
    ) -> SymbolObservationInput:
        regime_result = self._regime_engine.detect(market_input)
        volatility_result = self._volatility_engine.analyse(market_input)
        liquidity_result = self._liquidity_engine.analyse(market_input)
        session_result = self._session_engine.analyse(market_input)

        return SymbolObservationInput(
            instrument=instrument,
            volatility_state=volatility_result.state.value,
            volatility_quality=volatility_result.quality / 100.0,
            liquidity_state=liquidity_result.state.value,
            liquidity_quality=liquidity_result.quality / 100.0,
            session_quality=session_result.quality / 100.0,
            market_regime=self._simplified_regime(regime_result.regime.value, market_input),
            unix_ts=market_input.timestamp,
        )

    def _simplified_regime(self, mie_regime: str, market_input: MarketInput) -> str:
        """Map the MIE engine's 16-way regime classification onto the
        directional TREND_UP / TREND_DOWN / RANGE vocabulary the Brain's
        hypothesis formation already consumes (see
        `brain.py::_update_hypotheses`), using price slope for direction.
        """
        if mie_regime not in _TREND_LIKE_REGIMES:
            return "RANGE"
        closes = market_input.closes
        if len(closes) < 2:
            return "RANGE"
        return "TREND_UP" if closes[-1] >= closes[0] else "TREND_DOWN"

    # ── Access ────────────────────────────────────────────────────────────

    @property
    def brain(self) -> QDEBrain:
        return self._brain

    @property
    def observer(self) -> QDEMarketObserver:
        return self._observer
