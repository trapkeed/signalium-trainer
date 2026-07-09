"""S33-spec — Broker-aware Market Data Router.

Ties together three pieces that already exist independently:

  * ``ProviderPriority`` — which provider names apply for MANUAL vs
    AUTONOMOUS trading mode (and for training).
  * ``ProviderManager`` — health-based failover once a priority order is
    chosen (reused as-is, not reimplemented).
  * ``SymbolNormalizer`` — raw broker spelling -> canonical symbol.

Routing rules (S33):
  - Training always uses canonical symbols only. A non-canonical symbol
    passed with ``context="training"`` is a hard error — training data
    must never silently normalize, since normalization ambiguity in a
    dataset would silently mislabel history.
  - Live/manual inference: OANDA, fallback TIINGO.
  - Live/autonomous inference: EA_FEED, fallback OANDA, fallback TIINGO.
  - Any raw symbol used for inference is auto-normalized (and the
    normalization event logged) before being routed to a provider.
"""

from __future__ import annotations

from typing import Any, Literal

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.market_data.provider_interface import MarketDataProvider
from trainer.market_data.provider_manager import DataContext, ProviderManager
from trainer.market_data.provider_priority import ProviderName, ProviderPriority, TradingMode
from trainer.market_data.symbol_normalizer import SymbolNormalizer

RoutingContext = Literal["training", "inference"]


class SymbolNotCanonicalError(Exception):
    """Raised when training requests a symbol that is not already canonical.

    Training must consume only canonical symbols — silently normalizing a
    training-data request would let dataset ambiguity slip through
    unnoticed. Callers must normalize explicitly (and audit the result)
    before requesting training data.
    """


class MarketDataRouter:
    def __init__(
        self,
        providers: dict[str, MarketDataProvider],
        *,
        priority: ProviderPriority | None = None,
        normalizer: SymbolNormalizer | None = None,
        **provider_manager_kwargs: Any,
    ) -> None:
        """``providers`` maps provider name (case-insensitive, e.g. "oanda",
        "tiingo", "ea_feed") to an already-constructed ``MarketDataProvider``.
        Missing providers are simply skipped when building a priority list —
        so a deployment without an EA feed configured degrades gracefully to
        OANDA/TIINGO for autonomous mode rather than erroring at construction.
        """
        self._providers = {name.lower(): p for name, p in providers.items()}
        self._priority = priority if priority is not None else ProviderPriority()
        self._normalizer = normalizer if normalizer is not None else SymbolNormalizer()
        self._manager_kwargs = provider_manager_kwargs

        self._managers: dict[str, ProviderManager] = {
            "training": self._build_manager(self._priority.training_providers),
            "manual": self._build_manager(self._priority.for_mode(TradingMode.MANUAL)),
            "autonomous": self._build_manager(self._priority.for_mode(TradingMode.AUTONOMOUS)),
        }

    def _resolve_providers(self, names: list[ProviderName]) -> list[MarketDataProvider]:
        resolved = []
        for name in names:
            provider = self._providers.get(name.value.lower())
            if provider is not None:
                resolved.append(provider)
            else:
                logger.warning(
                    f"MarketDataRouter: provider {name.value!r} not configured, skipping in priority list"
                )
        return resolved

    def _build_manager(self, names: list[ProviderName]) -> ProviderManager:
        ordered = self._resolve_providers(names)
        # Each internal ProviderManager only ever uses DataContext.LIVE —
        # ProviderManager's own LIVE/TRAINING split is orthogonal to (and
        # reused independently of) the routing contexts here, so we always
        # feed the same ordered list into both of its slots and only ever
        # call with DataContext.LIVE.
        return ProviderManager(
            live_priority=ordered, training_priority=ordered, **self._manager_kwargs
        )

    def priority_for(
        self, mode: TradingMode | None = None, *, context: RoutingContext = "inference"
    ) -> list[str]:
        if context == "training":
            return [p.value for p in self._priority.training_providers]
        if mode is None:
            raise ValueError("mode is required for inference routing context")
        return [p.value for p in self._priority.for_mode(mode)]

    def active_provider_name(
        self, mode: TradingMode | None = None, *, context: RoutingContext = "inference"
    ) -> str:
        manager = self._manager_for(mode, context)
        return manager.active_provider(DataContext.LIVE).name()

    def _manager_for(self, mode: TradingMode | None, context: RoutingContext) -> ProviderManager:
        if context == "training":
            return self._managers["training"]
        if mode == TradingMode.MANUAL:
            return self._managers["manual"]
        if mode == TradingMode.AUTONOMOUS:
            return self._managers["autonomous"]
        raise ValueError(f"mode must be set for inference routing context, got {mode!r}")

    def resolve_symbol(self, raw_symbol: str, *, broker: str | None = None) -> str:
        """Auto-normalize a raw broker symbol for live inference use. Always
        logs the normalization event (via ``SymbolNormalizer.normalize``)."""
        result = self._normalizer.normalize(raw_symbol, broker=broker, fuzzy=True)
        if result.canonical is None:
            raise ValueError(f"Could not normalize symbol {raw_symbol!r} (broker={broker!r})")
        if not result.enabled:
            raise ValueError(f"Symbol {result.canonical!r} is disabled")
        return result.canonical

    def route(
        self,
        symbol: str,
        method_name: str,
        *args: Any,
        mode: TradingMode | None = None,
        context: RoutingContext = "inference",
        broker: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Normalize (inference only) and dispatch ``method_name`` to the
        highest-priority available provider for ``mode``/``context``."""
        if context == "training":
            canonical = symbol.upper()
            if (
                canonical not in self._normalizer._symbols
            ):  # noqa: SLF001 — same-package internal use
                raise SymbolNotCanonicalError(
                    f"Training data must use canonical symbols only; {symbol!r} is not canonical. "
                    f"Normalize explicitly via SymbolNormalizer before requesting training data."
                )
        else:
            canonical = self.resolve_symbol(symbol, broker=broker)

        manager = self._manager_for(mode, context)
        return manager.call(DataContext.LIVE, method_name, canonical, *args, **kwargs)

    def diagnostics(self) -> dict[str, Any]:
        """Snapshot for the admin/Signalium diagnostics surface: configured
        priority lists, which providers are actually wired up, and current
        active/health state per routing context."""
        return {
            "priority": self._priority.to_dict(),
            "configured_providers": sorted(self._providers.keys()),
            "contexts": {
                key: {
                    "active_provider": (
                        manager.active_provider(DataContext.LIVE).name()
                        if manager.health_snapshot(DataContext.LIVE)
                        else None
                    ),
                    "health": manager.health_snapshot(DataContext.LIVE),
                    "recent_failovers": manager.recent_failovers(),
                }
                for key, manager in self._managers.items()
            },
        }
