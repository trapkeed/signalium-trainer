"""S33-spec — Broker-aware provider priority configuration.

Distinct from ``ProviderManager``'s ``DataContext.LIVE/TRAINING`` split
(which governs failover behaviour once a priority list is chosen): this
module defines *which* priority list applies for a given trading mode, per
the S33 routing spec:

    if mode == MANUAL:
        use OANDA, fallback TIINGO

    if mode == AUTONOMOUS:
        use EA_FEED, fallback OANDA, fallback TIINGO

Training always uses its own fixed priority regardless of trading mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ProviderName(str, Enum):
    OANDA = "OANDA"
    TIINGO = "TIINGO"
    EA_FEED = "EA_FEED"


class TradingMode(str, Enum):
    MANUAL = "MANUAL"
    AUTONOMOUS = "AUTONOMOUS"


def _default_training_providers() -> list[ProviderName]:
    return [ProviderName.OANDA, ProviderName.TIINGO]


def _default_inference_priority_manual() -> list[ProviderName]:
    return [ProviderName.OANDA, ProviderName.TIINGO]


def _default_inference_priority_autonomous() -> list[ProviderName]:
    return [ProviderName.EA_FEED, ProviderName.OANDA, ProviderName.TIINGO]


@dataclass
class ProviderPriority:
    """Ordered provider-name lists per routing context. Defaults match the
    S33 spec exactly; callers may override for testing or future providers."""

    training_providers: list[ProviderName] = field(default_factory=_default_training_providers)
    inference_priority_manual: list[ProviderName] = field(
        default_factory=_default_inference_priority_manual
    )
    inference_priority_autonomous: list[ProviderName] = field(
        default_factory=_default_inference_priority_autonomous
    )

    def for_mode(self, mode: TradingMode) -> list[ProviderName]:
        """Inference priority list for the given trading mode."""
        if mode == TradingMode.MANUAL:
            return list(self.inference_priority_manual)
        if mode == TradingMode.AUTONOMOUS:
            return list(self.inference_priority_autonomous)
        raise ValueError(f"Unknown trading mode: {mode!r}")

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "training_providers": [p.value for p in self.training_providers],
            "inference_priority_manual": [p.value for p in self.inference_priority_manual],
            "inference_priority_autonomous": [p.value for p in self.inference_priority_autonomous],
        }
