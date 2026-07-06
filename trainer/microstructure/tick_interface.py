"""Phase 22 — Tick Feature Interface.

Stub interfaces for future tick-derived features.
All features are Optional — the system degrades gracefully to OHLC-only.

Supported (future):
Tick Velocity, Tick Density, Tick Imbalance, Tick Momentum,
Tick Volatility, Tick Acceleration, Tick Entropy.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.microstructure.micro_models import MicrostructureFeatures


@dataclass
class TickBar:
    """Minimal tick data container (future use)."""

    timestamp: float
    price: float
    size: float = 1.0
    side: str = "UNKNOWN"  # BUY | SELL | UNKNOWN

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "price": round(self.price, 5),
            "size": round(self.size, 2),
            "side": self.side,
        }


class TickInterface:
    """Interface for tick-derived microstructure features.

    All methods accept Optional[List[TickBar]] and return None when
    tick data is unavailable — OHLC features are always the fallback.

    Implementations should be drop-in additions; the rest of the pipeline
    checks `features.has_tick_features` before using tick results.
    """

    def is_available(self) -> bool:
        """Returns False until tick data integration is complete."""
        return False

    def tick_velocity(self, ticks: list[TickBar] | None) -> float | None:
        """Ticks per second — measures speed of market activity."""
        if not ticks or len(ticks) < 2:
            return None
        duration = ticks[-1].timestamp - ticks[0].timestamp
        if duration < 1e-6:
            return None
        return round(len(ticks) / duration, 4)

    def tick_density(self, ticks: list[TickBar] | None, price_range: float = 0.0) -> float | None:
        """Ticks per unit price range — concentration of activity."""
        if not ticks or price_range < 1e-10:
            return None
        return round(len(ticks) / price_range, 4)

    def tick_imbalance(self, ticks: list[TickBar] | None) -> float | None:
        """Directional imbalance: (buys - sells) / total.
        +1 = all buys, -1 = all sells, 0 = balanced.
        """
        if not ticks:
            return None
        buys = sum(t.size for t in ticks if t.side == "BUY")
        sells = sum(t.size for t in ticks if t.side == "SELL")
        total = buys + sells
        if total < 1e-10:
            return None
        return round((buys - sells) / total, 4)

    def tick_momentum(self, ticks: list[TickBar] | None, window: int = 20) -> float | None:
        """Exponentially weighted price momentum over recent ticks."""
        if not ticks or len(ticks) < 2:
            return None
        recent = ticks[-window:]
        if len(recent) < 2:
            return None
        price_change = recent[-1].price - recent[0].price
        price_range = max(t.price for t in recent) - min(t.price for t in recent)
        if price_range < 1e-10:
            return None
        return round(price_change / price_range, 4)

    def tick_volatility(self, ticks: list[TickBar] | None) -> float | None:
        """Standard deviation of tick price changes."""
        if not ticks or len(ticks) < 3:
            return None
        import statistics

        changes = [ticks[i].price - ticks[i - 1].price for i in range(1, len(ticks))]
        if not changes:
            return None
        return round(statistics.stdev(changes), 6)

    def tick_acceleration(self, ticks: list[TickBar] | None) -> float | None:
        """Second derivative of tick momentum — acceleration of price movement."""
        if not ticks or len(ticks) < 4:
            return None
        half = len(ticks) // 2
        early = ticks[:half]
        late = ticks[half:]
        early_mom = self.tick_momentum(early)
        late_mom = self.tick_momentum(late)
        if early_mom is None or late_mom is None:
            return None
        return round(late_mom - early_mom, 4)

    def tick_entropy(self, ticks: list[TickBar] | None) -> float | None:
        """Shannon entropy of tick direction distribution.

        High entropy = random/balanced market.
        Low entropy  = directionally biased market.
        """
        if not ticks or len(ticks) < 4:
            return None
        import math

        buy_count = sum(1 for t in ticks if t.side == "BUY")
        sell_count = sum(1 for t in ticks if t.side == "SELL")
        total = buy_count + sell_count
        if total < 1:
            return None
        p_buy = buy_count / total
        p_sell = sell_count / total
        entropy = 0.0
        for p in (p_buy, p_sell):
            if p > 1e-10:
                entropy -= p * math.log2(p)
        # Normalize to 0–1 (max entropy for 2 categories = 1.0)
        return round(entropy, 4)

    def enrich_features(
        self,
        features: MicrostructureFeatures,
        ticks: list[TickBar] | None,
    ) -> MicrostructureFeatures:
        """Attempt to enrich features with tick data.

        If tick data is unavailable, features are returned unchanged.
        """
        if not ticks:
            logger.debug("TickInterface: no tick data — skipping enrichment")
            return features

        prices = [t.price for t in ticks]
        price_range = max(prices) - min(prices) if len(prices) > 1 else 0.0

        features.tick_velocity = self.tick_velocity(ticks)
        features.tick_density = self.tick_density(ticks, price_range)
        features.tick_imbalance = self.tick_imbalance(ticks)
        features.tick_momentum = self.tick_momentum(ticks)
        features.tick_volatility = self.tick_volatility(ticks)
        features.tick_acceleration = self.tick_acceleration(ticks)
        features.tick_entropy = self.tick_entropy(ticks)

        logger.debug(
            f"TickInterface: enriched features with {len(ticks)} ticks, "
            f"velocity={features.tick_velocity}"
        )
        return features
