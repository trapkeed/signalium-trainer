"""Phase 22 — Order Flow Interface.

Interface stubs for future order flow data:
Volume Delta, Bid/Ask Imbalance, Cumulative Delta, Footprint,
Order Book Imbalance, Market Depth, Absorption, Iceberg Detection.

No implementation is required. These are architectural interfaces only,
designed to be drop-in additions when higher-resolution data is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OrderBookLevel:
    """One level of the order book."""

    price: float
    bid_size: float = 0.0
    ask_size: float = 0.0

    @property
    def imbalance(self) -> float:
        """Positive = more bids, negative = more asks. Range: -1 to +1."""
        total = self.bid_size + self.ask_size
        if total < 1e-10:
            return 0.0
        return round((self.bid_size - self.ask_size) / total, 4)


@dataclass
class OrderFlowBar:
    """Aggregated order flow data for one bar (future use)."""

    timestamp: float
    volume_delta: float = 0.0  # buy volume - sell volume
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    bid_imbalance: float = 0.0  # -1 to +1
    ask_imbalance: float = 0.0
    depth_levels: list[OrderBookLevel] = field(default_factory=list)

    @property
    def cumulative_delta(self) -> float:
        return self.volume_delta

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "volume_delta": round(self.volume_delta, 2),
            "buy_volume": round(self.buy_volume, 2),
            "sell_volume": round(self.sell_volume, 2),
            "bid_imbalance": round(self.bid_imbalance, 4),
            "depth_levels": len(self.depth_levels),
        }


class OrderFlowInterface:
    """Interface for order flow microstructure features.

    All methods return None when order flow data is unavailable.
    The pipeline checks for None and falls back to OHLC proxies.

    Implementation will be completed when broker execution statistics
    or Level II / DOM data become available.
    """

    def is_available(self) -> bool:
        """Returns False until order flow data integration is complete."""
        return False

    def volume_delta(self, bars: list[OrderFlowBar] | None) -> float | None:
        """Net directional volume pressure (+= buy pressure, -= sell pressure)."""
        if not bars:
            return None
        total_delta = sum(b.volume_delta for b in bars)
        total_vol = sum(b.buy_volume + b.sell_volume for b in bars)
        if total_vol < 1e-10:
            return None
        return round(total_delta / total_vol, 4)

    def bid_ask_imbalance(self, levels: list[OrderBookLevel] | None) -> float | None:
        """Order book imbalance across all levels."""
        if not levels:
            return None
        total_bid = sum(l.bid_size for l in levels)
        total_ask = sum(l.ask_size for l in levels)
        total = total_bid + total_ask
        if total < 1e-10:
            return None
        return round((total_bid - total_ask) / total, 4)

    def cumulative_delta(self, bars: list[OrderFlowBar] | None) -> float | None:
        """Running sum of volume delta — measures persistent buy/sell pressure."""
        if not bars:
            return None
        return round(sum(b.volume_delta for b in bars), 2)

    def footprint_imbalance(self, bar: OrderFlowBar | None) -> float | None:
        """Price-level imbalance from footprint chart data."""
        if not bar or not bar.depth_levels:
            return None
        imbalances = [level.imbalance for level in bar.depth_levels]
        return round(sum(imbalances) / len(imbalances), 4) if imbalances else None

    def order_book_imbalance(
        self, levels: list[OrderBookLevel] | None, top_n: int = 5
    ) -> float | None:
        """Top-of-book imbalance (most relevant to price)."""
        if not levels:
            return None
        top = levels[:top_n]
        return self.bid_ask_imbalance(top)

    def market_depth(self, levels: list[OrderBookLevel] | None) -> dict | None:
        """Total available liquidity on each side."""
        if not levels:
            return None
        return {
            "bid_depth": round(sum(l.bid_size for l in levels), 2),
            "ask_depth": round(sum(l.ask_size for l in levels), 2),
            "levels": len(levels),
        }

    def absorption_score(self, bars: list[OrderFlowBar] | None) -> float | None:
        """Detect absorption: high volume with small price change.

        High score = price is being absorbed (large player defending a level).
        """
        if not bars or len(bars) < 2:
            return None
        last = bars[-1]
        total_vol = last.buy_volume + last.sell_volume
        if total_vol < 1e-10:
            return None
        # Will be computed from price + volume data when available
        return None  # Implementation pending tick + DOM data

    def iceberg_probability(self, bars: list[OrderFlowBar] | None) -> float | None:
        """Estimate probability of iceberg orders at current price.

        Icebergs leave a signature of repeated small fills at the same price.
        Requires tick-level data — not yet implementable from OHLC.
        """
        if not bars:
            return None
        return None  # Implementation pending tick data

    def data_quality_report(self) -> dict:
        """Report on what order flow data is available."""
        return {
            "available": False,
            "volume_delta": "pending — requires tick data",
            "bid_ask_imbalance": "pending — requires Level II data",
            "cumulative_delta": "pending — requires tick data",
            "footprint": "pending — requires tick data",
            "order_book": "pending — requires Level II / DOM data",
            "absorption": "pending — requires tick + DOM data",
            "iceberg_detection": "pending — requires tick data",
            "current_source": "OHLC",
            "degradation": "graceful — all OHLC features fully functional",
        }
