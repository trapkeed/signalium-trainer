"""Phase 20 — Portfolio Instruments Registry.

Defines all supported instruments, their asset classes, currency exposures,
regions, and metadata. Supports unlimited future instrument registration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class AssetClass(str, Enum):
    FOREX = "forex"
    COMMODITY = "commodity"
    INDEX = "index"
    CRYPTO = "crypto"
    BOND = "bond"
    EQUITY = "equity"


class Region(str, Enum):
    NORTH_AMERICA = "north_america"
    EUROPE = "europe"
    ASIA_PACIFIC = "asia_pacific"
    GLOBAL = "global"
    MIDDLE_EAST = "middle_east"
    LATIN_AMERICA = "latin_america"


@dataclass
class InstrumentInfo:
    symbol: str
    asset_class: AssetClass
    base_currency: str
    quote_currency: str
    region: Region
    description: str = ""
    sector: str = ""
    is_inverse: bool = False
    tick_value: float = 1.0
    typical_spread: float = 0.0001
    tags: List[str] = field(default_factory=list)

    def currency_exposures(self) -> Dict[str, float]:
        if self.asset_class == AssetClass.FOREX:
            return {self.base_currency: 1.0, self.quote_currency: -1.0}
        if self.asset_class == AssetClass.COMMODITY:
            return {self.quote_currency: -1.0}
        if self.asset_class in (AssetClass.INDEX, AssetClass.EQUITY):
            return {self.quote_currency: -1.0}
        if self.asset_class == AssetClass.CRYPTO:
            return {self.quote_currency: -1.0}
        return {self.quote_currency: -1.0}

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "asset_class": self.asset_class.value,
            "base_currency": self.base_currency,
            "quote_currency": self.quote_currency,
            "region": self.region.value,
            "description": self.description,
            "sector": self.sector,
            "is_inverse": self.is_inverse,
            "tags": self.tags,
        }


@dataclass
class ModelForecast:
    instrument: str
    direction: str
    confidence: float
    expected_return: float
    expected_risk: float
    timestamp: str = ""
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.direction in ("BUY", "SELL", "NEUTRAL"), (
            f"direction must be BUY/SELL/NEUTRAL, got {self.direction!r}"
        )
        self.confidence = max(0.0, min(1.0, self.confidence))

    def signed_return(self) -> float:
        if self.direction == "BUY":
            return self.expected_return
        if self.direction == "SELL":
            return -self.expected_return
        return 0.0

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "direction": self.direction,
            "confidence": self.confidence,
            "expected_return": self.expected_return,
            "expected_risk": self.expected_risk,
            "timestamp": self.timestamp,
        }


_DEFAULT_INSTRUMENTS: List[dict] = [
    dict(symbol="EURUSD", asset_class=AssetClass.FOREX,   base="EUR", quote="USD", region=Region.EUROPE,        desc="Euro / US Dollar",          tags=["major", "eur", "usd"]),
    dict(symbol="GBPUSD", asset_class=AssetClass.FOREX,   base="GBP", quote="USD", region=Region.EUROPE,        desc="British Pound / US Dollar",  tags=["major", "gbp", "usd"]),
    dict(symbol="USDJPY", asset_class=AssetClass.FOREX,   base="USD", quote="JPY", region=Region.ASIA_PACIFIC,  desc="US Dollar / Japanese Yen",   tags=["major", "usd", "jpy"]),
    dict(symbol="USDCHF", asset_class=AssetClass.FOREX,   base="USD", quote="CHF", region=Region.EUROPE,        desc="US Dollar / Swiss Franc",    tags=["major", "usd", "chf", "safe_haven"]),
    dict(symbol="USDCAD", asset_class=AssetClass.FOREX,   base="USD", quote="CAD", region=Region.NORTH_AMERICA, desc="US Dollar / Canadian Dollar", tags=["major", "usd", "cad", "commodity_currency"]),
    dict(symbol="AUDUSD", asset_class=AssetClass.FOREX,   base="AUD", quote="USD", region=Region.ASIA_PACIFIC,  desc="Australian Dollar / US Dollar", tags=["major", "aud", "usd", "commodity_currency"]),
    dict(symbol="NZDUSD", asset_class=AssetClass.FOREX,   base="NZD", quote="USD", region=Region.ASIA_PACIFIC,  desc="New Zealand Dollar / US Dollar", tags=["major", "nzd", "usd", "commodity_currency"]),
    dict(symbol="EURJPY", asset_class=AssetClass.FOREX,   base="EUR", quote="JPY", region=Region.EUROPE,        desc="Euro / Japanese Yen",        tags=["cross", "eur", "jpy"]),
    dict(symbol="GBPJPY", asset_class=AssetClass.FOREX,   base="GBP", quote="JPY", region=Region.EUROPE,        desc="British Pound / Japanese Yen", tags=["cross", "gbp", "jpy"]),
    dict(symbol="XAUUSD", asset_class=AssetClass.COMMODITY, base="XAU", quote="USD", region=Region.GLOBAL,       desc="Gold / US Dollar",           tags=["commodity", "gold", "safe_haven", "usd"]),
    dict(symbol="XAGUSD", asset_class=AssetClass.COMMODITY, base="XAG", quote="USD", region=Region.GLOBAL,       desc="Silver / US Dollar",         tags=["commodity", "silver", "usd"]),
    dict(symbol="NAS100", asset_class=AssetClass.INDEX,   base="NAS", quote="USD", region=Region.NORTH_AMERICA, desc="NASDAQ 100",                 tags=["index", "tech", "usd", "risk_on"]),
    dict(symbol="US30",   asset_class=AssetClass.INDEX,   base="DOW", quote="USD", region=Region.NORTH_AMERICA, desc="Dow Jones 30",               tags=["index", "usd", "risk_on"]),
    dict(symbol="SPX500", asset_class=AssetClass.INDEX,   base="SPX", quote="USD", region=Region.NORTH_AMERICA, desc="S&P 500",                    tags=["index", "usd", "risk_on"]),
    dict(symbol="BTCUSD", asset_class=AssetClass.CRYPTO,  base="BTC", quote="USD", region=Region.GLOBAL,        desc="Bitcoin / US Dollar",        tags=["crypto", "usd", "risk_on"]),
    dict(symbol="ETHUSD", asset_class=AssetClass.CRYPTO,  base="ETH", quote="USD", region=Region.GLOBAL,        desc="Ethereum / US Dollar",       tags=["crypto", "usd", "risk_on"]),
]


class InstrumentRegistry:
    """Central registry for all tradeable instruments."""

    def __init__(self) -> None:
        self._instruments: Dict[str, InstrumentInfo] = {}

    @classmethod
    def default(cls) -> "InstrumentRegistry":
        reg = cls()
        for d in _DEFAULT_INSTRUMENTS:
            reg.register(InstrumentInfo(
                symbol=d["symbol"],
                asset_class=d["asset_class"],
                base_currency=d["base"],
                quote_currency=d["quote"],
                region=d["region"],
                description=d["desc"],
                tags=d["tags"],
            ))
        return reg

    def register(self, info: InstrumentInfo) -> None:
        self._instruments[info.symbol] = info

    def get(self, symbol: str) -> Optional[InstrumentInfo]:
        return self._instruments.get(symbol)

    def get_required(self, symbol: str) -> InstrumentInfo:
        info = self._instruments.get(symbol)
        if info is None:
            raise KeyError(f"Instrument {symbol!r} not registered.")
        return info

    def all_symbols(self) -> List[str]:
        return sorted(self._instruments.keys())

    def by_asset_class(self, asset_class: AssetClass) -> List[InstrumentInfo]:
        return [i for i in self._instruments.values() if i.asset_class == asset_class]

    def by_region(self, region: Region) -> List[InstrumentInfo]:
        return [i for i in self._instruments.values() if i.region == region]

    def by_tag(self, tag: str) -> List[InstrumentInfo]:
        return [i for i in self._instruments.values() if tag in i.tags]

    def by_currency(self, currency: str) -> List[InstrumentInfo]:
        return [
            i for i in self._instruments.values()
            if i.base_currency == currency or i.quote_currency == currency
        ]

    def total(self) -> int:
        return len(self._instruments)

    def asset_class_summary(self) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for info in self._instruments.values():
            key = info.asset_class.value
            summary[key] = summary.get(key, 0) + 1
        return summary

    def to_dict(self) -> Dict[str, dict]:
        return {sym: info.to_dict() for sym, info in self._instruments.items()}
