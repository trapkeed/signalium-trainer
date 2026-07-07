"""Phase 20 — Conflict Resolver.

Detects conflicting forecasts across instruments and recommends
appropriate actions: KEEP, REDUCE, IGNORE, DELAY.

Examples of conflicts:
  EURUSD BUY + USDCHF BUY → USD both sides (conflict)
  EURUSD BUY + GBPUSD BUY → USD short both (reinforcing)
  NAS100 BUY  + BTCUSD BUY → risk-on reinforcing
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from trainer.portfolio.portfolio_instruments import (
    InstrumentInfo,
    InstrumentRegistry,
    ModelForecast,
)


class ConflictType(str, Enum):
    DIRECT_CURRENCY = "direct_currency"
    INDIRECT_USD = "indirect_usd"
    SAFE_HAVEN_RISK = "safe_haven_risk"
    COMMODITY_CURRENCY = "commodity_currency"
    CORRELATED_OPPOSITE = "correlated_opposite"


class Resolution(str, Enum):
    KEEP = "keep"
    REDUCE = "reduce"
    IGNORE = "ignore"
    DELAY = "delay"


@dataclass
class ConflictDetail:
    instrument_a: str
    instrument_b: str
    conflict_type: ConflictType
    description: str
    severity: float
    resolution: Resolution
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "instrument_a": self.instrument_a,
            "instrument_b": self.instrument_b,
            "conflict_type": self.conflict_type.value,
            "description": self.description,
            "severity": round(self.severity, 4),
            "resolution": self.resolution.value,
            "reasoning": self.reasoning,
        }


@dataclass
class ReinforcementDetail:
    instrument_a: str
    instrument_b: str
    description: str
    strength: float

    def to_dict(self) -> dict:
        return {
            "instrument_a": self.instrument_a,
            "instrument_b": self.instrument_b,
            "description": self.description,
            "strength": round(self.strength, 4),
        }


@dataclass
class ConflictReport:
    conflicts: list[ConflictDetail]
    reinforcements: list[ReinforcementDetail]
    conflicted_instruments: set[str]
    clean_forecasts: list[ModelForecast]
    adjusted_forecasts: list[ModelForecast]

    def conflict_count(self) -> int:
        return len(self.conflicts)

    def reinforcement_count(self) -> int:
        return len(self.reinforcements)

    def has_conflicts(self) -> bool:
        return bool(self.conflicts)

    def to_dict(self) -> dict:
        return {
            "conflict_count": self.conflict_count(),
            "reinforcement_count": self.reinforcement_count(),
            "conflicted_instruments": sorted(self.conflicted_instruments),
            "conflicts": [c.to_dict() for c in self.conflicts],
            "reinforcements": [r.to_dict() for r in self.reinforcements],
            "clean_forecast_count": len(self.clean_forecasts),
            "adjusted_forecast_count": len(self.adjusted_forecasts),
        }


# Known conflict patterns (currency exposure overlap)
_DIRECT_CONFLICTS: list[tuple[str, str, str]] = [
    # (A, B, description) — these pairs have opposing implicit USD direction
    ("EURUSD", "USDCHF", "EUR/USD long = USD short; USD/CHF long = USD long → USD conflict"),
    ("EURUSD", "USDJPY", "EUR/USD long = USD short; USD/JPY long = USD long → USD conflict"),
    ("EURUSD", "USDCAD", "EUR/USD long = USD short; USD/CAD long = USD long → USD conflict"),
    ("GBPUSD", "USDCHF", "GBP/USD long = USD short; USD/CHF long = USD long → USD conflict"),
    ("GBPUSD", "USDJPY", "GBP/USD long = USD short; USD/JPY long = USD long → USD conflict"),
    ("GBPUSD", "USDCAD", "GBP/USD long = USD short; USD/CAD long = USD long → USD conflict"),
    ("AUDUSD", "USDCHF", "AUD/USD long = USD short; USD/CHF long = USD long → USD conflict"),
    ("AUDUSD", "USDJPY", "AUD/USD long = USD short; USD/JPY long = USD long → USD conflict"),
    ("NZDUSD", "USDCHF", "NZD/USD long = USD short; USD/CHF long = USD long → USD conflict"),
    ("NZDUSD", "USDJPY", "NZD/USD long = USD short; USD/JPY long = USD long → USD conflict"),
]

_SAFE_HAVEN_RISK_PAIRS: list[tuple[str, str]] = [
    ("XAUUSD", "NAS100"),
    ("XAUUSD", "SPX500"),
    ("XAUUSD", "BTCUSD"),
    ("USDCHF", "AUDUSD"),
    ("USDCHF", "NZDUSD"),
    ("USDJPY", "AUDUSD"),
    ("USDJPY", "NZDUSD"),
]

_RISK_ON_INSTRUMENTS: set[str] = {
    "NAS100",
    "US30",
    "SPX500",
    "BTCUSD",
    "ETHUSD",
    "AUDUSD",
    "NZDUSD",
}
_SAFE_HAVEN_INSTRUMENTS: set[str] = {"XAUUSD", "USDCHF", "USDJPY"}


def _usd_direction(fc: ModelForecast, info: InstrumentInfo) -> float:
    """Returns +1 if forecast is USD-bullish, -1 if USD-bearish, 0 if neutral."""
    if fc.direction == "NEUTRAL":
        return 0.0
    if info.base_currency == "USD":
        return 1.0 if fc.direction == "BUY" else -1.0
    if info.quote_currency == "USD":
        return -1.0 if fc.direction == "BUY" else 1.0
    return 0.0


class ConflictResolver:
    """Detects signal conflicts and recommends resolutions."""

    def __init__(self, registry: InstrumentRegistry | None = None) -> None:
        self._registry = registry or InstrumentRegistry.default()

    def resolve(self, forecasts: list[ModelForecast]) -> ConflictReport:
        fc_map: dict[str, ModelForecast] = {f.instrument: f for f in forecasts}
        conflicts: list[ConflictDetail] = []
        reinforcements: list[ReinforcementDetail] = []
        ignore_set: set[str] = set()
        reduce_set: set[str] = set()

        # Direct currency conflicts
        for a, b, desc in _DIRECT_CONFLICTS:
            if a not in fc_map or b not in fc_map:
                continue
            fc_a = fc_map[a]
            fc_b = fc_map[b]
            if fc_a.direction == "NEUTRAL" or fc_b.direction == "NEUTRAL":
                continue
            info_a = self._registry.get(a)
            info_b = self._registry.get(b)
            if info_a is None or info_b is None:
                continue
            usd_a = _usd_direction(fc_a, info_a)
            usd_b = _usd_direction(fc_b, info_b)
            if usd_a != 0 and usd_b != 0 and usd_a != usd_b:
                severity = (fc_a.confidence + fc_b.confidence) / 2
                resolution = self._resolve_priority(fc_a, fc_b)
                conflicts.append(
                    ConflictDetail(
                        instrument_a=a,
                        instrument_b=b,
                        conflict_type=ConflictType.DIRECT_CURRENCY,
                        description=desc,
                        severity=severity,
                        resolution=resolution,
                        reasoning=self._resolution_reasoning(resolution, fc_a, fc_b),
                    )
                )
                if resolution == Resolution.IGNORE:
                    ignore_set.add(b if fc_b.confidence < fc_a.confidence else a)
                elif resolution == Resolution.REDUCE:
                    reduce_set.add(b if fc_b.confidence < fc_a.confidence else a)

        # Safe haven vs risk-on conflicts
        for safe, risk in _SAFE_HAVEN_RISK_PAIRS:
            if safe not in fc_map or risk not in fc_map:
                continue
            fc_safe = fc_map[safe]
            fc_risk = fc_map[risk]
            if fc_safe.direction == "NEUTRAL" or fc_risk.direction == "NEUTRAL":
                continue
            safe_bullish = fc_safe.direction == "BUY"
            risk_bullish = fc_risk.direction == "BUY"
            if safe_bullish and risk_bullish:
                severity = (fc_safe.confidence + fc_risk.confidence) / 2 * 0.7
                conflicts.append(
                    ConflictDetail(
                        instrument_a=safe,
                        instrument_b=risk,
                        conflict_type=ConflictType.SAFE_HAVEN_RISK,
                        description=f"Safe-haven {safe} BUY conflicts with risk-on {risk} BUY",
                        severity=severity,
                        resolution=Resolution.REDUCE,
                        reasoning="Safe-haven and risk-on assets rarely both rally simultaneously; reduce smaller position.",
                    )
                )
                reduce_set.add(safe if fc_safe.confidence < fc_risk.confidence else risk)
            elif safe_bullish and not risk_bullish:
                reinforcements.append(
                    ReinforcementDetail(
                        instrument_a=safe,
                        instrument_b=risk,
                        description=f"Safe-haven {safe} BUY + {risk} SELL reinforce risk-off bias",
                        strength=(fc_safe.confidence + fc_risk.confidence) / 2,
                    )
                )
            elif not safe_bullish and risk_bullish:
                reinforcements.append(
                    ReinforcementDetail(
                        instrument_a=safe,
                        instrument_b=risk,
                        description=f"Safe-haven {safe} SELL + {risk} BUY reinforce risk-on bias",
                        strength=(fc_safe.confidence + fc_risk.confidence) / 2,
                    )
                )

        # Risk-on cluster — all same direction = reinforcing
        risk_on_in_forecasts = [(fc_map[s], s) for s in _RISK_ON_INSTRUMENTS if s in fc_map]
        if len(risk_on_in_forecasts) >= 2:
            dirs = [fc.direction for fc, _ in risk_on_in_forecasts]
            if all(d == dirs[0] for d in dirs) and dirs[0] != "NEUTRAL":
                strength = sum(fc.confidence for fc, _ in risk_on_in_forecasts) / len(
                    risk_on_in_forecasts
                )
                reinforcements.append(
                    ReinforcementDetail(
                        instrument_a=risk_on_in_forecasts[0][1],
                        instrument_b=risk_on_in_forecasts[-1][1],
                        description=f"All {len(risk_on_in_forecasts)} risk-on instruments agree ({dirs[0]}) — strong reinforcement",
                        strength=strength,
                    )
                )

        conflicted = {c.instrument_a for c in conflicts} | {c.instrument_b for c in conflicts}
        clean = [f for f in forecasts if f.instrument not in ignore_set]
        adjusted = self._apply_reduce(clean, reduce_set)

        return ConflictReport(
            conflicts=conflicts,
            reinforcements=reinforcements,
            conflicted_instruments=conflicted,
            clean_forecasts=clean,
            adjusted_forecasts=adjusted,
        )

    @staticmethod
    def _resolve_priority(fc_a: ModelForecast, fc_b: ModelForecast) -> Resolution:
        conf_diff = abs(fc_a.confidence - fc_b.confidence)
        if conf_diff > 0.3:
            return Resolution.IGNORE
        if conf_diff > 0.1:
            return Resolution.REDUCE
        if max(fc_a.confidence, fc_b.confidence) < 0.5:
            return Resolution.DELAY
        return Resolution.REDUCE

    @staticmethod
    def _resolution_reasoning(res: Resolution, a: ModelForecast, b: ModelForecast) -> str:
        if res == Resolution.IGNORE:
            loser = b.instrument if b.confidence < a.confidence else a.instrument
            return f"Confidence gap >30%: ignore {loser} signal."
        if res == Resolution.REDUCE:
            return "Similar confidence: reduce both positions by 30-50%."
        if res == Resolution.DELAY:
            return "Both low confidence (<50%): delay until clarity improves."
        return "Keep all positions; monitor correlation risk."

    @staticmethod
    def _apply_reduce(forecasts: list[ModelForecast], reduce_set: set[str]) -> list[ModelForecast]:
        result = []
        for fc in forecasts:
            if fc.instrument in reduce_set:
                result.append(
                    ModelForecast(
                        instrument=fc.instrument,
                        direction=fc.direction,
                        confidence=fc.confidence * 0.60,
                        expected_return=fc.expected_return * 0.60,
                        expected_risk=fc.expected_risk,
                        timestamp=fc.timestamp,
                        metadata={**fc.metadata, "conflict_reduced": True},
                    )
                )
            else:
                result.append(fc)
        return result
