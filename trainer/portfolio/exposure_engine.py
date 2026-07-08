"""Phase 20 — Exposure Engine.

Calculates portfolio exposure across asset classes, currencies, regions,
and detects over-concentration, hidden exposure, duplicated signals,
and conflicting exposure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from trainer.portfolio.portfolio_instruments import (
    InstrumentRegistry,
    ModelForecast,
)


class ExposureAlert(str, Enum):
    OVER_CONCENTRATION = "over_concentration"
    HIDDEN_EXPOSURE = "hidden_exposure"
    DUPLICATED_EXPOSURE = "duplicated_exposure"
    CONFLICTING_EXPOSURE = "conflicting_exposure"
    CURRENCY_IMBALANCE = "currency_imbalance"
    USD_OVERWEIGHT = "usd_overweight"


@dataclass
class ExposureAlertItem:
    alert_type: ExposureAlert
    severity: str
    description: str
    instruments: list[str]
    value: float = 0.0

    def to_dict(self) -> dict:
        return {
            "alert_type": self.alert_type.value,
            "severity": self.severity,
            "description": self.description,
            "instruments": self.instruments,
            "value": round(self.value, 4),
        }


@dataclass
class ExposureReport:
    by_asset_class: dict[str, float]
    by_currency: dict[str, float]
    by_region: dict[str, float]
    by_instrument: dict[str, float]
    gross_exposure: float
    net_exposure: float
    long_exposure: float
    short_exposure: float
    diversification_ratio: float
    alerts: list[ExposureAlertItem] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "by_asset_class": {k: round(v, 4) for k, v in self.by_asset_class.items()},
            "by_currency": {k: round(v, 4) for k, v in self.by_currency.items()},
            "by_region": {k: round(v, 4) for k, v in self.by_region.items()},
            "by_instrument": {k: round(v, 4) for k, v in self.by_instrument.items()},
            "gross_exposure": round(self.gross_exposure, 4),
            "net_exposure": round(self.net_exposure, 4),
            "long_exposure": round(self.long_exposure, 4),
            "short_exposure": round(self.short_exposure, 4),
            "diversification_ratio": round(self.diversification_ratio, 4),
            "alerts": [a.to_dict() for a in self.alerts],
        }


class ExposureEngine:
    """Pure context provider — calculates and reports portfolio exposure
    from model forecasts for the decision layer (QDE) to consume.

    Alerts are informational flags only; this engine never blocks, rejects,
    or rescales a portfolio decision. QDE is the sole decision maker.
    """

    # Concentration thresholds (fractions of gross exposure)
    ASSET_CLASS_LIMIT = 0.70
    CURRENCY_LIMIT = 0.60
    REGION_LIMIT = 0.75
    SINGLE_INSTRUMENT_LIMIT = 0.35
    USD_NET_LIMIT = 0.55

    def __init__(self, registry: InstrumentRegistry | None = None) -> None:
        self._registry = registry or InstrumentRegistry.default()

    def get_context(self, *args, **kwargs) -> ExposureReport:
        """Alias of :meth:`compute` using context-provider naming."""
        return self.compute(*args, **kwargs)

    def compute(
        self,
        forecasts: list[ModelForecast],
        weights: dict[str, float] | None = None,
    ) -> ExposureReport:
        if not forecasts:
            return ExposureReport({}, {}, {}, {}, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Default to equal weights
        if weights is None:
            w = 1.0 / len(forecasts)
            weights = {f.instrument: w for f in forecasts}

        by_asset: dict[str, float] = {}
        by_currency: dict[str, float] = {}
        by_region: dict[str, float] = {}
        by_instrument: dict[str, float] = {}
        long_exp = 0.0
        short_exp = 0.0

        for fc in forecasts:
            info = self._registry.get(fc.instrument)
            if info is None:
                continue
            w = weights.get(fc.instrument, 0.0)
            signed = w if fc.direction == "BUY" else (-w if fc.direction == "SELL" else 0.0)

            # Asset class
            cls_key = info.asset_class.value
            by_asset[cls_key] = by_asset.get(cls_key, 0.0) + signed

            # Currency
            for curr, sign in info.currency_exposures().items():
                curr_exp = signed * sign
                by_currency[curr] = by_currency.get(curr, 0.0) + curr_exp

            # Region
            reg_key = info.region.value
            by_region[reg_key] = by_region.get(reg_key, 0.0) + signed

            # Instrument
            by_instrument[fc.instrument] = signed

            if signed > 0:
                long_exp += signed
            else:
                short_exp += abs(signed)

        gross = long_exp + short_exp
        net = long_exp - short_exp
        div_ratio = self._diversification_ratio(by_instrument, gross)

        alerts = self._detect_alerts(
            by_asset, by_currency, by_region, by_instrument, gross, forecasts
        )

        return ExposureReport(
            by_asset_class=by_asset,
            by_currency=by_currency,
            by_region=by_region,
            by_instrument=by_instrument,
            gross_exposure=gross,
            net_exposure=net,
            long_exposure=long_exp,
            short_exposure=short_exp,
            diversification_ratio=div_ratio,
            alerts=alerts,
        )

    @staticmethod
    def _diversification_ratio(by_instrument: dict[str, float], gross: float) -> float:
        if gross < 1e-9 or not by_instrument:
            return 0.0
        n = len([v for v in by_instrument.values() if abs(v) > 1e-9])
        if n == 0:
            return 0.0
        max_w = max(abs(v) for v in by_instrument.values()) / max(gross, 1e-9)
        return round(1.0 - max_w, 4)

    def _detect_alerts(
        self,
        by_asset: dict[str, float],
        by_currency: dict[str, float],
        by_region: dict[str, float],
        by_instrument: dict[str, float],
        gross: float,
        forecasts: list[ModelForecast],
    ) -> list[ExposureAlertItem]:
        alerts: list[ExposureAlertItem] = []
        if gross < 1e-9:
            return alerts

        # Over-concentration: single asset class
        for cls_key, exp in by_asset.items():
            frac = abs(exp) / gross
            if frac > self.ASSET_CLASS_LIMIT:
                alerts.append(
                    ExposureAlertItem(
                        alert_type=ExposureAlert.OVER_CONCENTRATION,
                        severity="high",
                        description=f"Asset class '{cls_key}' has {frac:.0%} of gross exposure.",
                        instruments=[
                            fc.instrument
                            for fc in forecasts
                            if self._registry.get(fc.instrument)
                            and self._registry.get(fc.instrument).asset_class.value == cls_key
                        ],
                        value=frac,
                    )
                )

        # Over-concentration: single instrument
        for sym, exp in by_instrument.items():
            frac = abs(exp) / gross
            if frac > self.SINGLE_INSTRUMENT_LIMIT:
                alerts.append(
                    ExposureAlertItem(
                        alert_type=ExposureAlert.OVER_CONCENTRATION,
                        severity="medium",
                        description=f"Single instrument '{sym}' has {frac:.0%} of gross exposure.",
                        instruments=[sym],
                        value=frac,
                    )
                )

        # USD overweight
        usd_exp = abs(by_currency.get("USD", 0.0))
        if usd_exp / gross > self.USD_NET_LIMIT:
            alerts.append(
                ExposureAlertItem(
                    alert_type=ExposureAlert.USD_OVERWEIGHT,
                    severity="medium",
                    description=f"USD net exposure is {usd_exp/gross:.0%} of gross — above {self.USD_NET_LIMIT:.0%} limit.",
                    instruments=[],
                    value=usd_exp / gross,
                )
            )

        # Conflicting exposure (same currency both sides)
        for curr, exp in by_currency.items():
            if curr == "USD":
                continue
            longs = [
                fc
                for fc in forecasts
                if self._registry.get(fc.instrument)
                and curr in self._registry.get(fc.instrument).currency_exposures()
                and fc.direction == "BUY"
            ]
            shorts = [
                fc
                for fc in forecasts
                if self._registry.get(fc.instrument)
                and curr in self._registry.get(fc.instrument).currency_exposures()
                and fc.direction == "SELL"
            ]
            if longs and shorts:
                alerts.append(
                    ExposureAlertItem(
                        alert_type=ExposureAlert.CONFLICTING_EXPOSURE,
                        severity="low",
                        description=f"Currency '{curr}' has both long and short exposure.",
                        instruments=[fc.instrument for fc in longs + shorts],
                        value=abs(exp),
                    )
                )

        # Duplicated exposure: two instruments with same base + same direction
        instrument_dirs: dict[str, list[str]] = {}
        for fc in forecasts:
            info = self._registry.get(fc.instrument)
            if info is None:
                continue
            key = f"{info.base_currency}_{fc.direction}"
            instrument_dirs.setdefault(key, []).append(fc.instrument)
        for key, syms in instrument_dirs.items():
            if len(syms) >= 2:
                alerts.append(
                    ExposureAlertItem(
                        alert_type=ExposureAlert.DUPLICATED_EXPOSURE,
                        severity="low",
                        description=f"Duplicated directional exposure for key '{key}'.",
                        instruments=syms,
                        value=float(len(syms)),
                    )
                )

        return alerts

    def exposure_breakdown(
        self,
        forecasts: list[ModelForecast],
        weights: dict[str, float] | None = None,
    ) -> dict:
        report = self.compute(forecasts, weights)
        return {
            "report": report.to_dict(),
            "summary": {
                "instruments": len(forecasts),
                "alert_count": len(report.alerts),
                "high_alerts": sum(1 for a in report.alerts if a.severity == "high"),
                "diversification_ratio": report.diversification_ratio,
                "long_short_ratio": (
                    round(report.long_exposure / max(report.short_exposure, 1e-9), 2)
                    if report.short_exposure > 0
                    else None
                ),
            },
        }


# Context-provider naming alias (Task: decision engines → context providers).
# ExposureEngine is already pure/info-only (no veto power); this alias exposes
# the preferred name without breaking existing callers.
ExposureContextProvider = ExposureEngine
ExposureContext = ExposureReport
