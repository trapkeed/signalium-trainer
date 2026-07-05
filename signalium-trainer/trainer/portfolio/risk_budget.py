"""Phase 20 — Risk Budget Engine.

Defines and enforces configurable risk budgets across asset classes,
currencies, and individual exposures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from trainer.portfolio.portfolio_instruments import AssetClass, ModelForecast


class BudgetStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    BREACH = "breach"


@dataclass
class BudgetItem:
    name: str
    limit: float
    current: float
    unit: str = "fraction"

    @property
    def utilisation(self) -> float:
        return self.current / max(self.limit, 1e-12)

    @property
    def status(self) -> BudgetStatus:
        if self.utilisation >= 1.0:
            return BudgetStatus.BREACH
        if self.utilisation >= 0.85:
            return BudgetStatus.WARNING
        return BudgetStatus.OK

    @property
    def available(self) -> float:
        return max(0.0, self.limit - self.current)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "limit": round(self.limit, 4),
            "current": round(self.current, 4),
            "available": round(self.available, 4),
            "utilisation": round(self.utilisation, 4),
            "status": self.status.value,
            "unit": self.unit,
        }


@dataclass
class RiskBudgetConfig:
    max_usd_exposure: float = 0.60
    max_gold_exposure: float = 0.25
    max_crypto_exposure: float = 0.20
    max_index_exposure: float = 0.35
    max_single_instrument: float = 0.30
    max_correlated_exposure: float = 0.50
    max_total_gross: float = 1.50
    max_forex_exposure: float = 0.70
    max_commodity_exposure: float = 0.30
    max_region_exposure: float = 0.65
    max_long_only: float = 0.80
    max_short_only: float = 0.80

    @classmethod
    def conservative(cls) -> "RiskBudgetConfig":
        return cls(
            max_usd_exposure=0.50,
            max_gold_exposure=0.20,
            max_crypto_exposure=0.10,
            max_index_exposure=0.25,
            max_single_instrument=0.20,
            max_correlated_exposure=0.40,
            max_total_gross=1.20,
            max_forex_exposure=0.60,
            max_commodity_exposure=0.20,
            max_region_exposure=0.55,
        )

    @classmethod
    def aggressive(cls) -> "RiskBudgetConfig":
        return cls(
            max_usd_exposure=0.80,
            max_gold_exposure=0.40,
            max_crypto_exposure=0.35,
            max_index_exposure=0.50,
            max_single_instrument=0.45,
            max_correlated_exposure=0.70,
            max_total_gross=2.00,
            max_forex_exposure=0.90,
            max_commodity_exposure=0.45,
            max_region_exposure=0.80,
        )

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class BudgetCheckResult:
    items: List[BudgetItem]
    overall_status: BudgetStatus
    breaches: List[BudgetItem]
    warnings: List[BudgetItem]

    def is_feasible(self) -> bool:
        return self.overall_status != BudgetStatus.BREACH

    def to_dict(self) -> dict:
        return {
            "overall_status": self.overall_status.value,
            "is_feasible": self.is_feasible(),
            "breach_count": len(self.breaches),
            "warning_count": len(self.warnings),
            "items": [i.to_dict() for i in self.items],
            "breaches": [b.name for b in self.breaches],
            "warnings": [w.name for w in self.warnings],
        }


class RiskBudgetEngine:
    """Enforces configurable risk budgets on portfolio allocations."""

    def __init__(self, config: Optional[RiskBudgetConfig] = None) -> None:
        self.config = config or RiskBudgetConfig()

    def check(
        self,
        weights: Dict[str, float],
        forecasts: List[ModelForecast],
        by_currency: Optional[Dict[str, float]] = None,
        by_asset_class: Optional[Dict[str, float]] = None,
        gross_exposure: Optional[float] = None,
    ) -> BudgetCheckResult:
        items: List[BudgetItem] = []
        fc_map = {f.instrument: f for f in forecasts}

        gross = gross_exposure if gross_exposure is not None else sum(abs(w) for w in weights.values())

        # Total gross
        items.append(BudgetItem(
            name="max_total_gross",
            limit=self.config.max_total_gross,
            current=gross,
            unit="gross_exposure",
        ))

        # Per-instrument
        for sym, w in weights.items():
            if abs(w) > 0:
                items.append(BudgetItem(
                    name=f"instrument:{sym}",
                    limit=self.config.max_single_instrument,
                    current=abs(w) / max(gross, 1e-9),
                    unit="fraction_of_gross",
                ))

        # Currency exposures
        if by_currency:
            usd_exp = abs(by_currency.get("USD", 0.0))
            items.append(BudgetItem(
                name="max_usd_exposure",
                limit=self.config.max_usd_exposure,
                current=usd_exp / max(gross, 1e-9),
                unit="fraction_of_gross",
            ))
            gold_exp = abs(by_currency.get("XAU", 0.0))
            items.append(BudgetItem(
                name="max_gold_exposure",
                limit=self.config.max_gold_exposure,
                current=gold_exp / max(gross, 1e-9),
                unit="fraction_of_gross",
            ))

        # Asset class exposures
        if by_asset_class:
            crypto_exp = abs(by_asset_class.get(AssetClass.CRYPTO.value, 0.0))
            items.append(BudgetItem(
                name="max_crypto_exposure",
                limit=self.config.max_crypto_exposure,
                current=crypto_exp / max(gross, 1e-9),
                unit="fraction_of_gross",
            ))
            index_exp = abs(by_asset_class.get(AssetClass.INDEX.value, 0.0))
            items.append(BudgetItem(
                name="max_index_exposure",
                limit=self.config.max_index_exposure,
                current=index_exp / max(gross, 1e-9),
                unit="fraction_of_gross",
            ))
            forex_exp = abs(by_asset_class.get(AssetClass.FOREX.value, 0.0))
            items.append(BudgetItem(
                name="max_forex_exposure",
                limit=self.config.max_forex_exposure,
                current=forex_exp / max(gross, 1e-9),
                unit="fraction_of_gross",
            ))
            commodity_exp = abs(by_asset_class.get(AssetClass.COMMODITY.value, 0.0))
            items.append(BudgetItem(
                name="max_commodity_exposure",
                limit=self.config.max_commodity_exposure,
                current=commodity_exp / max(gross, 1e-9),
                unit="fraction_of_gross",
            ))

        breaches = [i for i in items if i.status == BudgetStatus.BREACH]
        warnings = [i for i in items if i.status == BudgetStatus.WARNING]
        overall = BudgetStatus.OK
        if breaches:
            overall = BudgetStatus.BREACH
        elif warnings:
            overall = BudgetStatus.WARNING

        return BudgetCheckResult(items=items, overall_status=overall,
                                  breaches=breaches, warnings=warnings)

    def scale_to_budget(
        self,
        weights: Dict[str, float],
        forecasts: List[ModelForecast],
    ) -> Dict[str, float]:
        gross = sum(abs(w) for w in weights.values())
        if gross < 1e-9:
            return dict(weights)

        max_allowed_gross = self.config.max_total_gross
        if gross > max_allowed_gross:
            scale = max_allowed_gross / gross
            weights = {k: v * scale for k, v in weights.items()}

        max_single = self.config.max_single_instrument
        new_gross = sum(abs(w) for w in weights.values())
        for sym, w in list(weights.items()):
            frac = abs(w) / max(new_gross, 1e-9)
            if frac > max_single:
                scale = max_single / frac
                weights[sym] = w * scale

        # Renormalize
        total = sum(abs(w) for w in weights.values())
        if total > 1e-9:
            scale = min(1.0, max_allowed_gross / total)
            weights = {k: v * scale for k, v in weights.items()}

        return weights

    def utilisation_report(
        self,
        weights: Dict[str, float],
        forecasts: List[ModelForecast],
        by_currency: Optional[Dict[str, float]] = None,
        by_asset_class: Optional[Dict[str, float]] = None,
    ) -> dict:
        result = self.check(weights, forecasts, by_currency, by_asset_class)
        return {
            "status": result.overall_status.value,
            "feasible": result.is_feasible(),
            "budget_config": self.config.to_dict(),
            "budget_check": result.to_dict(),
        }
