"""Phase 24 — Risk Budget Engine.

Supports 12 configurable risk budget types.
Tracks usage and raises alerts on breach.
"""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.icari.icari_models import RiskBudget
from trainer.icari.icari_types import DEFAULT_RISK_BUDGETS, RiskBudgetType


class RiskBudgetEngine:
    """Manages configurable risk budgets across all budget types."""

    def __init__(self, capital: float = 100_000.0):
        self._capital = capital
        self._budgets: Dict[str, RiskBudget] = {}
        self._load_defaults()

    def _load_defaults(self) -> None:
        for btype, limit in DEFAULT_RISK_BUDGETS.items():
            self._budgets[btype.value] = RiskBudget(
                budget_type=btype,
                limit=limit,
                used=0.0,
                label=f"Default {btype.value}",
            )

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def set_budget(
        self,
        budget_type: RiskBudgetType,
        limit: float,
        label: str = "",
    ) -> RiskBudget:
        """Set or update a risk budget limit."""
        budget = RiskBudget(
            budget_type=budget_type,
            limit=limit,
            used=self._budgets.get(budget_type.value, RiskBudget(budget_type, limit)).used,
            label=label or budget_type.value,
        )
        self._budgets[budget_type.value] = budget
        logger.debug(f"RiskBudget: set {budget_type.value} limit={limit:.4f}")
        return budget

    def get_budget(self, budget_type: RiskBudgetType) -> Optional[RiskBudget]:
        return self._budgets.get(budget_type.value)

    def all_budgets(self) -> List[RiskBudget]:
        return list(self._budgets.values())

    # ── Usage tracking ────────────────────────────────────────────────────────

    def consume(self, budget_type: RiskBudgetType, amount: float) -> Dict:
        """Record risk consumption against a budget."""
        budget = self._budgets.get(budget_type.value)
        if not budget:
            return {"error": f"Budget {budget_type.value} not found"}

        budget.used = budget.used + amount
        breached = budget.is_breached
        if breached:
            logger.warning(
                f"RiskBudget BREACH: {budget_type.value} used={budget.used:.4f} limit={budget.limit:.4f}"
            )

        return {
            "budget_type": budget_type.value,
            "consumed": round(amount, 4),
            "used": round(budget.used, 4),
            "limit": round(budget.limit, 4),
            "remaining": round(budget.remaining, 4),
            "breached": breached,
        }

    def release(self, budget_type: RiskBudgetType, amount: float) -> bool:
        """Release previously consumed budget (e.g. trade closed)."""
        budget = self._budgets.get(budget_type.value)
        if not budget:
            return False
        budget.used = max(0.0, budget.used - amount)
        return True

    def reset(self, budget_type: RiskBudgetType) -> bool:
        """Reset usage to zero for a budget (e.g. start of new period)."""
        budget = self._budgets.get(budget_type.value)
        if not budget:
            return False
        budget.used = 0.0
        return True

    def reset_all(self) -> None:
        for budget in self._budgets.values():
            budget.used = 0.0

    # ── Checks ────────────────────────────────────────────────────────────────

    def check(self, budget_type: RiskBudgetType, amount: float) -> Dict:
        """Check if consuming `amount` would breach the budget (dry run)."""
        budget = self._budgets.get(budget_type.value)
        if not budget:
            return {"allowed": False, "error": "Budget not found"}
        projected = budget.used + amount
        allowed = projected <= budget.limit
        return {
            "allowed": allowed,
            "current": round(budget.used, 4),
            "requested": round(amount, 4),
            "projected": round(projected, 4),
            "limit": round(budget.limit, 4),
            "remaining": round(budget.remaining, 4),
        }

    def breached_budgets(self) -> List[RiskBudget]:
        return [b for b in self._budgets.values() if b.is_breached]

    def near_limit_budgets(self, threshold: float = 0.80) -> List[RiskBudget]:
        return [
            b for b in self._budgets.values() if b.utilisation >= threshold and not b.is_breached
        ]

    # ── Capital-adjusted helpers ───────────────────────────────────────────────

    def limit_usd(self, budget_type: RiskBudgetType) -> float:
        b = self.get_budget(budget_type)
        return (b.limit * self._capital) if b else 0.0

    def used_usd(self, budget_type: RiskBudgetType) -> float:
        b = self.get_budget(budget_type)
        return (b.used * self._capital) if b else 0.0

    # ── Stats ─────────────────────────────────────────────────────────────────

    def summary(self) -> Dict:
        budgets = self.all_budgets()
        return {
            "total_budgets": len(budgets),
            "breached": len(self.breached_budgets()),
            "near_limit": len(self.near_limit_budgets()),
            "all": [b.to_dict() for b in budgets],
        }
