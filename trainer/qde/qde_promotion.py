"""Phase 27.6 — QDE Promotion Engine.

A new QDE version may only be promoted if it improves across all five dimensions:
  1. Risk Adjusted Return   (Sharpe / Sortino)
  2. Drawdown               (max drawdown reduction)
  3. Consistency            (win-rate / profit-factor across folds)
  4. Capital Efficiency     (return per margin unit)
  5. Decision Quality       (entry/exit precision)
"""

from __future__ import annotations

from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.qde.qde_models import (
    DimensionResult,
    QDEPromotionCriteria,
    QDEPromotionDecision,
    QDEStage,
)
from trainer.qde.qde_registry import QDERegistry


class QDEPromotionEngine:
    """Evaluates whether a challenger QDE version should replace the incumbent.

    Promotion requires the challenger to improve on *all five* dimensions
    relative to the current active (champion) version, subject to the
    thresholds defined in ``QDEPromotionCriteria``.

    If there is no incumbent (first-ever deployment), absolute floor checks
    are applied instead.
    """

    def __init__(
        self,
        registry: QDERegistry,
        criteria: QDEPromotionCriteria | None = None,
        auto_promote: bool = False,
    ) -> None:
        self._registry = registry
        self._criteria = criteria or QDEPromotionCriteria()
        self._auto_promote = auto_promote
        self._history: list[QDEPromotionDecision] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        challenger_version_id: str,
        notes: str = "",
    ) -> QDEPromotionDecision:
        """Run a full promotion evaluation for a challenger version.

        The challenger must be in DEPLOYMENT_CANDIDATE stage.
        """
        challenger = self._registry.get(challenger_version_id)
        if challenger is None:
            raise KeyError(f"Challenger version {challenger_version_id} not found")
        if challenger.stage != QDEStage.DEPLOYMENT_CANDIDATE:
            raise ValueError(
                f"Challenger must be in DEPLOYMENT_CANDIDATE stage, "
                f"got {challenger.stage.value}"
            )

        incumbent = self._registry.get_active()
        incumbent_metrics = incumbent.metrics if incumbent else {}

        results = [
            self._check_risk_adjusted_return(challenger.metrics, incumbent_metrics),
            self._check_drawdown(challenger.metrics, incumbent_metrics),
            self._check_consistency(challenger.metrics, incumbent_metrics),
            self._check_capital_efficiency(challenger.metrics, incumbent_metrics),
            self._check_decision_quality(challenger.metrics, incumbent_metrics),
        ]

        decision = QDEPromotionDecision.create(
            challenger_version_id=challenger_version_id,
            incumbent_version_id=incumbent.version_id if incumbent else None,
            dimension_results=results,
            auto_promote=self._auto_promote,
            notes=notes,
        )
        self._history.append(decision)
        logger.info(
            f"QDEPromotion.evaluate: {challenger_version_id} → "
            f"{'ELIGIBLE' if decision.is_eligible else 'NOT ELIGIBLE'}"
        )
        return decision

    def approve(
        self,
        decision: QDEPromotionDecision,
        approver: str = "",
    ) -> QDEPromotionDecision:
        """Manually approve an eligible promotion decision and activate the version."""
        decision.approve(approver=approver)
        self._activate(decision.challenger_version_id)
        logger.info(
            f"QDEPromotion.approve: {decision.challenger_version_id} approved by {approver!r}"
        )
        return decision

    def auto_evaluate_and_promote(
        self,
        challenger_version_id: str,
        notes: str = "",
    ) -> QDEPromotionDecision:
        """Evaluate and immediately promote if eligible (for automated pipelines)."""
        decision = self.evaluate(challenger_version_id, notes=notes)
        if decision.is_eligible:
            self._activate(challenger_version_id)
            decision.is_promoted = True
        return decision

    def history(self) -> list[dict[str, Any]]:
        return [d.to_dict() for d in self._history]

    def criteria(self) -> dict[str, Any]:
        return self._criteria.to_dict()

    # ── Dimension Checks ──────────────────────────────────────────────────────

    def _check_risk_adjusted_return(
        self,
        challenger: dict[str, float],
        incumbent: dict[str, float],
    ) -> DimensionResult:
        c_sharpe = challenger.get("sharpe_ratio", 0.0)
        i_sharpe = incumbent.get("sharpe_ratio", None)
        c_sortino = challenger.get("sortino_ratio", 0.0)
        c_rar = max(c_sharpe, c_sortino)

        if i_sharpe is not None:
            i_rar = max(i_sharpe, incumbent.get("sortino_ratio", 0.0))
            delta = c_rar - i_rar
            threshold = self._criteria.min_risk_adjusted_return_delta
            passed = delta >= threshold and c_rar >= self._criteria.min_sharpe
        else:
            # No incumbent — apply absolute floor
            delta = c_rar
            threshold = self._criteria.min_sharpe
            passed = c_rar >= threshold
            i_sharpe = None

        return DimensionResult(
            dimension="Risk Adjusted Return",
            passed=passed,
            challenger_value=c_rar,
            incumbent_value=i_sharpe,
            delta=delta,
            threshold=threshold,
            notes=f"sharpe={c_sharpe:.4f}, sortino={c_sortino:.4f}",
        )

    def _check_drawdown(
        self,
        challenger: dict[str, float],
        incumbent: dict[str, float],
    ) -> DimensionResult:
        c_dd = challenger.get("max_drawdown", 1.0)
        i_dd = incumbent.get("max_drawdown", None)

        abs_passed = c_dd <= self._criteria.max_drawdown_abs

        if i_dd is not None and self._criteria.drawdown_must_improve:
            delta = i_dd - c_dd  # positive = improvement
            threshold = self._criteria.min_risk_adjusted_return_delta  # 0.0 = must not worsen
            passed = delta >= threshold and abs_passed
        else:
            delta = self._criteria.max_drawdown_allowed - c_dd
            threshold = 0.0
            passed = abs_passed

        return DimensionResult(
            dimension="Drawdown",
            passed=passed,
            challenger_value=c_dd,
            incumbent_value=i_dd,
            delta=delta,
            threshold=threshold,
            notes=f"max_drawdown={c_dd:.4f} (abs limit={self._criteria.max_drawdown_abs:.4f})",
        )

    def _check_consistency(
        self,
        challenger: dict[str, float],
        incumbent: dict[str, float],
    ) -> DimensionResult:
        c_wr = challenger.get("win_rate", 0.0)
        c_pf = challenger.get("profit_factor", 0.0)
        c_val = (c_wr + min(c_pf / 3.0, 1.0)) / 2.0  # normalised score

        i_wr = incumbent.get("win_rate", None)
        i_pf = incumbent.get("profit_factor", None)

        if i_wr is not None and i_pf is not None:
            i_val = (i_wr + min(i_pf / 3.0, 1.0)) / 2.0
            delta = c_val - i_val
            threshold = self._criteria.min_consistency_delta
            abs_ok = (
                c_wr >= self._criteria.min_win_rate and c_pf >= self._criteria.min_profit_factor
            )
            passed = delta >= threshold and abs_ok
        else:
            delta = c_val
            threshold = (
                self._criteria.min_win_rate + min(self._criteria.min_profit_factor / 3.0, 1.0)
            ) / 2.0
            passed = (
                c_wr >= self._criteria.min_win_rate and c_pf >= self._criteria.min_profit_factor
            )

        return DimensionResult(
            dimension="Consistency",
            passed=passed,
            challenger_value=c_val,
            incumbent_value=(i_wr + min(i_pf / 3.0, 1.0)) / 2.0 if i_wr is not None else None,
            delta=delta,
            threshold=threshold,
            notes=f"win_rate={c_wr:.4f}, profit_factor={c_pf:.4f}",
        )

    def _check_capital_efficiency(
        self,
        challenger: dict[str, float],
        incumbent: dict[str, float],
    ) -> DimensionResult:
        c_ce = challenger.get("capital_efficiency", challenger.get("return_on_margin", 0.0))
        i_ce = incumbent.get("capital_efficiency", incumbent.get("return_on_margin", None))

        if i_ce is not None:
            delta = c_ce - i_ce
            threshold = self._criteria.min_capital_efficiency_delta
            passed = delta >= threshold
        else:
            delta = c_ce
            threshold = 0.0
            passed = c_ce >= 0.0

        return DimensionResult(
            dimension="Capital Efficiency",
            passed=passed,
            challenger_value=c_ce,
            incumbent_value=i_ce,
            delta=delta,
            threshold=threshold,
            notes=f"capital_efficiency={c_ce:.4f}",
        )

    def _check_decision_quality(
        self,
        challenger: dict[str, float],
        incumbent: dict[str, float],
    ) -> DimensionResult:
        c_dq = challenger.get(
            "decision_quality",
            challenger.get("precision_macro", challenger.get("accuracy", 0.0)),
        )
        i_dq = incumbent.get(
            "decision_quality",
            incumbent.get("precision_macro", incumbent.get("accuracy", None)),
        )

        if i_dq is not None:
            delta = c_dq - i_dq
            threshold = self._criteria.min_decision_quality_delta
            passed = delta >= threshold
        else:
            delta = c_dq
            threshold = 0.0
            passed = c_dq > 0.0

        return DimensionResult(
            dimension="Decision Quality",
            passed=passed,
            challenger_value=c_dq,
            incumbent_value=i_dq,
            delta=delta,
            threshold=threshold,
            notes=f"decision_quality={c_dq:.4f}",
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _activate(self, version_id: str) -> None:
        self._registry.set_active(version_id)
        logger.info(f"QDEPromotion: activated {version_id}")
