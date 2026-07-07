"""Phase 15 — Promotion Policy.

Configurable rule-based promotion engine for the Institutional Validation Platform.

A model may only be promoted to production (Champion status) if it passes
ALL configured threshold rules.  Promotion can be:
  - Automatic — engine grants promotion when all rules pass
  - Manual — engine flags eligibility, human must approve

Configurable rules:
  min_precision, min_recall, min_profit_factor, max_drawdown,
  min_sharpe, max_calibration_error, min_trade_count,
  min_confidence_quality, min_win_rate, min_accuracy, min_f1,
  min_validation_folds, max_ece, min_sortino

Usage::

    policy = PromotionPolicy(
        min_accuracy=0.55,
        min_sharpe=1.0,
        max_drawdown=0.20,
        min_profit_factor=1.5,
        require_manual_approval=False,
    )
    decision = policy.evaluate(
        model_id="quant_v1",
        metrics={"accuracy": 0.58, "sharpe_ratio": 1.2, ...},
        n_trades=150,
        n_folds=10,
    )
    if decision.is_eligible:
        print(decision.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Promotion rule
# ---------------------------------------------------------------------------


@dataclass
class PromotionRule:
    """A single configurable threshold rule.

    Attributes:
        metric: Metric name to check (must match key in metrics dict).
        threshold: Required threshold value.
        is_max: When True, metric must be <= threshold (e.g. drawdown).
        label: Human-readable rule name.
        critical: When True, failure blocks promotion (all rules are critical
            by default; non-critical rules generate warnings only).
    """

    metric: str
    threshold: float
    is_max: bool = False
    label: str = ""
    critical: bool = True

    def evaluate(self, value: float) -> bool:
        """Return True when the rule passes."""
        if self.is_max:
            return value <= self.threshold
        return value >= self.threshold

    def failure_message(self, value: float) -> str:
        op = "<=" if self.is_max else ">="
        name = self.label or self.metric
        return f"{name} = {value:.4f} (required {op} {self.threshold:.4f})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "threshold": self.threshold,
            "is_max": self.is_max,
            "label": self.label or self.metric,
            "critical": self.critical,
        }


# ---------------------------------------------------------------------------
# Promotion decision
# ---------------------------------------------------------------------------


@dataclass
class PromotionDecision:
    """Result of a promotion policy evaluation.

    Attributes:
        model_id: Model that was evaluated.
        is_eligible: True when all critical rules pass.
        is_promoted: True when eligible AND (automatic OR manually approved).
        failed_rules: List of failed rule descriptions.
        passed_rules: List of passed rule descriptions.
        warnings: Non-critical rule failures.
        requires_manual_approval: Whether human sign-off is needed.
        decided_at: ISO-8601 UTC timestamp.
        notes: Optional notes added at decision time.
    """

    model_id: str
    is_eligible: bool = False
    is_promoted: bool = False
    failed_rules: list[str] = field(default_factory=list)
    passed_rules: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    requires_manual_approval: bool = True
    decided_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    notes: str = ""

    def summary(self) -> str:
        status = "ELIGIBLE" if self.is_eligible else "NOT ELIGIBLE"
        promoted = "PROMOTED" if self.is_promoted else "NOT PROMOTED"
        lines = [
            f"=== Promotion Decision: {self.model_id} ===",
            f"  Status        : {status}",
            f"  Promotion     : {promoted}",
            f"  Decided At    : {self.decided_at}",
            "",
            f"  Passed Rules  : {len(self.passed_rules)}/{len(self.passed_rules) + len(self.failed_rules)}",
        ]
        for r in self.passed_rules:
            lines.append(f"    [PASS] {r}")
        if self.failed_rules:
            lines.append(f"  Failed Rules  : {len(self.failed_rules)}")
            for r in self.failed_rules:
                lines.append(f"    [FAIL] {r}")
        if self.warnings:
            lines.append(f"  Warnings      : {len(self.warnings)}")
            for w in self.warnings:
                lines.append(f"    [WARN] {w}")
        if self.notes:
            lines.append(f"  Notes         : {self.notes}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "is_eligible": self.is_eligible,
            "is_promoted": self.is_promoted,
            "failed_rules": self.failed_rules,
            "passed_rules": self.passed_rules,
            "warnings": self.warnings,
            "requires_manual_approval": self.requires_manual_approval,
            "decided_at": self.decided_at,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Promotion Policy
# ---------------------------------------------------------------------------


class PromotionPolicy:
    """Configurable rule-based model promotion policy.

    All thresholds are optional — only configured rules are checked.
    If no rules are configured, the policy is permissive (always passes).

    Args:
        min_accuracy: Minimum validation accuracy.
        min_precision: Minimum macro precision.
        min_recall: Minimum macro recall.
        min_f1: Minimum macro F1.
        min_profit_factor: Minimum profit factor.
        max_drawdown: Maximum allowed drawdown (e.g. 0.20 = 20%).
        min_sharpe: Minimum Sharpe ratio.
        min_sortino: Minimum Sortino ratio.
        max_calibration_error: Maximum ECE (lower is better).
        max_ece: Alias for max_calibration_error.
        min_trade_count: Minimum number of trades evaluated.
        min_confidence_quality: Minimum fraction of predictions above 0.60 confidence.
        min_win_rate: Minimum win rate.
        min_validation_folds: Minimum number of walk-forward folds.
        require_manual_approval: When True, ``is_promoted`` is always False;
            human must call ``.approve()``.
        custom_rules: Additional ``PromotionRule`` objects.
    """

    def __init__(
        self,
        min_accuracy: float | None = None,
        min_precision: float | None = None,
        min_recall: float | None = None,
        min_f1: float | None = None,
        min_profit_factor: float | None = None,
        max_drawdown: float | None = None,
        min_sharpe: float | None = None,
        min_sortino: float | None = None,
        max_calibration_error: float | None = None,
        max_ece: float | None = None,
        min_trade_count: int | None = None,
        min_confidence_quality: float | None = None,
        min_win_rate: float | None = None,
        min_validation_folds: int | None = None,
        require_manual_approval: bool = True,
        custom_rules: list[PromotionRule] | None = None,
    ) -> None:
        self._rules: list[PromotionRule] = []
        self._manual = require_manual_approval

        def _add(metric, threshold, is_max=False, label=""):
            if threshold is not None:
                self._rules.append(
                    PromotionRule(
                        metric=metric,
                        threshold=float(threshold),
                        is_max=is_max,
                        label=label,
                    )
                )

        _add("accuracy", min_accuracy, label="Minimum Accuracy")
        _add("precision_macro", min_precision, label="Minimum Precision")
        _add("recall_macro", min_recall, label="Minimum Recall")
        _add("f1_macro", min_f1, label="Minimum F1")
        _add("profit_factor", min_profit_factor, label="Minimum Profit Factor")
        _add("max_drawdown", max_drawdown, is_max=True, label="Maximum Drawdown")
        _add("sharpe_ratio", min_sharpe, label="Minimum Sharpe")
        _add("sortino_ratio", min_sortino, label="Minimum Sortino")
        _add("ece", max_ece or max_calibration_error, is_max=True, label="Maximum ECE")
        _add("n_trades", min_trade_count, label="Minimum Trade Count")
        _add("confidence_quality", min_confidence_quality, label="Minimum Confidence Quality")
        _add("win_rate", min_win_rate, label="Minimum Win Rate")
        _add("n_folds", min_validation_folds, label="Minimum Validation Folds")

        for rule in custom_rules or []:
            self._rules.append(rule)

    def add_rule(self, rule: PromotionRule) -> None:
        """Add a custom rule to the policy."""
        self._rules.append(rule)

    def evaluate(
        self,
        model_id: str,
        metrics: dict[str, float],
        n_trades: int = 0,
        n_folds: int = 0,
        notes: str = "",
    ) -> PromotionDecision:
        """Evaluate whether a model passes all promotion rules.

        Args:
            model_id: Model identifier for the decision record.
            metrics: Dict of metric name → value. Expected keys include any
                metric used in the configured rules.
            n_trades: Number of simulated trades evaluated.
            n_folds: Number of walk-forward folds completed.
            notes: Optional notes to attach to the decision.

        Returns:
            ``PromotionDecision``.
        """
        # Augment metrics with explicit counts
        full_metrics = dict(metrics)
        full_metrics["n_trades"] = float(n_trades)
        full_metrics["n_folds"] = float(n_folds)

        passed: list[str] = []
        failed: list[str] = []
        warnings: list[str] = []

        for rule in self._rules:
            val = full_metrics.get(rule.metric, 0.0)
            ok = rule.evaluate(float(val))
            msg = (
                rule.failure_message(float(val))
                if not ok
                else (f"{rule.label or rule.metric} = {val:.4f}")
            )
            if ok:
                passed.append(msg)
            elif rule.critical:
                failed.append(msg)
            else:
                warnings.append(msg)

        is_eligible = len(failed) == 0
        is_promoted = is_eligible and not self._manual

        return PromotionDecision(
            model_id=model_id,
            is_eligible=is_eligible,
            is_promoted=is_promoted,
            failed_rules=failed,
            passed_rules=passed,
            warnings=warnings,
            requires_manual_approval=self._manual,
            notes=notes,
        )

    def approve(self, decision: PromotionDecision, approver: str = "") -> PromotionDecision:
        """Grant manual approval to an eligible decision.

        Args:
            decision: A previously computed ``PromotionDecision``.
            approver: Name / ID of the approving researcher (for audit trail).

        Returns:
            Updated ``PromotionDecision`` with ``is_promoted=True``.

        Raises:
            ValueError: When the model is not eligible.
        """
        if not decision.is_eligible:
            raise ValueError(
                f"Cannot approve model {decision.model_id!r} — "
                f"it did not pass all required rules:\n"
                + "\n".join(f"  - {r}" for r in decision.failed_rules)
            )
        decision.is_promoted = True
        if approver:
            decision.notes = (decision.notes or "") + f" | Approved by: {approver}"
        return decision

    def describe(self) -> list[dict[str, Any]]:
        """Return all configured rules as a list of dicts."""
        return [r.to_dict() for r in self._rules]

    @property
    def n_rules(self) -> int:
        return len(self._rules)

    @property
    def requires_manual_approval(self) -> bool:
        return self._manual
