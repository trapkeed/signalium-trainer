"""Prediction heads for QuantModelV1 — Phase 14.6 Institutional Analytics.

Phase 14.5 heads (preserved, unchanged):
  ClassificationHead, ConfidenceHead, ExpectedMoveHead, StopLossHead,
  TakeProfitHead, HoldingTimeHead, TPProbabilityHead

Phase 14.6 — Institutional Trade Analytics heads (new):
  MFEHead                 — Maximum Favorable Excursion
  MAEExcursionHead        — Maximum Adverse Excursion
  TradeEfficiencyHead     — % of available move captured (0–100)
  RewardRiskHead          — Expected reward-to-risk ratio
  TradeSurvivalHead       — Survival probabilities at N bar horizons
  SLResilienceHead        — P(stop loss survives noise) ∈ [0, 1]
  TPAchievabilityHead     — P(TP achieved before invalidation) ∈ [0, 1]
  EntryQualityHead        — Entry quality score 0–100
  ExitQualityHead         — Exit quality score 0–100
  MarketStabilityHead     — 4-class stability classifier (logits)
  TradeDecayHead          — Decay probability + multi-step decay curve
  ProfitDistributionHead  — Mean, median, variance, skewness, CI (batch, 6)
  LossDistributionHead    — Mean, median, variance, tail risk (batch, 4)
  PositionSizeQualityHead — Capital allocation quality score 0–100
  UncertaintyHead         — Epistemic + aleatoric uncertainty (batch, 2)

Every head shares the same context vector from the backbone.  Zero
separate feature encoders; zero additional backbone passes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from trainer.framework.abstractions.head import (
    BaseClassificationHead,
    BaseConfidenceHead,
    BaseRegressionHead,
)
from trainer.models.quant_v1.config import (
    ClassificationHeadConfig,
    ConfidenceHeadConfig,
    DistributionHeadConfig,
    ExcursionHeadConfig,
    InstitutionalHeadConfig,
    MarketStabilityHeadConfig,
    RegressionHeadConfig,
    TPProbabilityHeadConfig,
    TradeDecayHeadConfig,
    TradeSurvivalHeadConfig,
    UncertaintyHeadConfig,
)


# ---------------------------------------------------------------------------
# Shared internal base classes
# ---------------------------------------------------------------------------


class _RegressionHead(BaseRegressionHead):
    """Scalar regression head shared by Phase 14.5 and 14.6 heads.

    Architecture:
        Linear(context_dim → hidden_dim) → GELU → Dropout
        → Linear(hidden_dim → 1) [→ Softplus if nonnegative]
    """

    def __init__(
        self,
        context_dim: int,
        hidden_dim: int,
        dropout: float,
        nonnegative: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        ]
        if nonnegative:
            layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class _ScaledScoreHead(nn.Module):
    """Head that produces a quality score in [0, 100] via 100 × Sigmoid.

    Architecture:
        Linear(context_dim → hidden_dim) → GELU → Dropout
        → Linear(hidden_dim → 1) → Sigmoid → ×100
    """

    def __init__(self, context_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context) * 100.0


class _BinaryProbHead(nn.Module):
    """Binary probability head — Sigmoid output ∈ [0, 1].

    Architecture:
        Linear(context_dim → hidden_dim) → GELU → Dropout
        → Linear(hidden_dim → 1) → Sigmoid
    """

    def __init__(self, context_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class _MultiOutputHead(nn.Module):
    """Multi-output head producing a vector of predictions per sample.

    Architecture:
        Linear(context_dim → hidden_dim) → GELU → Dropout
        → Linear(hidden_dim → out_dim)

    Subclasses apply their own per-element activations on top.
    """

    def __init__(
        self,
        context_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


# ---------------------------------------------------------------------------
# Phase 14.5 — Original heads (unchanged — backward compatible)
# ---------------------------------------------------------------------------


class ClassificationHead(BaseClassificationHead):
    """Maps the context vector to direction class logits.

    Architecture:
        Linear(context_dim → hidden_dim) → LayerNorm → GELU → Dropout
        → Linear(hidden_dim → num_classes)
    """

    def __init__(self, context_dim: int, config: ClassificationHeadConfig) -> None:
        super().__init__()
        self._config = config
        self.net = nn.Sequential(
            nn.Linear(context_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class ConfidenceHead(BaseConfidenceHead):
    """Produces a scalar confidence score in [0, 1].

    Architecture:
        Linear(context_dim → hidden_dim) → GELU → Dropout
        → Linear(hidden_dim → 1) → Sigmoid
    """

    def __init__(self, context_dim: int, config: ConfidenceHeadConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class ExpectedMoveHead(_RegressionHead):
    """Predicts expected price movement (signed, unrestricted)."""

    def __init__(self, context_dim: int, config: RegressionHeadConfig) -> None:
        super().__init__(context_dim, config.hidden_dim, config.dropout, nonnegative=False)


class StopLossHead(_RegressionHead):
    """Predicts Stop Loss distance (non-negative via Softplus)."""

    def __init__(self, context_dim: int, config: RegressionHeadConfig) -> None:
        super().__init__(context_dim, config.hidden_dim, config.dropout, nonnegative=True)


class TakeProfitHead(_RegressionHead):
    """Predicts Take Profit distance (non-negative via Softplus)."""

    def __init__(self, context_dim: int, config: RegressionHeadConfig) -> None:
        super().__init__(context_dim, config.hidden_dim, config.dropout, nonnegative=True)


class HoldingTimeHead(_RegressionHead):
    """Predicts expected holding duration (non-negative via Softplus)."""

    def __init__(self, context_dim: int, config: RegressionHeadConfig) -> None:
        super().__init__(context_dim, config.hidden_dim, config.dropout, nonnegative=True)


class TPProbabilityHead(_BinaryProbHead):
    """P(Take Profit reached before Stop Loss) ∈ [0, 1]."""

    def __init__(self, context_dim: int, config: TPProbabilityHeadConfig) -> None:
        super().__init__(context_dim, config.hidden_dim, config.dropout)


# ---------------------------------------------------------------------------
# Phase 14.6 — Institutional Trade Analytics heads
# ---------------------------------------------------------------------------


class MFEHead(_RegressionHead):
    """Maximum Favorable Excursion — largest expected favorable movement.

    Output: scalar ≥ 0 (Softplus), in the configured unit.
    Returns (batch, 1).

    Args:
        context_dim: Backbone output dimensionality.
        config: ExcursionHeadConfig with unit, hidden_dim, dropout.
    """

    def __init__(self, context_dim: int, config: ExcursionHeadConfig) -> None:
        super().__init__(context_dim, config.hidden_dim, config.dropout, nonnegative=True)
        self._config = config


class MAEExcursionHead(_RegressionHead):
    """Maximum Adverse Excursion — largest expected adverse movement.

    Output: scalar ≥ 0 (Softplus), in the configured unit.
    Returns (batch, 1).

    Args:
        context_dim: Backbone output dimensionality.
        config: ExcursionHeadConfig with unit, hidden_dim, dropout.
    """

    def __init__(self, context_dim: int, config: ExcursionHeadConfig) -> None:
        super().__init__(context_dim, config.hidden_dim, config.dropout, nonnegative=True)
        self._config = config


class TradeEfficiencyHead(_ScaledScoreHead):
    """Expected percentage of available move likely to be captured.

    Output: scalar ∈ [0, 100] via 100 × Sigmoid.
    Returns (batch, 1).

    Args:
        context_dim: Backbone output dimensionality.
        config: InstitutionalHeadConfig.
    """

    def __init__(self, context_dim: int, config: InstitutionalHeadConfig) -> None:
        super().__init__(context_dim, config.hidden_dim, config.dropout)


class RewardRiskHead(_RegressionHead):
    """Expected reward-to-risk ratio.

    Output: scalar ≥ 0 (Softplus) — e.g. 1.5, 2.0, 2.8, 3.6.
    Returns (batch, 1).

    Args:
        context_dim: Backbone output dimensionality.
        config: InstitutionalHeadConfig.
    """

    def __init__(self, context_dim: int, config: InstitutionalHeadConfig) -> None:
        super().__init__(context_dim, config.hidden_dim, config.dropout, nonnegative=True)


class TradeSurvivalHead(_MultiOutputHead):
    """Probability that the trade remains valid after N bars.

    Outputs one Sigmoid probability per horizon in config.survival_bars.
    Default: [5, 10, 20] → shape (batch, 3).

    Each output is independently Sigmoid-activated to produce a
    calibrated probability in [0, 1].

    Args:
        context_dim: Backbone output dimensionality.
        config: TradeSurvivalHeadConfig with survival_bars list.
    """

    def __init__(self, context_dim: int, config: TradeSurvivalHeadConfig) -> None:
        self._survival_bars = config.survival_bars
        out_dim = len(config.survival_bars)
        super().__init__(context_dim, config.hidden_dim, out_dim, config.dropout)
        self._sigmoid = nn.Sigmoid()

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Returns (batch, len(survival_bars)) probabilities, all in [0, 1]."""
        return self._sigmoid(self.net(context))

    @property
    def survival_bars(self) -> list[int]:
        return self._survival_bars


class SLResilienceHead(_BinaryProbHead):
    """P(stop loss survives normal market noise) ∈ [0, 1].

    Output: (batch, 1) Sigmoid probability.

    Args:
        context_dim: Backbone output dimensionality.
        config: InstitutionalHeadConfig.
    """

    def __init__(self, context_dim: int, config: InstitutionalHeadConfig) -> None:
        super().__init__(context_dim, config.hidden_dim, config.dropout)


class TPAchievabilityHead(_BinaryProbHead):
    """P(TP achieved before market conditions invalidate the setup) ∈ [0, 1].

    Output: (batch, 1) Sigmoid probability.

    Args:
        context_dim: Backbone output dimensionality.
        config: InstitutionalHeadConfig.
    """

    def __init__(self, context_dim: int, config: InstitutionalHeadConfig) -> None:
        super().__init__(context_dim, config.hidden_dim, config.dropout)


class EntryQualityHead(_ScaledScoreHead):
    """Entry quality score in [0, 100].

    Output: (batch, 1) via 100 × Sigmoid.
    Higher = better entry conditions.

    Args:
        context_dim: Backbone output dimensionality.
        config: InstitutionalHeadConfig.
    """

    def __init__(self, context_dim: int, config: InstitutionalHeadConfig) -> None:
        super().__init__(context_dim, config.hidden_dim, config.dropout)


class ExitQualityHead(_ScaledScoreHead):
    """Exit quality score in [0, 100].

    Output: (batch, 1) via 100 × Sigmoid.
    Higher = better exit conditions.

    Args:
        context_dim: Backbone output dimensionality.
        config: InstitutionalHeadConfig.
    """

    def __init__(self, context_dim: int, config: InstitutionalHeadConfig) -> None:
        super().__init__(context_dim, config.hidden_dim, config.dropout)


class MarketStabilityHead(_MultiOutputHead):
    """4-class market stability classifier.

    Classes:  0 = Stable, 1 = Moderately Stable, 2 = Unstable, 3 = Highly Unstable
    Output: logits of shape (batch, 4).  Apply Softmax for probabilities.
    Loss: CrossEntropyLoss (takes raw logits).

    Args:
        context_dim: Backbone output dimensionality.
        config: MarketStabilityHeadConfig.
    """

    def __init__(self, context_dim: int, config: MarketStabilityHeadConfig) -> None:
        super().__init__(context_dim, config.hidden_dim, config.num_classes, config.dropout)
        self._label_names = config.label_names
        self._softmax = nn.Softmax(dim=-1)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (batch, num_classes) for CE loss computation."""
        return self.net(context)

    def predict_proba(self, context: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities (batch, num_classes) for inference."""
        return self._softmax(self.net(context))

    @property
    def label_names(self) -> list[str]:
        return self._label_names


class TradeDecayHead(_MultiOutputHead):
    """Predicts how quickly setup quality deteriorates over time.

    Output shape: (batch, decay_steps + 1)
        - dim 0:   decay probability (overall) ∈ [0, 1]
        - dims 1…N: decay curve probabilities at each future step ∈ [0, 1]

    All outputs are Sigmoid-activated probabilities.
    Default: decay_steps=5 → output shape (batch, 6).

    Args:
        context_dim: Backbone output dimensionality.
        config: TradeDecayHeadConfig with decay_steps.
    """

    def __init__(self, context_dim: int, config: TradeDecayHeadConfig) -> None:
        self._decay_steps = config.decay_steps
        out_dim = config.decay_steps + 1  # [decay_prob, t1, t2, …, tN]
        super().__init__(context_dim, config.hidden_dim, out_dim, config.dropout)
        self._sigmoid = nn.Sigmoid()

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Returns (batch, decay_steps + 1) — all values ∈ [0, 1]."""
        return self._sigmoid(self.net(context))

    @property
    def decay_steps(self) -> int:
        return self._decay_steps


class ProfitDistributionHead(_MultiOutputHead):
    """Statistical profile of the expected profit distribution.

    Output shape: (batch, 6)
        - dim 0: mean          (unrestricted — signed)
        - dim 1: median        (unrestricted — signed)
        - dim 2: variance      (≥ 0, Softplus)
        - dim 3: skewness      (unrestricted)
        - dim 4: ci_lower      (unrestricted)
        - dim 5: ci_upper      (unrestricted)

    Softplus is applied selectively to the variance output only.

    Args:
        context_dim: Backbone output dimensionality.
        config: DistributionHeadConfig.
    """

    MEAN_IDX = 0
    MEDIAN_IDX = 1
    VAR_IDX = 2
    SKEW_IDX = 3
    CI_LOWER_IDX = 4
    CI_UPPER_IDX = 5
    OUT_DIM = 6

    def __init__(self, context_dim: int, config: DistributionHeadConfig) -> None:
        super().__init__(context_dim, config.hidden_dim, self.OUT_DIM, config.dropout)
        self._softplus = nn.Softplus()

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 6) distribution statistics."""
        raw = self.net(context)
        out = raw.clone()
        out[:, self.VAR_IDX] = self._softplus(raw[:, self.VAR_IDX])
        return out


class LossDistributionHead(_MultiOutputHead):
    """Statistical profile of the expected loss (adverse) distribution.

    Output shape: (batch, 4)
        - dim 0: mean          (unrestricted — typically negative for losses)
        - dim 1: median        (unrestricted)
        - dim 2: variance      (≥ 0, Softplus)
        - dim 3: tail_risk     (≥ 0, Softplus — expected downside beyond VaR)

    Args:
        context_dim: Backbone output dimensionality.
        config: DistributionHeadConfig.
    """

    MEAN_IDX = 0
    MEDIAN_IDX = 1
    VAR_IDX = 2
    TAIL_IDX = 3
    OUT_DIM = 4

    def __init__(self, context_dim: int, config: DistributionHeadConfig) -> None:
        super().__init__(context_dim, config.hidden_dim, self.OUT_DIM, config.dropout)
        self._softplus = nn.Softplus()

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 4) distribution statistics."""
        raw = self.net(context)
        out = raw.clone()
        out[:, self.VAR_IDX] = self._softplus(raw[:, self.VAR_IDX])
        out[:, self.TAIL_IDX] = self._softplus(raw[:, self.TAIL_IDX])
        return out


class PositionSizeQualityHead(_ScaledScoreHead):
    """Normalized capital allocation quality score in [0, 100].

    IMPORTANT: This head does NOT recommend lot size.  It produces a
    dimensionless score representing the quality of allocating additional
    capital to this opportunity.  The execution platform combines this
    score with account balance, risk percentage, broker constraints,
    margin requirements, portfolio exposure, and correlation limits to
    determine the actual position size.

    Output: (batch, 1) via 100 × Sigmoid.

    Args:
        context_dim: Backbone output dimensionality.
        config: InstitutionalHeadConfig.
    """

    def __init__(self, context_dim: int, config: InstitutionalHeadConfig) -> None:
        super().__init__(context_dim, config.hidden_dim, config.dropout)


class UncertaintyHead(_MultiOutputHead):
    """Simultaneous epistemic and aleatoric uncertainty estimation.

    Output shape: (batch, 2)
        - dim 0: epistemic uncertainty (≥ 0) — model/knowledge uncertainty
        - dim 1: aleatoric uncertainty (≥ 0) — inherent data noise

    Both outputs are Softplus-activated to guarantee non-negativity.
    Higher values indicate greater uncertainty in the respective component.

    Epistemic uncertainty decreases with more training data.
    Aleatoric uncertainty is irreducible noise inherent to market data.

    Args:
        context_dim: Backbone output dimensionality.
        config: UncertaintyHeadConfig.
    """

    EPISTEMIC_IDX = 0
    ALEATORIC_IDX = 1
    OUT_DIM = 2

    def __init__(self, context_dim: int, config: UncertaintyHeadConfig) -> None:
        super().__init__(context_dim, config.hidden_dim, self.OUT_DIM, config.dropout)
        self._softplus = nn.Softplus()

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 2) — [epistemic, aleatoric], both ≥ 0."""
        return self._softplus(self.net(context))
