"""ModelStatus — enumerated lifecycle states for the Champion/Challenger framework."""

from __future__ import annotations

from enum import Enum


class ModelStatus(str, Enum):
    """Lifecycle status of a model in the Champion/Challenger promotion framework.

    Transition diagram::

        EXPERIMENTAL ──► CANDIDATE ──► CHALLENGER ──► CHAMPION
                                                          │
                                                       RETIRED

    Rules:
        - Only one model per ``model_name`` may hold ``CHAMPION`` status.
        - ``RETIRED`` is a terminal state; retired models cannot be promoted.
        - ``EXPERIMENTAL`` models are excluded from the leaderboard by default.
        - ``CANDIDATE`` models have passed initial quality gates.
        - ``CHALLENGER`` models are actively competing with the current champion.
        - Promotion decisions are manual — no automatic promotion logic exists.
    """

    EXPERIMENTAL = "experimental"
    """New model under active development; not yet evaluated on the full dataset."""

    CANDIDATE = "candidate"
    """Has passed initial quality gates; eligible for benchmark comparison."""

    CHALLENGER = "challenger"
    """Actively competing with the current champion; A/B testing may be in progress."""

    CHAMPION = "champion"
    """Current production model — the best-performing model for a given name."""

    RETIRED = "retired"
    """Superseded or deprecated; kept for audit purposes only."""

    @property
    def is_active(self) -> bool:
        """True if the model is in an active (non-retired) state."""
        return self != ModelStatus.RETIRED

    @property
    def is_production_eligible(self) -> bool:
        """True if the model is eligible to serve production traffic."""
        return self in (ModelStatus.CHALLENGER, ModelStatus.CHAMPION)

    @property
    def rank(self) -> int:
        """Numeric rank for sorting (higher = more advanced in the lifecycle)."""
        return {
            ModelStatus.EXPERIMENTAL: 1,
            ModelStatus.CANDIDATE: 2,
            ModelStatus.CHALLENGER: 3,
            ModelStatus.CHAMPION: 4,
            ModelStatus.RETIRED: 0,
        }[self]
