"""Phase 18 — Champion / Challenger Management.

Maintains:
  - Champion Model (current production champion)
  - Candidate Models (challenger queue)
  - Historical Champions (audit trail)
  - Retired Champions

Supports:
  - Champion / challenger comparison
  - Promotion
  - Rollback
  - Champion audit trail

Usage::

    cc = ChampionChallengerManager(registry, lifecycle, governance)
    cc.set_champion("model_01", approver="head_quant")
    cc.add_challenger("model_02", reason="new backbone")
    comparison = cc.compare("model_01", "model_02")
    cc.promote_challenger("model_02", approver="head_quant")
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from trainer.mlops.registry import LifecycleState, ModelRegistry
from trainer.mlops.lifecycle import LifecycleManager
from trainer.mlops.governance import DeploymentGovernance

# ---------------------------------------------------------------------------
# Champion event
# ---------------------------------------------------------------------------


@dataclass
class ChampionEvent:
    """One event in the champion history."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = "promotion"  # promotion | rollback | retirement | comparison
    champion_id: str = ""
    challenger_id: Optional[str] = None
    actor: str = "system"
    reason: str = ""
    outcome: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "champion_id": self.champion_id,
            "challenger_id": self.challenger_id,
            "actor": self.actor,
            "reason": self.reason,
            "outcome": self.outcome,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Champion/challenger manager
# ---------------------------------------------------------------------------


class ChampionChallengerManager:
    """Governs champion model and challenger queue.

    Args:
        registry: ModelRegistry for model metadata.
        lifecycle: LifecycleManager for state transitions.
        governance: DeploymentGovernance for deployment records.
    """

    def __init__(
        self, registry: ModelRegistry, lifecycle: LifecycleManager, governance: DeploymentGovernance
    ) -> None:
        self.registry = registry
        self.lifecycle = lifecycle
        self.governance = governance
        self._current_champion_id: Optional[str] = None
        self._challengers: List[str] = []
        self._historical_champions: List[str] = []
        self._retired_champions: List[str] = []
        self._events: List[ChampionEvent] = []

    def set_champion(
        self, model_id: str, approver: str = "system", reason: str = ""
    ) -> ChampionEvent:
        """Promote a model to champion status.

        Demotes the current champion to ARCHIVED before promoting.
        """
        model = self.registry.get_or_raise(model_id)
        old_champion_id = self._current_champion_id

        if old_champion_id and old_champion_id != model_id:
            old_champion = self.registry.get(old_champion_id)
            if old_champion and old_champion.is_champion:
                self.lifecycle.transition(
                    old_champion,
                    LifecycleState.ARCHIVED,
                    actor=approver,
                    reason=f"Superseded by {model.full_name}",
                )
                self._historical_champions.append(old_champion_id)

        self.governance.deploy(
            model_id,
            target="champion",
            approver=approver,
            actor=approver,
            reason=reason or "Promoted to champion",
            deploy_type="manual",
        )
        self._current_champion_id = model_id

        if model_id in self._challengers:
            self._challengers.remove(model_id)

        event = ChampionEvent(
            event_type="promotion",
            champion_id=model_id,
            actor=approver,
            reason=reason,
            outcome=f"{model.full_name} is now champion",
            metadata={"old_champion": old_champion_id},
        )
        self._events.append(event)
        logger.info(f"Champion: {model.full_name} (by {approver})")
        return event

    def rollback_champion(
        self, actor: str = "ops", reason: str = "rollback"
    ) -> Optional[ChampionEvent]:
        """Roll back the current champion to ROLLED_BACK state."""
        if not self._current_champion_id:
            return None
        model = self.registry.get(self._current_champion_id)
        if model is None:
            return None

        self.governance.rollback(
            self._current_champion_id,
            actor=actor,
            reason=reason,
            emergency=True,
        )
        event = ChampionEvent(
            event_type="rollback",
            champion_id=self._current_champion_id,
            actor=actor,
            reason=reason,
            outcome=f"{model.full_name} rolled back",
        )
        self._events.append(event)
        self._historical_champions.append(self._current_champion_id)
        self._current_champion_id = None

        if self._historical_champions:
            prev_id = (
                self._historical_champions[-2] if len(self._historical_champions) >= 2 else None
            )
            if prev_id:
                logger.warning(
                    f"Champion {model.full_name} rolled back. "
                    f"Previous champion was {prev_id}. "
                    f"Manual intervention required to promote a new champion."
                )
        return event

    def add_challenger(self, model_id: str, reason: str = "") -> None:
        """Add a model to the challenger queue."""
        model = self.registry.get_or_raise(model_id)
        if model_id not in self._challengers:
            self._challengers.append(model_id)
        if model.lifecycle_state not in (
            LifecycleState.APPROVED,
            LifecycleState.CANDIDATE,
            LifecycleState.SHADOW,
        ):
            self.lifecycle.transition(
                model,
                LifecycleState.SHADOW,
                actor="system",
                reason=f"Added to challenger queue: {reason}",
            )
        logger.info(f"Challenger added: {model.full_name}")

    def promote_challenger(
        self, model_id: str, approver: str = "head_quant", reason: str = ""
    ) -> ChampionEvent:
        """Promote a challenger to champion."""
        if model_id not in self._challengers:
            raise ValueError(f"Model {model_id!r} is not a challenger.")
        self._challengers.remove(model_id)
        return self.set_champion(model_id, approver=approver, reason=reason)

    def retire_champion(
        self, model_id: str, actor: str = "system", reason: str = ""
    ) -> ChampionEvent:
        """Retire a champion model."""
        model = self.registry.get_or_raise(model_id)
        self.lifecycle.retire(model, actor=actor, reason=reason)
        self._retired_champions.append(model_id)
        if self._current_champion_id == model_id:
            self._current_champion_id = None
        event = ChampionEvent(
            event_type="retirement",
            champion_id=model_id,
            actor=actor,
            reason=reason,
            outcome=f"{model.full_name} retired",
        )
        self._events.append(event)
        return event

    def compare(
        self,
        model_a_id: str,
        model_b_id: str,
        metrics_a: Optional[Dict[str, float]] = None,
        metrics_b: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Compare two models on registered validation metrics.

        Returns a comparison dict with winner per metric and overall winner.
        """
        m_a = self.registry.get_or_raise(model_a_id)
        m_b = self.registry.get_or_raise(model_b_id)

        a_val = metrics_a or m_a.validation_report
        b_val = metrics_b or m_b.validation_report

        comparison_keys = [
            "accuracy",
            "sharpe_ratio",
            "profit_factor",
            "win_rate",
            "calibration_error",
            "max_drawdown",
            "validation_score",
        ]
        results: Dict[str, Any] = {}
        a_wins = 0
        b_wins = 0

        for key in comparison_keys:
            a_v = a_val.get(key)
            b_v = b_val.get(key)
            if a_v is None and b_v is None:
                continue
            if a_v is None:
                winner = model_b_id
                b_wins += 1
            elif b_v is None:
                winner = model_a_id
                a_wins += 1
            else:
                if key == "calibration_error":
                    winner = model_a_id if a_v <= b_v else model_b_id
                    if a_v <= b_v:
                        a_wins += 1
                    else:
                        b_wins += 1
                elif key == "max_drawdown":
                    winner = model_a_id if a_v >= b_v else model_b_id
                    if a_v >= b_v:
                        a_wins += 1
                    else:
                        b_wins += 1
                else:
                    winner = model_a_id if a_v >= b_v else model_b_id
                    if a_v >= b_v:
                        a_wins += 1
                    else:
                        b_wins += 1
            results[key] = {"model_a": a_v, "model_b": b_v, "winner": winner}

        overall_winner = model_a_id if a_wins >= b_wins else model_b_id
        comparison = {
            "model_a": {"model_id": model_a_id, "name": m_a.full_name, "wins": a_wins},
            "model_b": {"model_id": model_b_id, "name": m_b.full_name, "wins": b_wins},
            "overall_winner": overall_winner,
            "overall_winner_name": (
                m_a.full_name if overall_winner == model_a_id else m_b.full_name
            ),
            "metrics": results,
        }
        event = ChampionEvent(
            event_type="comparison",
            champion_id=model_a_id,
            challenger_id=model_b_id,
            outcome=f"Winner: {overall_winner}",
            metadata=comparison,
        )
        self._events.append(event)
        return comparison

    @property
    def current_champion(self) -> Optional[Dict[str, Any]]:
        if not self._current_champion_id:
            return None
        m = self.registry.get(self._current_champion_id)
        return m.to_dict() if m else None

    @property
    def challengers(self) -> List[Dict[str, Any]]:
        result = []
        for mid in self._challengers:
            m = self.registry.get(mid)
            if m:
                result.append(m.to_dict())
        return result

    @property
    def historical_champions(self) -> List[Dict[str, Any]]:
        result = []
        for mid in self._historical_champions:
            m = self.registry.get(mid)
            if m:
                result.append(m.to_dict())
        return result

    @property
    def retired_champions(self) -> List[Dict[str, Any]]:
        result = []
        for mid in self._retired_champions:
            m = self.registry.get(mid)
            if m:
                result.append(m.to_dict())
        return result

    def event_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in reversed(self._events[-limit:])]

    def summary(self) -> Dict[str, Any]:
        return {
            "current_champion": self.current_champion,
            "n_challengers": len(self._challengers),
            "n_historical_champions": len(self._historical_champions),
            "n_retired_champions": len(self._retired_champions),
            "n_events": len(self._events),
        }
