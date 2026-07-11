"""QDE Brain — decision memory bridge (Phase C6).

Persists ``CognitiveDecision`` objects to:

  1. **Episodic memory (C2)** — soft import; when available, each decision
     is stored as an ``Episode`` so future hypothesis generation and memory
     recall can learn from past decision contexts.

  2. **Local in-process deque** — always available; acts as a fallback
     and provides fast ``recent_decisions()`` access without an episodic
     index scan.

  3. **Working memory snapshot** — a compact entry is added to
     ``BrainState.working_memory`` via ``WorkingMemory.add()`` so the
     decision is immediately visible in the brain's short-term context.

Design constraints
------------------
* All C2 imports are soft (``try/except ImportError``).
* No ``QDEDecision`` imports.
* ``to_snapshot()`` / ``load_snapshot()`` dict persistence pattern.
* The bridge stores decisions, it does not produce them.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.qde.brain.decision.decision_types import CognitiveDecision
from trainer.qde.brain.working_memory import WorkingMemory

# C2 soft imports
try:
    from trainer.qde.brain.episodic.episodic_memory import (
        Episode,
        EpisodeFeatures,
        EpisodeOutcome,
        EpisodePortfolioState,
    )

    _HAS_EPISODIC = True
except ImportError:
    _HAS_EPISODIC = False

MAX_LOCAL_DECISIONS = 500


# -------------------------------------------------------------------
# Feature encoding
# -------------------------------------------------------------------


def _decision_to_features(decision: CognitiveDecision) -> dict:
    """Encode a ``CognitiveDecision`` into the 12-dim feature dict
    consumed by ``EpisodeFeatures.from_dict()``.
    """
    action_numeric = {
        "BUY": 1.0,
        "SELL": 0.0,
        "WAIT": 0.5,
        "MONITOR": 0.5,
        "IGNORE": 0.5,
    }
    dir_numeric = {
        "BULLISH": 1.0,
        "BEARISH": 0.0,
        "NEUTRAL": 0.5,
    }
    return {
        "volatility": decision.uncertainty_score,
        "momentum": dir_numeric.get(decision.direction.value, 0.5) * 2.0 - 1.0,
        "trend_strength": decision.conviction_score,
        "range_compression": decision.risk_signal,
        "absorption": decision.entry_urgency,
        "volume_balance": decision.size_signal,
        "session_quality": decision.calibrated_confidence,
        "microstructure": 1.0 - decision.uncertainty_score,
        "regime_numeric": action_numeric.get(decision.action.value, 0.5),
        "hypothesis_conf": decision.confidence,
        "attention_weight": 1.0 if decision.overconfidence_flagged else 0.0,
        "spread_norm": 1.0 if decision.ignored_evidence_flagged else 0.0,
    }


# -------------------------------------------------------------------
# Bridge
# -------------------------------------------------------------------


class DecisionMemoryBridge:
    """Persist ``CognitiveDecision`` objects to episodic memory and the
    local deque.

    Usage::

        bridge = DecisionMemoryBridge(episodic_memory=emem)
        episode_id = bridge.store(decision, brain_state=state, now=now)

        recent = bridge.recent_decisions("EURUSD", n=10)
    """

    def __init__(
        self,
        episodic_memory: object | None = None,
        max_local: int = MAX_LOCAL_DECISIONS,
    ) -> None:
        self._episodic = episodic_memory
        self._max_local = max_local
        self._local: dict[str, deque[CognitiveDecision]] = defaultdict(
            lambda: deque(maxlen=self._max_local)
        )
        self._wm = WorkingMemory()

    # ── Primary API ───────────────────────────────────────────────────

    def store(
        self,
        decision: CognitiveDecision,
        brain_state: object | None = None,
        now: float | None = None,
    ) -> str:
        """Persist the decision to all available stores.

        Returns the episode_id if episodic memory is available, or
        ``decision.decision_id`` otherwise.
        """
        now = now if now is not None else time.time()
        self._local[decision.instrument].append(decision)

        # Log to working memory if brain_state available
        if brain_state is not None:
            try:
                self._wm.add(
                    items=brain_state.working_memory,
                    instrument=decision.instrument,
                    kind="COGNITIVE_DECISION",
                    content=decision.to_dict(),
                    salience=0.80 if decision.is_actionable else 0.45,
                    now=now,
                )
            except Exception as exc:
                logger.debug(f"[DecisionMemoryBridge] WM log failed: {exc}")

        if not _HAS_EPISODIC or self._episodic is None:
            return decision.decision_id

        return self._store_episodic(decision, now)

    # ── Query ─────────────────────────────────────────────────────────

    def recent_decisions(
        self,
        instrument: str,
        n: int = 20,
        action: str | None = None,
    ) -> list[CognitiveDecision]:
        """Most-recent decisions for ``instrument``, newest first.

        ``action`` filters to a specific ``CognitiveAction`` value
        (e.g. "BUY", "WAIT").
        """
        all_dec = list(self._local.get(instrument, []))
        all_dec.reverse()
        if action:
            all_dec = [d for d in all_dec if d.action.value == action.upper()]
        return all_dec[:n]

    def decision_count(self, instrument: str) -> int:
        return len(self._local.get(instrument, []))

    def action_distribution(self, instrument: str) -> dict[str, int]:
        """Count of decisions by action type for ``instrument``."""
        counts: dict[str, int] = {}
        for d in self._local.get(instrument, []):
            counts[d.action.value] = counts.get(d.action.value, 0) + 1
        return counts

    # ── Persistence ───────────────────────────────────────────────────

    def to_snapshot(self) -> dict:
        return {
            "decisions": {instr: [d.to_dict() for d in deq] for instr, deq in self._local.items()}
        }

    def load_snapshot(self, data: dict) -> None:
        self._local.clear()
        for instr, decs in data.get("decisions", {}).items():
            for d in decs:
                try:
                    dec = CognitiveDecision.from_dict(d)
                    self._local[instr].append(dec)
                except Exception:
                    pass

    # ── Internal ──────────────────────────────────────────────────────

    def _store_episodic(self, decision: CognitiveDecision, now: float) -> str:
        try:
            feat = EpisodeFeatures.from_dict(_decision_to_features(decision))  # type: ignore[name-defined]
            outcome = EpisodeOutcome(  # type: ignore[name-defined]
                direction_correct=decision.is_actionable,
                reward_atr=decision.size_signal * decision.risk_signal,
                resolved=False,
            )
            ep = Episode(  # type: ignore[name-defined]
                instrument=decision.instrument,
                timestamp=now,
                features=feat,
                portfolio=EpisodePortfolioState(),  # type: ignore[name-defined]
                action=decision.action.value,
                confidence=decision.calibrated_confidence,
                outcome=outcome,
                trade_result_atr=0.0,
                reasoning=(
                    f"cognitive:{decision.action.value}:"
                    f"{decision.direction.value}:"
                    f"conv={decision.conviction_score:.2f}:"
                    f"unc={decision.uncertainty_score:.2f}"
                ),
            )
            self._episodic.write_memory(ep, now=now)  # type: ignore[union-attr]
            logger.debug(
                f"[DecisionMemoryBridge] stored episode {ep.episode_id} "
                f"for decision {decision.decision_id} ({decision.action.value}) "
                f"on {decision.instrument}"
            )
            return ep.episode_id
        except Exception as exc:
            logger.warning(f"[DecisionMemoryBridge] episodic store failed: {exc}")
            return decision.decision_id


__all__ = ["DecisionMemoryBridge"]
