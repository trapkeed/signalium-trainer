"""QDE Brain — internal state store (first layer).

This is a read/aggregation facade over the raw internal state that already
lives in `BrainState` (owned in-process by `QDEBrain`) and is persisted by
`QDEBrainDatabase` — it deliberately does **not** introduce a second copy of
that storage. Its job is to expose the specific first-layer views the QDE
brain must maintain per symbol:

  - recent structure        (`MarketNarrative.structure_notes`)
  - volatility evolution    (per-tick `OBSERVATION` items in working memory)
  - trend evolution         (per-tick `OBSERVATION` items in working memory)
  - liquidity observations  (per-tick `OBSERVATION` items in working memory)
  - session context         (`MarketNarrative.session_context`)
  - unfinished narratives   (non-terminal hypotheses + a non-neutral,
                             still-developing narrative)

Every per-tick `OBSERVATION` working-memory item already carries
`market_regime` / `volatility_state` / `liquidity_state` / `readiness_score`
(see `brain.QDEBrain.on_market_update`), so "evolution" here means reading
that existing history back out in chronological order — no new persistence
schema, no duplicated math.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from trainer.qde.brain.brain_types import is_terminal_hypothesis_status

if TYPE_CHECKING:
    from trainer.qde.brain.brain import QDEBrain
    from trainer.qde.brain.brain_models import BrainState

OBSERVATION_KIND = "OBSERVATION"


class InternalStateStore:
    """Typed, per-symbol view over `BrainState` for the Brain's first layer.

    Stateless itself (mirrors `WorkingMemory`/`LongTermMemory`) — it always
    reads through to whichever `QDEBrain` instance it wraps, so state stays
    single-sourced and persists exactly the way `QDEBrain`/`QDEBrainDatabase`
    already persist it.
    """

    def __init__(self, brain: "QDEBrain") -> None:
        self._brain = brain

    def _state(self, instrument: str) -> "BrainState":
        return self._brain.get_state(instrument)

    # ── Recent structure ─────────────────────────────────────────────────

    def recent_structure(self, instrument: str, n: int = 10) -> list[str]:
        """Most recent structural observations noted for this symbol."""
        state = self._state(instrument)
        if state.narrative is None:
            return []
        return list(state.narrative.structure_notes[-n:])

    # ── Volatility / trend / liquidity evolution ─────────────────────────

    def _observation_history(self, instrument: str, n: int) -> list[dict]:
        state = self._state(instrument)
        items = [i for i in state.working_memory if i.kind == OBSERVATION_KIND]
        items.sort(key=lambda i: i.created_at)
        return [i.content for i in items[-n:]]

    def volatility_evolution(self, instrument: str, n: int = 20) -> list[dict]:
        """Chronological sequence of recent volatility states/readiness for
        this symbol, e.g. [{"volatility_state": "HIGH", "readiness_score": 0.6}, ...].
        """
        return [
            {
                "volatility_state": obs.get("volatility_state"),
                "readiness_score": obs.get("readiness_score"),
            }
            for obs in self._observation_history(instrument, n)
        ]

    def trend_evolution(self, instrument: str, n: int = 20) -> list[dict]:
        """Chronological sequence of recent market-regime observations."""
        return [
            {
                "market_regime": obs.get("market_regime"),
                "observer_state": obs.get("observer_state"),
            }
            for obs in self._observation_history(instrument, n)
        ]

    def liquidity_observations(self, instrument: str, n: int = 20) -> list[dict]:
        """Chronological sequence of recent liquidity-state observations."""
        return [
            {"liquidity_state": obs.get("liquidity_state")}
            for obs in self._observation_history(instrument, n)
        ]

    # ── Session context ───────────────────────────────────────────────────

    def session_context(self, instrument: str) -> str:
        state = self._state(instrument)
        return state.narrative.session_context if state.narrative else ""

    # ── Unfinished narratives ─────────────────────────────────────────────

    def unfinished_narratives(self, instrument: str) -> list[dict]:
        """Hypotheses that haven't resolved yet, plus the current narrative
        if it's still telling a directional/developing story rather than a
        settled/neutral one.
        """
        state = self._state(instrument)
        open_items: list[dict] = [
            h.to_dict()
            for h in state.hypotheses.values()
            if not is_terminal_hypothesis_status(h.status)
        ]
        if state.narrative is not None:
            from trainer.qde.brain.brain_types import NarrativeTone

            if state.narrative.tone != NarrativeTone.NEUTRAL:
                open_items.append(
                    {
                        "narrative_id": state.narrative.narrative_id,
                        "summary": state.narrative.summary,
                        "tone": state.narrative.tone.value,
                    }
                )
        return open_items

    # ── Combined snapshot ─────────────────────────────────────────────────

    def snapshot(self, instrument: str, evolution_window: int = 20) -> dict:
        """The full first-layer view for one symbol in a single dict —
        the shape a dashboard/API endpoint would want to render.
        """
        return {
            "instrument": instrument,
            "recent_structure": self.recent_structure(instrument),
            "volatility_evolution": self.volatility_evolution(instrument, evolution_window),
            "trend_evolution": self.trend_evolution(instrument, evolution_window),
            "liquidity_observations": self.liquidity_observations(instrument, evolution_window),
            "session_context": self.session_context(instrument),
            "unfinished_narratives": self.unfinished_narratives(instrument),
        }
