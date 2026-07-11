"""QDE Brain — narrative store.

Defines the canonical `SymbolNarrative` data model (distinct from the
existing single-instance `MarketNarrative` summary in `brain_models.py`)
and the stateless `NarrativeStore` that operates over a per-symbol
``dict[str, SymbolNarrative]``.

Design contract
---------------
- `MarketNarrative` (brain_models) is QDE's *summary* narrative for a
  symbol — one coherent tone+regime+session description revised on every
  tick.  It is intentionally kept as-is; this module adds the *typed*
  narrative layer on top: up to one living `SymbolNarrative` per
  `NarrativeKind`, each carrying its own evidence list, confidence,
  age, regime/session context, and maturity.
- `NarrativeStore` follows the exact convention of `HypothesisTracker`
  and `OpportunityTracker`: storage lives on the caller's dict (normally
  owned by `MarketNarrativeEngine`), this class holds no state of its own.
- `SymbolNarrative.to_dict()` / `NarrativeStore.from_list_dict()` use the
  same ``to_dict()`` convention as the rest of brain_models so the caller
  can persist them via `QDEBrainDatabase` or any equivalent store.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

# ── Vocabulary ────────────────────────────────────────────────────────────


class NarrativeKind(str, Enum):
    """Named market narrative archetypes the QDE can recognise and track."""

    TREND_CONTINUATION = "TREND_CONTINUATION"  # existing trend sustaining
    REVERSAL = "REVERSAL"  # trend losing steam, setup flipping
    ACCUMULATION = "ACCUMULATION"  # quiet range, smart-money building
    DISTRIBUTION = "DISTRIBUTION"  # range after run, supply emerging
    BREAKOUT = "BREAKOUT"  # imminent or active range escape
    EXHAUSTION = "EXHAUSTION"  # move running out of fuel
    LIQUIDITY_GRAB = "LIQUIDITY_GRAB"  # stop-hunt / wick below/above range


class NarrativeMaturity(str, Enum):
    """How developed and actionable a narrative currently is.

    Parallel to `HypothesisMaturity` but scoped to narrative reasoning
    rather than execution readiness — a narrative reaching ESTABLISHED
    is what should *seed* a hypothesis, not place a trade directly.
    """

    FORMING = "FORMING"  # just detected, thin evidence
    DEVELOPING = "DEVELOPING"  # gathering evidence, moderate confidence
    ESTABLISHED = "ESTABLISHED"  # well-supported, should seed hypotheses
    FADING = "FADING"  # conflicting evidence outgrowing support
    EXPIRED = "EXPIRED"  # aged out or explicitly invalidated


# Per-kind default lifespan — directional narratives that have not been
# recently reinforced should fade faster than structural/cycle narratives.
_DEFAULT_MAX_AGE: dict[NarrativeKind, float] = {
    NarrativeKind.TREND_CONTINUATION: 8.0 * 3600,
    NarrativeKind.REVERSAL: 4.0 * 3600,
    NarrativeKind.ACCUMULATION: 12.0 * 3600,
    NarrativeKind.DISTRIBUTION: 12.0 * 3600,
    NarrativeKind.BREAKOUT: 2.0 * 3600,
    NarrativeKind.EXHAUSTION: 3.0 * 3600,
    NarrativeKind.LIQUIDITY_GRAB: 1.0 * 3600,
}

MAX_EVIDENCE_ITEMS = 20  # truncation cap per evidence list (mirrors brain_models)


# ── Data model ────────────────────────────────────────────────────────────


@dataclass
class SymbolNarrative:
    """One typed market narrative for one symbol.

    A symbol can hold at most one *active* narrative per `NarrativeKind` at
    a time; when a new signal for the same kind arrives the existing
    narrative is evolved in place (see `narrative_evolution.py`) rather than
    replaced from scratch, so its age and evidence history survive across
    ticks.
    """

    narrative_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    instrument: str = ""
    kind: NarrativeKind = NarrativeKind.TREND_CONTINUATION
    title: str = ""
    supporting_evidence: list[str] = field(default_factory=list)
    conflicting_evidence: list[str] = field(default_factory=list)
    confidence: float = 0.3
    maturity: NarrativeMaturity = NarrativeMaturity.FORMING
    # The market regime and session description captured when this narrative
    # was formed or last substantially reinforced — context for decay engines.
    regime_context: str = ""
    session_context: str = ""
    formed_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)
    reinforcement_count: int = 0
    max_age_seconds: float = 4.0 * 3600

    # ── Computed properties ──────────────────────────────────────────────

    def age_seconds(self, now: float | None = None) -> float:
        now = now if now is not None else time.time()
        return max(0.0, now - self.formed_at)

    def evidence_balance(self) -> float:
        """Supporting-to-total ratio: 1.0 = all supporting, 0.0 = all conflicting.
        Returns 0.5 when no evidence has been recorded yet.
        """
        total = len(self.supporting_evidence) + len(self.conflicting_evidence)
        if total == 0:
            return 0.5
        return len(self.supporting_evidence) / total

    def is_active(self) -> bool:
        return self.maturity != NarrativeMaturity.EXPIRED

    def is_actionable(self) -> bool:
        """True when mature enough to feed into hypothesis generation."""
        return self.maturity in (NarrativeMaturity.DEVELOPING, NarrativeMaturity.ESTABLISHED)

    # ── Mutation helpers ─────────────────────────────────────────────────

    def reinforce(self, evidence: str, boost: float = 0.12, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        self.supporting_evidence.append(evidence)
        if len(self.supporting_evidence) > MAX_EVIDENCE_ITEMS:
            self.supporting_evidence = self.supporting_evidence[-MAX_EVIDENCE_ITEMS:]
        self.confidence = min(1.0, self.confidence + boost)
        self.reinforcement_count += 1
        self.last_updated_at = now
        self._recompute_maturity()

    def weaken(self, evidence: str, penalty: float = 0.15, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        self.conflicting_evidence.append(evidence)
        if len(self.conflicting_evidence) > MAX_EVIDENCE_ITEMS:
            self.conflicting_evidence = self.conflicting_evidence[-MAX_EVIDENCE_ITEMS:]
        self.confidence = max(0.0, self.confidence - penalty)
        self.last_updated_at = now
        self._recompute_maturity()

    def expire(self, now: float | None = None) -> None:
        self.maturity = NarrativeMaturity.EXPIRED
        self.last_updated_at = now if now is not None else time.time()

    def _recompute_maturity(self) -> None:
        """Recompute maturity from current confidence + evidence balance.
        Does not revert EXPIRED — once expired, only explicit re-detection
        via `NarrativeBuilder` can open a fresh narrative for this kind.
        """
        if self.maturity == NarrativeMaturity.EXPIRED:
            return
        balance = self.evidence_balance()
        # Fading: conflicting evidence is dominating
        if balance < 0.35 and len(self.conflicting_evidence) >= 2:
            self.maturity = NarrativeMaturity.FADING
            return
        # Established: high confidence, repeated reinforcement, positive balance
        if self.confidence >= 0.70 and self.reinforcement_count >= 3 and balance >= 0.60:
            self.maturity = NarrativeMaturity.ESTABLISHED
        elif self.confidence >= 0.45 and self.reinforcement_count >= 1:
            self.maturity = NarrativeMaturity.DEVELOPING
        else:
            self.maturity = NarrativeMaturity.FORMING

    # ── Serialisation ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "narrative_id": self.narrative_id,
            "instrument": self.instrument,
            "kind": self.kind.value,
            "title": self.title,
            "supporting_evidence": self.supporting_evidence[-MAX_EVIDENCE_ITEMS:],
            "conflicting_evidence": self.conflicting_evidence[-MAX_EVIDENCE_ITEMS:],
            "confidence": round(self.confidence, 4),
            "maturity": self.maturity.value,
            "regime_context": self.regime_context,
            "session_context": self.session_context,
            "formed_at": round(self.formed_at, 3),
            "last_updated_at": round(self.last_updated_at, 3),
            "reinforcement_count": self.reinforcement_count,
            "max_age_seconds": self.max_age_seconds,
            "evidence_balance": round(self.evidence_balance(), 4),
            "age_seconds": round(self.age_seconds(), 1),
        }


# ── Store (stateless operations) ─────────────────────────────────────────


class NarrativeStore:
    """Stateless CRUD operations over a per-symbol ``dict[str, SymbolNarrative]``
    keyed by `NarrativeKind.value`.

    Follows the same convention as `HypothesisTracker` / `OpportunityTracker`:
    the dict lives on the engine that owns per-symbol state
    (`MarketNarrativeEngine._stores`), this class holds none of its own.
    """

    def upsert(self, store: dict[str, SymbolNarrative], narrative: SymbolNarrative) -> None:
        """Insert or replace the narrative for its kind."""
        store[narrative.kind.value] = narrative

    def get(self, store: dict[str, SymbolNarrative], kind: NarrativeKind) -> SymbolNarrative | None:
        return store.get(kind.value)

    def active(self, store: dict[str, SymbolNarrative]) -> list[SymbolNarrative]:
        return [n for n in store.values() if n.is_active()]

    def actionable(self, store: dict[str, SymbolNarrative]) -> list[SymbolNarrative]:
        return [n for n in store.values() if n.is_actionable()]

    def established(self, store: dict[str, SymbolNarrative]) -> list[SymbolNarrative]:
        return [n for n in store.values() if n.maturity == NarrativeMaturity.ESTABLISHED]

    def dominant(self, store: dict[str, SymbolNarrative]) -> SymbolNarrative | None:
        """The active narrative with the highest confidence."""
        candidates = self.active(store)
        if not candidates:
            return None
        return max(candidates, key=lambda n: n.confidence)

    def expire_stale(self, store: dict[str, SymbolNarrative], now: float | None = None) -> int:
        """Age out narratives that have exceeded their `max_age_seconds`.
        Returns the number expired.
        """
        now = now if now is not None else time.time()
        expired = 0
        for narrative in store.values():
            if narrative.is_active() and narrative.age_seconds(now) > narrative.max_age_seconds:
                narrative.expire(now)
                expired += 1
        return expired

    def to_list_dict(self, store: dict[str, SymbolNarrative]) -> list[dict]:
        return [n.to_dict() for n in store.values()]

    @staticmethod
    def from_list_dict(raw: list[dict]) -> dict[str, SymbolNarrative]:
        """Reconstruct a store dict from its `to_list_dict()` form."""
        result: dict[str, SymbolNarrative] = {}
        for r in raw:
            try:
                n = SymbolNarrative(
                    narrative_id=r["narrative_id"],
                    instrument=r["instrument"],
                    kind=NarrativeKind(r["kind"]),
                    title=r["title"],
                    supporting_evidence=r.get("supporting_evidence", []),
                    conflicting_evidence=r.get("conflicting_evidence", []),
                    confidence=float(r["confidence"]),
                    maturity=NarrativeMaturity(r["maturity"]),
                    regime_context=r.get("regime_context", ""),
                    session_context=r.get("session_context", ""),
                    formed_at=float(r["formed_at"]),
                    last_updated_at=float(r["last_updated_at"]),
                    reinforcement_count=int(r.get("reinforcement_count", 0)),
                    max_age_seconds=float(r.get("max_age_seconds", 4.0 * 3600)),
                )
                result[n.kind.value] = n
            except (KeyError, ValueError):
                pass
        return result


__all__ = [
    "NarrativeKind",
    "NarrativeMaturity",
    "SymbolNarrative",
    "NarrativeStore",
    "_DEFAULT_MAX_AGE",
    "MAX_EVIDENCE_ITEMS",
]
