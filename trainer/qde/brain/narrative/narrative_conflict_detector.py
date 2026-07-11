"""QDE Brain — narrative conflict detector.

Identifies logical contradictions between simultaneously-active
`SymbolNarrative` instances for one symbol, classifying each conflict by
type (directional, structural, temporal) and severity.

Why conflicts matter
--------------------
Multiple active narratives are not always compatible.  A QDE reasoning from
TREND_CONTINUATION and REVERSAL simultaneously has a contradictory internal
picture — the combined signal is weaker than either alone.  Surfacing
conflicts explicitly lets:

  * `MarketNarrativeEngine.narrative_conviction_boost` discount conviction
    when the narrative picture is divided rather than coherent.
  * `HypothesisEngine` and `ConvictionEngine` consumers know to treat a
    high-conflict state as ambiguous, preferring NO_TRADE.
  * Opportunity ranking penalise symbols with conflicting narratives (even
    if each individual narrative looks promising in isolation).

Conflict types
--------------
DIRECTIONAL  — the two narrative kinds imply opposite price expectations
               (e.g. TREND_CONTINUATION vs REVERSAL, ACCUMULATION vs
               DISTRIBUTION, BREAKOUT vs EXHAUSTION).
STRUCTURAL   — the two narrative kinds are structurally incompatible even
               if not directly opposite (e.g. LIQUIDITY_GRAB vs
               TREND_CONTINUATION: a stop-hunt undermines a clean trend).
TEMPORAL     — one narrative implies the move is just starting and the
               other implies it is ending (e.g. BREAKOUT vs EXHAUSTION).

This module produces raw conflict readings only — it never alters any
narrative's confidence or maturity (that is `NarrativeEvolution`'s domain),
and never makes trade decisions (that stays at `qde_inference`).
"""

from __future__ import annotations

from dataclasses import dataclass

from trainer.qde.brain.narrative.narrative_store import (
    NarrativeKind,
    NarrativeMaturity,
    SymbolNarrative,
)

# ── Conflict taxonomy ─────────────────────────────────────────────────────

# frozenset({KindA, KindB}) -> (conflict_type, base_severity)
# Severity is the maximum conflict score for fully-established narratives;
# actual severity is scaled by each narrative's confidence.
_DIRECTIONAL_CONFLICTS: dict[frozenset, float] = {
    frozenset({NarrativeKind.TREND_CONTINUATION, NarrativeKind.REVERSAL}): 0.90,
    frozenset({NarrativeKind.ACCUMULATION, NarrativeKind.DISTRIBUTION}): 0.85,
    frozenset({NarrativeKind.BREAKOUT, NarrativeKind.EXHAUSTION}): 0.80,
}

_STRUCTURAL_CONFLICTS: dict[frozenset, float] = {
    frozenset({NarrativeKind.TREND_CONTINUATION, NarrativeKind.LIQUIDITY_GRAB}): 0.55,
    frozenset({NarrativeKind.BREAKOUT, NarrativeKind.LIQUIDITY_GRAB}): 0.60,
    frozenset({NarrativeKind.ACCUMULATION, NarrativeKind.LIQUIDITY_GRAB}): 0.45,
}

_TEMPORAL_CONFLICTS: dict[frozenset, float] = {
    frozenset({NarrativeKind.BREAKOUT, NarrativeKind.EXHAUSTION}): 0.75,
    frozenset({NarrativeKind.TREND_CONTINUATION, NarrativeKind.EXHAUSTION}): 0.70,
    frozenset({NarrativeKind.REVERSAL, NarrativeKind.ACCUMULATION}): 0.50,
    frozenset({NarrativeKind.REVERSAL, NarrativeKind.DISTRIBUTION}): 0.50,
}

# Directionality implied by each kind: +1 bullish, -1 bearish, 0 neutral.
# Used to compute net directional bias when conflicts are present.
_KIND_BIAS: dict[NarrativeKind, float] = {
    NarrativeKind.TREND_CONTINUATION: 0.0,  # direction depends on current regime
    NarrativeKind.REVERSAL: 0.0,  # same — depends on which way trend runs
    NarrativeKind.ACCUMULATION: 0.6,  # typically bullish buildup
    NarrativeKind.DISTRIBUTION: -0.6,  # typically bearish buildup
    NarrativeKind.BREAKOUT: 0.0,  # direction-agnostic
    NarrativeKind.EXHAUSTION: -0.3,  # bearish undertone (move failing)
    NarrativeKind.LIQUIDITY_GRAB: 0.0,  # ambiguous direction
}


# ── Output dataclass ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class NarrativeConflict:
    """One detected contradiction between two active narratives."""

    kind_a: NarrativeKind
    kind_b: NarrativeKind
    conflict_type: str  # "DIRECTIONAL" | "STRUCTURAL" | "TEMPORAL"
    base_severity: float  # 0-1 from the taxonomy table
    severity: float  # 0-1, scaled by the two narratives' confidence
    description: str

    def to_dict(self) -> dict:
        return {
            "kind_a": self.kind_a.value,
            "kind_b": self.kind_b.value,
            "conflict_type": self.conflict_type,
            "base_severity": round(self.base_severity, 4),
            "severity": round(self.severity, 4),
            "description": self.description,
        }


# ── Detector ──────────────────────────────────────────────────────────────


class NarrativeConflictDetector:
    """Detect logical contradictions between active narratives for one symbol.

    Stateless — safe to share and reuse across symbols.
    """

    def detect(self, narratives: list[SymbolNarrative]) -> list[NarrativeConflict]:
        """Return all detected conflicts among the given narratives.

        Only *active* (non-EXPIRED), *actionable* (DEVELOPING or ESTABLISHED)
        narratives participate — a FORMING narrative has too thin evidence to
        be considered a committed position worth conflicting with.
        """
        candidates = [
            n for n in narratives if n.is_active() and n.maturity != NarrativeMaturity.FORMING
        ]
        if len(candidates) < 2:
            return []

        conflicts: list[NarrativeConflict] = []
        seen: set[frozenset] = set()

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                na, nb = candidates[i], candidates[j]
                pair = frozenset({na.kind, nb.kind})
                if pair in seen:
                    continue
                seen.add(pair)

                conflict = self._check_pair(na, nb, pair)
                if conflict is not None:
                    conflicts.append(conflict)

        # Sort by descending severity so callers see the worst first.
        return sorted(conflicts, key=lambda c: c.severity, reverse=True)

    def conflict_intensity(self, conflicts: list[NarrativeConflict]) -> float:
        """Aggregate 0-1 measure of how conflicted the overall narrative picture is.

        0.0 = no conflicts; 1.0 = maximum conflict across all active narratives.
        Uses the max single-conflict severity rather than a sum so one extreme
        conflict dominates rather than many mild ones compounding.
        """
        if not conflicts:
            return 0.0
        return max(c.severity for c in conflicts)

    def net_bias(self, narratives: list[SymbolNarrative]) -> float:
        """Confidence-weighted directional bias across all active narratives.

        Returns a value in [-1, 1]: positive = bullish lean, negative = bearish,
        ~0 = ambiguous or balanced.  Useful for `opportunity_rank_signal` and
        for seeding hypothesis direction when multiple narratives agree.
        """
        active = [n for n in narratives if n.is_active()]
        if not active:
            return 0.0
        total_weight = sum(n.confidence for n in active)
        if total_weight < 1e-9:
            return 0.0
        weighted = sum(_KIND_BIAS[n.kind] * n.confidence for n in active)
        return max(-1.0, min(1.0, weighted / total_weight))

    def coherence_score(
        self,
        narratives: list[SymbolNarrative],
        conflicts: list[NarrativeConflict] | None = None,
    ) -> float:
        """0-1 score for how internally consistent the narrative picture is.

        1.0 = no conflicts, all active narratives agree.
        0.0 = maximally conflicted.
        """
        if conflicts is None:
            conflicts = self.detect(narratives)
        return max(0.0, 1.0 - self.conflict_intensity(conflicts))

    # ── Internal ─────────────────────────────────────────────────────────

    def _check_pair(
        self,
        na: SymbolNarrative,
        nb: SymbolNarrative,
        pair: frozenset,
    ) -> NarrativeConflict | None:
        # Check all three conflict tables; a pair can appear in multiple tables
        # (e.g. BREAKOUT/EXHAUSTION is both DIRECTIONAL and TEMPORAL).  We
        # report the highest-severity type.
        best: tuple[str, float] | None = None

        for table, conflict_type in (
            (_DIRECTIONAL_CONFLICTS, "DIRECTIONAL"),
            (_STRUCTURAL_CONFLICTS, "STRUCTURAL"),
            (_TEMPORAL_CONFLICTS, "TEMPORAL"),
        ):
            base = table.get(pair)
            if base is None:
                continue
            if best is None or base > best[1]:
                best = (conflict_type, base)

        if best is None:
            return None

        conflict_type, base_severity = best
        # Scale severity by the geometric mean of the two confidences — a
        # FORMING narrative with 0.3 confidence barely counts as a committed
        # position, so its conflict with a well-established narrative is mild.
        scaled = base_severity * (na.confidence * nb.confidence) ** 0.5

        description = (
            f"{na.kind.value} ({na.maturity.value}, conf={na.confidence:.0%}) "
            f"conflicts with "
            f"{nb.kind.value} ({nb.maturity.value}, conf={nb.confidence:.0%}) "
            f"[{conflict_type}]"
        )

        return NarrativeConflict(
            kind_a=na.kind,
            kind_b=nb.kind,
            conflict_type=conflict_type,
            base_severity=base_severity,
            severity=round(min(1.0, scaled), 4),
            description=description,
        )


__all__ = ["NarrativeConflict", "NarrativeConflictDetector"]
