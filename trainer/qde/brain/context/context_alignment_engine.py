"""QDE Brain — multi-timeframe context: alignment engine.

Pure, stateless computation of cross-timeframe relationship patterns.

`ContextAlignmentEngine.compute()` takes the current per-timeframe
observations (from `TimeframeMemory`) and produces a `CrossTimeframeContext`
that describes *how* the timeframes currently relate to each other — not
*what* each one "means" in isolation.

Pattern vocabulary
------------------
ALIGNED      — All contributing timeframes share the same momentum direction.
PULLBACK     — The coarser group has a direction; the finer group is counter-
               moving.  This is contextual, not contradictory: an H4 upward
               momentum with M15 downward momentum is a pullback, not a signal
               conflict.
CONTINUATION — Aligned AND the finer group's momentum has been strengthening
               in the dominant direction.
EXHAUSTION   — Aligned but momentum has been weakening across timeframes —
               the move is losing energy.
TRANSITION   — One or more timeframes recently crossed zero — a directional
               change is underway.
MIXED        — Insufficient data or no clear relational pattern.

No hardcoded directional rules
-------------------------------
Every branch in `_classify` compares numeric quantities derived from
`TimeframeObservation` (momentum, alignment_score, momentum_trend) — never
"if H4 is bullish" or "if D1 regime is TREND_UP".  The `dominant_direction`
field in the output is always `sign(coarse_group_weighted_momentum)`, which
emerges from the actual bar data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from trainer.qde.brain.context.timeframe_memory import (
    ALL_CONTEXT_TIMEFRAMES,
    CONTEXT_TIMEFRAME_SECONDS,
    ContextTimeframe,
    TimeframeMemory,
    TimeframeObservation,
    coarse_timeframes,
    fine_timeframes,
)

# ── AlignmentPattern ──────────────────────────────────────────────────────────


class AlignmentPattern(str, Enum):
    """Relational pattern between the coarse and fine timeframe groups.

    Names describe *relationships* (what the timeframes are doing relative
    to each other), not directional assertions ("price is going up").
    """

    ALIGNED = "ALIGNED"
    PULLBACK = "PULLBACK"
    CONTINUATION = "CONTINUATION"
    EXHAUSTION = "EXHAUSTION"
    TRANSITION = "TRANSITION"
    MIXED = "MIXED"


# ── CrossTimeframeContext ─────────────────────────────────────────────────────


@dataclass
class CrossTimeframeContext:
    """The QDE's current understanding of where price sits in the larger story.

    All numeric values are derived from observational quantities without
    hardcoded directional rules:

    - `dominant_direction`  = sign of the coarse-group weighted-average
      momentum, emerging from bar data alone.
    - `pullback_depth`      = range-position distance between the coarse and
      fine groups, a continuous approximation of how far the counter-move
      has penetrated the coarser structure.
    - `alignment_score`     = fraction of adjacent timeframe pairs that share
      a momentum direction, measuring agreement across the full stack.

    `summary` is a human-readable, relational description (e.g. "H4/D1
    momentum positive, H1/M15/M5 momentum negative — counter-move within
    larger picture") that does not assert which direction is "the trend".
    """

    instrument: str = ""
    as_of: float = field(default_factory=time.time)

    pattern: AlignmentPattern = AlignmentPattern.MIXED
    pattern_confidence: float = 0.0     # 0–1

    alignment_score: float = 0.0        # 0 = fully opposed, 1 = fully aligned

    # Coarse group (MN/W1/D1/H4): duration-weighted average momentum
    coarse_momentum: float = 0.0
    # Fine group (H1/M15/M5): duration-weighted average momentum
    fine_momentum: float = 0.0
    # Sign of coarse_momentum: +1 / 0 / −1
    dominant_direction: int = 0

    # How far the fine-TF counter-move has penetrated the coarse-TF range.
    # 0 = no counter-move; 1 = range positions fully inverted.
    pullback_depth: float = 0.0

    # Per-timeframe raw observations at time of computation (serialised)
    per_timeframe: dict[str, dict] = field(default_factory=dict)

    # Relational summary — no directional labels, only relational descriptions
    summary: str = ""

    timeframes_contributing: int = 0

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "as_of": round(self.as_of, 3),
            "pattern": self.pattern.value,
            "pattern_confidence": round(self.pattern_confidence, 4),
            "alignment_score": round(self.alignment_score, 4),
            "coarse_momentum": round(self.coarse_momentum, 6),
            "fine_momentum": round(self.fine_momentum, 6),
            "dominant_direction": self.dominant_direction,
            "pullback_depth": round(self.pullback_depth, 4),
            "per_timeframe": self.per_timeframe,
            "summary": self.summary,
            "timeframes_contributing": self.timeframes_contributing,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CrossTimeframeContext:
        return cls(
            instrument=data.get("instrument", ""),
            as_of=data.get("as_of", 0.0),
            pattern=AlignmentPattern(data.get("pattern", AlignmentPattern.MIXED.value)),
            pattern_confidence=data.get("pattern_confidence", 0.0),
            alignment_score=data.get("alignment_score", 0.0),
            coarse_momentum=data.get("coarse_momentum", 0.0),
            fine_momentum=data.get("fine_momentum", 0.0),
            dominant_direction=data.get("dominant_direction", 0),
            pullback_depth=data.get("pullback_depth", 0.0),
            per_timeframe=data.get("per_timeframe", {}),
            summary=data.get("summary", ""),
            timeframes_contributing=data.get("timeframes_contributing", 0),
        )


# ── ContextAlignmentEngine ────────────────────────────────────────────────────

# Minimum fresh timeframes before we emit anything other than MIXED.
_MIN_CONTRIBUTING = 3

# Momentum below this threshold (absolute) is treated as "flat" for
# directional sign classification.  Slightly above _MOMENTUM_FLAT in
# timeframe_memory so that very small noise values don't produce
# spurious sign disagreements.
_SIGN_THRESHOLD = 0.12

# Alignment score thresholds
_ALIGNMENT_HIGH = 0.65   # above this → strongly aligned
_ALIGNMENT_LOW  = 0.35   # below this → clear disagreement

# Exhaustion: the normalised strengthening score must exceed this to
# classify as EXHAUSTION rather than ALIGNED.
_EXHAUSTION_THRESHOLD = 0.25

# Continuation: fine-TF momentum trend must be positive and above this.
_CONTINUATION_TREND_MIN = 0.04

# Maximum staleness multiplier before a timeframe is excluded from
# contribution (beyond the per-observation is_fresh check).
_MAX_STALENESS = 2.5

import math as _math


def _sign_t(x: float, threshold: float = _SIGN_THRESHOLD) -> int:
    return 0 if abs(x) < threshold else (1 if x > 0 else -1)


def _sign(x: float) -> int:
    return 0 if abs(x) < 1e-9 else (1 if x > 0 else -1)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _weighted_momentum(
    obs: dict[ContextTimeframe, TimeframeObservation],
) -> float:
    """Duration-weighted average momentum across a group of timeframes.

    Coarser timeframes have larger `CONTEXT_TIMEFRAME_SECONDS` values and
    therefore larger `log1p` weights — their slower-moving signals dominate
    the group average, which is the right behaviour: monthly momentum should
    outweigh 5-minute momentum when summarising the coarse group.
    """
    total_w = total_m = 0.0
    for tf, o in obs.items():
        w = _math.log1p(CONTEXT_TIMEFRAME_SECONDS.get(tf, 3600))
        total_w += w
        total_m += w * o.momentum
    return total_m / total_w if total_w > 1e-12 else 0.0


class ContextAlignmentEngine:
    """Pure, stateless cross-timeframe pattern computer.

    Instantiate once; call `compute()` on every tick that produces updated
    observations.  Holds no per-symbol state — all history queries are
    delegated back to the caller's `TimeframeMemory`.
    """

    def compute(
        self,
        instrument: str,
        observations: dict[ContextTimeframe, TimeframeObservation],
        memory: TimeframeMemory,
        now: float | None = None,
    ) -> CrossTimeframeContext:
        """Compute the cross-timeframe context for `instrument`.

        Args:
            instrument:   Symbol being analysed.
            observations: Current per-TF observations from `TimeframeMemory`.
            memory:       History store for slope / crossing queries.
            now:          Epoch seconds; defaults to `time.time()`.
        """
        ts = now if now is not None else time.time()

        fresh = {
            tf: obs
            for tf, obs in observations.items()
            if obs.is_fresh(_MAX_STALENESS)
        }

        if len(fresh) < _MIN_CONTRIBUTING:
            return CrossTimeframeContext(
                instrument=instrument,
                as_of=ts,
                pattern=AlignmentPattern.MIXED,
                pattern_confidence=0.0,
                summary="insufficient data",
                timeframes_contributing=len(fresh),
                per_timeframe={tf.value: o.to_dict() for tf, o in observations.items()},
            )

        coarse_obs = {tf: o for tf, o in fresh.items() if tf in coarse_timeframes()}
        fine_obs   = {tf: o for tf, o in fresh.items() if tf in fine_timeframes()}

        coarse_mom = _weighted_momentum(coarse_obs)
        fine_mom   = _weighted_momentum(fine_obs) if fine_obs else 0.0

        dom_dir    = _sign_t(coarse_mom)
        alignment  = self._alignment_score(fresh)
        pb_depth   = self._pullback_depth(coarse_obs, fine_obs)

        any_transition = any(
            memory.recently_crossed_zero(instrument, tf, lookback=4)
            for tf in fresh
        )

        coarse_trends = [memory.momentum_trend(instrument, tf) for tf in coarse_obs]
        fine_trends   = [memory.momentum_trend(instrument, tf) for tf in fine_obs]
        avg_coarse_trend = _mean(coarse_trends)
        avg_fine_trend   = _mean(fine_trends)

        pattern, confidence = self._classify(
            coarse_mom=coarse_mom,
            fine_mom=fine_mom,
            alignment=alignment,
            dom_dir=dom_dir,
            any_transition=any_transition,
            avg_coarse_trend=avg_coarse_trend,
            avg_fine_trend=avg_fine_trend,
            n_fresh=len(fresh),
        )

        summary = self._summarize(pattern, coarse_mom, fine_mom, dom_dir, fresh)

        return CrossTimeframeContext(
            instrument=instrument,
            as_of=ts,
            pattern=pattern,
            pattern_confidence=round(confidence, 4),
            alignment_score=round(alignment, 4),
            coarse_momentum=round(coarse_mom, 6),
            fine_momentum=round(fine_mom, 6),
            dominant_direction=dom_dir,
            pullback_depth=round(pb_depth, 4),
            per_timeframe={tf.value: o.to_dict() for tf, o in observations.items()},
            summary=summary,
            timeframes_contributing=len(fresh),
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _alignment_score(
        self,
        obs: dict[ContextTimeframe, TimeframeObservation],
    ) -> float:
        """Fraction of adjacent TF pairs (coarsest → finest) whose momentum
        signs agree.

        1.0 = every adjacent pair agrees; 0.0 = every adjacent pair disagrees.
        Flat observations (|momentum| < threshold) contribute 0.5 — neutral.
        """
        ordered = [obs[tf] for tf in ALL_CONTEXT_TIMEFRAMES if tf in obs]
        if len(ordered) < 2:
            return 0.5
        agreements = 0.0
        pairs = len(ordered) - 1
        for a, b in zip(ordered[:-1], ordered[1:]):
            sa = _sign_t(a.momentum)
            sb = _sign_t(b.momentum)
            if sa == 0 or sb == 0:
                agreements += 0.5
            elif sa == sb:
                agreements += 1.0
        return agreements / pairs

    def _pullback_depth(
        self,
        coarse_obs: dict[ContextTimeframe, TimeframeObservation],
        fine_obs:   dict[ContextTimeframe, TimeframeObservation],
    ) -> float:
        """Continuous approximation of how far the fine-TF counter-move has
        penetrated the coarse-TF structure, using range_position.

        If the coarse group's average range_position is 0.80 (near highs)
        and the fine group's is 0.50 (mid-range), the pullback depth is 0.30
        — price has retraced 30 % of the coarse-TF range in the fine group.

        Returns 0.0 when either group has no data.
        """
        if not coarse_obs or not fine_obs:
            return 0.0
        coarse_rp = _mean([o.range_position for o in coarse_obs.values()])
        fine_rp   = _mean([o.range_position for o in fine_obs.values()])
        return min(1.0, abs(coarse_rp - fine_rp))

    def _classify(
        self,
        coarse_mom: float,
        fine_mom: float,
        alignment: float,
        dom_dir: int,
        any_transition: bool,
        avg_coarse_trend: float,
        avg_fine_trend: float,
        n_fresh: int,
    ) -> tuple[AlignmentPattern, float]:
        """Choose the best-fit pattern and a calibrated confidence score.

        Every branch here compares numeric quantities — momentum values,
        alignment ratios, trend slopes.  No branch reads a directional label
        or a timeframe name.
        """
        coarse_sign = _sign_t(coarse_mom)
        fine_sign   = _sign_t(fine_mom)

        # ── TRANSITION ────────────────────────────────────────────────────
        # A zero-crossing happened recently; the context is changing but
        # hasn't settled into a new pattern yet.
        if any_transition:
            # Confidence is higher when alignment is still low (confusion) or
            # when both groups are now disagreeing.
            conf = 0.55 + 0.25 * (1.0 - alignment)
            return AlignmentPattern.TRANSITION, min(conf, 0.88)

        # ── No direction at all ───────────────────────────────────────────
        if coarse_sign == 0 and fine_sign == 0:
            return AlignmentPattern.MIXED, 0.4

        # ── PULLBACK ─────────────────────────────────────────────────────
        # Coarse and fine have opposite signs: fine TF is counter-moving
        # within the coarse-TF directional context.
        if coarse_sign != 0 and fine_sign != 0 and coarse_sign != fine_sign:
            total = abs(coarse_mom) + abs(fine_mom)
            opposition = abs(coarse_mom - fine_mom) / (total + 1e-9)
            conf = 0.50 + 0.42 * opposition
            return AlignmentPattern.PULLBACK, min(conf, 0.95)

        # ── Same-direction or one side flat ──────────────────────────────

        # Normalised exhaustion score: how strongly is momentum decelerating?
        # Positive = decelerating (exhaustion); negative = accelerating.
        mom_magnitude = abs(coarse_mom) + abs(fine_mom)
        if mom_magnitude > 1e-9:
            raw_slope = avg_coarse_trend + avg_fine_trend
            exhaustion_score = -raw_slope / mom_magnitude
        else:
            exhaustion_score = 0.0

        # EXHAUSTION: aligned direction but momentum measurably weakening
        if (
            alignment > _ALIGNMENT_LOW
            and exhaustion_score > _EXHAUSTION_THRESHOLD
        ):
            conf = 0.45 + 0.35 * min(1.0, exhaustion_score)
            return AlignmentPattern.EXHAUSTION, min(conf, 0.85)

        # CONTINUATION: aligned AND fine-TF momentum recently strengthening
        if (
            alignment > _ALIGNMENT_HIGH
            and fine_sign != 0
            and avg_fine_trend > _CONTINUATION_TREND_MIN
            and _sign_t(avg_fine_trend) == coarse_sign
        ):
            conf = 0.50 + 0.40 * alignment
            return AlignmentPattern.CONTINUATION, min(conf, 0.92)

        # ALIGNED: same direction, no special sub-pattern
        if alignment > _ALIGNMENT_HIGH and coarse_sign != 0:
            conf = 0.40 + 0.50 * alignment
            return AlignmentPattern.ALIGNED, min(conf, 0.90)

        return AlignmentPattern.MIXED, 0.35

    def _summarize(
        self,
        pattern: AlignmentPattern,
        coarse_mom: float,
        fine_mom: float,
        dom_dir: int,
        obs: dict[ContextTimeframe, TimeframeObservation],
    ) -> str:
        """Human-readable, relational description.

        Uses timeframe group names (e.g. "MN/W1/D1/H4") and directional
        adjectives ("positive", "negative", "flat") — deliberately avoids
        asserting "the trend is X" or "the market is bullish".
        """
        coarse_present = [tf.value for tf in coarse_timeframes() if tf in obs]
        fine_present   = [tf.value for tf in fine_timeframes()   if tf in obs]
        coarse_names   = "/".join(coarse_present) or "coarse"
        fine_names     = "/".join(fine_present)   or "fine"

        def _dir(m: float) -> str:
            s = _sign_t(m)
            return "positive" if s > 0 else ("negative" if s < 0 else "flat")

        n = len([o for o in obs.values() if o.bars_available > 0])

        if pattern == AlignmentPattern.PULLBACK:
            return (
                f"{coarse_names} momentum {_dir(coarse_mom)}, "
                f"{fine_names} momentum {_dir(fine_mom)} "
                f"— counter-move within larger picture ({n} TFs)"
            )
        if pattern == AlignmentPattern.CONTINUATION:
            return (
                f"{coarse_names} and {fine_names} momentum both {_dir(coarse_mom)} "
                f"and fine-TF strengthening — continuation ({n} TFs)"
            )
        if pattern == AlignmentPattern.ALIGNED:
            return (
                f"Cross-timeframe momentum aligned {_dir(coarse_mom)} "
                f"({coarse_names}, {fine_names}) ({n} TFs)"
            )
        if pattern == AlignmentPattern.EXHAUSTION:
            return (
                f"Momentum {_dir(coarse_mom)} across timeframes but weakening "
                f"({coarse_names}, {fine_names}) ({n} TFs)"
            )
        if pattern == AlignmentPattern.TRANSITION:
            return (
                f"Recent momentum direction change detected "
                f"({coarse_names}, {fine_names}) ({n} TFs)"
            )
        return f"Mixed signals across {n} contributing timeframes"
