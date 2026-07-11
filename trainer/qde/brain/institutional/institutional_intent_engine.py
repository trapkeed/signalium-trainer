"""QDE Brain — institutional: learned intent engine.

``InstitutionalIntentEngine`` is the learning layer that answers the
question the structural detectors cannot:

    "In historically *similar* market conditions, what institutional
     behaviour patterns actually preceded the price move that followed?"

The three structural detectors
(``AccumulationDetector``, ``DistributionDetector``,
``InventoryTransferDetector``) produce structural *priors* — what the
bar pattern structurally resembles.  The engine blends those priors with
historically observed outcomes for similar conditions, progressively
replacing the structural guess with empirically grounded beliefs.

Six probabilistic beliefs
--------------------------
The engine maintains six beliefs over the current market state,
all expressed as continuous probabilities in [0, 1]:

  * ``accumulation_prob``    — gradual building of a large long position
  * ``distribution_prob``    — gradual exit of a long / entry of a short
  * ``transfer_prob``        — inventory transfer from institutional to retail
  * ``stop_hunt_prob``       — engineered sweep of structural levels
  * ``liquidity_acq_prob``   — directional move toward resting-order zones
  * ``rebalance_prob``       — portfolio adjustment without directional intent

These are NOT a probability distribution over mutually exclusive
hypotheses.  They are six independent assessments.  All six can be
high simultaneously (e.g. a stop hunt that also acts as a liquidity
acquisition for the institutional participant's accumulated position).

Learning loop
--------------
Call sequence for one completed trade:

1. At opportunity identification time::

       conditions = InstitutionalConditions(
           instrument="EURUSD",
           range_compression=0.42,
           absorption_strength=0.35,
           ...
       )
       # engine scores the conditions:
       beliefs = engine.score(conditions, direction=1)

2. After the trade closes (or after N bars)::

       outcome = InstitutionalOutcome(
           accumulation_confirmed=True,
           direction_correct=True,
           reward_atr=0.6,
       )
       engine.record_outcome(conditions, outcome)

The engine stores the (conditions, outcome) pair and uses it to
weight future estimates for similar conditions toward the historically
observed pattern.

Similarity metric
-----------------
Cosine similarity on the 8-dimensional ``InstitutionalConditions``
vector.  Consistent with the projection and reward_risk packages.
All 8 features are in [0, 1] so the cosine is well-defined.

Integration signals
-------------------
``context_for_inference()`` returns exactly 18 ``inst_*`` keys for
``QDEInferenceInput.brain_context``:

    inst_accumulation_prob        0-1
    inst_distribution_prob        0-1
    inst_transfer_prob            0-1
    inst_stop_hunt_prob           0-1
    inst_liquidity_acq_prob       0-1
    inst_rebalance_prob           0-1
    inst_dominant_intent_id       float-encoded: 0=accum 1=distrib 2=xfer 3=hunt 4=liq 5=rebal 6=neutral
    inst_intent_certainty         max_prob − second_max (0-1, separation)
    inst_direction_alignment      0-1 (0=opposing 0.5=neutral 1=aligned)
    inst_conviction_modifier      -0.25..+0.25
    inst_opportunity_boost        0-1
    inst_timing_confidence        0-1
    inst_absorption_strength      0-1 (structural absorption)
    inst_range_compression        0-1 (structural range compression)
    inst_volume_signature         0-1 (volume anomaly)
    inst_sweep_recency            0-1 (recency of last structural sweep)
    inst_confidence               0-1 (based on similar-record count)
    inst_is_structural            1.0=pure structural; 0.0=history used

How each named concern is influenced
--------------------------------------
**Narratives**
    ``inst_dominant_intent_id`` encodes the dominant belief so the
    narrative engine can label the current market context.
    ``inst_intent_certainty`` communicates how confident the label is.
    High certainty + clear intent → the narrative gains a "smart money
    acting" thread.

**Conviction**
    ``inst_conviction_modifier`` is added to the per-setup conviction
    score (capped at ±0.25).  When the dominant institutional intent
    *aligns* with the proposed trade direction (e.g. accumulation
    detected + going long), conviction is boosted.  When opposing
    (distribution detected + going long), conviction is reduced.

**Opportunity ranking**
    ``inst_opportunity_boost`` lifts the opportunity score when aligned
    institutional intent is present.  It is used as an additive
    adjustment in the ranker's composite score, not as a replacement
    for any existing component.

**Execution timing**
    ``inst_timing_confidence`` is highest immediately after a confirmed
    stop-hunt reversal or at the point where accumulation / distribution
    reaches peak absorption.  Low in the middle of a transfer or when
    the dominant intent is rebalancing (no edge to any entry timing).

Design rules
------------
* No ``QDEDecision`` imports.
* No imports from ``trainer.qde.brain.projection`` or any other sibling
  brain package (this package is independently self-contained).
* No ICT-specific rules.  All signal names are structurally neutral.
* Multi-symbol: one engine instance, memory keyed by instrument.
"""

from __future__ import annotations

import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.qde.brain.institutional.accumulation_detector import (
    AccumulationDetector,
    AccumulationInput,
)
from trainer.qde.brain.institutional.distribution_detector import (
    DistributionDetector,
    DistributionInput,
)
from trainer.qde.brain.institutional.inventory_transfer_detector import (
    InventoryTransferDetector,
    InventoryTransferInput,
    SweepEvent,
)

# ── Hyper-parameters ──────────────────────────────────────────────────────────

_DEFAULT_MAX_RECORDS = 2000  # per-symbol FIFO capacity
_DEFAULT_K = 30  # top-K for weighted estimates
_MIN_INFORMATIVE = 5  # confidence floor sample count
_RECENCY_TAU_DAYS = 14.0  # exponential recency decay
_N_DIMS = 8  # InstitutionalConditions vector dimension

# Integration signal bounds
_CONVICTION_MOD_MIN = -0.25
_CONVICTION_MOD_MAX = 0.25

# Dominant-intent float encoding
_INTENT_IDS = {
    "accumulation": 0.0,
    "distribution": 1.0,
    "transfer": 2.0,
    "stop_hunt": 3.0,
    "liquidity_acq": 4.0,
    "rebalance": 5.0,
    "neutral": 6.0,
}


# ── InstitutionalConditions ───────────────────────────────────────────────────


@dataclass
class InstitutionalConditions:
    """8-dimensional market conditions fingerprint for institutional similarity.

    All eight features are in [0, 1] for well-defined cosine similarity.

    ``range_compression``
        0-1: how compressed recent price range is vs. baseline.

    ``absorption_strength``
        0-1: level-test absorption (support or resistance).

    ``volume_buy_dominance``
        0-1: volume skew toward buyers.  0.5 = balanced.

    ``momentum_norm``
        0-1 normalisation of signed momentum: ``0.5 + 0.5 * momentum``
        so that −1 → 0, 0 → 0.5, +1 → 1.

    ``session_quality``
        0-1 from the session intelligence engine.

    ``microstructure_quality``
        0-1 from the MIE order-flow quality signal.

    ``sweep_recency``
        0-1 recency of the last structural sweep (0=none/old, 1=very
        recent).  Populated from ``InventoryTransferEstimate.sweep_recency``
        or from explicit ``SweepEvent`` records.

    ``level_proximity``
        0-1 distance to the nearest known structural level.
    """

    instrument: str = ""
    range_compression: float = 0.0
    absorption_strength: float = 0.0
    volume_buy_dominance: float = 0.5
    momentum_norm: float = 0.5  # 0.5 + 0.5 * raw_momentum
    session_quality: float = 0.80
    microstructure_quality: float = 0.80
    sweep_recency: float = 0.0
    level_proximity: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @staticmethod
    def normalize_momentum(raw_momentum: float) -> float:
        """Map signed momentum [−1, +1] → [0, 1]."""
        return max(0.0, min(1.0, 0.5 + 0.5 * raw_momentum))

    def as_vector(self) -> list[float]:
        """8-dimensional cosine-similarity feature vector.

        All values in [0, 1]; no normalization required before cosine.
        """
        return [
            self.range_compression,
            self.absorption_strength,
            self.volume_buy_dominance,
            self.momentum_norm,
            self.session_quality,
            self.microstructure_quality,
            self.sweep_recency,
            self.level_proximity,
        ]

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "range_compression": round(self.range_compression, 4),
            "absorption_strength": round(self.absorption_strength, 4),
            "volume_buy_dominance": round(self.volume_buy_dominance, 4),
            "momentum_norm": round(self.momentum_norm, 4),
            "session_quality": round(self.session_quality, 4),
            "microstructure_quality": round(self.microstructure_quality, 4),
            "sweep_recency": round(self.sweep_recency, 4),
            "level_proximity": round(self.level_proximity, 4),
            "timestamp": round(self.timestamp, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> InstitutionalConditions:
        return cls(
            instrument=d.get("instrument", ""),
            range_compression=d.get("range_compression", 0.0),
            absorption_strength=d.get("absorption_strength", 0.0),
            volume_buy_dominance=d.get("volume_buy_dominance", 0.5),
            momentum_norm=d.get("momentum_norm", 0.5),
            session_quality=d.get("session_quality", 0.80),
            microstructure_quality=d.get("microstructure_quality", 0.80),
            sweep_recency=d.get("sweep_recency", 0.0),
            level_proximity=d.get("level_proximity", 0.0),
            timestamp=d.get("timestamp", 0.0),
        )


# ── InstitutionalOutcome ──────────────────────────────────────────────────────


@dataclass
class InstitutionalOutcome:
    """Ground-truth outcome recorded after a trade closes.

    These labels let the engine learn which structural conditions
    actually preceded each type of institutional behaviour.

    ``accumulation_confirmed``
        True if price subsequently broke upward from the range,
        consistent with an accumulation phase completing.

    ``distribution_confirmed``
        True if price subsequently broke downward, consistent with
        a completed distribution phase.

    ``hunt_reversal_confirmed``
        True if a stop sweep was followed by a clear directional
        reversal in the trade's intended direction.

    ``liquidity_confirmed``
        True if price moved into and filled the identified liquidity
        zone, then showed directional follow-through.

    ``direction_correct``
        True if the trade direction matched the eventual dominant move.
        This is the ground truth for all six intent types.

    ``reward_atr``
        Realized reward in ATR units (signed; positive = profitable).

    ``bars_to_resolution``
        How many bars elapsed before the intent resolved (0 = unknown).
    """

    accumulation_confirmed: bool = False
    distribution_confirmed: bool = False
    hunt_reversal_confirmed: bool = False
    liquidity_confirmed: bool = False
    direction_correct: bool = False
    reward_atr: float = 0.0
    bars_to_resolution: int = 0

    def to_dict(self) -> dict:
        return {
            "accumulation_confirmed": self.accumulation_confirmed,
            "distribution_confirmed": self.distribution_confirmed,
            "hunt_reversal_confirmed": self.hunt_reversal_confirmed,
            "liquidity_confirmed": self.liquidity_confirmed,
            "direction_correct": self.direction_correct,
            "reward_atr": round(self.reward_atr, 6),
            "bars_to_resolution": self.bars_to_resolution,
        }

    @classmethod
    def from_dict(cls, d: dict) -> InstitutionalOutcome:
        return cls(
            accumulation_confirmed=d.get("accumulation_confirmed", False),
            distribution_confirmed=d.get("distribution_confirmed", False),
            hunt_reversal_confirmed=d.get("hunt_reversal_confirmed", False),
            liquidity_confirmed=d.get("liquidity_confirmed", False),
            direction_correct=d.get("direction_correct", False),
            reward_atr=d.get("reward_atr", 0.0),
            bars_to_resolution=d.get("bars_to_resolution", 0),
        )


# ── InstitutionalRecord ───────────────────────────────────────────────────────


@dataclass
class InstitutionalRecord:
    """One completed (conditions → outcome) learning observation."""

    record_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    conditions: InstitutionalConditions = field(default_factory=InstitutionalConditions)
    outcome: InstitutionalOutcome = field(default_factory=InstitutionalOutcome)
    recorded_at: float = field(default_factory=time.time)

    def recency_weight(self, now: float, tau_days: float = _RECENCY_TAU_DAYS) -> float:
        elapsed_days = max(0.0, now - self.recorded_at) / 86400.0
        return math.exp(-elapsed_days / max(tau_days, 1e-9))

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "conditions": self.conditions.to_dict(),
            "outcome": self.outcome.to_dict(),
            "recorded_at": round(self.recorded_at, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> InstitutionalRecord:
        return cls(
            record_id=d.get("record_id", str(uuid.uuid4())[:12]),
            conditions=InstitutionalConditions.from_dict(d.get("conditions", {})),
            outcome=InstitutionalOutcome.from_dict(d.get("outcome", {})),
            recorded_at=d.get("recorded_at", 0.0),
        )


# ── InstitutionalMemory ───────────────────────────────────────────────────────


class InstitutionalMemory:
    """Per-symbol rolling store of institutional intent observations.

    One bounded FIFO per instrument, keyed by instrument name.  No
    business logic here — all estimation in ``InstitutionalIntentEngine``.
    """

    def __init__(self, max_records: int = _DEFAULT_MAX_RECORDS) -> None:
        self._max = max(1, max_records)
        self._store: dict[str, deque[InstitutionalRecord]] = {}

    def record(
        self,
        instrument: str,
        conditions: InstitutionalConditions,
        outcome: InstitutionalOutcome,
        now: float | None = None,
    ) -> InstitutionalRecord:
        now = now if now is not None else time.time()
        rec = InstitutionalRecord(conditions=conditions, outcome=outcome, recorded_at=now)
        q = self._store.setdefault(instrument, deque(maxlen=self._max))
        q.append(rec)
        return rec

    def records(self, instrument: str) -> list[InstitutionalRecord]:
        return list(self._store.get(instrument, []))

    def record_count(self, instrument: str) -> int:
        return len(self._store.get(instrument, []))

    def instruments(self) -> list[str]:
        return list(self._store.keys())

    def to_snapshot(self, instrument: str) -> dict:
        return {
            "max_records": self._max,
            "records": [r.to_dict() for r in self.records(instrument)],
        }

    def load_snapshot(self, instrument: str, data: dict) -> None:
        q: deque[InstitutionalRecord] = deque(maxlen=self._max)
        for d in data.get("records", []):
            try:
                q.append(InstitutionalRecord.from_dict(d))
            except Exception:
                pass
        self._store[instrument] = q


# ── InstitutionalBeliefs ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class InstitutionalBeliefs:
    """Full probabilistic belief state over institutional intent.

    All 18 fields are exposed as ``inst_*`` keys via
    ``context_for_inference()``.

    Six probabilistic beliefs (0-1)
    --------------------------------
    Each is an independent probability; they do not sum to 1.

    Integration signals
    -------------------
    ``inst_conviction_modifier``    −0.25..+0.25, added to conviction score
    ``inst_opportunity_boost``      0-1, additive boost for opportunity ranking
    ``inst_timing_confidence``      0-1, execution timing quality signal

    Dominant intent encoding
    -------------------------
    ``inst_dominant_intent_id``     0=accumulation 1=distribution 2=transfer
                                    3=stop_hunt 4=liquidity_acq 5=rebalance 6=neutral
    ``inst_intent_certainty``       max_prob − second_max_prob (separation)
    ``inst_direction_alignment``    0=opposing 0.5=neutral 1=aligned with direction

    Structural passthrough signals
    -------------------------------
    ``inst_absorption_strength``    raw structural absorption
    ``inst_range_compression``      raw structural range compression
    ``inst_volume_signature``       raw volume anomaly
    ``inst_sweep_recency``          raw sweep recency

    Metadata
    ---------
    ``inst_confidence``             0-1 based on similar-record count
    ``inst_is_structural``          1.0=pure structural; 0.0=history used
    """

    # Six probabilistic beliefs
    inst_accumulation_prob: float = 0.0
    inst_distribution_prob: float = 0.0
    inst_transfer_prob: float = 0.0
    inst_stop_hunt_prob: float = 0.0
    inst_liquidity_acq_prob: float = 0.0
    inst_rebalance_prob: float = 0.0

    # Dominant intent
    inst_dominant_intent_id: float = 6.0  # neutral default
    inst_intent_certainty: float = 0.0
    inst_direction_alignment: float = 0.5  # neutral

    # Integration signals
    inst_conviction_modifier: float = 0.0
    inst_opportunity_boost: float = 0.0
    inst_timing_confidence: float = 0.0

    # Structural passthrough
    inst_absorption_strength: float = 0.0
    inst_range_compression: float = 0.0
    inst_volume_signature: float = 0.0
    inst_sweep_recency: float = 0.0

    # Metadata
    inst_confidence: float = 0.0
    inst_is_structural: float = 1.0

    def to_dict(self) -> dict:
        return {
            "inst_accumulation_prob": round(self.inst_accumulation_prob, 4),
            "inst_distribution_prob": round(self.inst_distribution_prob, 4),
            "inst_transfer_prob": round(self.inst_transfer_prob, 4),
            "inst_stop_hunt_prob": round(self.inst_stop_hunt_prob, 4),
            "inst_liquidity_acq_prob": round(self.inst_liquidity_acq_prob, 4),
            "inst_rebalance_prob": round(self.inst_rebalance_prob, 4),
            "inst_dominant_intent_id": round(self.inst_dominant_intent_id, 1),
            "inst_intent_certainty": round(self.inst_intent_certainty, 4),
            "inst_direction_alignment": round(self.inst_direction_alignment, 4),
            "inst_conviction_modifier": round(self.inst_conviction_modifier, 4),
            "inst_opportunity_boost": round(self.inst_opportunity_boost, 4),
            "inst_timing_confidence": round(self.inst_timing_confidence, 4),
            "inst_absorption_strength": round(self.inst_absorption_strength, 4),
            "inst_range_compression": round(self.inst_range_compression, 4),
            "inst_volume_signature": round(self.inst_volume_signature, 4),
            "inst_sweep_recency": round(self.inst_sweep_recency, 4),
            "inst_confidence": round(self.inst_confidence, 4),
            "inst_is_structural": round(self.inst_is_structural, 1),
        }


# ── InstitutionalIntentEngine ─────────────────────────────────────────────────


class InstitutionalIntentEngine:
    """Learned multi-symbol institutional intent engine.

    Blends structural priors from the three detectors with historically
    observed outcomes for similar conditions.  Learns over time which
    structural patterns actually preceded each type of institutional
    behaviour for each instrument.

    Multi-symbol: one engine instance manages all watched instruments.
    """

    def __init__(
        self,
        max_records: int = _DEFAULT_MAX_RECORDS,
        k: int = _DEFAULT_K,
        recency_tau_days: float = _RECENCY_TAU_DAYS,
        min_informative: int = _MIN_INFORMATIVE,
    ) -> None:
        self._memory = InstitutionalMemory(max_records=max_records)
        self._acc_det = AccumulationDetector()
        self._dist_det = DistributionDetector()
        self._xfer_det = InventoryTransferDetector()
        self._k = k
        self._tau = recency_tau_days
        self._min = min_informative

    # ── Learning ──────────────────────────────────────────────────────────

    def record_outcome(
        self,
        conditions: InstitutionalConditions,
        outcome: InstitutionalOutcome,
        now: float | None = None,
    ) -> InstitutionalRecord:
        """Append one completed (conditions → intent outcome) observation.

        Called once per completed trade or resolved market event.  Use
        ``InstitutionalOutcome`` fields to label what was observed.
        """
        now = now if now is not None else time.time()
        rec = self._memory.record(conditions.instrument, conditions, outcome, now)
        logger.debug(
            f"[IIE] {conditions.instrument} "
            f"accum={outcome.accumulation_confirmed} "
            f"dist={outcome.distribution_confirmed} "
            f"hunt={outcome.hunt_reversal_confirmed} "
            f"reward={outcome.reward_atr:.3f} ATR recorded."
        )
        return rec

    # ── Scoring ───────────────────────────────────────────────────────────

    def score(
        self,
        conditions: InstitutionalConditions,
        direction: int = 1,
        sweep_events: list[SweepEvent] | None = None,
        now: float | None = None,
    ) -> InstitutionalBeliefs:
        """Compute the full institutional intent belief state.

        Parameters
        ----------
        conditions:
            Current market conditions fingerprint.
        direction:
            +1 = long, −1 = short.  Used to determine direction
            alignment of any detected intent.
        sweep_events:
            Optional list of recent structural sweep events from the
            liquidity layer (``SweepEvent`` descriptors).  Passed to
            the inventory transfer detector.
        now:
            Override current time (for testing / replay).
        """
        now = now if now is not None else time.time()

        # ── Structural priors ─────────────────────────────────────────────
        acc_est = self._acc_det.detect(
            AccumulationInput(
                range_compression=conditions.range_compression,
                absorption_strength=conditions.absorption_strength,
                volume_buy_dominance=conditions.volume_buy_dominance,
                momentum=conditions.momentum_norm * 2.0 - 1.0,  # un-normalize
                session_quality=conditions.session_quality,
                microstructure_quality=conditions.microstructure_quality,
                level_proximity=conditions.level_proximity,
            )
        )
        dist_est = self._dist_det.detect(
            DistributionInput(
                range_compression=conditions.range_compression,
                rejection_strength=conditions.absorption_strength,  # shared signal
                volume_sell_dominance=1.0 - conditions.volume_buy_dominance,
                momentum=conditions.momentum_norm * 2.0 - 1.0,
                session_quality=conditions.session_quality,
                microstructure_quality=conditions.microstructure_quality,
                level_proximity=conditions.level_proximity,
            )
        )
        xfer_est = self._xfer_det.detect(
            InventoryTransferInput(
                sweep_events=sweep_events or [],
                momentum=conditions.momentum_norm * 2.0 - 1.0,
                session_quality=conditions.session_quality,
                microstructure_quality=conditions.microstructure_quality,
                level_proximity=conditions.level_proximity,
                level_proximity_direction=1,  # default: approaching
            )
        )

        # Structural six-belief vector
        s_acc = acc_est.accumulation_prob
        s_dist = dist_est.distribution_prob
        s_xfer = xfer_est.transfer_prob
        s_hunt = xfer_est.stop_hunt_prob
        s_liq = xfer_est.liquidity_acq_prob
        s_rebal = xfer_est.rebalance_prob

        # ── Historical lookup ─────────────────────────────────────────────
        records = self._memory.records(conditions.instrument)
        confidence = 0.0
        h_acc = s_acc
        h_dist = s_dist
        h_xfer = s_xfer
        h_hunt = s_hunt
        h_liq = s_liq
        h_rebal = s_rebal
        n_used = 0

        if records:
            scored = _score_records(conditions.as_vector(), records, self._tau, now)
            top_k = scored[: self._k]
            if top_k:
                n_used = len(top_k)
                confidence = min(1.0, n_used / max(self._min, 1))
                ws = [w for w, _ in top_k]
                recs_k = [r for _, r in top_k]

                h_acc = _wfrac(ws, [r.outcome.accumulation_confirmed for r in recs_k])
                h_dist = _wfrac(ws, [r.outcome.distribution_confirmed for r in recs_k])
                h_hunt = _wfrac(ws, [r.outcome.hunt_reversal_confirmed for r in recs_k])
                h_liq = _wfrac(ws, [r.outcome.liquidity_confirmed for r in recs_k])
                h_dir = _wfrac(ws, [r.outcome.direction_correct for r in recs_k])

                # Transfer and rebalance from directional correctness proxy
                # (no explicit outcome label; infer from reward and direction)
                pos_rewards = [r.outcome.reward_atr > 0 for r in recs_k]
                h_xfer = _wfrac(
                    ws,
                    [
                        r and p
                        for r, p in zip(pos_rewards, [r.outcome.direction_correct for r in recs_k])
                    ],
                )
                h_rebal = 1.0 - max(h_acc, h_dist, h_hunt, h_liq)  # residual

        # ── Blend structural + historical ─────────────────────────────────
        c = confidence
        s = 1.0 - c

        acc = s * s_acc + c * h_acc
        dist = s * s_dist + c * h_dist
        xfer = s * s_xfer + c * h_xfer
        hunt = s * s_hunt + c * h_hunt
        liq = s * s_liq + c * h_liq
        rebal = s * s_rebal + c * h_rebal

        # Clamp all beliefs
        acc, dist, xfer, hunt, liq, rebal = (
            max(0.0, min(1.0, v)) for v in (acc, dist, xfer, hunt, liq, rebal)
        )

        # ── Dominant intent + certainty ───────────────────────────────────
        named = {
            "accumulation": acc,
            "distribution": dist,
            "transfer": xfer,
            "stop_hunt": hunt,
            "liquidity_acq": liq,
            "rebalance": rebal,
        }
        dom_name, dom_prob, second_prob = _dominant(named)
        intent_id = _INTENT_IDS[dom_name]
        intent_cert = max(0.0, dom_prob - second_prob)

        # ── Direction alignment ───────────────────────────────────────────
        dir_align = _direction_alignment(dom_name, acc, dist, hunt, liq, direction)

        # ── Integration signals ───────────────────────────────────────────
        conv_mod = _conviction_modifier(dir_align, dom_prob, confidence)
        opp_boost = _opportunity_boost(dir_align, dom_prob, hunt, acc, liq)
        timing = _timing_confidence(
            hunt,
            acc,
            dist,
            liq,
            xfer,
            rebal,
            acc_est.absorption_strength,
            xfer_est.sweep_recency,
            confidence,
        )

        return InstitutionalBeliefs(
            inst_accumulation_prob=round(acc, 4),
            inst_distribution_prob=round(dist, 4),
            inst_transfer_prob=round(xfer, 4),
            inst_stop_hunt_prob=round(hunt, 4),
            inst_liquidity_acq_prob=round(liq, 4),
            inst_rebalance_prob=round(rebal, 4),
            inst_dominant_intent_id=intent_id,
            inst_intent_certainty=round(intent_cert, 4),
            inst_direction_alignment=round(dir_align, 4),
            inst_conviction_modifier=round(conv_mod, 4),
            inst_opportunity_boost=round(opp_boost, 4),
            inst_timing_confidence=round(timing, 4),
            inst_absorption_strength=round(acc_est.absorption_strength, 4),
            inst_range_compression=round(conditions.range_compression, 4),
            inst_volume_signature=round(xfer_est.volume_signature, 4),
            inst_sweep_recency=round(conditions.sweep_recency, 4),
            inst_confidence=round(confidence, 4),
            inst_is_structural=1.0 if confidence == 0.0 else 0.0,
        )

    def context_for_inference(
        self,
        conditions: InstitutionalConditions,
        direction: int = 1,
        sweep_events: list[SweepEvent] | None = None,
        now: float | None = None,
    ) -> dict:
        """Return exactly 18 ``inst_*`` keys for ``QDEInferenceInput.brain_context``.

        Keys
        ----
        inst_accumulation_prob        0-1: P(accumulation)
        inst_distribution_prob        0-1: P(distribution)
        inst_transfer_prob            0-1: P(inventory transfer)
        inst_stop_hunt_prob           0-1: P(stop hunt)
        inst_liquidity_acq_prob       0-1: P(liquidity acquisition)
        inst_rebalance_prob           0-1: P(position rebalancing)
        inst_dominant_intent_id       float 0-6 encoding of dominant intent
        inst_intent_certainty         max_prob − second_max (separation)
        inst_direction_alignment      0=opposing 0.5=neutral 1=aligned
        inst_conviction_modifier      −0.25..+0.25
        inst_opportunity_boost        0-1
        inst_timing_confidence        0-1
        inst_absorption_strength      structural absorption signal
        inst_range_compression        structural range compression
        inst_volume_signature         volume anomaly signal
        inst_sweep_recency            recency of last structural sweep
        inst_confidence               0-1 based on similar-record count
        inst_is_structural            1.0=pure structural; 0.0=history used
        """
        b = self.score(conditions, direction, sweep_events, now)
        return b.to_dict()

    # ── Persistence ───────────────────────────────────────────────────────

    def to_snapshot(self, instrument: str) -> dict:
        return self._memory.to_snapshot(instrument)

    def load_snapshot(self, instrument: str, data: dict) -> None:
        self._memory.load_snapshot(instrument, data)

    def record_count(self, instrument: str) -> int:
        return self._memory.record_count(instrument)

    def memory(self) -> InstitutionalMemory:
        return self._memory


# ── Internal helpers ──────────────────────────────────────────────────────────


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity ∈ [−1, 1].  Returns 0.0 for zero vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def _score_records(
    current_vec: list[float],
    records: list[InstitutionalRecord],
    tau_days: float,
    now: float,
) -> list[tuple[float, InstitutionalRecord]]:
    """Score records by cosine_sim × recency, sorted highest first."""
    scored: list[tuple[float, InstitutionalRecord]] = []
    for rec in records:
        sim = max(0.0, _cosine_sim(current_vec, rec.conditions.as_vector()))
        recency = rec.recency_weight(now, tau_days)
        w = sim * recency
        if w > 1e-12:
            scored.append((w, rec))

    if not scored:
        for rec in records:
            r = rec.recency_weight(now, tau_days)
            if r > 1e-3:
                scored.append((r, rec))

    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


def _wfrac(weights: list[float], flags: list[bool]) -> float:
    """Weighted fraction of True flags.  Returns 0.0 on empty input."""
    total_w = sum(weights)
    if total_w < 1e-12:
        return 0.0
    pos_w = sum(w for w, f in zip(weights, flags) if f)
    return max(0.0, min(1.0, pos_w / total_w))


def _dominant(
    named: dict[str, float],
) -> tuple[str, float, float]:
    """Return (dominant_intent_name, max_prob, second_max_prob).

    Returns ``("neutral", 0.0, 0.0)`` when all probs are ≤ 0.1.
    """
    pairs = sorted(named.items(), key=lambda p: p[1], reverse=True)
    top_name, top_prob = pairs[0]
    second_prob = pairs[1][1] if len(pairs) > 1 else 0.0
    if top_prob <= 0.10:
        return "neutral", 0.0, 0.0
    return top_name, top_prob, second_prob


def _direction_alignment(
    dom_name: str,
    acc: float,
    dist: float,
    hunt: float,
    liq: float,
    direction: int,
) -> float:
    """0-1 alignment of the dominant intent with the proposed trade direction.

    Alignment rules:

    * ``accumulation`` aligns with long (+1), opposes short (−1)
    * ``distribution`` aligns with short (−1), opposes long (+1)
    * ``stop_hunt`` is aligned: a stop hunt creates the opportunity to
      enter in the reversal direction (we align it with direction)
    * ``liquidity_acq`` is direction-neutral (0.5) — the acquisition
      move precedes the eventual direction, but we don't know which way
    * ``transfer`` and ``rebalance`` are neutral (0.5)

    Returns 0-1 (0=opposing, 0.5=neutral, 1=aligned).
    """
    if dom_name == "accumulation":
        return 1.0 if direction == 1 else 0.0
    if dom_name == "distribution":
        return 1.0 if direction == -1 else 0.0
    if dom_name == "stop_hunt":
        # Stop hunt: the opportunity is to enter after the reversal;
        # alignment is positive for any direction trade (timing-dependent)
        return 0.7
    # transfer, rebalance, liquidity_acq, neutral → weakly neutral
    # Slight positive if any directional intent is non-trivially present
    dir_signal = acc if direction == 1 else dist
    return 0.5 + 0.2 * dir_signal


def _conviction_modifier(
    dir_align: float,
    dom_prob: float,
    confidence: float,
) -> float:
    """−0.25..+0.25 conviction modifier.

    Strong aligned intent at high confidence → +0.25.
    Strong opposing intent at high confidence → −0.25.
    Structural priors (confidence=0) produce a reduced but non-zero
    modifier so that the structural model still influences conviction
    before any history accumulates.

    ``dir_align`` ∈ [0, 1]: mapped to [−1, +1] by 2*(align − 0.5).

    Scale: structural weight is 0.4 of the max; history lifts it to 1.0.
    """
    alignment_signed = 2.0 * (dir_align - 0.5)  # −1..+1
    structural_weight = 0.4  # base modifier from structural alone
    history_weight = structural_weight + (1.0 - structural_weight) * confidence
    raw = alignment_signed * dom_prob * history_weight
    return max(_CONVICTION_MOD_MIN, min(_CONVICTION_MOD_MAX, raw * _CONVICTION_MOD_MAX))


def _opportunity_boost(
    dir_align: float,
    dom_prob: float,
    hunt_prob: float,
    acc_prob: float,
    liq_prob: float,
) -> float:
    """0-1 opportunity boost for cross-symbol ranker.

    Boost is highest when:
    * A confirmed stop hunt is present (creates the entry opportunity)
    * Strong accumulation aligns with direction (best timing to go long)
    * Liquidity acquisition is underway (about to reach the target zone)

    Returns 0 when dir_align < 0.5 (opposing intent detected).
    """
    if dir_align < 0.5:
        return 0.0  # opposing institutional intent → no boost
    alignment_weight = 2.0 * (dir_align - 0.5)  # 0..1
    # Hunt reversal is the highest-conviction entry signal
    hunt_component = hunt_prob * 0.50
    # Accumulation alignment
    acc_component = acc_prob * 0.30
    # Liquidity acquisition approaching
    liq_component = liq_prob * 0.20
    raw_boost = (hunt_component + acc_component + liq_component) * alignment_weight
    return max(0.0, min(1.0, raw_boost))


def _timing_confidence(
    hunt: float,
    acc: float,
    dist: float,
    liq: float,
    xfer: float,
    rebal: float,
    absorption: float,
    sweep_rec: float,
    confidence: float,
) -> float:
    """0-1 execution timing confidence.

    High when structural signals indicate a specific, identifiable
    inflection point:

    * Confirmed stop-hunt sweep + reversal → highest timing signal
      (the reversal is the entry trigger)
    * Peak absorption (absorption near max) + accumulation high → good
      timing (accumulation complete, expansion about to begin)
    * Liquidity acquisition reaching a zone → pre-zone entry timing
    * Distribution / transfer / rebalancing → low timing confidence
      (no clear inflection point)

    Scaled by confidence so structural-only estimates stay modest.
    """
    # Hunt reversal: sweep_recency AND hunt probability → immediate timing
    hunt_timing = sweep_rec * hunt

    # Absorption peak: high absorption + high accumulation or distribution
    phase_timing = absorption * max(acc, dist)

    # Liquidity zone approach: elevated as price nears the zone
    liq_timing = liq * 0.6

    # Noise from transfer/rebalance: no clear timing signal
    noise_penalty = xfer * 0.3 + rebal * 0.2

    raw = max(hunt_timing, phase_timing, liq_timing) - noise_penalty * 0.2
    # Scale by confidence: structural only → more modest signal
    scale = 0.5 + 0.5 * confidence
    return max(0.0, min(1.0, raw * scale))
