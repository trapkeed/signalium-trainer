"""QDE Brain — regime transition: top-level orchestrator.

``RegimeTransitionEngine`` is the single entry point for the rest of the
Brain.  It owns one ``TransitionMemory`` and one ``TransitionPredictor``
for all symbols and exposes five integration points:

1. **Ingestion** — ``on_observation(instrument, inp)`` records a tick's
   feature snapshot.  Call once per ``SymbolObservationInput`` received.

2. **Confirmed transition** — ``record_transition(instrument, from_regime,
   to_regime)`` is called by the upstream pipeline (e.g.
   ``BrainStateManager``) when the MIE regime classifier detects a label
   change.  The engine immediately computes and caches the pre-transition
   feature vector so future prediction calls are fast.

3. **Inference context** — ``context_for_inference(instrument)`` returns a
   compact dict with ``regime_*`` keys for merging into
   ``QDEInferenceInput.brain_context``.

4. **Conviction modifier** — ``conviction_modifier(instrument)`` returns a
   signed ±0.20 adjustment.  High-entropy predictions (no clear transition
   signal) push toward −0.10 (regime uncertainty); a concentrated signal
   in either direction pushes toward the range edge whose magnitude scales
   with the predicted transition's severity.

5. **Attention signal** — ``attention_signal(instrument)`` returns 0-1.
   Callers add this to ``OpportunitySignals.unusual_behavior`` to elevate
   a symbol's attention score when a regime change looks imminent.

6. **Risk appetite modifier** — ``risk_appetite_modifier(instrument)``
   returns −1..+1.  ``VOL_EXPANSION`` predictions push negative (tighten
   risk); ``VOL_CONTRACTION`` and ``RANGE_FORMATION`` push positive
   (loosen risk).  Directional trend signals are near-zero here (they are
   handled via conviction, not sizing).

Design rules
------------
* Does not call ``MarketRegimeEngine`` — the already-classified
  ``market_regime`` string from ``SymbolObservationInput`` is used as-is.
* Does not import ``QDEDecision`` or any execution-layer type.
* No hardcoded probability thresholds or slope cutoffs — all integration
  signals are linear or affine functions of the ``TransitionProbabilityMap``
  outputs so threshold sensitivity is fully delegated to the learned
  distribution.
"""

from __future__ import annotations

import time

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.qde.brain.regime.transition_memory import (
    RegimeSnapshot,
    TransitionMemory,
    TransitionTarget,
)
from trainer.qde.brain.regime.transition_predictor import (
    _DEFAULT_DECAY_TAU_HOURS,
    _DEFAULT_WINDOW,
    _MIN_INFORMATIVE,
    TransitionPredictor,
    TransitionProbabilityMap,
)

# ── Conviction modifier bounds ────────────────────────────────────────────────

_MAX_CONVICTION_BOOST = 0.20  # cap on positive modifier (concentrated signal)
_MAX_CONVICTION_DRAG = 0.10  # cap on negative modifier (high uncertainty)

# Targets that each directionally increase or decrease conviction:
# "trending" targets = clearer environment → small boost when dominant
_BULLISH_CONVICTION_TARGETS = {
    TransitionTarget.TREND_FORMATION,
    TransitionTarget.BREAKOUT,
    TransitionTarget.VOL_CONTRACTION,  # calmer → easier to hold conviction
}
_BEARISH_CONVICTION_TARGETS = {
    TransitionTarget.TREND_EXHAUSTION,
    TransitionTarget.VOL_EXPANSION,  # chaotic → harder to hold conviction
}

# ── Risk appetite modifier bounds ─────────────────────────────────────────────

_MAX_RISK_DRAG = -0.40  # VOL_EXPANSION dominant → tighten significantly
_MAX_RISK_BOOST = 0.20  # VOL_CONTRACTION dominant → loosen slightly

# ── Attention signal targets (any imminent transition is worth watching) ──────

_HIGH_ATTENTION_TARGETS = set(TransitionTarget)  # all — any shift is salient


# ── Default prediction cache sentinel ─────────────────────────────────────────


def _empty_probs() -> dict[str, float]:
    n = len(TransitionTarget)
    return {t.value: 1.0 / n for t in TransitionTarget}


# ── RegimeTransitionEngine ────────────────────────────────────────────────────


class RegimeTransitionEngine:
    """Per-symbol regime-transition forecasting engine.

    Holds ``TransitionMemory`` + ``TransitionPredictor`` and provides all
    five integration points for the QDE brain pipeline.

    Thread safety: not thread-safe.  Callers that share one instance across
    threads should serialise ``on_observation`` / ``record_transition``
    calls externally.
    """

    def __init__(
        self,
        snapshot_depth: int = 200,
        max_transitions: int = 500,
        window: int = _DEFAULT_WINDOW,
        recency_tau_hours: float = _DEFAULT_DECAY_TAU_HOURS,
        min_samples: int = _MIN_INFORMATIVE,
    ) -> None:
        self._memory = TransitionMemory(
            snapshot_depth=snapshot_depth,
            max_transitions=max_transitions,
        )
        self._predictor = TransitionPredictor()
        self._window = window
        self._tau = recency_tau_hours
        self._min = min_samples

        # Prediction cache: instrument → (timestamp, TransitionProbabilityMap)
        # Re-computed lazily on the next predict() call after a new snapshot
        # or transition arrives.
        self._cache: dict[str, tuple[float, TransitionProbabilityMap]] = {}
        self._dirty: set[str] = set()  # instruments whose cache is stale

    # ── Ingestion ─────────────────────────────────────────────────────────

    def on_observation(
        self,
        instrument: str,
        inp: object,  # SymbolObservationInput
        now: float | None = None,
    ) -> None:
        """Record one tick's features for ``instrument``.

        ``inp`` is a ``SymbolObservationInput``; typed as ``object`` to
        avoid a hard import in this package — the duck-typed attribute
        access matches the real class exactly.
        """
        now = now if now is not None else time.time()
        snap = RegimeSnapshot.from_observation(instrument, inp, now)
        self._memory.push_snapshot(instrument, snap)
        self._dirty.add(instrument)

    def record_transition(
        self,
        instrument: str,
        from_regime: str,
        to_regime: str,
        target: TransitionTarget | str | None = None,
        now: float | None = None,
    ) -> None:
        """Notify the engine that the upstream classifier detected a regime
        change for ``instrument``.

        Immediately computes and caches the pre-transition feature vector
        on the new ``TransitionRecord`` so prediction calls stay O(K)
        where K = number of historical records, not O(K × window).

        ``target`` is optional: if omitted (or ``None``), the engine infers
        it from ``(from_regime, to_regime)`` via ``infer_target``.
        """
        now = now if now is not None else time.time()

        if isinstance(target, str):
            try:
                target = TransitionTarget(target)
            except ValueError:
                target = None

        record = self._memory.record_transition(instrument, from_regime, to_regime, target, now)

        # Cache the feature vector on the record immediately
        snaps = self._memory.snapshots(instrument)
        fv = self._predictor.extract_feature_vector(snaps, desired_window=self._window)
        record.feature_vector = fv.as_list()

        self._dirty.add(instrument)
        logger.debug(
            f"[RTE] {instrument}: {from_regime} → {to_regime} " f"({record.target.value}) recorded"
        )

    # ── Prediction ────────────────────────────────────────────────────────

    def predict(
        self,
        instrument: str,
        now: float | None = None,
    ) -> TransitionProbabilityMap:
        """Return the current ``TransitionProbabilityMap`` for ``instrument``.

        Results are cached per instrument and recomputed only when new
        data has arrived (marked by ``on_observation`` or
        ``record_transition``).
        """
        now = now if now is not None else time.time()

        if instrument not in self._dirty:
            cached = self._cache.get(instrument)
            if cached is not None:
                return cached[1]

        prob_map = self._predictor.predict(
            instrument=instrument,
            memory=self._memory,
            window=self._window,
            recency_tau_hours=self._tau,
            min_samples=self._min,
            now=now,
        )
        self._cache[instrument] = (now, prob_map)
        self._dirty.discard(instrument)
        return prob_map

    # ── Integration point 1: inference context ────────────────────────────

    def context_for_inference(
        self,
        instrument: str,
        now: float | None = None,
    ) -> dict:
        """Return ``regime_*`` keys for ``QDEInferenceInput.brain_context``.

        All values are scalars or short strings — no nested objects.

        Keys
        ----
        regime_current               — most recently observed regime label
        regime_transition_count      — total confirmed transitions seen
        regime_last_transition_age_h — hours since last transition (−1 if none)
        regime_top_target            — TransitionTarget with highest probability
        regime_top_prob              — probability of top target (0-1)
        regime_entropy               — distribution entropy (0=concentrated, 1=uniform)
        regime_confidence            — how many historical samples inform the estimate
        regime_trend_form_prob       — P(TREND_FORMATION)
        regime_trend_exh_prob        — P(TREND_EXHAUSTION)
        regime_vol_expand_prob       — P(VOL_EXPANSION)
        regime_vol_contract_prob     — P(VOL_CONTRACTION)
        regime_breakout_prob         — P(BREAKOUT)
        regime_range_form_prob       — P(RANGE_FORMATION)
        regime_sample_count          — number of historical transitions used
        regime_atr_slope             — ATR slope from current window (raw units)
        regime_spread_slope          — spread slope from current window
        regime_momentum_slope        — momentum slope from current window
        regime_vol_rank_slope        — vol-state-rank slope from current window
        regime_trend_valid_ratio     — fraction of recent ticks with trend_valid
        regime_atr_cv                — ATR coefficient of variation (compression signal)
        """
        now = now if now is not None else time.time()
        pm = self.predict(instrument, now)
        probs = pm.probabilities
        fv = pm.feature_vector

        # Last transition age
        last = self._memory.last_transition(instrument)
        age_h = last.age_hours(now) if last else -1.0

        ctx: dict = {
            "regime_current": self._memory.current_regime(instrument),
            "regime_transition_count": self._memory.transition_count(instrument),
            "regime_last_transition_age_h": round(age_h, 2),
            "regime_top_target": pm.top_target,
            "regime_top_prob": round(pm.top_probability, 4),
            "regime_entropy": round(pm.entropy, 4),
            "regime_confidence": round(pm.confidence, 4),
            "regime_sample_count": pm.sample_count,
            # Per-target probabilities
            "regime_trend_form_prob": round(
                probs.get(TransitionTarget.TREND_FORMATION.value, 0.0), 4
            ),
            "regime_trend_exh_prob": round(
                probs.get(TransitionTarget.TREND_EXHAUSTION.value, 0.0), 4
            ),
            "regime_vol_expand_prob": round(
                probs.get(TransitionTarget.VOL_EXPANSION.value, 0.0), 4
            ),
            "regime_vol_contract_prob": round(
                probs.get(TransitionTarget.VOL_CONTRACTION.value, 0.0), 4
            ),
            "regime_breakout_prob": round(probs.get(TransitionTarget.BREAKOUT.value, 0.0), 4),
            "regime_range_form_prob": round(
                probs.get(TransitionTarget.RANGE_FORMATION.value, 0.0), 4
            ),
        }

        # Feature vector components (raw, for model use)
        if fv is not None:
            ctx.update(
                {
                    "regime_atr_slope": round(fv.atr_slope, 6),
                    "regime_spread_slope": round(fv.spread_slope, 6),
                    "regime_momentum_slope": round(fv.momentum_slope, 6),
                    "regime_vol_rank_slope": round(fv.vol_rank_slope, 6),
                    "regime_trend_valid_ratio": round(fv.trend_valid_ratio, 4),
                    "regime_atr_cv": round(fv.atr_cv, 6),
                }
            )
        else:
            ctx.update(
                {
                    "regime_atr_slope": 0.0,
                    "regime_spread_slope": 0.0,
                    "regime_momentum_slope": 0.0,
                    "regime_vol_rank_slope": 0.0,
                    "regime_trend_valid_ratio": 0.5,
                    "regime_atr_cv": 0.0,
                }
            )

        return ctx

    # ── Integration point 2: conviction modifier ──────────────────────────

    def conviction_modifier(
        self,
        instrument: str,
        now: float | None = None,
    ) -> float:
        """Return a signed ±0.20 conviction adjustment.

        The modifier is a linear combination of the transition probability
        mass on "environment-clarifying" targets (positive) and
        "environment-chaotic" targets (negative), scaled by confidence
        and modulated by distribution entropy.

        High entropy (no clear signal) → pulls toward −``_MAX_CONVICTION_DRAG``.
        Concentrated signal in a trending direction → toward +``_MAX_CONVICTION_BOOST``.
        Concentrated signal in an exhaustion/vol-expansion direction →
            toward −``_MAX_CONVICTION_BOOST``.
        """
        now = now if now is not None else time.time()
        pm = self.predict(instrument, now)
        p = pm.probabilities

        bull_mass = sum(p.get(t.value, 0.0) for t in _BULLISH_CONVICTION_TARGETS)
        bear_mass = sum(p.get(t.value, 0.0) for t in _BEARISH_CONVICTION_TARGETS)
        net_signal = bull_mass - bear_mass  # −1..+1

        # Scale by concentration (1 − entropy) and by data confidence
        concentration = 1.0 - pm.entropy  # 0 = uniform, 1 = spike
        confidence = pm.confidence

        # Without data, entropy=1 → concentration=0 → modifier is only the
        # uncertainty drag, which is also zero (confidence=0). Net = 0.0.
        if confidence < 1e-6:
            return 0.0

        # Positive net → boost (up to _MAX_CONVICTION_BOOST)
        # Negative net → drag (up to _MAX_CONVICTION_DRAG)
        if net_signal >= 0:
            modifier = net_signal * concentration * confidence * _MAX_CONVICTION_BOOST
        else:
            modifier = net_signal * concentration * confidence * _MAX_CONVICTION_BOOST
            # Asymmetric cap: drag is smaller than boost (uncertainty is softer)
            modifier = max(modifier, -_MAX_CONVICTION_DRAG)

        # Extra entropy drag: even positive-signal predictions are slightly
        # penalised if the distribution is very flat
        entropy_drag = pm.entropy * _MAX_CONVICTION_DRAG * 0.5 * confidence
        modifier -= entropy_drag

        return round(max(-_MAX_CONVICTION_BOOST, min(_MAX_CONVICTION_BOOST, modifier)), 4)

    # ── Integration point 3: attention signal ─────────────────────────────

    def attention_signal(
        self,
        instrument: str,
        now: float | None = None,
    ) -> float:
        """Return a 0-1 salience score for the attention allocator.

        High when: a single transition target dominates AND confidence is
        reasonable AND the predicted target is imminent relative to the
        last observed regime stability.

        Callers add this to ``OpportunitySignals.unusual_behavior`` or
        pass it directly to the ``AttentionScoreAllocator``.
        """
        now = now if now is not None else time.time()
        pm = self.predict(instrument, now)

        # Raw salience = concentration × confidence
        concentration = 1.0 - pm.entropy
        salience = concentration * pm.confidence

        # Recency of last transition: if a transition was very recent, this
        # symbol is already in a transition — attention should be high.
        last = self._memory.last_transition(instrument)
        if last is not None:
            age_h = last.age_hours(now)
            # Decay from 1.0 at age=0h to ~0.14 at age=48h (half-life ≈ 20h)
            recent_boost = max(0.0, 1.0 - age_h / 48.0)
            salience = min(1.0, salience + 0.3 * recent_boost)

        return round(min(1.0, max(0.0, salience)), 4)

    # ── Integration point 4: risk appetite modifier ───────────────────────

    def risk_appetite_modifier(
        self,
        instrument: str,
        now: float | None = None,
    ) -> float:
        """Return a −1..+1 risk appetite adjustment.

        Negative = tighten risk (reduce position sizes / wider stops less
        appropriate).  Positive = looser risk environment.

        ``VOL_EXPANSION`` dominant → pushes toward ``_MAX_RISK_DRAG``
            (−0.40 at full confidence and concentration).
        ``VOL_CONTRACTION`` dominant → pushes toward ``_MAX_RISK_BOOST``
            (+0.20).
        ``TREND_FORMATION`` / ``BREAKOUT`` → small positive.
        ``TREND_EXHAUSTION`` → small negative.
        Other targets → near zero (not a sizing signal on their own).

        All modifiers are scaled by confidence and concentration so the
        output is near zero until the engine has observed enough history
        to be informative.
        """
        now = now if now is not None else time.time()
        pm = self.predict(instrument, now)
        p = pm.probabilities

        scale = pm.confidence * (1.0 - pm.entropy * 0.5)

        vol_expand = p.get(TransitionTarget.VOL_EXPANSION.value, 0.0)
        vol_contract = p.get(TransitionTarget.VOL_CONTRACTION.value, 0.0)
        trend_form = p.get(TransitionTarget.TREND_FORMATION.value, 0.0)
        breakout = p.get(TransitionTarget.BREAKOUT.value, 0.0)
        trend_exh = p.get(TransitionTarget.TREND_EXHAUSTION.value, 0.0)

        raw = (
            vol_contract * abs(_MAX_RISK_BOOST)  # calmer → loosen
            + trend_form * 0.10  # trending emerging → slight positive
            + breakout * 0.10  # breakout → slight positive
            - vol_expand * abs(_MAX_RISK_DRAG)  # expansion → tighten strongly
            - trend_exh * 0.10  # exhaustion → tighten slightly
        )
        raw = raw * scale

        return round(max(_MAX_RISK_DRAG, min(_MAX_RISK_BOOST, raw)), 4)

    # ── Accessors ─────────────────────────────────────────────────────────

    def map_for(self, instrument: str) -> TransitionMemory:
        """Return the underlying ``TransitionMemory`` (read-only contract)."""
        return self._memory

    def current_regime(self, instrument: str) -> str:
        return self._memory.current_regime(instrument)

    def transition_count(self, instrument: str) -> int:
        return self._memory.transition_count(instrument)

    # ── Persistence ───────────────────────────────────────────────────────

    def to_snapshot(self, instrument: str) -> dict:
        return self._memory.to_snapshot(instrument)

    def load_snapshot(self, instrument: str, data: dict) -> None:
        self._memory.load_snapshot(instrument, data)
        self._dirty.add(instrument)
        # Rebuild feature vectors on cached transition records that were
        # loaded without them (serialised records store the vector, so
        # this is a safety pass for any that were saved before the field
        # was added or that had empty vectors).
        for rec in self._memory.transitions(instrument):
            if not rec.feature_vector:
                fv = self._predictor.extract_feature_vector(
                    rec.pre_snapshots, desired_window=self._window
                )
                rec.feature_vector = fv.as_list()
