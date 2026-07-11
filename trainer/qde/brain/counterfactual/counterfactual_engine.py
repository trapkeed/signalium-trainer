"""C16 Counterfactual Consciousness — engine and public callables.

Three public callables (spec):

    generate_alternatives(primary_thesis, secondary_thesis, uncertainty_score)
        → list[CounterfactualScenario]

    evaluate_alternative(scenario, new_evidence, current_uncertainty)
        → float  (updated probability weight, 0-1)

    switch_thesis(state, trigger, new_primary_side, new_primary_thesis,
                  new_secondary_thesis, ...)
        → ThesisSwitchResult

``CounterfactualEngine`` — stateful wrapper (one per instrument).

Design
------
All three callables are near-pure (modulo timestamp).  The stateful
``CounterfactualEngine`` holds a ``CounterfactualState`` per instrument
and delegates to the module-level functions.

generate_alternatives
----------------------
Seeds 2-4 scenarios from the primary thesis's ``conflicting_evidence``
(already computed by ``ThesisManager``) — no evidence re-derivation.
Generates:

    REVERSAL      — the opposing direction (seeded by conflicting items)
    SIDEWAYS      — neither direction resolves (uncertainty-weighted)
    ACCELERATION  — same direction as primary, amplified
    EXHAUSTION    — same direction winding down (only when primary strong)

Probability weights are priors adjusted by:
    * primary thesis strength (higher primary → lower reversal P)
    * secondary thesis strength (higher secondary → higher reversal P)
    * uncertainty score (higher uncertainty → higher sideways P)

evaluate_alternative
---------------------
Scores how well new evidence supports a scenario:

    score = match_ratio * (1 - invalidation_ratio) * scenario.probability_weight

Where:
    match_ratio         = matches(trigger_conditions ∩ new_evidence) /
                          max(1, len(trigger_conditions))
    invalidation_ratio  = matches(invalidation_signals ∩ new_evidence) /
                          max(1, len(invalidation_signals))

Substring matching is used for natural-language evidence strings.
Updated probability_weight is not mutated on the immutable
``CounterfactualScenario``; the score is returned as a float the caller
uses to rank / filter scenarios.

switch_thesis
--------------
1. Extract ``FlipLesson`` from the old primary's conflicting_evidence.
2. Build ``ThesisFlipRecord``.
3. Detect reversal: if the new primary side == the previous previous
   primary side AND elapsed < REVERSAL_WINDOW_SECONDS.
4. Record ``ThesisReversalRecord`` if reversal detected.
5. Generate new ``CounterfactualScenario`` list for new primary thesis.
6. Append ``ProbabilityPoint`` to evolution timeline.
7. Return ``ThesisSwitchResult``.

Probability evolution / uncertainty spike detection
----------------------------------------------------
``record_probability_point()`` is called whenever primary/secondary
strengths are sampled.  It appends a ``ProbabilityPoint`` and, if the
uncertainty delta vs the previous point exceeds ``UNCERTAINTY_SPIKE_DELTA``,
records an ``UncertaintySpikeRecord``.

Recommended switch detection
-----------------------------
``recommend_switch()`` returns True when:
    secondary_p > primary_p + SWITCH_MARGIN   (secondary clearly stronger), OR
    primary_p   < PRIMARY_CONFIDENCE_FLOOR     (primary collapsed below floor)
"""

from __future__ import annotations

import time
import uuid

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.qde.brain.brain_types import BeliefDirection
from trainer.qde.brain.counterfactual.counterfactual_types import (
    MAX_FLIP_RECORDS,
    MAX_PROBABILITY_POINTS,
    MAX_REVERSAL_RECORDS,
    MAX_SPIKE_RECORDS,
    PRIMARY_CONFIDENCE_FLOOR,
    REVERSAL_WINDOW_SECONDS,
    SWITCH_MARGIN,
    UNCERTAINTY_SPIKE_DELTA,
    CounterfactualScenario,
    CounterfactualState,
    FlipLesson,
    FlipTrigger,
    ProbabilityPoint,
    ScenarioType,
    ThesisFlipRecord,
    ThesisReversalRecord,
    ThesisSwitchResult,
    UncertaintySpikeRecord,
    _SCENARIO_BASE_WEIGHTS,
)
from trainer.qde.brain.thesis_manager import Thesis


# ── Internal helpers ──────────────────────────────────────────────────────────


def _make_id(prefix: str = "") -> str:
    return f"{prefix}{str(uuid.uuid4())[:8]}"


def _opposite_direction(side: BeliefDirection) -> BeliefDirection:
    if side == BeliefDirection.BULLISH:
        return BeliefDirection.BEARISH
    if side == BeliefDirection.BEARISH:
        return BeliefDirection.BULLISH
    return BeliefDirection.NEUTRAL


def _evidence_matches(evidence_list: list[str], target_items: list[str]) -> int:
    """Count how many items in ``target_items`` appear (substring) in any
    evidence string in ``evidence_list``.  Case-insensitive.
    """
    matches = 0
    for target in target_items:
        tl = target.lower()
        if any(tl in ev.lower() for ev in evidence_list):
            matches += 1
    return matches


def _extract_flip_lesson(
    old_primary_conflicting: list[str],
    trigger_detail:          str,
    state:                   CounterfactualState,
    now:                     float,
) -> FlipLesson:
    """Derive a ``FlipLesson`` from the conflicting evidence that was
    present in the old primary thesis before the flip.

    All items in ``old_primary_conflicting`` are "ignored warnings" —
    signals that contradicted the primary thesis but were not acted on.
    ``time_to_flip`` is the elapsed time since the *previous* flip (or
    state creation if no prior flip) — a proxy for how long warnings
    were present before the switch was triggered.  Clamped to ≥ 0 so
    that synthetic timestamps earlier than ``state.created_at`` do not
    produce negative values.
    """
    reference_time = state.last_flip_at or state.created_at
    time_to_flip = max(0.0, now - reference_time)
    could_have_switched_earlier = (
        bool(old_primary_conflicting)
        and time_to_flip >= REVERSAL_WINDOW_SECONDS
    )
    return FlipLesson(
        ignored_warnings=list(old_primary_conflicting),
        key_trigger=trigger_detail or "unknown",
        time_to_flip=round(time_to_flip, 1),
        could_have_switched_earlier=could_have_switched_earlier,
    )


def _build_reversal_scenario(
    primary_side:       BeliefDirection,
    conflicting_items:  list[str],
    reversal_weight:    float,
    now:                float,
) -> CounterfactualScenario:
    """Generate a REVERSAL scenario from conflicting evidence items."""
    opposite = _opposite_direction(primary_side)
    trigger_conditions = [
        f"price breaks structure in {opposite.value.lower()} direction",
        "momentum divergence confirms direction shift",
    ]
    if conflicting_items:
        trigger_conditions = list(conflicting_items[:3]) + trigger_conditions[:1]
    return CounterfactualScenario(
        scenario_id=_make_id("rev_"),
        scenario_type=ScenarioType.REVERSAL,
        direction=opposite,
        trigger_conditions=trigger_conditions[:4],
        probability_weight=round(max(0.05, min(0.70, reversal_weight)), 4),
        expected_observations=[
            f"sequence of {opposite.value.lower()} candles with volume",
            "key support/resistance level breached",
            "opposing thesis conflict evidence accumulating",
        ],
        invalidation_signals=[
            f"price holds {primary_side.value.lower()} structure",
            "volume fails to confirm directional move",
        ],
        confidence_if_correct=round(min(0.85, reversal_weight + 0.40), 4),
        source_evidence=list(conflicting_items[:5]),
        created_at=now,
    )


def _build_sideways_scenario(
    primary_side:    BeliefDirection,
    uncertainty:     float,
    sideways_weight: float,
    now:             float,
) -> CounterfactualScenario:
    """Generate a SIDEWAYS/consolidation scenario, weighted by uncertainty."""
    return CounterfactualScenario(
        scenario_id=_make_id("sw_"),
        scenario_type=ScenarioType.SIDEWAYS,
        direction=BeliefDirection.NEUTRAL,
        trigger_conditions=[
            "price oscillates within defined range",
            "volume contracts, no breakout momentum",
            "both bull and bear evidence roughly equal",
        ],
        probability_weight=round(max(0.05, min(0.50, sideways_weight)), 4),
        expected_observations=[
            "narrowing price range",
            "declining ATR",
            "equal closes on both sides of range midpoint",
        ],
        invalidation_signals=[
            f"decisive breakout in {primary_side.value.lower()} direction with volume",
            "expansion of ATR beyond range",
        ],
        confidence_if_correct=round(0.55 + uncertainty * 0.15, 4),
        source_evidence=[],
        created_at=now,
    )


def _build_acceleration_scenario(
    primary_side:   BeliefDirection,
    primary_p:      float,
    accel_weight:   float,
    now:            float,
) -> CounterfactualScenario:
    """Generate an ACCELERATION scenario — same direction, faster."""
    return CounterfactualScenario(
        scenario_id=_make_id("acc_"),
        scenario_type=ScenarioType.ACCELERATION,
        direction=primary_side,
        trigger_conditions=[
            f"strong {primary_side.value.lower()} momentum with expanding volume",
            "momentum indicator at multi-period high",
            "no meaningful pullback / consolidation",
        ],
        probability_weight=round(max(0.05, min(0.50, accel_weight)), 4),
        expected_observations=[
            "series of impulsive bars",
            "volume > 1.5× average",
            "price extension beyond normal distribution",
        ],
        invalidation_signals=[
            "sharp pullback > 50% of prior move",
            "volume collapses on continuation attempt",
        ],
        confidence_if_correct=round(min(0.90, primary_p + 0.25), 4),
        source_evidence=[],
        created_at=now,
    )


def _build_exhaustion_scenario(
    primary_side:    BeliefDirection,
    primary_p:       float,
    exhaust_weight:  float,
    now:             float,
) -> CounterfactualScenario:
    """Generate an EXHAUSTION scenario — same direction, winding down."""
    return CounterfactualScenario(
        scenario_id=_make_id("ex_"),
        scenario_type=ScenarioType.EXHAUSTION,
        direction=primary_side,
        trigger_conditions=[
            "momentum divergence on continuation attempt",
            "volume declining on new extremes",
            "price approaching major structural level",
        ],
        probability_weight=round(max(0.05, min(0.40, exhaust_weight)), 4),
        expected_observations=[
            "smaller-range candles at extremes",
            "RSI/momentum divergence",
            "market depth thins at extension",
        ],
        invalidation_signals=[
            f"fresh {primary_side.value.lower()} momentum with volume expansion",
            "price breaks to new extreme with breadth",
        ],
        confidence_if_correct=round(min(0.75, primary_p - 0.10), 4),
        source_evidence=[],
        created_at=now,
    )


def _compute_scenario_weights(
    primary_p:   float,
    secondary_p: float,
    uncertainty: float,
) -> dict[ScenarioType, float]:
    """Adjust base scenario weights by current market context.

    Higher primary → lower reversal P.
    Higher secondary → higher reversal P.
    Higher uncertainty → higher sideways P at expense of continuation.
    """
    base = dict(_SCENARIO_BASE_WEIGHTS)

    # Reversal weight: pulled up by secondary strength, pushed down by primary strength
    base[ScenarioType.REVERSAL] = round(
        base[ScenarioType.REVERSAL]
        * (1 + secondary_p - primary_p),
        4,
    )
    # Sideways weight: elevated by uncertainty
    base[ScenarioType.SIDEWAYS] = round(
        base[ScenarioType.SIDEWAYS] + 0.30 * uncertainty,
        4,
    )
    # Acceleration weight: elevated by primary strength
    base[ScenarioType.ACCELERATION] = round(
        base[ScenarioType.ACCELERATION] * (0.5 + primary_p),
        4,
    )
    # Exhaustion: only meaningful when primary is already strong
    base[ScenarioType.EXHAUSTION] = round(
        base[ScenarioType.EXHAUSTION] * primary_p,
        4,
    )
    # Clamp all to [0.05, 0.70]
    for k in base:
        base[k] = max(0.05, min(0.70, base[k]))
    return base


# ── Public callable: generate_alternatives ───────────────────────────────────


def generate_alternatives(
    primary_thesis:   Thesis,
    secondary_thesis: Thesis | None = None,
    uncertainty_score: float = 0.0,
    *,
    _state:           CounterfactualState | None = None,
    now:              float | None = None,
) -> list[CounterfactualScenario]:
    """Generate alternative counterfactual scenarios for the primary thesis.

    Asks "what if I am wrong?" and produces 2-4 concrete scenarios the
    QDE should monitor alongside its primary commitment.

    The REVERSAL scenario is seeded directly from
    ``primary_thesis.conflicting_evidence`` — items that are already
    contradicting the primary thesis are the natural seeds for the
    alternative scenario.

    Parameters
    ----------
    primary_thesis:
        The currently committed thesis (``Thesis`` from ``ThesisManager``).
    secondary_thesis:
        The opposing thesis.  When supplied, its strength informs the
        reversal probability weight.
    uncertainty_score:
        Composite uncertainty (0-1) from ``UncertaintyEngine`` or any
        other source.  Elevated uncertainty increases sideways scenario P.
    _state:
        Optional per-instrument state; if provided, the generated
        scenarios are stored on ``_state.current_scenarios``.
    now:
        Override current timestamp.

    Returns
    -------
    list[CounterfactualScenario] — 2-4 scenarios ordered:
        REVERSAL, SIDEWAYS, ACCELERATION, EXHAUSTION (conditional).
    """
    now = now if now is not None else time.time()
    primary_p   = primary_thesis.strength()
    secondary_p = secondary_thesis.strength() if secondary_thesis is not None else 0.40

    weights = _compute_scenario_weights(primary_p, secondary_p, uncertainty_score)

    scenarios: list[CounterfactualScenario] = []

    # 1. REVERSAL — always generated; highest priority alternative
    scenarios.append(
        _build_reversal_scenario(
            primary_side=primary_thesis.side,
            conflicting_items=list(primary_thesis.conflicting_evidence),
            reversal_weight=weights[ScenarioType.REVERSAL],
            now=now,
        )
    )

    # 2. SIDEWAYS — always generated; elevated under uncertainty
    scenarios.append(
        _build_sideways_scenario(
            primary_side=primary_thesis.side,
            uncertainty=uncertainty_score,
            sideways_weight=weights[ScenarioType.SIDEWAYS],
            now=now,
        )
    )

    # 3. ACCELERATION — when primary is reasonably strong
    if primary_p >= 0.45:
        scenarios.append(
            _build_acceleration_scenario(
                primary_side=primary_thesis.side,
                primary_p=primary_p,
                accel_weight=weights[ScenarioType.ACCELERATION],
                now=now,
            )
        )

    # 4. EXHAUSTION — only when primary is already at high conviction
    if primary_p >= 0.60:
        scenarios.append(
            _build_exhaustion_scenario(
                primary_side=primary_thesis.side,
                primary_p=primary_p,
                exhaust_weight=weights[ScenarioType.EXHAUSTION],
                now=now,
            )
        )

    if _state is not None:
        _state.current_scenarios = scenarios
        _state.last_updated_at   = now

    return scenarios


# ── Public callable: evaluate_alternative ────────────────────────────────────


def evaluate_alternative(
    scenario:           CounterfactualScenario,
    new_evidence:       list[str],
    current_uncertainty: float = 0.0,
    *,
    _state:             CounterfactualState | None = None,
) -> float:
    """Score how well ``new_evidence`` supports the given scenario.

    Returns an updated probability weight in [0, 1].  This is NOT stored
    back onto the immutable ``CounterfactualScenario``; callers should
    use the returned value to rank / filter scenarios.

    Scoring formula
    ---------------
        match_ratio        = evidence_matches(trigger_conditions) /
                             max(1, len(trigger_conditions))
        invalidation_ratio = evidence_matches(invalidation_signals) /
                             max(1, len(invalidation_signals))
        raw_score          = match_ratio × (1 − invalidation_ratio)
        updated_weight     = 0.60 × scenario.probability_weight
                           + 0.40 × raw_score

    The 60/40 blend means new evidence can move the scenario's weight but
    not completely override the structural prior.

    Uncertainty bonus: SIDEWAYS scenarios get a +0.10 × uncertainty bonus
    because elevated uncertainty inherently increases the probability of
    directional indecision.

    Parameters
    ----------
    scenario:
        A ``CounterfactualScenario`` from ``generate_alternatives()``.
    new_evidence:
        List of evidence strings observed since the scenario was generated.
    current_uncertainty:
        Current composite uncertainty (0-1).

    Returns
    -------
    float in [0, 1] — updated probability weight for this scenario.
    """
    trigger_count = max(1, len(scenario.trigger_conditions))
    inval_count   = max(1, len(scenario.invalidation_signals))

    trigger_matches = _evidence_matches(new_evidence, scenario.trigger_conditions)
    inval_matches   = _evidence_matches(new_evidence, scenario.invalidation_signals)

    match_ratio  = trigger_matches / trigger_count
    inval_ratio  = inval_matches   / inval_count
    raw_score    = match_ratio * (1.0 - inval_ratio)

    # Blend: prior weight + new evidence signal
    updated = 0.60 * scenario.probability_weight + 0.40 * raw_score

    # Uncertainty bonus for SIDEWAYS
    if scenario.scenario_type == ScenarioType.SIDEWAYS:
        updated += 0.10 * current_uncertainty

    updated = round(max(0.0, min(1.0, updated)), 4)

    if _state is not None:
        _state.last_updated_at = time.time()

    return updated


# ── Public callable: switch_thesis ───────────────────────────────────────────


def switch_thesis(
    state:               CounterfactualState,
    trigger:             FlipTrigger,
    new_primary_side:    BeliefDirection,
    new_primary_thesis:  Thesis,
    new_secondary_thesis: Thesis,
    trigger_detail:      str        = "",
    uncertainty:         float      = 0.0,
    now:                 float | None = None,
) -> ThesisSwitchResult:
    """Perform a thesis switch: primary ↔ secondary.

    Steps
    -----
    1. Extract ``FlipLesson`` from the old primary's conflicting evidence.
    2. Build and store ``ThesisFlipRecord``.
    3. Detect reversal (flip-back within ``REVERSAL_WINDOW_SECONDS``).
    4. Build and store ``ThesisReversalRecord`` if reversal.
    5. Generate fresh ``CounterfactualScenario`` list for new primary.
    6. Record a ``ProbabilityPoint`` post-switch.
    7. Return ``ThesisSwitchResult``.

    Parameters
    ----------
    state:
        Per-instrument ``CounterfactualState`` (mutated in place).
    trigger:
        What caused the switch (``FlipTrigger``).
    new_primary_side:
        The ``BeliefDirection`` that becomes primary after the switch.
    new_primary_thesis:
        The ``Thesis`` object for the new primary side.
    new_secondary_thesis:
        The ``Thesis`` object for the new secondary side.
    trigger_detail:
        Brief free-text description of what triggered the switch.
    uncertainty:
        Current composite uncertainty (0-1) for scenario weighting.
    now:
        Override current timestamp.

    Returns
    -------
    ``ThesisSwitchResult`` — flip record + new scenarios + reversal info.
    """
    now = now if now is not None else time.time()

    old_primary_side = state.primary_side
    prev_primary_p   = new_secondary_thesis.strength()   # old primary is now secondary
    new_primary_p    = new_primary_thesis.strength()

    # 1. FlipLesson
    lesson = _extract_flip_lesson(
        old_primary_conflicting=list(
            new_secondary_thesis.conflicting_evidence  # old primary's conflicting evidence
        ),
        trigger_detail=trigger_detail,
        state=state,
        now=now,
    )

    # 2. Build ThesisFlipRecord
    flip_id = _make_id("flip_")
    flip_record = ThesisFlipRecord(
        flip_id=flip_id,
        previous_primary_side=old_primary_side,
        new_primary_side=new_primary_side,
        trigger=trigger,
        trigger_detail=trigger_detail or trigger.value,
        primary_p_before=prev_primary_p,
        secondary_p_before=new_primary_p,
        evidence_snapshot=list(new_secondary_thesis.conflicting_evidence[:10]),
        flip_lesson=lesson,
        flipped_at=now,
    )
    # Append with cap
    state.flip_records.append(flip_record)
    if len(state.flip_records) > MAX_FLIP_RECORDS:
        state.flip_records = state.flip_records[-MAX_FLIP_RECORDS:]

    # 3. Reversal detection: are we flipping back to the side we were at before the LAST flip?
    is_reversal = False
    reversal_record: ThesisReversalRecord | None = None

    if state.flip_records and len(state.flip_records) >= 2:
        previous_flip = state.flip_records[-2]
        seconds_elapsed = now - previous_flip.flipped_at
        if (
            previous_flip.previous_primary_side == new_primary_side
            and seconds_elapsed <= REVERSAL_WINDOW_SECONDS
        ):
            is_reversal = True
            rev_id = _make_id("rev_")
            reversal_record = ThesisReversalRecord(
                reversal_id=rev_id,
                returning_to_side=new_primary_side,
                intermediate_side=old_primary_side,
                seconds_elapsed=round(seconds_elapsed, 1),
                flip_id=previous_flip.flip_id,
                reversed_at=now,
            )
            state.reversal_records.append(reversal_record)
            if len(state.reversal_records) > MAX_REVERSAL_RECORDS:
                state.reversal_records = state.reversal_records[-MAX_REVERSAL_RECORDS:]

    # 4. Update state sides
    state.primary_side   = new_primary_side
    state.secondary_side = new_secondary_thesis.side
    state.last_flip_at   = now

    # 5. Generate new scenarios for new primary
    new_scenarios = generate_alternatives(
        primary_thesis=new_primary_thesis,
        secondary_thesis=new_secondary_thesis,
        uncertainty_score=uncertainty,
        _state=state,
        now=now,
    )

    # 6. Record ProbabilityPoint post-switch
    _append_probability_point(
        state=state,
        primary_p=new_primary_p,
        secondary_p=prev_primary_p,
        uncertainty=uncertainty,
        now=now,
    )

    state.last_updated_at = now

    logger.debug(
        f"[Counterfactual] {state.instrument} thesis flip: "
        f"{old_primary_side.value} → {new_primary_side.value} "
        f"({'REVERSAL' if is_reversal else 'NEW'}) trigger={trigger.value}"
    )

    return ThesisSwitchResult(
        flip_record=flip_record,
        new_scenarios=new_scenarios,
        is_reversal=is_reversal,
        reversal_record=reversal_record,
    )


# ── Probability evolution helpers ─────────────────────────────────────────────


def _append_probability_point(
    state:       CounterfactualState,
    primary_p:   float,
    secondary_p: float,
    uncertainty: float,
    now:         float,
) -> None:
    """Append a ``ProbabilityPoint`` and detect uncertainty spikes."""
    # Spike detection before appending
    prev = state.last_probability_point()
    if prev is not None:
        delta = uncertainty - prev.uncertainty
        if delta >= UNCERTAINTY_SPIKE_DELTA:
            _record_spike(state=state, before=prev.uncertainty, after=uncertainty, now=now)

    point = ProbabilityPoint(
        timestamp=now,
        primary_p=round(primary_p, 4),
        secondary_p=round(secondary_p, 4),
        uncertainty=round(uncertainty, 4),
    )
    state.probability_evolution.append(point)
    if len(state.probability_evolution) > MAX_PROBABILITY_POINTS:
        state.probability_evolution = state.probability_evolution[-MAX_PROBABILITY_POINTS:]


def _record_spike(
    state:  CounterfactualState,
    before: float,
    after:  float,
    now:    float,
    context: str = "",
) -> None:
    """Record an ``UncertaintySpikeRecord``."""
    # Map composite to level string without importing UncertaintyLevel to avoid coupling
    if after >= 0.75:
        level = "EXTREME"
    elif after >= 0.55:
        level = "HIGH"
    elif after >= 0.35:
        level = "MODERATE"
    else:
        level = "LOW"

    spike = UncertaintySpikeRecord(
        spike_id=_make_id("spk_"),
        instrument=state.instrument,
        uncertainty_before=round(before, 4),
        uncertainty_after=round(after, 4),
        delta=round(after - before, 4),
        level_after=level,
        context=context or f"uncertainty jumped {after - before:+.2f}",
        spiked_at=now,
    )
    state.uncertainty_spikes.append(spike)
    if len(state.uncertainty_spikes) > MAX_SPIKE_RECORDS:
        state.uncertainty_spikes = state.uncertainty_spikes[-MAX_SPIKE_RECORDS:]


def record_probability_point(
    state:       CounterfactualState,
    primary_p:   float,
    secondary_p: float,
    uncertainty: float,
    now:         float | None = None,
) -> ProbabilityPoint:
    """Public helper: sample current thesis strengths and append to evolution.

    Call this on every market-data update tick or whenever thesis strengths
    change.  Returns the new ``ProbabilityPoint``.
    """
    now = now if now is not None else time.time()
    _append_probability_point(state, primary_p, secondary_p, uncertainty, now)
    state.last_updated_at = now
    return state.probability_evolution[-1]


def recommend_switch(
    state:       CounterfactualState,
    primary_p:   float,
    secondary_p: float,
) -> tuple[bool, str]:
    """Return ``(should_switch, reason)`` based on current strength readings.

    Supports "fast thesis switching" by continuously checking whether the
    secondary has overtaken the primary with enough margin.

    Returns
    -------
    (True, reason) when a switch is recommended; (False, "") otherwise.
    """
    spread = primary_p - secondary_p

    if secondary_p > primary_p + SWITCH_MARGIN:
        return True, (
            f"secondary_dominance: secondary_p={secondary_p:.3f} > "
            f"primary_p={primary_p:.3f} + margin={SWITCH_MARGIN}"
        )

    if primary_p < PRIMARY_CONFIDENCE_FLOOR:
        return True, (
            f"confidence_collapse: primary_p={primary_p:.3f} < "
            f"floor={PRIMARY_CONFIDENCE_FLOOR}"
        )

    return False, ""


# ── CounterfactualEngine (stateful wrapper) ───────────────────────────────────


class CounterfactualEngine:
    """Stateful wrapper for the three C16 public callables.

    One instance per instrument.  Maintains a ``CounterfactualState``
    and delegates to the module-level functions.

    Parameters
    ----------
    instrument:
        Symbol label for logging and state snapshots.
    initial_primary_side:
        Initial primary thesis direction.
    initial_secondary_side:
        Initial secondary thesis direction.

    Typical usage
    -------------
    engine = CounterfactualEngine(instrument="EURUSD")

    # On each tick: sample current thesis strengths
    engine.record_probability_point(
        primary_p=bull_thesis.strength(),
        secondary_p=bear_thesis.strength(),
        uncertainty=uncertainty_score.composite,
    )

    # Should we switch?
    should_switch, reason = engine.recommend_switch(
        primary_p=bull_thesis.strength(),
        secondary_p=bear_thesis.strength(),
    )
    if should_switch:
        result = engine.switch_thesis(
            trigger=FlipTrigger.SECONDARY_DOMINANCE,
            new_primary_side=BeliefDirection.BEARISH,
            new_primary_thesis=bear_thesis,
            new_secondary_thesis=bull_thesis,
            trigger_detail=reason,
        )

    # Periodically: generate alternatives and evaluate
    scenarios = engine.generate_alternatives(bull_thesis, bear_thesis, uncertainty=0.35)
    for scenario in scenarios:
        score = engine.evaluate_alternative(scenario, new_evidence=["bearish divergence"])
    """

    def __init__(
        self,
        instrument:            str             = "",
        initial_primary_side:  BeliefDirection = BeliefDirection.BULLISH,
        initial_secondary_side: BeliefDirection = BeliefDirection.BEARISH,
    ) -> None:
        self.instrument = instrument
        self._state = CounterfactualState(
            instrument=instrument,
            primary_side=initial_primary_side,
            secondary_side=initial_secondary_side,
        )

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def state(self) -> CounterfactualState:
        return self._state

    @property
    def primary_side(self) -> BeliefDirection:
        return self._state.primary_side

    @property
    def secondary_side(self) -> BeliefDirection:
        return self._state.secondary_side

    # ── Three public callables ────────────────────────────────────────────

    def generate_alternatives(
        self,
        primary_thesis:    Thesis,
        secondary_thesis:  Thesis | None = None,
        uncertainty:       float         = 0.0,
        now:               float | None  = None,
    ) -> list[CounterfactualScenario]:
        """Generate alternative scenarios for the current primary thesis."""
        return generate_alternatives(
            primary_thesis=primary_thesis,
            secondary_thesis=secondary_thesis,
            uncertainty_score=uncertainty,
            _state=self._state,
            now=now,
        )

    def evaluate_alternative(
        self,
        scenario:            CounterfactualScenario,
        new_evidence:        list[str],
        current_uncertainty: float = 0.0,
    ) -> float:
        """Score how well new evidence supports a scenario."""
        return evaluate_alternative(
            scenario=scenario,
            new_evidence=new_evidence,
            current_uncertainty=current_uncertainty,
            _state=self._state,
        )

    def switch_thesis(
        self,
        trigger:               FlipTrigger,
        new_primary_side:      BeliefDirection,
        new_primary_thesis:    Thesis,
        new_secondary_thesis:  Thesis,
        trigger_detail:        str         = "",
        uncertainty:           float       = 0.0,
        now:                   float | None = None,
    ) -> ThesisSwitchResult:
        """Perform a thesis switch and record the flip."""
        return switch_thesis(
            state=self._state,
            trigger=trigger,
            new_primary_side=new_primary_side,
            new_primary_thesis=new_primary_thesis,
            new_secondary_thesis=new_secondary_thesis,
            trigger_detail=trigger_detail,
            uncertainty=uncertainty,
            now=now,
        )

    # ── Probability evolution ─────────────────────────────────────────────

    def record_probability_point(
        self,
        primary_p:   float,
        secondary_p: float,
        uncertainty: float      = 0.0,
        now:         float | None = None,
    ) -> ProbabilityPoint:
        """Sample current thesis strengths and append to evolution timeline."""
        return record_probability_point(
            state=self._state,
            primary_p=primary_p,
            secondary_p=secondary_p,
            uncertainty=uncertainty,
            now=now,
        )

    def recommend_switch(
        self,
        primary_p:   float,
        secondary_p: float,
    ) -> tuple[bool, str]:
        """Check whether the secondary has overtaken the primary."""
        return recommend_switch(self._state, primary_p, secondary_p)

    # ── History access ────────────────────────────────────────────────────

    def flip_records(self) -> list[ThesisFlipRecord]:
        return list(self._state.flip_records)

    def reversal_records(self) -> list[ThesisReversalRecord]:
        return list(self._state.reversal_records)

    def uncertainty_spikes(self) -> list[UncertaintySpikeRecord]:
        return list(self._state.uncertainty_spikes)

    def probability_evolution(self) -> list[ProbabilityPoint]:
        return list(self._state.probability_evolution)

    def current_scenarios(self) -> list[CounterfactualScenario]:
        return list(self._state.current_scenarios)

    def snapshot(self) -> dict:
        """Compact snapshot for reporting."""
        return self._state.to_snapshot()


__all__ = [
    # Public callables
    "generate_alternatives",
    "evaluate_alternative",
    "switch_thesis",
    # Supporting helpers
    "record_probability_point",
    "recommend_switch",
    # Stateful wrapper
    "CounterfactualEngine",
]
