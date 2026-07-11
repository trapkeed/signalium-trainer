"""C15 Behavioral Control Layer — engine and public callables.

Three public callables (spec):

    evaluate_behavior(signals)              → BehavioralAssessment
    adjust_state(current_state, assessment) → CognitiveState
    behavior_report(engine)                 → dict

``BehavioralControlEngine`` — stateful wrapper (one per instrument or per
                               process depending on symbol-isolation strategy).

Design
------
All three callables are intentionally pure or near-pure.  The stateful
``BehavioralControlEngine`` is a thin wrapper that holds ``_state`` (the
current ``CognitiveState``) and ``_history`` (a bounded list of
``BehavioralRecord`` objects) and delegates to the module-level functions.

Five behavioral pathologies measured
--------------------------------------
1. Confidence inflation
   Signed mean of (predicted_confidence − actual_outcome) over the recent
   window.  When the QDE consistently predicts 0.70 but wins at 0.55, the
   inflation = 0.15.  Reuses the same semantic intent as
   ``MetacognitiveAssessment.calibration_error`` (unsigned) but adds direction
   and rolling trend.

2. Overconfidence rate
   Fraction of high-confidence trades (≥ 0.65) that resulted in a loss
   (outcome < 0.5).  Distinct from ``OverconfidenceTracker`` (which fires
   per debate) — this is the rolling behavioural rate over closed trades.

3. Drawdown response quality
   Measures whether aggressiveness declined when drawdown increased.
   Score 1.0 = aggressiveness fell below baseline during drawdown (ideal).
   Score 0.0 = aggressiveness was elevated at or above 0.60 during a
   significant drawdown (pathological).

4. Aggressiveness drift
   Signed slope of a linear fit over recent aggressiveness scores, normalised
   to [-1, +1].  Positive = slow creep toward more aggressive.  Computed
   without importing scipy — pure rolling arithmetic.

5. Risk escalation
   Approximate second derivative of recent per-trade risk: whether the rate
   of risk increase is itself accelerating.  Computed as
   (Δrisk_second_half − Δrisk_first_half) / window, normalised to [0, 1].

State-transition FSM (adjust_state)
-------------------------------------
Priority order (first matching rule wins):

    HARD DEFENSIVE rules (always override):
        consecutive_losses >= LOSING_STREAK_DEFENSIVE  → DEFENSIVE
        drawdown >= DRAWDOWN_DEFENSIVE_THRESHOLD        → DEFENSIVE
        pathology_score >= PATHOLOGY_DEFENSIVE_THRESHOLD → DEFENSIVE

    EUPHORIA guard (post-win):
        consecutive_wins >= WINNING_STREAK_EUPHORIA     → force CAUTION
        (AGGRESSIVE AND euphoria_risk > EUPHORIA_RISK_THRESHOLD) → CAUTION

    RECOVERY rules (improve from DEFENSIVE):
        DEFENSIVE AND consecutive_losses == 0 AND drawdown < DRAWDOWN_CAUTION_THRESHOLD
            → CAUTION

    CAUTION rules:
        consecutive_losses >= LOSING_STREAK_CAUTION     → ≥CAUTION
        pathology_score >= PATHOLOGY_CAUTION_THRESHOLD  → CAUTION
        CAUTION AND all clear                           → NORMAL

    AGGRESSIVE promotion:
        consecutive_wins >= WINNING_STREAK_AGGRESSIVE
            AND pathology_score < 0.15
            AND current_state == NORMAL                 → AGGRESSIVE

    Otherwise: current_state unchanged.
"""

from __future__ import annotations

import time
import uuid
from collections import deque

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.qde.brain.behavioral.behavioral_types import (
    AGGRESSION_DRIFT_SEVERE,
    DRAWDOWN_CAUTION_THRESHOLD,
    DRAWDOWN_DEFENSIVE_THRESHOLD,
    EUPHORIA_RISK_THRESHOLD,
    INFLATION_SEVERE_THRESHOLD,
    LOSING_STREAK_CAUTION,
    LOSING_STREAK_DEFENSIVE,
    MAX_BEHAVIORAL_HISTORY,
    OVERCONFIDENCE_RATE_SEVERE,
    PATHOLOGY_CAUTION_THRESHOLD,
    PATHOLOGY_DEFENSIVE_THRESHOLD,
    RISK_ESCALATION_SEVERE,
    WINNING_STREAK_AGGRESSIVE,
    WINNING_STREAK_EUPHORIA,
    BehavioralAssessment,
    BehavioralRecord,
    BehavioralSignals,
    CognitiveState,
    state_severity,
)

# ── Internal signal computation ───────────────────────────────────────────────


def _confidence_inflation(
    confidences: list[float],
    outcomes: list[float],
) -> float:
    """Signed mean of (predicted - actual) over the paired window.

    Returns 0.0 when fewer than 2 pairs are available.
    Range: nominally [-1, +1] but clamped to [0, 1] for the pathology score
    (negative inflation = QDE is being conservative = not a pathology).
    """
    pairs = list(zip(confidences, outcomes))
    if len(pairs) < 2:
        return 0.0
    return round(sum(c - o for c, o in pairs) / len(pairs), 4)


def _overconfidence_rate(
    confidences: list[float],
    outcomes: list[float],
    conf_threshold: float = 0.65,
) -> float:
    """Fraction of high-confidence trades (≥ conf_threshold) that lost (< 0.5).

    Returns 0.0 when no high-confidence trades exist in the window.
    """
    high_conf_pairs = [(c, o) for c, o in zip(confidences, outcomes) if c >= conf_threshold]
    if not high_conf_pairs:
        return 0.0
    lost = sum(1 for _, o in high_conf_pairs if o < 0.5)
    return round(lost / len(high_conf_pairs), 4)


def _drawdown_response_quality(
    current_drawdown: float,
    drawdown_peak_aggressiveness: float,
    recent_aggressiveness: list[float],
) -> float:
    """Measure whether aggressiveness declined appropriately during drawdown.

    Returns a quality score in [0, 1]:
      1.0 = no drawdown, or aggressiveness was already low
      0.0 = significant drawdown AND aggressiveness stayed high

    Logic:
    - Below DRAWDOWN_CAUTION_THRESHOLD: trivially good (score=1.0).
    - Above threshold: score = max(0, 1 - drawdown_peak_aggressiveness / 0.60).
      A system that capped its aggressiveness at 0.30 during an 8% drawdown
      scores (1 - 0.30/0.60) = 0.5; one that ran at 0.60+ scores 0.
    """
    if current_drawdown < DRAWDOWN_CAUTION_THRESHOLD:
        return 1.0

    # Also factor in trend of recent aggressiveness
    recent_mean = (
        sum(recent_aggressiveness) / len(recent_aggressiveness) if recent_aggressiveness else 0.5
    )
    # Worst case: both peak AND recent mean are high
    worst_agg = max(drawdown_peak_aggressiveness, recent_mean)
    cap = 0.60  # "acceptable" aggressiveness ceiling during drawdown
    quality = max(0.0, 1.0 - worst_agg / cap)
    return round(quality, 4)


def _aggressiveness_drift(scores: list[float]) -> float:
    """Signed slope of aggressiveness over recent history, normalised to [-1, +1].

    Uses a simple least-squares slope approximation without importing scipy.
    Positive = trending more aggressive.
    Returns 0.0 when fewer than 3 observations are available.
    """
    n = len(scores)
    if n < 3:
        return 0.0
    # Least-squares slope for y = scores, x = 0..n-1
    x_mean = (n - 1) / 2.0
    y_mean = sum(scores) / n
    numerator = sum((i - x_mean) * (scores[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    slope = numerator / denominator  # units: aggressiveness-change per step
    # Normalise: a slope of 0.05/step over a 20-step window = 1.0 drift
    normalised = slope * (n - 1) / 1.0  # total change over window
    return round(max(-1.0, min(1.0, normalised)), 4)


def _risk_escalation_rate(risk_series: list[float]) -> float:
    """Approximate second derivative of risk: is risk increasing faster?

    Splits the series in half, computes the average rate-of-change for each
    half, and returns the normalised increase: (second_half_slope −
    first_half_slope) clamped to [0, 1].

    Returns 0.0 when fewer than 4 observations are available.
    """
    n = len(risk_series)
    if n < 4:
        return 0.0

    def _slope(window: list[float]) -> float:
        k = len(window)
        if k < 2:
            return 0.0
        return (window[-1] - window[0]) / (k - 1)

    mid = n // 2
    first_slope = _slope(risk_series[:mid])
    second_slope = _slope(risk_series[mid:])

    # Escalation requires the second half to have a genuinely positive slope
    # (risk is actually rising), not merely "less falling" than the first half.
    # A series that declines steeply then flattens is *decelerating*, not escalating.
    if second_slope <= 0:
        return 0.0

    escalation = second_slope - max(0.0, first_slope)  # excess over first-half rise
    return round(max(0.0, min(1.0, escalation * 4.0)), 4)  # ×4 for sensitivity


def _euphoria_risk(consecutive_wins: int, confidence_inflation: float) -> float:
    """Risk of post-win overconfidence (euphoria).

    Driven by long winning streaks combined with confidence inflation.
    Range: [0, 1].
    """
    streak_component = min(1.0, max(0.0, (consecutive_wins - 3) / 6.0))
    inflation_component = max(0.0, confidence_inflation)  # only positive inflation counts
    return round(min(1.0, 0.60 * streak_component + 0.40 * inflation_component), 4)


def _loss_spiral_risk(
    consecutive_losses: int,
    drawdown: float,
    drawdown_response_quality: float,
) -> float:
    """Risk of revenge trading / loss spiral.

    Elevated when losses are accumulating AND the drawdown response is poor
    (aggressiveness not declining).
    Range: [0, 1].
    """
    loss_component = min(1.0, consecutive_losses / 6.0)
    drawdown_component = min(1.0, drawdown / DRAWDOWN_DEFENSIVE_THRESHOLD)
    # Invert quality: poor response amplifies the spiral risk
    response_amplifier = 1.0 - drawdown_response_quality
    return round(
        min(
            1.0,
            0.40 * loss_component
            + 0.30 * drawdown_component
            + 0.30 * response_amplifier * max(loss_component, drawdown_component),
        ),
        4,
    )


def _pathology_score(
    confidence_inflation: float,
    overconfidence_level: float,
    drawdown_response_quality: float,
    aggressiveness_drift: float,
    risk_escalation_rate: float,
    metacognition_overconfidence: float = 0.0,
    metacognition_caution: float = 0.0,
) -> float:
    """Weighted composite pathology score in [0, 1].

    A score near 0 = behaviorally healthy.
    A score near 1 = multiple pathological patterns firing simultaneously.

    Weights:
        0.25 confidence inflation   (primary driver — calibration failure)
        0.20 overconfidence rate    (high-confidence + loss combos)
        0.20 drawdown response      (inverted: poor response = higher pathology)
        0.15 aggressiveness drift   (slow creep; only counts positive drift)
        0.10 risk escalation        (second-derivative acceleration)
        0.10 metacognition signal   (optional pre-computed composite)
    """
    inflation_contrib = max(0.0, confidence_inflation)  # only positive inflation
    drift_contrib = max(0.0, aggressiveness_drift)  # only positive drift
    response_contrib = 1.0 - drawdown_response_quality  # invert (poor = high)
    meta_contrib = max(metacognition_overconfidence, metacognition_caution * 0.5)

    raw = (
        0.25 * min(1.0, inflation_contrib / INFLATION_SEVERE_THRESHOLD)
        + 0.20 * min(1.0, overconfidence_level / OVERCONFIDENCE_RATE_SEVERE)
        + 0.20 * response_contrib
        + 0.15 * min(1.0, drift_contrib / AGGRESSION_DRIFT_SEVERE)
        + 0.10 * min(1.0, risk_escalation_rate / RISK_ESCALATION_SEVERE)
        + 0.10 * min(1.0, meta_contrib)
    )
    return round(min(1.0, max(0.0, raw)), 4)


def _recommend_state(
    consecutive_losses: int,
    consecutive_wins: int,
    current_drawdown: float,
    pathology: float,
    euphoria: float,
    loss_spiral: float,
    current_state: CognitiveState | None = None,
) -> tuple[CognitiveState, str]:
    """Derive the recommended CognitiveState from the behavioral signals.

    Returns (state, trigger_description).  The ``current_state`` is used to
    resolve CAUTION→NORMAL recovery and DEFENSIVE→CAUTION recovery correctly.
    """
    # ── Hard defensive overrides ──────────────────────────────────────────
    if consecutive_losses >= LOSING_STREAK_DEFENSIVE:
        return CognitiveState.DEFENSIVE, f"losing_streak={consecutive_losses}"
    if current_drawdown >= DRAWDOWN_DEFENSIVE_THRESHOLD:
        return CognitiveState.DEFENSIVE, f"drawdown={current_drawdown:.1%}"
    if pathology >= PATHOLOGY_DEFENSIVE_THRESHOLD:
        return CognitiveState.DEFENSIVE, f"pathology={pathology:.2f}"

    # ── Euphoria guard ────────────────────────────────────────────────────
    if consecutive_wins >= WINNING_STREAK_EUPHORIA:
        return CognitiveState.CAUTION, (f"euphoria_guard: winning_streak={consecutive_wins}")
    if euphoria >= EUPHORIA_RISK_THRESHOLD and current_state == CognitiveState.AGGRESSIVE:
        return CognitiveState.CAUTION, f"euphoria_risk={euphoria:.2f}"

    # ── Caution triggers ──────────────────────────────────────────────────
    if consecutive_losses >= LOSING_STREAK_CAUTION:
        return CognitiveState.CAUTION, f"losing_streak={consecutive_losses}"
    if pathology >= PATHOLOGY_CAUTION_THRESHOLD:
        return CognitiveState.CAUTION, f"pathology={pathology:.2f}"
    if current_drawdown >= DRAWDOWN_CAUTION_THRESHOLD:
        return CognitiveState.CAUTION, f"drawdown={current_drawdown:.1%}"

    # ── Recovery: DEFENSIVE → CAUTION ────────────────────────────────────
    if (
        current_state == CognitiveState.DEFENSIVE
        and consecutive_losses == 0
        and current_drawdown < DRAWDOWN_CAUTION_THRESHOLD
    ):
        return CognitiveState.CAUTION, "recovering_from_defensive"

    # ── AGGRESSIVE promotion ──────────────────────────────────────────────
    if (
        consecutive_wins >= WINNING_STREAK_AGGRESSIVE
        and pathology < 0.15
        and euphoria < EUPHORIA_RISK_THRESHOLD
        and (current_state in (CognitiveState.NORMAL, None))
    ):
        return CognitiveState.AGGRESSIVE, (f"clean_winning_streak={consecutive_wins}")

    # ── CAUTION → NORMAL recovery ─────────────────────────────────────────
    if (
        current_state == CognitiveState.CAUTION
        and consecutive_losses < LOSING_STREAK_CAUTION
        and pathology < PATHOLOGY_CAUTION_THRESHOLD
        and current_drawdown < DRAWDOWN_CAUTION_THRESHOLD
    ):
        return CognitiveState.NORMAL, "caution_resolved"

    # ── AGGRESSIVE → NORMAL natural fade ─────────────────────────────────
    if current_state == CognitiveState.AGGRESSIVE and consecutive_wins < WINNING_STREAK_AGGRESSIVE:
        return CognitiveState.NORMAL, "winning_streak_faded"

    # ── No change ─────────────────────────────────────────────────────────
    fallback = current_state if current_state is not None else CognitiveState.NORMAL
    return fallback, "no_trigger"


# ── Public callable: evaluate_behavior ───────────────────────────────────────


def evaluate_behavior(
    signals: BehavioralSignals,
    current_state: CognitiveState | None = None,
) -> BehavioralAssessment:
    """Evaluate the QDE's current behavioral health from recent trade signals.

    This is a stateless computation — it takes ``BehavioralSignals`` and
    returns a ``BehavioralAssessment``.  The ``BehavioralControlEngine``
    stores history and provides the ``current_state`` context; standalone
    callers can pass ``current_state=None`` to get a context-free assessment.

    Parameters
    ----------
    signals:
        Input signals — see ``BehavioralSignals`` for field documentation.
    current_state:
        Current ``CognitiveState`` of the brain.  Used by ``_recommend_state``
        to handle recovery transitions correctly.  When None, the FSM assumes
        NORMAL as the baseline.

    Returns
    -------
    BehavioralAssessment — with all five pathology scores, composite signals,
    and a recommended CognitiveState.
    """
    # ── 1. Confidence inflation ───────────────────────────────────────────
    inflation = _confidence_inflation(
        signals.recent_trade_confidences,
        signals.recent_trade_outcomes,
    )
    # Blend with pre-computed calibration_error if provided (unsigned →
    # treat as positive inflation when no directional signal is available)
    if not signals.recent_trade_confidences and signals.calibration_error > 0:
        inflation = signals.calibration_error

    # ── 2. Overconfidence rate ────────────────────────────────────────────
    overconf = _overconfidence_rate(
        signals.recent_trade_confidences,
        signals.recent_trade_outcomes,
    )
    # Blend with pre-computed metacognition signal
    if signals.overconfidence_score > overconf:
        overconf = round((overconf + signals.overconfidence_score) / 2.0, 4)

    # ── 3. Drawdown response quality ──────────────────────────────────────
    dq = _drawdown_response_quality(
        signals.current_drawdown_fraction,
        signals.drawdown_peak_aggressiveness,
        signals.recent_aggressiveness_scores,
    )

    # ── 4. Aggressiveness drift ───────────────────────────────────────────
    drift = _aggressiveness_drift(signals.recent_aggressiveness_scores)

    # ── 5. Risk escalation rate ───────────────────────────────────────────
    escalation = _risk_escalation_rate(signals.recent_risk_per_trade)

    # ── Composite scores ──────────────────────────────────────────────────
    pathology = _pathology_score(
        confidence_inflation=inflation,
        overconfidence_level=overconf,
        drawdown_response_quality=dq,
        aggressiveness_drift=drift,
        risk_escalation_rate=escalation,
        metacognition_overconfidence=signals.overconfidence_score,
        metacognition_caution=signals.metacognition_caution_level,
    )
    euphoria = _euphoria_risk(signals.consecutive_wins, inflation)
    loss_spiral = _loss_spiral_risk(
        signals.consecutive_losses,
        signals.current_drawdown_fraction,
        dq,
    )

    # ── Recommended state ─────────────────────────────────────────────────
    rec_state, trigger = _recommend_state(
        consecutive_losses=signals.consecutive_losses,
        consecutive_wins=signals.consecutive_wins,
        current_drawdown=signals.current_drawdown_fraction,
        pathology=pathology,
        euphoria=euphoria,
        loss_spiral=loss_spiral,
        current_state=current_state,
    )

    reasoning = (
        f"inflation={inflation:+.2f} | overconf={overconf:.2f} | "
        f"dq={dq:.2f} | drift={drift:+.2f} | esc={escalation:.2f} | "
        f"pathology={pathology:.2f} | euphoria={euphoria:.2f} | "
        f"spiral={loss_spiral:.2f} | trigger={trigger}"
    )

    return BehavioralAssessment(
        confidence_inflation=inflation,
        overconfidence_level=overconf,
        drawdown_response_quality=dq,
        aggressiveness_drift=drift,
        risk_escalation_rate=escalation,
        pathology_score=pathology,
        euphoria_risk=euphoria,
        loss_spiral_risk=loss_spiral,
        recommended_state=rec_state,
        reasoning=reasoning,
    )


# ── Public callable: adjust_state ────────────────────────────────────────────


def adjust_state(
    current_state: CognitiveState,
    assessment: BehavioralAssessment,
) -> CognitiveState:
    """Apply the behavioral assessment to determine the next CognitiveState.

    This is a near-pure function: it does not read external state beyond the
    two arguments, and it never raises.

    The recommended state from ``evaluate_behavior`` is not applied blindly —
    this function enforces monotone severity rules to prevent oscillation:

      * The state can always worsen (NORMAL → DEFENSIVE in one step) when a
        hard trigger fires.
      * Recovery is throttled: we move at most one level per call
        (DEFENSIVE → CAUTION, not DEFENSIVE → NORMAL).

    Parameters
    ----------
    current_state:
        The brain's current ``CognitiveState``.
    assessment:
        Output of ``evaluate_behavior()``.

    Returns
    -------
    The new ``CognitiveState`` to adopt.
    """
    rec = assessment.recommended_state

    # Worsening is always immediate — safety rails do not throttle.
    if state_severity(rec) > state_severity(current_state):
        return rec

    # Recovery is throttled to at most one level per call.
    if state_severity(rec) < state_severity(current_state):
        new_severity = state_severity(current_state) - 1
        # Reverse-lookup severity → state
        for state, sev in {
            CognitiveState.AGGRESSIVE: 0,
            CognitiveState.NORMAL: 1,
            CognitiveState.CAUTION: 2,
            CognitiveState.DEFENSIVE: 3,
        }.items():
            if sev == new_severity:
                return state
        return current_state

    # Same severity → adopt the recommendation (handles same-severity lateral moves)
    return rec


# ── Public callable: behavior_report ─────────────────────────────────────────


def behavior_report(
    engine: "BehavioralControlEngine",
    now: float | None = None,
) -> dict:
    """Generate a comprehensive behavioral report from a BehavioralControlEngine.

    Parameters
    ----------
    engine:
        The ``BehavioralControlEngine`` instance to report on.
    now:
        Override current time.

    Returns
    -------
    dict with keys:
        ``current_state``       — current CognitiveState value.
        ``state_history``       — last 20 records with transitions highlighted.
        ``transition_count``    — total state transitions in history.
        ``time_in_state``       — dict mapping each state to record count.
        ``latest_assessment``   — most recent BehavioralAssessment dict (or None).
        ``pathology_trend``     — list of recent pathology_score values (last 20).
        ``euphoria_events``     — count of AGGRESSIVE→CAUTION transitions.
        ``loss_spiral_events``  — count of entries with loss_spiral_risk > 0.5.
        ``generated_at``        — timestamp.
        ``instrument``          — instrument label from the engine.
    """
    now = now if now is not None else time.time()
    history = engine.history()

    time_in_state: dict[str, int] = {s.value: 0 for s in CognitiveState}
    transition_count = 0
    euphoria_events = 0
    loss_spiral_events = 0
    pathology_trend: list[float] = []

    for rec in history:
        time_in_state[rec.new_state.value] += 1
        if rec.is_transition:
            transition_count += 1
        if (
            rec.previous_state == CognitiveState.AGGRESSIVE
            and rec.new_state == CognitiveState.CAUTION
        ):
            euphoria_events += 1
        if rec.assessment.loss_spiral_risk > 0.5:
            loss_spiral_events += 1

    # Pathology trend over recent history (last 20 entries)
    recent = history[-20:] if len(history) > 20 else history
    pathology_trend = [round(r.assessment.pathology_score, 4) for r in recent]

    latest = history[-1].assessment.to_dict() if history else None

    return {
        "instrument": engine.instrument,
        "current_state": engine.state.value,
        "state_history": [r.to_dict() for r in history[-20:]],
        "transition_count": transition_count,
        "time_in_state": time_in_state,
        "latest_assessment": latest,
        "pathology_trend": pathology_trend,
        "euphoria_events": euphoria_events,
        "loss_spiral_events": loss_spiral_events,
        "history_size": len(history),
        "generated_at": round(now, 3),
    }


# ── BehavioralControlEngine (stateful wrapper) ────────────────────────────────


class BehavioralControlEngine:
    """Stateful wrapper for the three C15 public callables.

    One instance per instrument (or per process if symbol isolation is handled
    externally).  Holds the current ``CognitiveState`` and a bounded deque of
    ``BehavioralRecord`` objects.

    Parameters
    ----------
    instrument:
        Symbol label for reporting (does not affect logic).
    initial_state:
        Starting cognitive state (default: NORMAL).

    Typical usage
    -------------
    engine = BehavioralControlEngine(instrument="EURUSD")

    # After each closed trade or periodic tick:
    signals = BehavioralSignals(
        recent_trade_confidences=[0.72, 0.68, 0.75],
        recent_trade_outcomes=[0.0, 0.0, 1.0],
        recent_aggressiveness_scores=[0.55, 0.60, 0.65],
        consecutive_losses=2,
        current_drawdown_fraction=0.06,
    )
    assessment = engine.evaluate_behavior(signals)
    new_state  = engine.adjust_state(assessment)

    print(f"state: {new_state.value}")   # e.g. "CAUTION"

    report = engine.behavior_report()
    """

    def __init__(
        self,
        instrument: str = "",
        initial_state: CognitiveState = CognitiveState.NORMAL,
    ) -> None:
        self.instrument = instrument
        self._state = initial_state
        self._history: deque[BehavioralRecord] = deque(maxlen=MAX_BEHAVIORAL_HISTORY)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def state(self) -> CognitiveState:
        """Current CognitiveState."""
        return self._state

    # ── Three public callables ────────────────────────────────────────────

    def evaluate_behavior(self, signals: BehavioralSignals) -> BehavioralAssessment:
        """Evaluate behavioral health from the given signals.

        Passes ``self._state`` as context so recovery transitions resolve
        correctly.  Does NOT update internal state.

        See module-level ``evaluate_behavior`` for full documentation.
        """
        return evaluate_behavior(signals, current_state=self._state)

    def adjust_state(
        self,
        assessment: BehavioralAssessment,
        session_label: str = "",
        now: float | None = None,
    ) -> CognitiveState:
        """Apply the assessment to determine and record the next CognitiveState.

        Updates ``self._state`` and appends a ``BehavioralRecord`` to history.

        Parameters
        ----------
        assessment:
            Output of ``evaluate_behavior()``.
        session_label:
            Current session identifier for the history record.
        now:
            Override current time.

        Returns
        -------
        The new ``CognitiveState`` (may be unchanged).
        """
        now = now if now is not None else time.time()
        previous = self._state
        new_state = adjust_state(previous, assessment)

        trigger_desc = (
            assessment.reasoning.split("trigger=")[-1]
            if "trigger=" in assessment.reasoning
            else "n/a"
        )
        if new_state != previous:
            trigger_desc = f"{previous.value}→{new_state.value}: {trigger_desc}"
            logger.debug(
                f"[BehavioralControl] {self.instrument} state transition: "
                f"{previous.value} → {new_state.value} ({trigger_desc})"
            )

        record = BehavioralRecord(
            record_id=str(uuid.uuid4())[:8],
            previous_state=previous,
            new_state=new_state,
            assessment=assessment,
            session_label=session_label,
            recorded_at=now,
            trigger=trigger_desc,
        )
        self._history.append(record)
        self._state = new_state
        return new_state

    def behavior_report(self, now: float | None = None) -> dict:
        """Generate a comprehensive behavioral report.

        See module-level ``behavior_report`` for full documentation.
        """
        return behavior_report(self, now=now)

    # ── History access ────────────────────────────────────────────────────

    def history(self) -> list[BehavioralRecord]:
        """Return the full history as a list (newest last)."""
        return list(self._history)

    def recent_states(self, n: int = 10) -> list[CognitiveState]:
        """Return the ``n`` most recent states (newest last)."""
        return [r.new_state for r in list(self._history)[-n:]]

    def transition_count(self) -> int:
        """Total number of state transitions in history."""
        return sum(1 for r in self._history if r.is_transition)

    def time_in_state(self, state: CognitiveState) -> int:
        """Number of history records where the new state was ``state``."""
        return sum(1 for r in self._history if r.new_state == state)

    def summary(self) -> dict:
        """Compact summary for logging."""
        return {
            "instrument": self.instrument,
            "current_state": self._state.value,
            "history_size": len(self._history),
            "transition_count": self.transition_count(),
            "time_in_defensive": self.time_in_state(CognitiveState.DEFENSIVE),
            "time_in_caution": self.time_in_state(CognitiveState.CAUTION),
            "time_in_aggressive": self.time_in_state(CognitiveState.AGGRESSIVE),
        }


__all__ = [
    # Public callables
    "evaluate_behavior",
    "adjust_state",
    "behavior_report",
    # Stateful wrapper
    "BehavioralControlEngine",
]
