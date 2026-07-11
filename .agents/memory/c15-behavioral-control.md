---
name: C15 Behavioral Control Layer
description: Prevents pathological QDE behavior. Tracks 5 behavioral signals, maintains CognitiveState (CAUTION/NORMAL/AGGRESSIVE/DEFENSIVE), exposes evaluate_behavior/adjust_state/behavior_report.
---

## Package location

`trainer/qde/brain/behavioral/` — new package, nothing in existing brain files modified.

- `behavioral_types.py` — CognitiveState enum, thresholds, BehavioralSignals, BehavioralAssessment, BehavioralRecord
- `behavioral_engine.py` — five signal functions + three public callables + BehavioralControlEngine
- `__init__.py` — re-exports all public symbols

## CognitiveState vs AccountAction

`CognitiveState` (CAUTION/NORMAL/AGGRESSIVE/DEFENSIVE) is the **brain-level internal operating mode** — modulates reasoning thresholds, patience requirements, attention allocation.

`AccountAction` (`qde_account_types`) is the **account-level posture** — size, bar, exposure limits.

They are read by different downstream consumers and must NOT be conflated.

## Five tracked pathologies

| Signal | Function | Key formula |
|--------|----------|-------------|
| Confidence inflation | `_confidence_inflation` | signed mean(predicted − actual) |
| Overconfidence rate | `_overconfidence_rate` | fraction of high-conf (≥0.65) trades that lost |
| Drawdown response quality | `_drawdown_response_quality` | 1 − peak_aggressiveness/0.60 during drawdown |
| Aggressiveness drift | `_aggressiveness_drift` | least-squares slope normalised to [−1, +1] |
| Risk escalation | `_risk_escalation_rate` | second-derivative approx (only when second_half_slope > 0) |

## Risk escalation semantic rule

`_risk_escalation_rate` only fires when the **second half of the series has a positive slope** (risk actually rising). A series that declines steeply then flattens is "decelerating", not escalating — returning 0.0 prevents false positives.

## Pathology composite weights

```
0.25 × confidence inflation (only positive)
0.20 × overconfidence rate
0.20 × drawdown response (inverted: poor = high)
0.15 × aggressiveness drift (only positive)
0.10 × risk escalation
0.10 × metacognition blend (overconfidence_score + 0.5 × caution_level)
```

## State-transition FSM (adjust_state priority order)

1. HARD DEFENSIVE: consecutive_losses ≥ 3 OR drawdown ≥ 8% OR pathology ≥ 0.55
2. EUPHORIA GUARD: winning_streak ≥ 6 → CAUTION; AGGRESSIVE + euphoria_risk > 0.55 → CAUTION
3. CAUTION triggers: losses ≥ 2 OR pathology ≥ 0.25 OR drawdown ≥ 5%
4. DEFENSIVE recovery: consecutive_losses == 0 AND drawdown < 5% → CAUTION
5. AGGRESSIVE promotion: wins ≥ 4 AND pathology < 0.15 AND NORMAL current
6. CAUTION resolved: all clear → NORMAL
7. AGGRESSIVE fade: winning_streak < 4 → NORMAL

## Recovery throttling in adjust_state()

Worsening is always **immediate** (NORMAL → DEFENSIVE in one call).
Recovery is **throttled to one severity level per call** (DEFENSIVE → CAUTION, not DEFENSIVE → NORMAL). This prevents oscillation.

## Thresholds

- `LOSING_STREAK_CAUTION = 2`, `LOSING_STREAK_DEFENSIVE = 3` (mirrors qde_account_types)
- `WINNING_STREAK_AGGRESSIVE = 4`, `WINNING_STREAK_EUPHORIA = 6`
- `DRAWDOWN_CAUTION_THRESHOLD = 0.05`, `DRAWDOWN_DEFENSIVE_THRESHOLD = 0.08`
- `PATHOLOGY_CAUTION_THRESHOLD = 0.25`, `PATHOLOGY_DEFENSIVE_THRESHOLD = 0.55`
- `EUPHORIA_RISK_THRESHOLD = 0.55`
- `MAX_BEHAVIORAL_HISTORY = 200` (deque maxlen)

## Integration with MetacognitionEngine

`BehavioralSignals` accepts pre-computed `MetacognitiveAssessment` fields:
- `overconfidence_score` → blended into overconfidence_level
- `calibration_error` → used as confidence inflation when no paired trade data
- `metacognition_caution_level` → blended into pathology via meta_contrib

Pass these to avoid recomputing metacognition inside the behavioral engine.
