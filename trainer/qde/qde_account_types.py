"""Phase S8 — Account Awareness: enums and named risk-band thresholds.

QDE Phase S7 gave the model an always-on per-*symbol* memory. Phase S8 adds
an always-on per-*account* memory sitting alongside it: the model should
reason not just about "is this symbol tradeable right now" but "should I be
trading at all, right now, given the state of the account".

`AccountAction` is the model's account-level posture — advisory context for
the existing decision engines, in the same spirit as `ObserverState` in
`qde_observer_types.py`: it never itself places or blocks an order. Only
`NO_TRADE` carries a `forces_no_trade` override (see `qde_account_models.py`)
because a fund manager who is over a hard capital-protection line does not
"suggest" caution, they stop.
"""

from __future__ import annotations

from enum import Enum


class AccountAction(str, Enum):
    """The account-level posture the model recommends right now."""

    AGGRESSIVE = "AGGRESSIVE"  # capital healthy, momentum favorable -> lean in
    NORMAL = "NORMAL"  # baseline posture, no notable tailwind or headwind
    DEFENSIVE = "DEFENSIVE"  # early warning signs -> smaller size, higher bar
    REDUCE_RISK = "REDUCE_RISK"  # actively cutting exposure / risk budget
    NO_TRADE = "NO_TRADE"  # stop opening new risk for the remainder of the window
    WAIT = "WAIT"  # conditions inconclusive -> hold, gather more information


# ── Risk-band thresholds (named constants, no magic numbers in the pipeline) ─

# Drawdown (current equity drawdown from high-water-mark), as a fraction (0-1).
DRAWDOWN_SOFT_LIMIT_PCT = 0.08  # approaching -> caution, may still trade smaller
DRAWDOWN_HARD_LIMIT_PCT = 0.15  # breached -> NO_TRADE regardless of confidence
DRAWDOWN_APPROACH_BAND_PCT = 0.02  # within this band of the soft limit counts as "approaching"

# Daily / weekly loss limits, as a fraction of starting balance for the period.
DAILY_LOSS_SOFT_LIMIT_PCT = 0.03
DAILY_LOSS_HARD_LIMIT_PCT = 0.06
WEEKLY_LOSS_SOFT_LIMIT_PCT = 0.08
WEEKLY_LOSS_HARD_LIMIT_PCT = 0.15

# Margin level (equity / used margin * 100). Below warning -> defensive;
# below critical -> forced no-trade (broker margin-call territory).
MARGIN_LEVEL_WARNING_PCT = 200.0
MARGIN_LEVEL_CRITICAL_PCT = 120.0

# Free margin as a fraction of equity; low free margin limits new position size.
FREE_MARGIN_LOW_PCT = 0.25

# Consecutive-loss / consecutive-win streak thresholds.
LOSING_STREAK_DEFENSIVE_THRESHOLD = 3  # 3 losses in a row -> start dialing back
LOSING_STREAK_SEVERE_THRESHOLD = 6  # 6 losses in a row -> deeply defensive (matches S8 example)
WINNING_STREAK_CONFIDENCE_THRESHOLD = 4  # sustained wins -> modest confidence lift

# Risk budget utilization (fraction of allocated risk budget already consumed).
RISK_UTILIZATION_HIGH_PCT = 0.75
RISK_UTILIZATION_CRITICAL_PCT = 0.90

# Portfolio correlation (0-1, 1 = all positions moving together = no diversification).
PORTFOLIO_CORRELATION_HIGH = 0.70

# Trade frequency: ratio of trades-in-window to the account's own rolling
# baseline. > multiplier => overtrading pattern, independent of any single
# trade's quality.
TRADE_FREQUENCY_OVERTRADE_MULTIPLIER = 2.0

# Risk signal bounds (output is always clamped to this range).
RISK_SIGNAL_MIN = 0.0
RISK_SIGNAL_MAX = 1.0
RISK_SIGNAL_BASELINE = 0.5  # "normal" posture, no data yet / neutral evidence

ACCOUNT_SCHEMA_VERSION = "s8_v1"
