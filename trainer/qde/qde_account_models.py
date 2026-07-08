"""Phase S8 — Account Awareness data models.

`AccountSnapshot` is the raw, per-tick account context schema — exactly the
inputs listed in the S8 spec (balance, equity, margin, pnl, exposure,
correlation, drawdown, streaks, risk utilization, trade frequency). It is
intentionally flat and unopinionated: no thresholds, no judgement calls.

`AccountFeatures` is the engineered/derived view produced by
`qde_account_pipeline.AccountFeaturePipeline.compute_features()` — the
account-level analogue of `ObserverMemory`: things like drawdown-from-peak,
pnl trend, and trade-frequency-vs-baseline that require history, not just
the latest tick.

`AccountRiskAssessment` is the model's account-level judgement: a
`risk_signal` (0-1, continuous "how much risk is warranted right now") plus
a discrete `AccountAction`. This is advisory context for the trading
decision engines, the same way `ObservationResult` is for entries — the one
exception is `forces_no_trade`, which downstream engines are expected to
treat as a hard stop (capital protection overrides direction confidence,
per the S8 goal), everything else is a signal to weigh, not obey blindly.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from trainer.qde.qde_account_types import ACCOUNT_SCHEMA_VERSION, AccountAction

DEFAULT_ACCOUNT_ID = "default"


# ── AccountSnapshot (raw context schema) ─────────────────────────────────────


@dataclass
class AccountSnapshot:
    """One point-in-time reading of the account, exactly as the S8 spec lists."""

    account_id: str = DEFAULT_ACCOUNT_ID

    # ── Balance / equity / margin ───────────────────────────────────────────
    balance: float = 0.0
    equity: float = 0.0
    margin_level_pct: float = 0.0  # equity / used_margin * 100; 0 == no open positions
    free_margin: float = 0.0

    # ── P&L ──────────────────────────────────────────────────────────────────
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0  # relative to balance at start of day
    weekly_pnl: float = 0.0
    weekly_pnl_pct: float = 0.0  # relative to balance at start of week

    # ── Portfolio state ──────────────────────────────────────────────────────
    open_exposure: float = 0.0  # gross exposure, account currency
    portfolio_correlation: float = 0.0  # 0-1, avg pairwise correlation of open positions
    current_drawdown_pct: float = 0.0  # from equity high-water-mark, 0-1

    # ── Behavioral streaks ───────────────────────────────────────────────────
    winning_streak: int = 0
    losing_streak: int = 0

    # ── Risk & activity ──────────────────────────────────────────────────────
    risk_utilization_pct: float = 0.0  # fraction of allocated risk budget consumed, 0-1
    trades_today: int = 0
    trades_this_week: int = 0

    unix_ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "account_id": self.account_id,
            "balance": round(self.balance, 2),
            "equity": round(self.equity, 2),
            "margin_level_pct": round(self.margin_level_pct, 2),
            "free_margin": round(self.free_margin, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_pnl_pct": round(self.daily_pnl_pct, 4),
            "weekly_pnl": round(self.weekly_pnl, 2),
            "weekly_pnl_pct": round(self.weekly_pnl_pct, 4),
            "open_exposure": round(self.open_exposure, 2),
            "portfolio_correlation": round(self.portfolio_correlation, 4),
            "current_drawdown_pct": round(self.current_drawdown_pct, 4),
            "winning_streak": self.winning_streak,
            "losing_streak": self.losing_streak,
            "risk_utilization_pct": round(self.risk_utilization_pct, 4),
            "trades_today": self.trades_today,
            "trades_this_week": self.trades_this_week,
            "unix_ts": round(self.unix_ts, 3),
        }

    @classmethod
    def from_dict(cls, data: dict) -> AccountSnapshot:
        return cls(
            account_id=data.get("account_id", DEFAULT_ACCOUNT_ID),
            balance=data.get("balance", 0.0),
            equity=data.get("equity", 0.0),
            margin_level_pct=data.get("margin_level_pct", 0.0),
            free_margin=data.get("free_margin", 0.0),
            daily_pnl=data.get("daily_pnl", 0.0),
            daily_pnl_pct=data.get("daily_pnl_pct", 0.0),
            weekly_pnl=data.get("weekly_pnl", 0.0),
            weekly_pnl_pct=data.get("weekly_pnl_pct", 0.0),
            open_exposure=data.get("open_exposure", 0.0),
            portfolio_correlation=data.get("portfolio_correlation", 0.0),
            current_drawdown_pct=data.get("current_drawdown_pct", 0.0),
            winning_streak=data.get("winning_streak", 0),
            losing_streak=data.get("losing_streak", 0),
            risk_utilization_pct=data.get("risk_utilization_pct", 0.0),
            trades_today=data.get("trades_today", 0),
            trades_this_week=data.get("trades_this_week", 0),
            unix_ts=data.get("unix_ts", time.time()),
        )


# ── AccountFeatures (engineered view, requires history) ──────────────────────


@dataclass
class AccountFeatures:
    """Derived, history-aware features computed from an `AccountSnapshot`
    plus the account's recent `AccountSnapshot` history.

    These are what the pipeline's `assess_risk()` actually reasons over —
    raw snapshot fields answer "what is true right now", these answer "what
    does that mean given recent trajectory".
    """

    account_id: str = DEFAULT_ACCOUNT_ID

    equity_high_water_mark: float = 0.0
    drawdown_from_peak_pct: float = 0.0  # recomputed from HWM, may differ from snapshot's own field
    approaching_soft_drawdown_limit: bool = False
    breached_hard_drawdown_limit: bool = False

    pnl_trend: float = 0.0  # smoothed daily_pnl_pct momentum, -1..1
    daily_loss_limit_breached: bool = False
    weekly_loss_limit_breached: bool = False

    streak_signal: int = 0  # signed: +winning_streak, -losing_streak
    is_severe_losing_streak: bool = False

    margin_pressure: float = 0.0  # 0 (safe) .. 1 (critical), derived from margin_level_pct
    margin_critical: bool = False

    risk_utilization_trend: float = 0.0  # rate of change vs recent history, -1..1
    risk_utilization_critical: bool = False

    trade_frequency_baseline: float = 0.0  # rolling avg trades/day over history
    trade_frequency_ratio: float = 1.0  # today's trades / baseline (1.0 == normal)
    is_overtrading: bool = False

    portfolio_correlation_high: bool = False

    unix_ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "account_id": self.account_id,
            "equity_high_water_mark": round(self.equity_high_water_mark, 2),
            "drawdown_from_peak_pct": round(self.drawdown_from_peak_pct, 4),
            "approaching_soft_drawdown_limit": self.approaching_soft_drawdown_limit,
            "breached_hard_drawdown_limit": self.breached_hard_drawdown_limit,
            "pnl_trend": round(self.pnl_trend, 4),
            "daily_loss_limit_breached": self.daily_loss_limit_breached,
            "weekly_loss_limit_breached": self.weekly_loss_limit_breached,
            "streak_signal": self.streak_signal,
            "is_severe_losing_streak": self.is_severe_losing_streak,
            "margin_pressure": round(self.margin_pressure, 4),
            "margin_critical": self.margin_critical,
            "risk_utilization_trend": round(self.risk_utilization_trend, 4),
            "risk_utilization_critical": self.risk_utilization_critical,
            "trade_frequency_baseline": round(self.trade_frequency_baseline, 2),
            "trade_frequency_ratio": round(self.trade_frequency_ratio, 4),
            "is_overtrading": self.is_overtrading,
            "portfolio_correlation_high": self.portfolio_correlation_high,
            "unix_ts": round(self.unix_ts, 3),
        }


# ── AccountRiskAssessment (model output) ─────────────────────────────────────


@dataclass
class AccountRiskAssessment:
    """The account-level judgement produced for one `AccountSnapshot`.

    `forces_no_trade` is the one field downstream decision engines must
    treat as a hard override rather than advisory context: per the S8 goal,
    capital protection comes before profit generation, so a hard
    capital-protection breach must suppress new entries even when direction
    confidence is high.
    """

    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    account_id: str = DEFAULT_ACCOUNT_ID
    risk_signal: float = 0.5  # 0 (max defensive) .. 1 (max aggressive)
    action: AccountAction = AccountAction.NORMAL
    forces_no_trade: bool = False
    reason_codes: list[str] = field(default_factory=list)
    reasoning: str = ""
    schema_version: str = ACCOUNT_SCHEMA_VERSION
    unix_ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "assessment_id": self.assessment_id,
            "account_id": self.account_id,
            "risk_signal": round(self.risk_signal, 4),
            "action": self.action.value,
            "forces_no_trade": self.forces_no_trade,
            "reason_codes": list(self.reason_codes),
            "reasoning": self.reasoning,
            "schema_version": self.schema_version,
            "unix_ts": round(self.unix_ts, 3),
        }
