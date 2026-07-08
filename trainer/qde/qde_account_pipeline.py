"""Phase S8 — Account feature pipeline.

Two pure stages, mirroring `qde_observer.py`'s convention of no owned
scheduler/loop (the caller drives cadence, typically once per tick / once
per position-management review):

  1. `compute_features()`: `AccountSnapshot` + recent history ->
     `AccountFeatures`. All the "does this need memory" math lives here
     (drawdown from a rolling high-water-mark, pnl trend, trade-frequency
     vs an account-specific rolling baseline) so `AccountSnapshot` itself
     can stay a flat, stateless schema.

  2. `assess_risk()`: `AccountFeatures` -> `AccountRiskAssessment`. This is
     the "experienced fund manager" heuristic core: protect capital first,
     generate profit second. It is a scored, explainable heuristic (not a
     black box) so its reasoning can be inspected and iterated on, the same
     way `qde_observer.py`'s readiness scoring is.

Both stages are pure functions of their inputs — no hidden state, no I/O.
Persistence and telemetry are the caller's responsibility (see
`qde_account_database.py` / `qde_account_telemetry.py`), exactly like the
`QDEMarketObserver` / `QDEObserverDatabase` / `ObserverMonitor` split.
"""

from __future__ import annotations

from trainer.qde.qde_account_models import AccountFeatures, AccountRiskAssessment, AccountSnapshot
from trainer.qde.qde_account_types import (
    DAILY_LOSS_HARD_LIMIT_PCT,
    DRAWDOWN_APPROACH_BAND_PCT,
    DRAWDOWN_HARD_LIMIT_PCT,
    DRAWDOWN_SOFT_LIMIT_PCT,
    FREE_MARGIN_LOW_PCT,
    LOSING_STREAK_DEFENSIVE_THRESHOLD,
    LOSING_STREAK_SEVERE_THRESHOLD,
    MARGIN_LEVEL_CRITICAL_PCT,
    MARGIN_LEVEL_WARNING_PCT,
    PORTFOLIO_CORRELATION_HIGH,
    RISK_SIGNAL_BASELINE,
    RISK_SIGNAL_MAX,
    RISK_SIGNAL_MIN,
    RISK_UTILIZATION_CRITICAL_PCT,
    RISK_UTILIZATION_HIGH_PCT,
    TRADE_FREQUENCY_OVERTRADE_MULTIPLIER,
    WEEKLY_LOSS_HARD_LIMIT_PCT,
    WINNING_STREAK_CONFIDENCE_THRESHOLD,
    AccountAction,
)


def _clamp(x: float, lo: float = RISK_SIGNAL_MIN, hi: float = RISK_SIGNAL_MAX) -> float:
    return max(lo, min(hi, x))


class AccountFeaturePipeline:
    """Stateless transform: `AccountSnapshot` (+ history) -> features -> assessment."""

    def compute_features(
        self,
        snapshot: AccountSnapshot,
        history: list[dict] | None = None,
    ) -> AccountFeatures:
        """`history` is oldest -> newest list of `{"snapshot": ..., "assessment": ...}`
        dicts as returned by `QDEAccountDatabase.list_history()`. May be empty
        (cold start) — every derived feature degrades gracefully to a
        neutral value in that case rather than raising.
        """
        history = history or []
        past_snapshots = [h["snapshot"] for h in history]

        # ── Equity high-water-mark & drawdown-from-peak ─────────────────────
        # History-derived HWM is only as good as the history we have. With a
        # cold start (no/thin history) it trivially equals current equity,
        # which would misreport "new equity high" even when the account is
        # genuinely underwater. The snapshot's own `current_drawdown_pct`
        # (caller/broker-reported, always authoritative) is used to infer an
        # implied peak and we take whichever view shows the deeper drawdown.
        equities = [s.get("equity", 0.0) for s in past_snapshots] + [snapshot.equity]
        history_hwm = max([e for e in equities if e > 0], default=snapshot.equity)

        reported_dd = max(0.0, min(0.999, snapshot.current_drawdown_pct))
        implied_hwm = (
            snapshot.equity / (1 - reported_dd)
            if reported_dd > 0 and snapshot.equity > 0
            else snapshot.equity
        )

        hwm = max(history_hwm, implied_hwm)
        drawdown_from_peak = ((hwm - snapshot.equity) / hwm) if hwm > 0 else 0.0
        drawdown_from_peak = max(0.0, drawdown_from_peak, reported_dd)

        approaching_soft = (
            (DRAWDOWN_SOFT_LIMIT_PCT - DRAWDOWN_APPROACH_BAND_PCT)
            <= drawdown_from_peak
            < DRAWDOWN_HARD_LIMIT_PCT
        )
        breached_hard = drawdown_from_peak >= DRAWDOWN_HARD_LIMIT_PCT

        # ── P&L trend (simple momentum of daily_pnl_pct over recent history) ─
        past_daily_pnls = [s.get("daily_pnl_pct", 0.0) for s in past_snapshots[-10:]]
        if past_daily_pnls:
            avg_past = sum(past_daily_pnls) / len(past_daily_pnls)
            pnl_trend = _clamp(snapshot.daily_pnl_pct - avg_past, -1.0, 1.0)
        else:
            pnl_trend = _clamp(snapshot.daily_pnl_pct, -1.0, 1.0)

        daily_loss_breached = snapshot.daily_pnl_pct <= -DAILY_LOSS_HARD_LIMIT_PCT
        weekly_loss_breached = snapshot.weekly_pnl_pct <= -WEEKLY_LOSS_HARD_LIMIT_PCT

        # ── Streaks ──────────────────────────────────────────────────────────
        streak_signal = snapshot.winning_streak - snapshot.losing_streak
        severe_losing_streak = snapshot.losing_streak >= LOSING_STREAK_SEVERE_THRESHOLD

        # ── Margin pressure ──────────────────────────────────────────────────
        if snapshot.margin_level_pct <= 0:
            # No open positions => no margin usage => no pressure.
            margin_pressure = 0.0
        else:
            span = MARGIN_LEVEL_WARNING_PCT - MARGIN_LEVEL_CRITICAL_PCT
            margin_pressure = _clamp(
                (MARGIN_LEVEL_WARNING_PCT - snapshot.margin_level_pct) / span if span > 0 else 0.0
            )
        margin_critical = 0 < snapshot.margin_level_pct <= MARGIN_LEVEL_CRITICAL_PCT

        # ── Risk utilization trend ──────────────────────────────────────────
        past_utils = [s.get("risk_utilization_pct", 0.0) for s in past_snapshots[-10:]]
        if past_utils:
            avg_util = sum(past_utils) / len(past_utils)
            util_trend = _clamp(snapshot.risk_utilization_pct - avg_util, -1.0, 1.0)
        else:
            util_trend = 0.0
        util_critical = snapshot.risk_utilization_pct >= RISK_UTILIZATION_CRITICAL_PCT

        # ── Trade-frequency vs the account's own rolling baseline ───────────
        past_trade_counts = [s.get("trades_today", 0) for s in past_snapshots[-14:]]
        baseline = (
            sum(past_trade_counts) / len(past_trade_counts)
            if past_trade_counts
            else float(snapshot.trades_today)
        )
        if baseline > 0:
            freq_ratio = snapshot.trades_today / baseline
        else:
            freq_ratio = 1.0 if snapshot.trades_today == 0 else float(snapshot.trades_today)
        is_overtrading = (
            freq_ratio >= TRADE_FREQUENCY_OVERTRADE_MULTIPLIER and snapshot.trades_today >= 3
        )

        return AccountFeatures(
            account_id=snapshot.account_id,
            equity_high_water_mark=hwm,
            drawdown_from_peak_pct=drawdown_from_peak,
            approaching_soft_drawdown_limit=approaching_soft,
            breached_hard_drawdown_limit=breached_hard,
            pnl_trend=pnl_trend,
            daily_loss_limit_breached=daily_loss_breached,
            weekly_loss_limit_breached=weekly_loss_breached,
            streak_signal=streak_signal,
            is_severe_losing_streak=severe_losing_streak,
            margin_pressure=margin_pressure,
            margin_critical=margin_critical,
            risk_utilization_trend=util_trend,
            risk_utilization_critical=util_critical,
            trade_frequency_baseline=baseline,
            trade_frequency_ratio=freq_ratio,
            is_overtrading=is_overtrading,
            portfolio_correlation_high=snapshot.portfolio_correlation >= PORTFOLIO_CORRELATION_HIGH,
            unix_ts=snapshot.unix_ts,
        )

    def assess_risk(
        self,
        snapshot: AccountSnapshot,
        features: AccountFeatures,
    ) -> AccountRiskAssessment:
        """Fund-manager heuristic: capital protection first, profit second.

        Scoring walks from the neutral baseline, applying deductions for
        every headwind and modest additions for tailwinds, then clamps.
        Hard-stop conditions short-circuit straight to NO_TRADE /
        `forces_no_trade=True` regardless of the accumulated score, because
        those are capital-protection lines, not opinions to be averaged
        away by an otherwise-good day.
        """
        reasons: list[str] = []

        # ── Hard stops (override everything, including direction confidence) ─
        if features.breached_hard_drawdown_limit:
            reasons.append(
                f"drawdown_from_peak={features.drawdown_from_peak_pct:.2%} "
                f">= hard_limit={DRAWDOWN_HARD_LIMIT_PCT:.0%}"
            )
            return AccountRiskAssessment(
                account_id=snapshot.account_id,
                risk_signal=RISK_SIGNAL_MIN,
                action=AccountAction.NO_TRADE,
                forces_no_trade=True,
                reason_codes=["DRAWDOWN_HARD_LIMIT_BREACHED"],
                reasoning="; ".join(reasons),
                unix_ts=snapshot.unix_ts,
            )

        if features.margin_critical:
            return AccountRiskAssessment(
                account_id=snapshot.account_id,
                risk_signal=RISK_SIGNAL_MIN,
                action=AccountAction.NO_TRADE,
                forces_no_trade=True,
                reason_codes=["MARGIN_LEVEL_CRITICAL"],
                reasoning=f"margin_level={snapshot.margin_level_pct:.1f}% <= critical={MARGIN_LEVEL_CRITICAL_PCT:.0f}%",
                unix_ts=snapshot.unix_ts,
            )

        if features.daily_loss_limit_breached or features.weekly_loss_limit_breached:
            code = (
                "DAILY_LOSS_LIMIT_BREACHED"
                if features.daily_loss_limit_breached
                else "WEEKLY_LOSS_LIMIT_BREACHED"
            )
            return AccountRiskAssessment(
                account_id=snapshot.account_id,
                risk_signal=RISK_SIGNAL_MIN,
                action=AccountAction.NO_TRADE,
                forces_no_trade=True,
                reason_codes=[code],
                reasoning=f"pnl limit breached ({code})",
                unix_ts=snapshot.unix_ts,
            )

        if features.approaching_soft_drawdown_limit:
            # Matches the S8 example verbatim: NO_TRADE even if direction
            # confidence remains high.
            reasons.append(
                f"drawdown_from_peak={features.drawdown_from_peak_pct:.2%} "
                f"approaching soft_limit={DRAWDOWN_SOFT_LIMIT_PCT:.0%}"
            )
            return AccountRiskAssessment(
                account_id=snapshot.account_id,
                risk_signal=_clamp(RISK_SIGNAL_BASELINE * 0.3),
                action=AccountAction.NO_TRADE,
                forces_no_trade=True,
                reason_codes=["DRAWDOWN_APPROACHING_SOFT_LIMIT"],
                reasoning="; ".join(reasons),
                unix_ts=snapshot.unix_ts,
            )

        # ── Soft, additive scoring for everything else ───────────────────────
        score = RISK_SIGNAL_BASELINE

        if features.is_severe_losing_streak:
            score -= 0.35
            reasons.append(f"severe_losing_streak={snapshot.losing_streak}")
        elif snapshot.losing_streak >= LOSING_STREAK_DEFENSIVE_THRESHOLD:
            score -= 0.15
            reasons.append(f"losing_streak={snapshot.losing_streak}")

        if snapshot.winning_streak >= WINNING_STREAK_CONFIDENCE_THRESHOLD:
            score += 0.10
            reasons.append(f"winning_streak={snapshot.winning_streak}")

        if (
            features.drawdown_from_peak_pct <= 0.001
            and snapshot.equity >= features.equity_high_water_mark
        ):
            score += 0.15
            reasons.append("new_equity_high")

        score += 0.10 * features.pnl_trend
        if features.pnl_trend > 0:
            reasons.append(f"positive_pnl_trend={features.pnl_trend:.2f}")
        elif features.pnl_trend < 0:
            reasons.append(f"negative_pnl_trend={features.pnl_trend:.2f}")

        if features.margin_pressure > 0:
            score -= 0.30 * features.margin_pressure
            if (
                snapshot.margin_level_pct > 0
                and snapshot.margin_level_pct < MARGIN_LEVEL_WARNING_PCT
            ):
                reasons.append(f"margin_level={snapshot.margin_level_pct:.1f}% below warning")

        if snapshot.free_margin > 0 and snapshot.equity > 0:
            free_margin_pct = snapshot.free_margin / snapshot.equity
            if free_margin_pct < FREE_MARGIN_LOW_PCT:
                score -= 0.15
                reasons.append(f"free_margin_pct={free_margin_pct:.2%} low")

        if features.risk_utilization_critical:
            score -= 0.25
            reasons.append(f"risk_utilization={snapshot.risk_utilization_pct:.0%} critical")
        elif snapshot.risk_utilization_pct >= RISK_UTILIZATION_HIGH_PCT:
            score -= 0.10
            reasons.append(f"risk_utilization={snapshot.risk_utilization_pct:.0%} high")

        if features.is_overtrading:
            score -= 0.20
            reasons.append(f"trade_frequency_ratio={features.trade_frequency_ratio:.2f}x baseline")

        if features.portfolio_correlation_high:
            score -= 0.10
            reasons.append(f"portfolio_correlation={snapshot.portfolio_correlation:.2f} high")

        risk_signal = _clamp(score)

        if risk_signal <= 0.20:
            action = AccountAction.REDUCE_RISK
        elif risk_signal <= 0.40:
            action = AccountAction.DEFENSIVE
        elif risk_signal < 0.60:
            action = AccountAction.NORMAL
        elif risk_signal < 0.80:
            action = AccountAction.AGGRESSIVE
        else:
            action = AccountAction.AGGRESSIVE

        # Inconclusive/thin-evidence cold start -> WAIT rather than a
        # confident-looking NORMAL/AGGRESSIVE call.
        if (
            not reasons
            and snapshot.trades_today == 0
            and snapshot.losing_streak == 0
            and snapshot.winning_streak == 0
        ):
            action = AccountAction.WAIT
            reasons.append("insufficient_evidence")

        return AccountRiskAssessment(
            account_id=snapshot.account_id,
            risk_signal=risk_signal,
            action=action,
            forces_no_trade=False,
            reason_codes=_derive_reason_codes(features, snapshot),
            reasoning="; ".join(reasons) if reasons else "neutral baseline, no notable signal",
            unix_ts=snapshot.unix_ts,
        )


def _derive_reason_codes(features: AccountFeatures, snapshot: AccountSnapshot) -> list[str]:
    codes: list[str] = []
    if features.is_severe_losing_streak:
        codes.append("SEVERE_LOSING_STREAK")
    elif snapshot.losing_streak >= LOSING_STREAK_DEFENSIVE_THRESHOLD:
        codes.append("LOSING_STREAK")
    if snapshot.winning_streak >= WINNING_STREAK_CONFIDENCE_THRESHOLD:
        codes.append("WINNING_STREAK")
    if (
        features.drawdown_from_peak_pct <= 0.001
        and snapshot.equity >= features.equity_high_water_mark
    ):
        codes.append("NEW_EQUITY_HIGH")
    if features.margin_pressure > 0.3:
        codes.append("MARGIN_PRESSURE")
    if features.risk_utilization_critical:
        codes.append("RISK_UTILIZATION_CRITICAL")
    elif snapshot.risk_utilization_pct >= RISK_UTILIZATION_HIGH_PCT:
        codes.append("RISK_UTILIZATION_HIGH")
    if features.is_overtrading:
        codes.append("OVERTRADING")
    if features.portfolio_correlation_high:
        codes.append("HIGH_CORRELATION")
    return codes
