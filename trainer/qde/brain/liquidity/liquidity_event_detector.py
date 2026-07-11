"""QDE Brain — structural liquidity: event detector.

Pure, stateless detection: given a bar sequence and an existing
`LiquidityMap`, produces new `LiquidityLevel` candidates and
`LiquidityEvent` records describing what changed.

Detection principles
--------------------
* Everything is derived from price relationships and bar patterns.
  No hardcoded ICT / Smart-Money rules — no assertion like "equal highs
  above current price are liquidity targets for sellers".
* Each detector returns neutral structural observations:
  - "This bar's high is the maximum in a 5-bar window" → SWING_HIGH
  - "Three swing highs within 0.1% of each other" → EQUAL_HIGH cluster
  - "Bar high exceeded level, close returned below" → LEVEL_SWEPT
* The QDE model learns what these observations predict from outcome data.
  This module never decides what they mean.

Efficiency
----------
Swing and period-extreme detection only scans recent bars
(up to 4 × lookback from the tail) to avoid O(n²) full re-scans on
every tick.  Sweep / trap detection uses the last max(sweep_bars,
trap_bars, 20) bars for the same reason.
"""

from __future__ import annotations

import time as _time_module

from trainer.qde.brain.liquidity.liquidity_memory import (
    LevelSide,
    LevelStatus,
    LiquidityEvent,
    LiquidityEventType,
    LiquidityLevel,
    LiquidityLevelType,
    LiquidityMap,
    OHLCBar,
)

# ── Tuneable defaults ─────────────────────────────────────────────────────────

_SWING_LOOKBACK = 5  # bars each side to confirm a swing
_EQUAL_TOL_PCT = 0.001  # 0.1%: levels within this → equal high/low
_DEDUP_TOL_PCT = 0.002  # 0.2%: skip new level if existing one this close
_SWEEP_REVERSAL_BARS = 3  # bars for close-back to count as a sweep
_TRAP_BARS = 5  # bars for reversal after apparent breakout close
_CLUSTER_TOL_PCT = 0.003  # 0.3%: levels within this → STOP_CLUSTER
_MIN_CLUSTER_SIZE = 3
_PERIOD_DEDUP_TOL_PCT = 0.002  # 0.2%: skip period extreme if map already has it


# ── Detector ──────────────────────────────────────────────────────────────────


class LiquidityEventDetector:
    """Pure, stateless structural event detector.

    All methods are side-effect-free: they read bar sequences and the
    current map and return lists of new levels / events.  The caller
    (``LiquidityMapEngine``) decides what to merge into the map.

    Equal-level marking is the one exception: the ``_mark_equal_levels``
    pass mutates ``level_type`` and ``cluster_member_count`` on the
    passed-in level objects in-place (and those objects already belong
    to the caller) because it is a classification step over existing data,
    not a creation step.
    """

    def __init__(
        self,
        swing_lookback: int = _SWING_LOOKBACK,
        equal_tol_pct: float = _EQUAL_TOL_PCT,
        dedup_tol_pct: float = _DEDUP_TOL_PCT,
        sweep_reversal_bars: int = _SWEEP_REVERSAL_BARS,
        trap_bars: int = _TRAP_BARS,
        cluster_tol_pct: float = _CLUSTER_TOL_PCT,
        min_cluster_size: int = _MIN_CLUSTER_SIZE,
        period_dedup_tol: float = _PERIOD_DEDUP_TOL_PCT,
    ) -> None:
        self._swing_lb = swing_lookback
        self._eq_tol = equal_tol_pct
        self._dedup_tol = dedup_tol_pct
        self._sweep_bars = sweep_reversal_bars
        self._trap_bars = trap_bars
        self._cluster_tol = cluster_tol_pct
        self._min_cluster = min_cluster_size
        self._period_dedup = period_dedup_tol

    # ── Main entry point ──────────────────────────────────────────────────

    def scan(
        self,
        instrument: str,
        bars: list[OHLCBar],
        existing_map: LiquidityMap,
        now: float | None = None,
    ) -> tuple[list[LiquidityLevel], list[LiquidityEvent]]:
        """Full detection pass over the bar history.

        Returns ``(new_levels, new_events)``.  The engine decides whether
        to add these to the map or deduplicate against existing entries.

        The method is safe to call on every tick; it only scans a bounded
        recent window regardless of total bar-history length.
        """
        ts = now if now is not None else _time_module.time()

        if len(bars) < self._swing_lb * 2 + 1:
            return [], []

        new_levels: list[LiquidityLevel] = []
        new_events: list[LiquidityEvent] = []

        # Precompute sets of already-known prices to avoid re-emitting them.
        known_high: set[float] = {
            lv.price
            for lv in existing_map.levels.values()
            if lv.side == LevelSide.HIGH and lv.is_active()
        }
        known_low: set[float] = {
            lv.price
            for lv in existing_map.levels.values()
            if lv.side == LevelSide.LOW and lv.is_active()
        }

        # 1. Swing highs / lows from recent bars.
        swing_highs, swing_lows = self._detect_swings(instrument, bars, ts, known_high, known_low)
        new_levels.extend(swing_highs)
        new_levels.extend(swing_lows)

        for lv in swing_highs + swing_lows:
            new_events.append(
                LiquidityEvent(
                    instrument=instrument,
                    event_type=LiquidityEventType.LEVEL_FORMED,
                    level_id=lv.level_id,
                    level_type=lv.level_type,
                    level_price=lv.price,
                    bar_timestamp=lv.source_bar_timestamp,
                    detected_at=ts,
                    context=f"swing({self._swing_lb}-bar window)",
                )
            )

        # 2. Equal-level classification over new + existing active levels.
        all_high = [
            lv
            for lv in existing_map.levels.values()
            if lv.side == LevelSide.HIGH and lv.is_active()
        ] + swing_highs
        all_low = [
            lv for lv in existing_map.levels.values() if lv.side == LevelSide.LOW and lv.is_active()
        ] + swing_lows

        eq_events = self._mark_equal_levels(instrument, all_high, all_low, ts)
        new_events.extend(eq_events)

        # 3. Session / daily / weekly extremes from bar history.
        period_levels, period_events = self._detect_period_extremes(
            instrument, bars, ts, existing_map
        )
        new_levels.extend(period_levels)
        new_events.extend(period_events)

        # 4. Sweeps and breakout traps against existing intact levels.
        sweep_evts, trap_evts = self._detect_sweeps_and_traps(instrument, bars, existing_map, ts)
        new_events.extend(sweep_evts)
        new_events.extend(trap_evts)

        # 5. Stop clusters over the union of existing + new active levels.
        all_active = [lv for lv in existing_map.levels.values() if lv.is_active()] + new_levels
        cluster_levels, cluster_evts = self._detect_stop_clusters(
            instrument, all_active, ts, existing_map
        )
        new_levels.extend(cluster_levels)
        new_events.extend(cluster_evts)

        return new_levels, new_events

    # ── Swing detection ───────────────────────────────────────────────────

    def _detect_swings(
        self,
        instrument: str,
        bars: list[OHLCBar],
        ts: float,
        known_high: set[float],
        known_low: set[float],
    ) -> tuple[list[LiquidityLevel], list[LiquidityLevel]]:
        """Detect new swing highs and lows in the most recent bars.

        A bar at index ``i`` is a swing high when its ``high`` is strictly
        the maximum of ``bars[i-lb … i+lb]``.  Symmetric by design —
        no directional bias in the detection criterion.

        Only scans the last ``4 × swing_lookback`` bars from the end
        (minus the ``lb``-bar confirmation tail) to bound each-tick cost.
        """
        lb = self._swing_lb
        n = len(bars)
        start = max(lb, n - lb * 4)
        end = n - lb  # need lb bars after to confirm the swing

        highs: list[LiquidityLevel] = []
        lows: list[LiquidityLevel] = []

        for i in range(start, end):
            lo_i = max(0, i - lb)
            hi_i = min(n, i + lb + 1)
            window_h = [bars[j].high for j in range(lo_i, hi_i)]
            window_l = [bars[j].low for j in range(lo_i, hi_i)]
            bh = bars[i].high
            bl = bars[i].low

            if bh == max(window_h) and not self._price_near_set(bh, known_high, self._dedup_tol):
                lv = LiquidityLevel(
                    instrument=instrument,
                    level_type=LiquidityLevelType.SWING_HIGH,
                    side=LevelSide.HIGH,
                    price=bh,
                    strength=0.40,
                    formed_at=ts,
                    source_bar_timestamp=bars[i].timestamp,
                )
                highs.append(lv)
                known_high.add(bh)

            if bl == min(window_l) and not self._price_near_set(bl, known_low, self._dedup_tol):
                lv = LiquidityLevel(
                    instrument=instrument,
                    level_type=LiquidityLevelType.SWING_LOW,
                    side=LevelSide.LOW,
                    price=bl,
                    strength=0.40,
                    formed_at=ts,
                    source_bar_timestamp=bars[i].timestamp,
                )
                lows.append(lv)
                known_low.add(bl)

        return highs, lows

    # ── Equal-level classification ─────────────────────────────────────────

    def _mark_equal_levels(
        self,
        instrument: str,
        high_levels: list[LiquidityLevel],
        low_levels: list[LiquidityLevel],
        ts: float,
    ) -> list[LiquidityEvent]:
        """Classify swing pairs/clusters as EQUAL_HIGH / EQUAL_LOW.

        Mutates ``level_type`` and ``cluster_member_count`` in-place on the
        passed-in objects — only for levels that are being promoted to an
        equal-level classification for the first time.

        Returns ``CLUSTER_FORMED`` events for newly discovered groups.
        """
        events: list[LiquidityEvent] = []

        for group, eq_type in [
            (high_levels, LiquidityLevelType.EQUAL_HIGH),
            (low_levels, LiquidityLevelType.EQUAL_LOW),
        ]:
            clusters = self._cluster_by_price(group, self._eq_tol)
            for cluster in clusters:
                if len(cluster) < 2:
                    continue
                for lv in cluster:
                    if lv.level_type not in (
                        LiquidityLevelType.EQUAL_HIGH,
                        LiquidityLevelType.EQUAL_LOW,
                    ):
                        lv.level_type = eq_type
                        lv.cluster_member_count = len(cluster)
                        lv.strength = min(1.0, lv.strength + 0.10 * len(cluster))
                intact = [lv for lv in cluster if lv.is_active()]
                if len(intact) >= 2:
                    avg_p = sum(lv.price for lv in intact) / len(intact)
                    events.append(
                        LiquidityEvent(
                            instrument=instrument,
                            event_type=LiquidityEventType.CLUSTER_FORMED,
                            level_id=intact[0].level_id,
                            level_type=eq_type,
                            level_price=avg_p,
                            bar_timestamp=ts,
                            detected_at=ts,
                            context=(f"{len(intact)} equal levels " f"within {self._eq_tol:.2%}"),
                        )
                    )

        return events

    # ── Period extremes (session / daily / weekly) ────────────────────────

    def _detect_period_extremes(
        self,
        instrument: str,
        bars: list[OHLCBar],
        ts: float,
        existing_map: LiquidityMap,
    ) -> tuple[list[LiquidityLevel], list[LiquidityEvent]]:
        """Identify the highest and lowest price in each session block,
        calendar day, and calendar week.

        Levels are only emitted if no existing active level in the map is
        already within ``_period_dedup_tol`` of the candidate price.
        """
        if not bars:
            return [], []

        from collections import defaultdict

        def _day_key(t: float) -> str:
            s = _time_module.gmtime(t)
            return f"{s.tm_year}-{s.tm_yday:03d}"

        def _week_key(t: float) -> str:
            s = _time_module.gmtime(t)
            return f"{s.tm_year}-W{s.tm_yday // 7:02d}"

        def _session_key(t: float) -> str:
            hour = _time_module.gmtime(t).tm_hour
            day = _day_key(t)
            if hour < 8:
                return f"as-{day}"
            if hour < 16:
                return f"lo-{day}"
            return f"ny-{day}"

        day_bars: dict[str, list[OHLCBar]] = defaultdict(list)
        week_bars: dict[str, list[OHLCBar]] = defaultdict(list)
        session_bars: dict[str, list[OHLCBar]] = defaultdict(list)

        for bar in bars:
            if bar.timestamp > 0:
                day_bars[_day_key(bar.timestamp)].append(bar)
                week_bars[_week_key(bar.timestamp)].append(bar)
                session_bars[_session_key(bar.timestamp)].append(bar)

        # Dedup only against levels already present in the map — prevents
        # re-emitting a level that a previous scan already captured.
        # Do NOT accumulate newly-discovered period levels into this set:
        # valid consecutive-day extremes may be within tolerance of each
        # other and should all be emitted.
        already_in_map: set[float] = {
            lv.price for lv in existing_map.levels.values() if lv.is_active()
        }

        new_levels: list[LiquidityLevel] = []
        new_events: list[LiquidityEvent] = []

        for period_dict, high_type, low_type in [
            (day_bars, LiquidityLevelType.DAILY_HIGH, LiquidityLevelType.DAILY_LOW),
            (week_bars, LiquidityLevelType.WEEKLY_HIGH, LiquidityLevelType.WEEKLY_LOW),
            (session_bars, LiquidityLevelType.SESSION_HIGH, LiquidityLevelType.SESSION_LOW),
        ]:
            for _key, period_b in period_dict.items():
                if not period_b:
                    continue
                ph = max(b.high for b in period_b)
                pl = min(b.low for b in period_b)
                tsh = next(b.timestamp for b in period_b if b.high == ph)
                tsl = next(b.timestamp for b in period_b if b.low == pl)

                for p, lt, side, src_t in [
                    (ph, high_type, LevelSide.HIGH, tsh),
                    (pl, low_type, LevelSide.LOW, tsl),
                ]:
                    if self._price_near_set(p, already_in_map, self._period_dedup):
                        continue
                    lv = LiquidityLevel(
                        instrument=instrument,
                        level_type=lt,
                        side=side,
                        price=p,
                        strength=0.55,  # period extremes start slightly stronger
                        formed_at=ts,
                        source_bar_timestamp=src_t,
                    )
                    new_levels.append(lv)
                    new_events.append(
                        LiquidityEvent(
                            instrument=instrument,
                            event_type=LiquidityEventType.LEVEL_FORMED,
                            level_id=lv.level_id,
                            level_type=lt,
                            level_price=p,
                            bar_timestamp=src_t,
                            detected_at=ts,
                            context=lt.value.lower(),
                        )
                    )

        return new_levels, new_events

    # ── Sweep + breakout-trap detection ───────────────────────────────────

    def _detect_sweeps_and_traps(
        self,
        instrument: str,
        bars: list[OHLCBar],
        existing_map: LiquidityMap,
        ts: float,
    ) -> tuple[list[LiquidityEvent], list[LiquidityEvent]]:
        """Detect level sweeps and breakout traps in recent bars.

        Sweep:
          HIGH-side level: bar.high > level.price AND bar.close < level.price
          LOW-side  level: bar.low  < level.price AND bar.close > level.price

        Breakout trap:
          HIGH-side: bar.close > level.price, but a subsequent close
          within ``_trap_bars`` bars returns below it.
          LOW-side: symmetric.

        Only intact or tested levels are checked to avoid double-reporting
        already-resolved levels.
        """
        sweep_events: list[LiquidityEvent] = []
        trap_events: list[LiquidityEvent] = []

        if len(bars) < 2:
            return sweep_events, trap_events

        window = bars[-max(self._sweep_bars + self._trap_bars + 4, 24) :]
        intact = [
            lv
            for lv in existing_map.levels.values()
            if lv.status in (LevelStatus.INTACT, LevelStatus.TESTED)
        ]

        for lv in intact:
            p = lv.price
            for i, bar in enumerate(window):
                if lv.side == LevelSide.HIGH:
                    # Sweep: wick above and close back below
                    if bar.high > p * (1.0 + 1e-5) and bar.close < p:
                        sweep_events.append(
                            self._sweep_event(
                                instrument,
                                lv,
                                bar,
                                ts,
                                f"high={bar.high:.5f} > {p:.5f}, close={bar.close:.5f}",
                            )
                        )
                        lv.status = LevelStatus.SWEPT
                        lv.sweep_count += 1
                        lv.last_status_change_at = ts
                        break

                    # Breakout trap: close above, then returns below
                    if bar.close > p:
                        subsequent = window[i + 1 : i + 1 + self._trap_bars]
                        if any(b.close < p for b in subsequent):
                            trap_events.append(
                                self._trap_event(
                                    instrument,
                                    lv,
                                    bar,
                                    ts,
                                    f"close={bar.close:.5f} > {p:.5f}, returned",
                                )
                            )
                            lv.level_type = LiquidityLevelType.BREAKOUT_TRAP
                            lv.last_status_change_at = ts
                            break

                elif lv.side == LevelSide.LOW:
                    # Sweep: wick below and close back above
                    if bar.low < p * (1.0 - 1e-5) and bar.close > p:
                        sweep_events.append(
                            self._sweep_event(
                                instrument,
                                lv,
                                bar,
                                ts,
                                f"low={bar.low:.5f} < {p:.5f}, close={bar.close:.5f}",
                            )
                        )
                        lv.status = LevelStatus.SWEPT
                        lv.sweep_count += 1
                        lv.last_status_change_at = ts
                        break

                    # Breakout trap: close below, then returns above
                    if bar.close < p:
                        subsequent = window[i + 1 : i + 1 + self._trap_bars]
                        if any(b.close > p for b in subsequent):
                            trap_events.append(
                                self._trap_event(
                                    instrument,
                                    lv,
                                    bar,
                                    ts,
                                    f"close={bar.close:.5f} < {p:.5f}, returned",
                                )
                            )
                            lv.level_type = LiquidityLevelType.BREAKOUT_TRAP
                            lv.last_status_change_at = ts
                            break

        return sweep_events, trap_events

    # ── Stop-cluster detection ─────────────────────────────────────────────

    def _detect_stop_clusters(
        self,
        instrument: str,
        all_levels: list[LiquidityLevel],
        ts: float,
        existing_map: LiquidityMap,
    ) -> tuple[list[LiquidityLevel], list[LiquidityEvent]]:
        """Find dense price clusters of ≥ ``_min_cluster_size`` active
        levels within ``_cluster_tol_pct`` of each other.

        Each cluster becomes one ``STOP_CLUSTER`` pseudo-level at the
        cluster's mean price, unless a cluster-type level already exists
        nearby in the map.
        """
        new_levels: list[LiquidityLevel] = []
        events: list[LiquidityEvent] = []

        if len(all_levels) < self._min_cluster:
            return new_levels, events

        existing_cluster_prices: set[float] = {
            lv.price
            for lv in existing_map.levels.values()
            if lv.level_type == LiquidityLevelType.STOP_CLUSTER
        }

        clusters = self._cluster_by_price(all_levels, self._cluster_tol)
        for cluster in clusters:
            if len(cluster) < self._min_cluster:
                continue
            avg_p = sum(lv.price for lv in cluster) / len(cluster)
            if self._price_near_set(avg_p, existing_cluster_prices, self._cluster_tol):
                continue

            sc = LiquidityLevel(
                instrument=instrument,
                level_type=LiquidityLevelType.STOP_CLUSTER,
                side=LevelSide.NONE,
                price=avg_p,
                strength=min(1.0, 0.50 + 0.08 * len(cluster)),
                formed_at=ts,
                cluster_member_count=len(cluster),
                notes=f"{len(cluster)} levels ±{self._cluster_tol:.2%}",
            )
            new_levels.append(sc)
            existing_cluster_prices.add(avg_p)
            events.append(
                LiquidityEvent(
                    instrument=instrument,
                    event_type=LiquidityEventType.STOP_CLUSTER_FOUND,
                    level_id=sc.level_id,
                    level_type=LiquidityLevelType.STOP_CLUSTER,
                    level_price=avg_p,
                    bar_timestamp=ts,
                    detected_at=ts,
                    context=f"{len(cluster)} levels within {self._cluster_tol:.2%}",
                )
            )

        return new_levels, events

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _price_near_set(price: float, known: set[float], tol: float) -> bool:
        """True if ``price`` is within ``tol`` of any value in ``known``."""
        return any(abs(price - kp) / max(price, 1e-9) <= tol for kp in known)

    @staticmethod
    def _cluster_by_price(levels: list[LiquidityLevel], tol: float) -> list[list[LiquidityLevel]]:
        """Group levels into price-adjacent clusters.

        Two levels belong to the same cluster if the price difference
        relative to the cluster's *first* member price is within ``tol``.
        Levels are sorted by price before grouping.
        """
        if not levels:
            return []
        sorted_l = sorted(levels, key=lambda lv: lv.price)
        clusters: list[list[LiquidityLevel]] = []
        current: list[LiquidityLevel] = [sorted_l[0]]
        for lv in sorted_l[1:]:
            ref = current[0].price
            if abs(lv.price - ref) / max(ref, 1e-9) <= tol:
                current.append(lv)
            else:
                clusters.append(current)
                current = [lv]
        clusters.append(current)
        return clusters

    @staticmethod
    def _sweep_event(
        instrument: str,
        lv: LiquidityLevel,
        bar: OHLCBar,
        ts: float,
        ctx: str,
    ) -> LiquidityEvent:
        return LiquidityEvent(
            instrument=instrument,
            event_type=LiquidityEventType.LEVEL_SWEPT,
            level_id=lv.level_id,
            level_type=lv.level_type,
            level_price=lv.price,
            bar_timestamp=bar.timestamp,
            detected_at=ts,
            context=ctx,
        )

    @staticmethod
    def _trap_event(
        instrument: str,
        lv: LiquidityLevel,
        bar: OHLCBar,
        ts: float,
        ctx: str,
    ) -> LiquidityEvent:
        return LiquidityEvent(
            instrument=instrument,
            event_type=LiquidityEventType.BREAKOUT_TRAP,
            level_id=lv.level_id,
            level_type=lv.level_type,
            level_price=lv.price,
            bar_timestamp=bar.timestamp,
            detected_at=ts,
            context=ctx,
        )
