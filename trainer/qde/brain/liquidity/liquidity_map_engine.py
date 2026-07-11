"""QDE Brain — structural liquidity: top-level orchestrator.

``LiquidityMapEngine`` is the single entry point for callers.  It owns a
``LiquidityMemory`` (bar history + per-symbol maps), orchestrates the
``LiquidityEventDetector`` on every bar, and exposes four integration
points consumed by the QDE brain:

  context_for_inference   → ``liq_*`` dict merged into brain_context
  conviction_modifier     → ±0.25 structural adjustment on a hypothesis
  narrative_context       → enrichment dict for NarrativeBuilder
  opportunity_rank_signal → 0-1 structural quality for opportunity ranking
  execution_timing_signal → 0-1 structural proximity / timing quality

Design constraints
------------------
* No hardcoded ICT / Smart-Money rules.  Signals are derived from
  neutral structural observations (density, proximity, sweep recency)
  not from directional assertions ("equal highs = sell").
* ``QDEDecision`` is never imported.  Callers decide what to do with the
  signals; this engine only describes structural state.
* Stateless helpers (Tracker, Detector) are created once and reused.
"""

from __future__ import annotations

import time

from trainer.qde.brain.liquidity.liquidity_event_detector import LiquidityEventDetector
from trainer.qde.brain.liquidity.liquidity_memory import (
    LevelSide,
    LevelStatus,
    LiquidityEventType,
    LiquidityMemory,
    OHLCBar,
)
from trainer.qde.brain.liquidity.liquidity_tracker import LiquidityTracker

# ── Tuneable defaults ─────────────────────────────────────────────────────────

_TOUCH_TOL_PCT     = 0.001   # 0.1%: counts as "touching" a level
_DENSITY_RADIUS    = 0.02    # 2%: radius for density calculations
_BLOCKING_DISTANCE = 0.01    # 1%: range used by conviction modifier
_SWEEP_BOOST       = 0.08    # conviction boost when recent sweeps exist
_MAX_CONVICTION_MOD = 0.25   # cap on conviction_modifier magnitude


class LiquidityMapEngine:
    """Structural liquidity orchestrator.

    Lifecycle:
      1. ``on_bar`` / ``on_bars`` — ingest OHLCV bars; detection runs on
         each push.
      2. ``context_for_inference`` — call per tick to get ``liq_*`` keys.
      3. ``conviction_modifier`` — call when scoring a hypothesis.
      4. ``narrative_context`` — call inside NarrativeBuilder integration.
      5. ``opportunity_rank_signal`` / ``execution_timing_signal`` — call
         when ranking or timing an opportunity.
    """

    def __init__(
        self,
        bar_history_depth:   int   = 500,
        max_levels:          int   = 300,
        touch_tol_pct:       float = _TOUCH_TOL_PCT,
        density_radius:      float = _DENSITY_RADIUS,
        blocking_distance:   float = _BLOCKING_DISTANCE,
        sweep_recency_boost: float = _SWEEP_BOOST,
        max_conviction_mod:  float = _MAX_CONVICTION_MOD,
        # Detector parameters — defaults match the detector's own defaults.
        swing_lookback:      int   = 5,
        equal_tol_pct:       float = 0.001,
        cluster_tol_pct:     float = 0.003,
        min_cluster_size:    int   = 3,
    ) -> None:
        self._memory    = LiquidityMemory(bar_history_depth, max_levels)
        self._detector  = LiquidityEventDetector(
            swing_lookback=swing_lookback,
            equal_tol_pct=equal_tol_pct,
            cluster_tol_pct=cluster_tol_pct,
            min_cluster_size=min_cluster_size,
        )
        self._tracker   = LiquidityTracker()

        self._touch_tol    = touch_tol_pct
        self._density_rad  = density_radius
        self._block_dist   = blocking_distance
        self._sweep_boost  = sweep_recency_boost
        self._max_mod      = max_conviction_mod

    # ── Ingestion ─────────────────────────────────────────────────────────

    def on_bar(
        self,
        instrument: str,
        bar: OHLCBar,
        now: float | None = None,
    ) -> None:
        """Ingest one new OHLCV bar and run a detection pass.

        The detection pass is incremental: only the recent tail of bars
        is re-examined, so per-tick cost is bounded.
        """
        self._memory.push_bar(instrument, bar)
        self._run_detection(instrument, now=now)

    def on_bars(
        self,
        instrument: str,
        bars: list[OHLCBar],
        now: float | None = None,
    ) -> None:
        """Batch-ingest bars and run one detection pass at the end.

        More efficient than calling ``on_bar`` in a loop when loading
        historical data, since detection runs only once.
        """
        self._memory.push_bars(instrument, bars)
        self._run_detection(instrument, now=now)

    def _run_detection(self, instrument: str, now: float | None = None) -> None:
        bars = self._memory.bars(instrument)
        mp   = self._memory.map_for(instrument)
        ts   = now if now is not None else time.time()

        new_levels, new_events = self._detector.scan(instrument, bars, mp, now=ts)

        # Merge new levels (skip if a duplicate is already in the map).
        existing_prices = {lv.price for lv in mp.levels.values() if lv.is_active()}
        for lv in new_levels:
            if not any(
                abs(lv.price - ep) / max(lv.price, 1e-9) <= self._detector._dedup_tol
                for ep in existing_prices
            ):
                mp.levels[lv.level_id] = lv
                existing_prices.add(lv.price)

        # Append events.
        for ev in new_events:
            mp.events.append(ev)

        # Check for touches on the current bar's close, update status.
        last_bar = mp.last_bar
        if last_bar and last_bar.close > 0:
            touched = self._tracker.check_touch(mp, last_bar.close, self._touch_tol)
            for lid in touched:
                self._tracker.mark_tested(mp, lid, now=ts)

        # Prune broken levels if map is getting large.
        self._memory.prune_broken_levels(instrument)
        self._tracker.prune_old_broken(mp)

    # ── Integration point 1: brain_context dict ───────────────────────────

    def context_for_inference(
        self,
        instrument: str,
        current_price: float | None = None,
    ) -> dict:
        """Return ``liq_*`` namespaced dict for merging into brain_context.

        If ``current_price`` is None, uses the close of the last bar
        stored for the instrument.

        Keys
        ----
        liq_total_active_levels   int   — total active levels in map
        liq_intact_count          int   — INTACT levels
        liq_swept_count           int   — ever-swept levels
        liq_density_above         float — levels-per-% above price (2% window)
        liq_density_below         float — levels-per-% below price (2% window)
        liq_density_asymmetry     float — signed (above − below) / total, -1..+1
        liq_nearest_above_pct     float — dist to nearest level above, as fraction
        liq_nearest_below_pct     float — dist to nearest level below, as fraction
        liq_void_above_pct        float — same as nearest_above_pct (alias)
        liq_void_below_pct        float — same as nearest_below_pct (alias)
        liq_structural_density    float — 0-1 density within 2% radius
        liq_sweep_recency_score   float — 0-1 recency of sweep activity
        liq_recent_sweeps_1h      int   — sweep events in last 1 h
        liq_nearest_above_type    str   — level_type of nearest level above
        liq_nearest_below_type    str   — level_type of nearest level below
        liq_targets_above         list  — top-5 structural targets above
        liq_targets_below         list  — top-5 structural targets below
        liq_level_type_counts     dict  — counts per LiquidityLevelType
        liq_last_event_type       str   — most recent event type or ""
        """
        mp    = self._memory.map_for(instrument)
        price = current_price or (mp.last_bar.close if mp.last_bar else 0.0)
        now   = time.time()

        if price <= 0.0:
            return self._empty_context()

        na = self._tracker.nearest_above(mp, price)
        nb = self._tracker.nearest_below(mp, price)

        active_lvls = [lv for lv in mp.levels.values() if lv.is_active()]
        intact_cnt  = sum(1 for lv in active_lvls if lv.status == LevelStatus.INTACT)
        swept_cnt   = sum(1 for lv in mp.levels.values() if lv.sweep_count > 0)

        last_ev = self._tracker.most_recent_event(mp)

        return {
            "liq_total_active_levels": len(active_lvls),
            "liq_intact_count":        intact_cnt,
            "liq_swept_count":         swept_cnt,
            "liq_density_above":       round(
                self._tracker.density_above(mp, price, self._density_rad), 4
            ),
            "liq_density_below":       round(
                self._tracker.density_below(mp, price, self._density_rad), 4
            ),
            "liq_density_asymmetry":   round(
                self._tracker.density_asymmetry(mp, price, self._density_rad), 4
            ),
            "liq_nearest_above_pct":   round(
                self._tracker.void_above(mp, price), 5
            ),
            "liq_nearest_below_pct":   round(
                self._tracker.void_below(mp, price), 5
            ),
            "liq_void_above_pct":      round(
                self._tracker.void_above(mp, price), 5
            ),
            "liq_void_below_pct":      round(
                self._tracker.void_below(mp, price), 5
            ),
            "liq_structural_density":  round(
                self._tracker.structural_density_score(mp, price, self._density_rad), 4
            ),
            "liq_sweep_recency_score": round(
                self._tracker.sweep_recency_score(mp, now=now), 4
            ),
            "liq_recent_sweeps_1h":    self._tracker.recent_sweep_count(
                mp, lookback_seconds=3600, now=now
            ),
            "liq_nearest_above_type":  na.level_type.value if na else "",
            "liq_nearest_below_type":  nb.level_type.value if nb else "",
            "liq_targets_above":       self._tracker.where_may_liquidity_exist_next(
                mp, price, direction=+1
            ),
            "liq_targets_below":       self._tracker.where_may_liquidity_exist_next(
                mp, price, direction=-1
            ),
            "liq_level_type_counts":   self._tracker.level_type_counts(mp),
            "liq_last_event_type":     last_ev.event_type.value if last_ev else "",
        }

    @staticmethod
    def _empty_context() -> dict:
        return {
            "liq_total_active_levels": 0,
            "liq_intact_count":        0,
            "liq_swept_count":         0,
            "liq_density_above":       0.0,
            "liq_density_below":       0.0,
            "liq_density_asymmetry":   0.0,
            "liq_nearest_above_pct":   0.05,
            "liq_nearest_below_pct":   0.05,
            "liq_void_above_pct":      0.05,
            "liq_void_below_pct":      0.05,
            "liq_structural_density":  0.0,
            "liq_sweep_recency_score": 0.0,
            "liq_recent_sweeps_1h":    0,
            "liq_nearest_above_type":  "",
            "liq_nearest_below_type":  "",
            "liq_targets_above":       [],
            "liq_targets_below":       [],
            "liq_level_type_counts":   {},
            "liq_last_event_type":     "",
        }

    # ── Integration point 2: conviction modifier ──────────────────────────

    def conviction_modifier(
        self,
        instrument:     str,
        current_price:  float,
        direction:      int,
        distance_pct:   float | None = None,
    ) -> float:
        """Structural ± modifier on hypothesis conviction.

        Range: [-``max_conviction_mod``, +``max_conviction_mod``].

        ``direction``: +1 = bullish, -1 = bearish, 0 = neutral (→ 0.0).

        Logic (no hardcoded rules):
          * ``clearance``: fraction of the blocking distance in ``direction``
            that is *free* of intact levels.  More clearance = higher modifier.
          * ``sweep_boost``: small positive adjustment when sweeps occurred
            recently (a structural event resolved nearby), regardless of
            direction.
          * ``asymmetry_penalty``: when density heavily favours the opposing
            side the modifier is slightly reduced.

        The result is a suggestion to the conviction engine — a ±0.25 nudge
        at most — not an override.
        """
        if direction == 0 or current_price <= 0.0:
            return 0.0

        dist = distance_pct if distance_pct is not None else self._block_dist
        mp   = self._memory.map_for(instrument)
        now  = time.time()

        # How many intact levels are blocking the intended direction?
        blocking = self._tracker.blocking_count(
            mp, current_price, direction, distance_pct=dist
        )
        # Clearance: 0 blocking → 1.0; ≥4 blocking → ~0.0
        clearance = max(0.0, 1.0 - blocking / 4.0)

        # Sweep recency: a recent sweep indicates resolved liquidity nearby.
        sweep_score = self._tracker.sweep_recency_score(mp, now=now)
        sweep_adj   = sweep_score * self._sweep_boost

        # Density asymmetry penalty: if density heavily favours opposing side,
        # reduce the modifier slightly.
        asym = self._tracker.density_asymmetry(mp, current_price, self._density_rad)
        # asym > 0 → more levels above;  asym < 0 → more levels below
        # For BULLISH (+1): high asym = headwind → penalty
        # For BEARISH (-1): low  asym = headwind → penalty
        asym_penalty = max(0.0, direction * asym) * 0.15

        raw = (clearance * self._max_mod + sweep_adj) - asym_penalty
        return max(-self._max_mod, min(self._max_mod, raw))

    # ── Integration point 3: narrative context ────────────────────────────

    def narrative_context(
        self,
        instrument:    str,
        current_price: float | None = None,
    ) -> dict:
        """Structural context dict for narrative enrichment.

        Returned keys feed into ``NarrativeBuilder`` as additional evidence
        signals.  Values are neutral observations — presence of equal-level
        clusters, recent sweeps, density imbalance — not assertions about
        direction.

        Keys
        ----
        liq_equal_high_count    int   — INTACT equal highs in map
        liq_equal_low_count     int   — INTACT equal lows in map
        liq_stop_cluster_count  int   — STOP_CLUSTER levels in map
        liq_breakout_trap_count int   — BREAKOUT_TRAP levels in map
        liq_recent_sweep        bool  — sweep in last 2 h
        liq_sweep_side          str   — "HIGH"/"LOW"/"" side of last sweep
        liq_near_high_level     bool  — price within 0.3% of a HIGH level
        liq_near_low_level      bool  — price within 0.3% of a LOW level
        liq_density_imbalance   str   — "ABOVE"/"BELOW"/"BALANCED"
        liq_weekly_high_intact  bool  — intact weekly high in map
        liq_weekly_low_intact   bool  — intact weekly low in map
        liq_daily_high_intact   bool  — intact daily high in map
        liq_daily_low_intact    bool  — intact daily low in map
        """
        mp    = self._memory.map_for(instrument)
        price = current_price or (mp.last_bar.close if mp.last_bar else 0.0)
        now   = time.time()

        near_tol = 0.003  # 0.3% for "near a level" check

        equal_highs = sum(
            1 for lv in mp.levels.values()
            if lv.level_type == LiquidityLevelType.EQUAL_HIGH and lv.is_active()
        )
        equal_lows = sum(
            1 for lv in mp.levels.values()
            if lv.level_type == LiquidityLevelType.EQUAL_LOW and lv.is_active()
        )
        stop_clusters = sum(
            1 for lv in mp.levels.values()
            if lv.level_type == LiquidityLevelType.STOP_CLUSTER
        )
        trap_count = sum(
            1 for lv in mp.levels.values()
            if lv.level_type == LiquidityLevelType.BREAKOUT_TRAP
        )

        last_sweep = self._tracker.most_recent_sweep(mp)
        recent_sweep = (
            last_sweep is not None
            and (now - last_sweep.detected_at) < 2 * 3600
        )
        sweep_side = ""
        if last_sweep and recent_sweep:
            lv = mp.levels.get(last_sweep.level_id)
            sweep_side = lv.side.value if lv else ""

        near_high = price > 0 and any(
            lv.side == LevelSide.HIGH and lv.is_active()
            and abs(lv.price - price) / max(price, 1e-9) <= near_tol
            for lv in mp.levels.values()
        )
        near_low = price > 0 and any(
            lv.side == LevelSide.LOW and lv.is_active()
            and abs(lv.price - price) / max(price, 1e-9) <= near_tol
            for lv in mp.levels.values()
        )

        asym = 0.0
        if price > 0:
            asym = self._tracker.density_asymmetry(mp, price, self._density_rad)
        if asym > 0.25:
            imbalance = "ABOVE"
        elif asym < -0.25:
            imbalance = "BELOW"
        else:
            imbalance = "BALANCED"

        def _has_intact(lt: LiquidityLevelType) -> bool:
            return any(
                lv.level_type == lt and lv.status == LevelStatus.INTACT
                for lv in mp.levels.values()
            )

        return {
            "liq_equal_high_count":    equal_highs,
            "liq_equal_low_count":     equal_lows,
            "liq_stop_cluster_count":  stop_clusters,
            "liq_breakout_trap_count": trap_count,
            "liq_recent_sweep":        recent_sweep,
            "liq_sweep_side":          sweep_side,
            "liq_near_high_level":     near_high,
            "liq_near_low_level":      near_low,
            "liq_density_imbalance":   imbalance,
            "liq_weekly_high_intact":  _has_intact(LiquidityLevelType.WEEKLY_HIGH),
            "liq_weekly_low_intact":   _has_intact(LiquidityLevelType.WEEKLY_LOW),
            "liq_daily_high_intact":   _has_intact(LiquidityLevelType.DAILY_HIGH),
            "liq_daily_low_intact":    _has_intact(LiquidityLevelType.DAILY_LOW),
        }

    # ── Integration point 4: opportunity ranking signal ───────────────────

    def opportunity_rank_signal(
        self,
        instrument:    str,
        current_price: float,
        direction:     int,
    ) -> float:
        """0-1 structural quality signal for opportunity ranking.

        Higher = structurally cleaner conditions in the intended direction.

        Components (no directional rules — just structural observations):
          * ``clearance_score``: fraction of blocking distance free of intact
            levels (0 obstacles → 1.0).
          * ``sweep_resolution``: recent sweeps slightly increase score
            (structural liquidity was recently resolved).
          * ``proximity_bonus``: being close to (but not on) a level of the
            *opposing* side within a generous range suggests a potential
            reaction point nearby, increasing structural interest slightly.

        ``direction``: +1 = bullish, -1 = bearish.  0 → 0.0.
        """
        if direction == 0 or current_price <= 0.0:
            return 0.0

        mp  = self._memory.map_for(instrument)
        now = time.time()

        # Structural clearance in intended direction (1% window)
        blocking = self._tracker.blocking_count(
            mp, current_price, direction, distance_pct=self._block_dist
        )
        clearance_score = max(0.0, 1.0 - blocking / 4.0)

        # Recent sweep resolution (0-1)
        sweep_score = self._tracker.sweep_recency_score(mp, now=now)

        # Proximity to opposing-side level (potential structure behind the move)
        if direction > 0:
            opp_level = self._tracker.nearest_below(mp, current_price)
        else:
            opp_level = self._tracker.nearest_above(mp, current_price)

        proximity_bonus = 0.0
        if opp_level:
            dist = abs(opp_level.price - current_price) / max(current_price, 1e-9)
            if dist <= 0.005:
                proximity_bonus = 0.2 * (1.0 - dist / 0.005)

        raw = (
            clearance_score * 0.55
            + sweep_score    * 0.25
            + proximity_bonus * 0.20
        )
        return max(0.0, min(1.0, raw))

    # ── Integration point 5: execution timing signal ──────────────────────

    def execution_timing_signal(
        self,
        instrument:    str,
        current_price: float,
    ) -> float:
        """0-1 structural timing quality at current price.

        High → price is close to a significant intact level (potential
        structural reaction point).
        Low  → price is in a structural void far from any level.

        This is deliberately direction-agnostic: being near a level is
        structurally significant regardless of the trade direction — the
        model learns whether "near a high-side level" before a long trade
        is good or bad.
        """
        if current_price <= 0.0:
            return 0.0

        mp = self._memory.map_for(instrument)
        if not mp.levels:
            return 0.0

        # Find the nearest level on either side
        above_pct = self._tracker.void_above(mp, current_price)
        below_pct = self._tracker.void_below(mp, current_price)
        nearest   = min(above_pct, below_pct)

        # Strength of the nearest level
        na = self._tracker.nearest_above(mp, current_price)
        nb = self._tracker.nearest_below(mp, current_price)
        if na and above_pct < below_pct:
            nearest_strength = self._tracker.strength_score(na)
        elif nb:
            nearest_strength = self._tracker.strength_score(nb)
        else:
            nearest_strength = 0.0

        # Proximity score: 0% distance → 1.0;  1% distance → 0.0
        proximity = max(0.0, 1.0 - nearest / 0.01)

        return max(0.0, min(1.0, proximity * 0.70 + nearest_strength * 0.30))

    # ── Convenience accessors ─────────────────────────────────────────────

    def map_for(self, instrument: str):
        """Direct access to the LiquidityMap for inspection / testing."""
        return self._memory.map_for(instrument)

    def bars(self, instrument: str) -> list[OHLCBar]:
        """Bar history stored for this instrument."""
        return self._memory.bars(instrument)

    # ── Persistence ───────────────────────────────────────────────────────

    def to_snapshot(self, instrument: str) -> dict:
        return self._memory.to_snapshot(instrument)

    def load_snapshot(self, instrument: str, data: dict) -> None:
        self._memory.load_snapshot(instrument, data)


# ── Import shorthand ─────────────────────────────────────────────────────────
# Needed here to avoid a forward-reference in narrative_context above.
from trainer.qde.brain.liquidity.liquidity_memory import LiquidityLevelType  # noqa: E402
