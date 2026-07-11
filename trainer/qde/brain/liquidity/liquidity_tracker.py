"""QDE Brain — structural liquidity: stateless tracker / analytics.

Mirrors the ``HypothesisTracker`` / ``OpportunityTracker`` pattern: the
``LiquidityTracker`` class holds no state of its own and operates purely
over a caller-supplied ``LiquidityMap`` (owned by ``LiquidityMemory``).

All analytics are *observational*:
  - "There are N intact levels above price within 1%"
  - "The nearest level below is 0.3% away"
  - "3 sweeps occurred in the last hour"

None of them assert what those observations mean for direction.  The
``LiquidityMapEngine`` composes these primitives into integration signals
without encoding trading rules.
"""

from __future__ import annotations

import time

from trainer.qde.brain.liquidity.liquidity_memory import (
    LevelStatus,
    LiquidityEvent,
    LiquidityEventType,
    LiquidityLevel,
    LiquidityMap,
)


class LiquidityTracker:
    """Stateless analytics and mutation helpers over a ``LiquidityMap``.

    All public methods are pure functions of their arguments except the
    ``mark_*`` and ``update_strength`` helpers, which mutate the
    ``LiquidityLevel`` objects stored in the map (by design — the map owns
    the objects, and the tracker provides controlled-mutation operations).
    """

    # ── Level retrieval ───────────────────────────────────────────────────

    def levels_above(
        self,
        mp: LiquidityMap,
        price: float,
        only_active: bool = True,
    ) -> list[LiquidityLevel]:
        """Active levels strictly above ``price``, sorted nearest first."""
        result = [
            lv
            for lv in mp.levels.values()
            if lv.price > price and (not only_active or lv.is_active())
        ]
        return sorted(result, key=lambda lv: lv.price)

    def levels_below(
        self,
        mp: LiquidityMap,
        price: float,
        only_active: bool = True,
    ) -> list[LiquidityLevel]:
        """Active levels strictly below ``price``, sorted nearest first."""
        result = [
            lv
            for lv in mp.levels.values()
            if lv.price < price and (not only_active or lv.is_active())
        ]
        return sorted(result, key=lambda lv: lv.price, reverse=True)

    def levels_near(
        self,
        mp: LiquidityMap,
        price: float,
        distance_pct: float = 0.005,
        only_active: bool = True,
    ) -> list[LiquidityLevel]:
        """Active levels within ``distance_pct`` of ``price`` on either side."""
        return [
            lv
            for lv in mp.levels.values()
            if abs(lv.price - price) / max(price, 1e-9) <= distance_pct
            and (not only_active or lv.is_active())
        ]

    def nearest_above(self, mp: LiquidityMap, price: float) -> LiquidityLevel | None:
        above = self.levels_above(mp, price)
        return above[0] if above else None

    def nearest_below(self, mp: LiquidityMap, price: float) -> LiquidityLevel | None:
        below = self.levels_below(mp, price)
        return below[0] if below else None

    def strongest_above(
        self,
        mp: LiquidityMap,
        price: float,
        lookup_pct: float = 0.05,
    ) -> LiquidityLevel | None:
        """Highest-strength active level above price within ``lookup_pct``."""
        candidates = [
            lv
            for lv in mp.levels.values()
            if lv.price > price
            and (lv.price - price) / max(price, 1e-9) <= lookup_pct
            and lv.is_active()
        ]
        return max(candidates, key=lambda lv: lv.strength) if candidates else None

    def strongest_below(
        self,
        mp: LiquidityMap,
        price: float,
        lookup_pct: float = 0.05,
    ) -> LiquidityLevel | None:
        """Highest-strength active level below price within ``lookup_pct``."""
        candidates = [
            lv
            for lv in mp.levels.values()
            if lv.price < price
            and (price - lv.price) / max(price, 1e-9) <= lookup_pct
            and lv.is_active()
        ]
        return max(candidates, key=lambda lv: lv.strength) if candidates else None

    # ── Density & structural metrics ──────────────────────────────────────

    def density_above(
        self,
        mp: LiquidityMap,
        price: float,
        lookback_pct: float = 0.02,
    ) -> float:
        """Count of active levels above price within ``lookback_pct``,
        normalized by lookback range so values are comparable across
        instruments with different price scales.

        Returns levels-per-percent (e.g. 3 levels in 2% range → 1.5).
        """
        count = sum(
            1
            for lv in mp.levels.values()
            if lv.price > price
            and (lv.price - price) / max(price, 1e-9) <= lookback_pct
            and lv.is_active()
        )
        return count / max(lookback_pct * 100, 1e-9)

    def density_below(
        self,
        mp: LiquidityMap,
        price: float,
        lookback_pct: float = 0.02,
    ) -> float:
        """Count of active levels below price within ``lookback_pct``,
        normalized identically to ``density_above``.
        """
        count = sum(
            1
            for lv in mp.levels.values()
            if lv.price < price
            and (price - lv.price) / max(price, 1e-9) <= lookback_pct
            and lv.is_active()
        )
        return count / max(lookback_pct * 100, 1e-9)

    def density_asymmetry(
        self,
        mp: LiquidityMap,
        price: float,
        lookback_pct: float = 0.02,
    ) -> float:
        """Signed density asymmetry: positive when more levels are above,
        negative when more are below.

        Range: -1 to +1.  Zero means equal density on both sides.
        """
        da = self.density_above(mp, price, lookback_pct)
        db = self.density_below(mp, price, lookback_pct)
        total = da + db
        if total < 1e-9:
            return 0.0
        return (da - db) / total

    def structural_density_score(
        self,
        mp: LiquidityMap,
        price: float,
        radius_pct: float = 0.02,
    ) -> float:
        """0-1 measure of how densely packed levels are near ``price``.

        10 levels in the radius → 1.0; 0 levels → 0.0.
        """
        near = self.levels_near(mp, price, distance_pct=radius_pct)
        return min(1.0, len(near) / 10.0)

    def void_above(self, mp: LiquidityMap, price: float) -> float:
        """Fractional distance to the nearest active level above price.

        Capped at 0.05 (5%) — beyond that the structural influence is
        negligible.
        """
        above = self.nearest_above(mp, price)
        if above is None:
            return 0.05
        return min(0.05, (above.price - price) / max(price, 1e-9))

    def void_below(self, mp: LiquidityMap, price: float) -> float:
        """Fractional distance to the nearest active level below price."""
        below = self.nearest_below(mp, price)
        if below is None:
            return 0.05
        return min(0.05, (price - below.price) / max(price, 1e-9))

    def blocking_count(
        self,
        mp: LiquidityMap,
        price: float,
        direction: int,
        distance_pct: float = 0.01,
    ) -> int:
        """Count of active levels within ``distance_pct`` in ``direction``.

        ``direction``: +1 = above, −1 = below.  Used by the engine to
        measure how many structural obstacles exist between current price
        and a short-term target.  Not directional in and of itself.
        """
        if direction > 0:
            return sum(
                1
                for lv in mp.levels.values()
                if lv.is_active()
                and lv.price > price
                and (lv.price - price) / max(price, 1e-9) <= distance_pct
            )
        return sum(
            1
            for lv in mp.levels.values()
            if lv.is_active()
            and lv.price < price
            and (price - lv.price) / max(price, 1e-9) <= distance_pct
        )

    # ── Sweep analytics ───────────────────────────────────────────────────

    def recent_sweep_count(
        self,
        mp: LiquidityMap,
        lookback_seconds: float = 3600.0,
        now: float | None = None,
    ) -> int:
        """Count of sweep events in the last ``lookback_seconds``."""
        now = now if now is not None else time.time()
        return sum(
            1
            for ev in mp.events
            if ev.event_type == LiquidityEventType.LEVEL_SWEPT
            and (now - ev.detected_at) <= lookback_seconds
        )

    def sweep_recency_score(
        self,
        mp: LiquidityMap,
        now: float | None = None,
    ) -> float:
        """0-1 score of recent sweep activity.

        Combines last-1h and last-4h counts with heavier weight on recency.
        Higher = more sweeps happened recently, regardless of direction.
        """
        now = now if now is not None else time.time()
        c1h = self.recent_sweep_count(mp, 3600, now=now)
        c4h = self.recent_sweep_count(mp, 4 * 3600, now=now)
        return min(1.0, c1h * 0.35 + c4h * 0.10)

    def most_recent_sweep(self, mp: LiquidityMap) -> LiquidityEvent | None:
        """Most recently detected sweep event, or ``None``."""
        sweeps = [ev for ev in mp.events if ev.event_type == LiquidityEventType.LEVEL_SWEPT]
        return max(sweeps, key=lambda ev: ev.detected_at) if sweeps else None

    def most_recent_event(
        self,
        mp: LiquidityMap,
        event_type: LiquidityEventType | None = None,
    ) -> LiquidityEvent | None:
        """Most recent event of the given type, or any event if None."""
        pool = (
            [ev for ev in mp.events if ev.event_type == event_type]
            if event_type
            else list(mp.events)
        )
        return max(pool, key=lambda ev: ev.detected_at) if pool else None

    # ── Strength scoring ──────────────────────────────────────────────────

    def strength_score(
        self,
        lv: LiquidityLevel,
        now: float | None = None,
    ) -> float:
        """Composite structural strength of a level.

        Combines the stored strength with test-count reinforcement and a
        slow 7-day age decay (levels don't lose relevance overnight).
        """
        now = now if now is not None else time.time()
        age_hours = lv.age_seconds(now) / 3600.0
        age_decay = max(0.50, 1.0 - age_hours / (24.0 * 7.0))
        test_bonus = min(0.30, lv.test_count * 0.05)
        return min(1.0, (lv.strength + test_bonus) * age_decay)

    # ── Level mutations ───────────────────────────────────────────────────

    def update_strength(
        self,
        mp: LiquidityMap,
        level_id: str,
        delta: float,
        now: float | None = None,
    ) -> None:
        """Adjust ``strength`` by ``delta``. +ve = reinforced; –ve = weakened."""
        lv = mp.levels.get(level_id)
        if lv:
            lv.strength = max(0.0, min(1.0, lv.strength + delta))
            lv.last_tested_at = now if now is not None else time.time()

    def mark_tested(
        self,
        mp: LiquidityMap,
        level_id: str,
        now: float | None = None,
    ) -> None:
        """Record that price approached but did not break this level."""
        now = now if now is not None else time.time()
        lv = mp.levels.get(level_id)
        if lv and lv.status == LevelStatus.INTACT:
            lv.status = LevelStatus.TESTED
            lv.test_count += 1
            lv.last_tested_at = now
            lv.strength = min(1.0, lv.strength + 0.05)

    def mark_broken(
        self,
        mp: LiquidityMap,
        level_id: str,
        now: float | None = None,
    ) -> None:
        """Mark a level as decisively broken (no reversal after close-through)."""
        lv = mp.levels.get(level_id)
        if lv:
            lv.status = LevelStatus.BROKEN
            lv.last_status_change_at = now if now is not None else time.time()

    def prune_old_broken(
        self,
        mp: LiquidityMap,
        older_than_seconds: float = 7 * 24 * 3600,
        now: float | None = None,
    ) -> int:
        """Remove BROKEN levels older than ``older_than_seconds``.

        Returns the number of levels removed.
        """
        now = now if now is not None else time.time()
        before = len(mp.levels)
        for lid in [
            lid
            for lid, lv in mp.levels.items()
            if lv.status == LevelStatus.BROKEN and lv.age_seconds(now) > older_than_seconds
        ]:
            del mp.levels[lid]
        return before - len(mp.levels)

    def check_touch(
        self,
        mp: LiquidityMap,
        price: float,
        touch_tolerance_pct: float = 0.001,
    ) -> list[str]:
        """Return IDs of INTACT levels that ``price`` is currently touching."""
        return [
            lid
            for lid, lv in mp.levels.items()
            if lv.status == LevelStatus.INTACT
            and abs(lv.price - price) / max(price, 1e-9) <= touch_tolerance_pct
        ]

    # ── Projection: where may liquidity exist next ─────────────────────────

    def where_may_liquidity_exist_next(
        self,
        mp: LiquidityMap,
        price: float,
        direction: int,
        top_n: int = 5,
    ) -> list[dict]:
        """Structured list of likely next structural targets in ``direction``.

        ``direction``: +1 = looking above, -1 = looking below.

        Returns a list of dicts describing candidate levels sorted by
        strength, NOT by proximity.  The model decides whether (and how)
        these targets are relevant — this method only identifies candidates.

        Keys per entry:
          price, level_type, status, strength, distance_pct,
          test_count, cluster_member_count.
        """
        if direction > 0:
            candidates = self.levels_above(mp, price)
        else:
            candidates = self.levels_below(mp, price)

        top = sorted(candidates[:30], key=lambda lv: lv.strength, reverse=True)[:top_n]
        return [
            {
                "price": round(lv.price, 5),
                "level_type": lv.level_type.value,
                "status": lv.status.value,
                "strength": round(lv.strength, 3),
                "distance_pct": round(abs(lv.price - price) / max(price, 1e-9), 5),
                "test_count": lv.test_count,
                "cluster_member_count": lv.cluster_member_count,
            }
            for lv in top
        ]

    # ── Summary helpers (used by engine for brain_context) ─────────────────

    def level_type_counts(
        self,
        mp: LiquidityMap,
        only_active: bool = True,
    ) -> dict[str, int]:
        """Count of levels per type (for telemetry / brain_context)."""
        counts: dict[str, int] = {}
        for lv in mp.levels.values():
            if only_active and not lv.is_active():
                continue
            key = lv.level_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def status_counts(self, mp: LiquidityMap) -> dict[str, int]:
        """Count of levels per status."""
        counts: dict[str, int] = {}
        for lv in mp.levels.values():
            key = lv.status.value
            counts[key] = counts.get(key, 0) + 1
        return counts
