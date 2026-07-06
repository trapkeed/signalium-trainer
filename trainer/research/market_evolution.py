"""Phase 19 — Market Evolution Monitor.

Detects and records structural changes in market behaviour:

  - Long-term volatility shifts
  - Changing trend persistence
  - Liquidity changes
  - Session behaviour changes
  - Instrument behaviour changes
  - Macro regime transitions
  - Changing correlation structures

All detections are stored historically.
No automated action is taken — detections are advisory.

Usage::

    mem = MarketEvolutionMonitor()
    mem.record_regime_snapshot("EURUSD", "1H", regime="trending_up",
                                volatility=0.12, trend_strength=0.68)
    detections = mem.detect_evolution("EURUSD", "1H")
    report = mem.evolution_report("EURUSD")
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

# ---------------------------------------------------------------------------
# Regime snapshot
# ---------------------------------------------------------------------------


@dataclass
class RegimeSnapshot:
    """One market regime observation for an instrument/timeframe pair."""

    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    instrument: str = ""
    timeframe: str = ""
    regime: str = "unknown"
    volatility: float = 0.0
    trend_strength: float = 0.0
    liquidity_score: float = 0.5
    session: str = ""
    correlation_score: float = 0.0
    volume_normalized: float = 1.0
    spread_normalized: float = 1.0
    macro_indicator: str = ""
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "regime": self.regime,
            "volatility": self.volatility,
            "trend_strength": self.trend_strength,
            "liquidity_score": self.liquidity_score,
            "session": self.session,
            "correlation_score": self.correlation_score,
            "volume_normalized": self.volume_normalized,
            "spread_normalized": self.spread_normalized,
            "macro_indicator": self.macro_indicator,
            "notes": self.notes,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Evolution detection
# ---------------------------------------------------------------------------


@dataclass
class EvolutionDetection:
    """One detected market structural change."""

    detection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    instrument: str = ""
    timeframe: str = ""
    change_type: str = ""  # volatility_shift, trend_change, liquidity_change, etc.
    severity: str = "low"  # low / medium / high
    description: str = ""
    baseline_value: float = 0.0
    current_value: float = 0.0
    change_magnitude: float = 0.0
    regime_before: str = ""
    regime_after: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "change_type": self.change_type,
            "severity": self.severity,
            "description": self.description,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "change_magnitude": self.change_magnitude,
            "regime_before": self.regime_before,
            "regime_after": self.regime_after,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Market evolution monitor
# ---------------------------------------------------------------------------


class MarketEvolutionMonitor:
    """Detects and stores structural market changes over time.

    Args:
        baseline_window: Number of snapshots used as baseline.
        recent_window: Number of recent snapshots for comparison.
        volatility_shift_threshold: Magnitude for volatility shift detection.
        trend_shift_threshold: Magnitude for trend shift detection.
        liquidity_shift_threshold: Magnitude for liquidity change detection.
    """

    def __init__(
        self,
        baseline_window: int = 100,
        recent_window: int = 20,
        volatility_shift_threshold: float = 0.05,
        trend_shift_threshold: float = 0.15,
        liquidity_shift_threshold: float = 0.20,
    ) -> None:
        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self.vol_threshold = volatility_shift_threshold
        self.trend_threshold = trend_shift_threshold
        self.liquidity_threshold = liquidity_shift_threshold
        self._buffers: Dict[str, Deque[RegimeSnapshot]] = {}
        self._all_snapshots: Dict[str, List[RegimeSnapshot]] = {}
        self._detections: List[EvolutionDetection] = []

    def _key(self, instrument: str, timeframe: str) -> str:
        return f"{instrument}:{timeframe}"

    def record_regime_snapshot(
        self,
        instrument: str,
        timeframe: str,
        regime: str = "unknown",
        volatility: float = 0.0,
        trend_strength: float = 0.0,
        liquidity_score: float = 0.5,
        session: str = "",
        correlation_score: float = 0.0,
        volume_normalized: float = 1.0,
        spread_normalized: float = 1.0,
        macro_indicator: str = "",
    ) -> RegimeSnapshot:
        """Record one market regime observation."""
        snap = RegimeSnapshot(
            instrument=instrument,
            timeframe=timeframe,
            regime=regime,
            volatility=volatility,
            trend_strength=trend_strength,
            liquidity_score=liquidity_score,
            session=session,
            correlation_score=correlation_score,
            volume_normalized=volume_normalized,
            spread_normalized=spread_normalized,
            macro_indicator=macro_indicator,
        )
        key = self._key(instrument, timeframe)
        total_window = self.baseline_window + self.recent_window
        if key not in self._buffers:
            self._buffers[key] = deque(maxlen=total_window)
            self._all_snapshots[key] = []
        self._buffers[key].append(snap)
        self._all_snapshots[key].append(snap)
        return snap

    def detect_evolution(self, instrument: str, timeframe: str) -> List[EvolutionDetection]:
        """Detect structural changes for an instrument/timeframe pair.

        Returns a list of EvolutionDetection objects if changes are detected.
        No automated action is taken.
        """
        key = self._key(instrument, timeframe)
        buf = list(self._buffers.get(key, []))
        n = len(buf)
        min_required = self.baseline_window // 2 + self.recent_window
        if n < min_required:
            return []

        baseline = buf[: n - self.recent_window]
        recent = buf[-self.recent_window :]
        detections = []

        def _mean(vals):
            return sum(vals) / len(vals) if vals else 0.0

        b_vol = _mean([s.volatility for s in baseline])
        r_vol = _mean([s.volatility for s in recent])
        if abs(r_vol - b_vol) > self.vol_threshold:
            severity = "high" if abs(r_vol - b_vol) > self.vol_threshold * 2 else "medium"
            d = EvolutionDetection(
                instrument=instrument,
                timeframe=timeframe,
                change_type="volatility_shift",
                severity=severity,
                description=(
                    f"Volatility shifted from {b_vol:.3f} to {r_vol:.3f} "
                    f"(Δ={r_vol - b_vol:+.3f})"
                ),
                baseline_value=round(b_vol, 4),
                current_value=round(r_vol, 4),
                change_magnitude=round(abs(r_vol - b_vol), 4),
            )
            detections.append(d)
            self._detections.append(d)

        b_trend = _mean([s.trend_strength for s in baseline])
        r_trend = _mean([s.trend_strength for s in recent])
        if abs(r_trend - b_trend) > self.trend_threshold:
            severity = "high" if abs(r_trend - b_trend) > self.trend_threshold * 2 else "medium"
            d = EvolutionDetection(
                instrument=instrument,
                timeframe=timeframe,
                change_type="trend_persistence_change",
                severity=severity,
                description=(f"Trend persistence shifted from {b_trend:.3f} to {r_trend:.3f}"),
                baseline_value=round(b_trend, 4),
                current_value=round(r_trend, 4),
                change_magnitude=round(abs(r_trend - b_trend), 4),
            )
            detections.append(d)
            self._detections.append(d)

        b_liq = _mean([s.liquidity_score for s in baseline])
        r_liq = _mean([s.liquidity_score for s in recent])
        if abs(r_liq - b_liq) > self.liquidity_threshold:
            d = EvolutionDetection(
                instrument=instrument,
                timeframe=timeframe,
                change_type="liquidity_change",
                severity="medium",
                description=(f"Liquidity shifted from {b_liq:.3f} to {r_liq:.3f}"),
                baseline_value=round(b_liq, 4),
                current_value=round(r_liq, 4),
                change_magnitude=round(abs(r_liq - b_liq), 4),
            )
            detections.append(d)
            self._detections.append(d)

        b_regimes = {s.regime for s in baseline}
        r_regimes = {s.regime for s in recent}
        if r_regimes != b_regimes:
            new_regimes = r_regimes - b_regimes
            lost_regimes = b_regimes - r_regimes
            if new_regimes or lost_regimes:
                d = EvolutionDetection(
                    instrument=instrument,
                    timeframe=timeframe,
                    change_type="regime_transition",
                    severity="high",
                    description=(f"Regime transition: new={new_regimes}, " f"lost={lost_regimes}"),
                    metadata={"new_regimes": list(new_regimes), "lost_regimes": list(lost_regimes)},
                )
                detections.append(d)
                self._detections.append(d)

        return detections

    def detect_all(self) -> List[EvolutionDetection]:
        """Run detection across all instrument/timeframe pairs."""
        all_detections = []
        for key in self._buffers:
            instrument, timeframe = key.split(":", 1)
            all_detections.extend(self.detect_evolution(instrument, timeframe))
        return all_detections

    def evolution_report(self, instrument: str, timeframe: Optional[str] = None) -> Dict[str, Any]:
        """Generate evolution report for an instrument."""
        keys = [k for k in self._buffers if k.startswith(f"{instrument}:")]
        if timeframe:
            keys = [self._key(instrument, timeframe)]
        total_snaps = sum(len(self._all_snapshots.get(k, [])) for k in keys)
        instrument_detections = [
            d.to_dict() for d in self._detections if d.instrument == instrument
        ]
        return {
            "instrument": instrument,
            "timeframe": timeframe or "all",
            "total_snapshots": total_snaps,
            "n_detections": len(instrument_detections),
            "detections": instrument_detections,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def global_report(self) -> Dict[str, Any]:
        """Generate a global market evolution report."""
        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        for d in self._detections:
            by_type[d.change_type] = by_type.get(d.change_type, 0) + 1
            by_severity[d.severity] = by_severity.get(d.severity, 0) + 1
        return {
            "n_pairs_monitored": len(self._buffers),
            "total_detections": len(self._detections),
            "by_type": by_type,
            "by_severity": by_severity,
            "recent_detections": [d.to_dict() for d in self._detections[-10:]],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @property
    def n_detections(self) -> int:
        return len(self._detections)
