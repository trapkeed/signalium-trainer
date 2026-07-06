"""Phase 24.9 — News Intelligence."""

from __future__ import annotations

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.mie.mie_models import MarketInput, NewsIntelligenceResult
from trainer.mie.mie_types import NewsImpact, clamp

# Expected dict keys from news_events list items
_IMPACT_KEY = "impact"  # HIGH | MEDIUM | LOW | NONE
_TIMING_KEY = "timing"  # upcoming | recent | past
_LABEL_KEY = "label"  # human-readable event name


def _parse_impact(raw: str) -> NewsImpact:
    mapping = {
        "HIGH": NewsImpact.HIGH,
        "MEDIUM": NewsImpact.MEDIUM,
        "LOW": NewsImpact.LOW,
    }
    return mapping.get(str(raw).upper(), NewsImpact.NONE)


class NewsIntelligenceEngine:
    """
    Evaluates news events and returns a NewsImpact classification and
    numeric impact score 0-100 (higher = greater market disruption risk).

    Impact weights:
    - HIGH upcoming   : 25 pts each (capped at 3 events → 75 pts)
    - HIGH recent     : 15 pts each
    - MEDIUM upcoming : 10 pts each
    - MEDIUM recent   : 5 pts each
    - LOW any         : 2 pts each
    """

    _WEIGHTS: dict[str, float] = {
        "HIGH_upcoming": 25.0,
        "HIGH_recent": 15.0,
        "HIGH_past": 8.0,
        "MEDIUM_upcoming": 10.0,
        "MEDIUM_recent": 5.0,
        "MEDIUM_past": 2.0,
        "LOW_upcoming": 2.0,
        "LOW_recent": 1.0,
        "LOW_past": 0.5,
    }

    # ── Public API ─────────────────────────────────────────────────────────

    def evaluate(self, inp: MarketInput) -> NewsIntelligenceResult:
        events = inp.news_events or []
        score = 0.0
        upcoming_high = 0
        recent_high = 0
        details: list[str] = []

        for event in events:
            impact_str = str(event.get(_IMPACT_KEY, "NONE")).upper()
            timing_str = str(event.get(_TIMING_KEY, "past")).lower()
            label = str(event.get(_LABEL_KEY, "unknown"))

            key = f"{impact_str}_{timing_str}"
            pts = self._WEIGHTS.get(key, 0.0)
            score += pts

            if impact_str == "HIGH":
                if timing_str == "upcoming":
                    upcoming_high += 1
                elif timing_str == "recent":
                    recent_high += 1

            if pts > 0:
                details.append(f"{impact_str} {timing_str}: {label} (+{pts:.0f})")

        score = clamp(score, 0.0, 100.0)
        impact = self._classify_impact(score, upcoming_high, recent_high)

        logger.debug(f"[NI] {inp.symbol} events={len(events)} score={score:.1f} → {impact.value}")
        return NewsIntelligenceResult(
            impact=impact,
            impact_score=score,
            event_count=len(events),
            upcoming_high=upcoming_high,
            recent_high=recent_high,
            details=details,
        )

    # ── Internal ───────────────────────────────────────────────────────────

    def _classify_impact(
        self,
        score: float,
        upcoming_high: int,
        recent_high: int,
    ) -> NewsImpact:
        if upcoming_high >= 1 or score >= 50:
            return NewsImpact.HIGH
        if score >= 20 or recent_high >= 1:
            return NewsImpact.MEDIUM
        if score >= 5:
            return NewsImpact.LOW
        return NewsImpact.NONE
