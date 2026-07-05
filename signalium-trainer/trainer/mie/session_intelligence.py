"""Phase 24.9 — Session Intelligence."""

from __future__ import annotations

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from trainer.mie.mie_types import SessionType, SESSION_QUALITY_SCORES, clamp
from trainer.mie.mie_models import MarketInput, SessionResult

# UTC hour ranges for each trading session
_SESSION_HOURS = {
    SessionType.ASIAN:    (0,  8),    # 00:00 – 07:59 UTC
    SessionType.LONDON:   (7,  16),   # 07:00 – 15:59 UTC
    SessionType.NEW_YORK: (12, 21),   # 12:00 – 20:59 UTC
    SessionType.OVERLAP:  (12, 16),   # 12:00 – 15:59 UTC (London/NY overlap)
}

# Hours typically considered high-volume
_HIGH_VOLUME_HOURS = set(range(7, 21))


def _detect_session(hour: int) -> SessionType:
    """Classify UTC hour into a SessionType (overlap takes priority)."""
    h = hour % 24
    if _SESSION_HOURS[SessionType.OVERLAP][0] <= h < _SESSION_HOURS[SessionType.OVERLAP][1]:
        return SessionType.OVERLAP
    if _SESSION_HOURS[SessionType.LONDON][0] <= h < _SESSION_HOURS[SessionType.LONDON][1]:
        return SessionType.LONDON
    if _SESSION_HOURS[SessionType.NEW_YORK][0] <= h < _SESSION_HOURS[SessionType.NEW_YORK][1]:
        return SessionType.NEW_YORK
    if _SESSION_HOURS[SessionType.ASIAN][0] <= h < _SESSION_HOURS[SessionType.ASIAN][1]:
        return SessionType.ASIAN
    return SessionType.OFF_HOURS


class SessionIntelligenceEngine:
    """
    Classifies the current trading session and returns a session quality
    score 0-100.

    Quality is derived from:
    - Base session quality (lookup table)
    - Whether it is a high-volume hour
    - Overlap bonus
    """

    # ── Public API ─────────────────────────────────────────────────────────

    def analyse(self, inp: MarketInput) -> SessionResult:
        session    = _detect_session(inp.session_hour)
        base_q     = SESSION_QUALITY_SCORES.get(session, 25.0)
        is_overlap = session == SessionType.OVERLAP
        is_hv      = inp.session_hour % 24 in _HIGH_VOLUME_HOURS

        quality = base_q
        if is_hv and session not in (SessionType.OFF_HOURS,):
            quality = min(100.0, quality + 5.0)

        logger.debug(f"[SI] hour={inp.session_hour} → {session.value} q={quality:.1f}")
        return SessionResult(
            session=session,
            quality=clamp(quality, 0.0, 100.0),
            is_overlap=is_overlap,
            is_high_volume=is_hv,
        )
