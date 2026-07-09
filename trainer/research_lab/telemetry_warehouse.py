"""S20 — Telemetry Warehouse.

High-level read/write API over the eight telemetry streams. Downstream consumers
(Failure Investigation Engine, Model Health Monitor, Retraining Recommendation
Engine) use this class — never the raw DB layer directly.

Write path: Signalium → POST /telemetry/<stream> → TelemetryWarehouse.ingest_*()
Read path:  TelemetryWarehouse.query_*() → downstream research systems

Thread-safe. Backed by the same ResearchLabDatabase as the rest of the Research Lab.
"""

from __future__ import annotations

from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.telemetry_models import (
    CounterfactualEvent,
    DecisionEvent,
    ExecutionEvent,
    InferenceEvent,
    LifecycleEvent,
    PositionEvent,
    ReasoningEvent,
    TelemetryFailureEvent,
    TelemetryQuery,
    TradeEvent,
)


class TelemetryWarehouse:
    """Append-only warehouse for all eight telemetry streams.

    Usage::

        warehouse = TelemetryWarehouse(db)

        # Write
        event = warehouse.ingest_decision(model_version="v1", symbol="XAUUSD", ...)
        event = warehouse.ingest_trade(...)

        # Read
        decisions = warehouse.query_decisions(TelemetryQuery(symbol="XAUUSD", since=t0))
        trades    = warehouse.query_trades(TelemetryQuery(model_version="v1"))
    """

    def __init__(self, db: ResearchLabDatabase) -> None:
        self._db = db

    # ── Write ─────────────────────────────────────────────────────────────────

    def ingest_decision(self, **kwargs: Any) -> DecisionEvent:
        event = DecisionEvent.new(**kwargs)
        self._db.insert_telemetry_event("tel_decisions", event.event_id, event.to_dict())
        logger.debug("Telemetry: ingested decision {} for {}", event.event_id, event.symbol)
        return event

    def ingest_trade(self, **kwargs: Any) -> TradeEvent:
        event = TradeEvent.new(**kwargs)
        self._db.insert_telemetry_event("tel_trades", event.event_id, event.to_dict())
        logger.debug("Telemetry: ingested trade {} for {}", event.event_id, event.symbol)
        return event

    def ingest_position(self, **kwargs: Any) -> PositionEvent:
        event = PositionEvent.new(**kwargs)
        self._db.insert_telemetry_event("tel_positions", event.event_id, event.to_dict())
        logger.debug("Telemetry: ingested position event {} for {}", event.event_id, event.symbol)
        return event

    def ingest_failure(self, **kwargs: Any) -> TelemetryFailureEvent:
        event = TelemetryFailureEvent.new(**kwargs)
        self._db.insert_telemetry_event("tel_failures", event.event_id, event.to_dict())
        logger.warning(
            "Telemetry: ingested failure {} ({}) for {}",
            event.event_id,
            event.failure_type,
            event.symbol,
        )
        return event

    def ingest_reasoning(self, **kwargs: Any) -> ReasoningEvent:
        event = ReasoningEvent.new(**kwargs)
        self._db.insert_telemetry_event("tel_reasoning", event.event_id, event.to_dict())
        logger.debug("Telemetry: ingested reasoning {} for {}", event.event_id, event.symbol)
        return event

    def ingest_execution(self, **kwargs: Any) -> ExecutionEvent:
        event = ExecutionEvent.new(**kwargs)
        self._db.insert_telemetry_event("tel_execution", event.event_id, event.to_dict())
        logger.debug("Telemetry: ingested execution {} for {}", event.event_id, event.symbol)
        return event

    def ingest_counterfactual(self, **kwargs: Any) -> CounterfactualEvent:
        event = CounterfactualEvent.new(**kwargs)
        self._db.insert_telemetry_event("tel_counterfactuals", event.event_id, event.to_dict())
        logger.debug("Telemetry: ingested counterfactual {} for {}", event.event_id, event.symbol)
        return event

    def ingest_lifecycle(self, **kwargs: Any) -> LifecycleEvent:
        event = LifecycleEvent.new(**kwargs)
        self._db.insert_telemetry_event("tel_lifecycle", event.event_id, event.to_dict())
        logger.debug(
            "Telemetry: ingested lifecycle {} ({}) for {}", event.event_id, event.kind, event.symbol
        )
        return event

    def ingest_inference(self, **kwargs: Any) -> InferenceEvent:
        event = InferenceEvent.new(**kwargs)
        self._db.insert_telemetry_event("tel_inference", event.event_id, event.to_dict())
        logger.debug(
            "Telemetry: ingested inference sample {} for {}", event.event_id, event.model_version
        )
        return event

    # ── Read ──────────────────────────────────────────────────────────────────

    def query_decisions(self, q: TelemetryQuery | None = None) -> list[dict[str, Any]]:
        return self._db.query_telemetry("tel_decisions", q or TelemetryQuery())

    def query_trades(self, q: TelemetryQuery | None = None) -> list[dict[str, Any]]:
        return self._db.query_telemetry("tel_trades", q or TelemetryQuery())

    def query_positions(self, q: TelemetryQuery | None = None) -> list[dict[str, Any]]:
        return self._db.query_telemetry("tel_positions", q or TelemetryQuery())

    def query_failures(self, q: TelemetryQuery | None = None) -> list[dict[str, Any]]:
        return self._db.query_telemetry("tel_failures", q or TelemetryQuery())

    def query_reasoning(self, q: TelemetryQuery | None = None) -> list[dict[str, Any]]:
        return self._db.query_telemetry("tel_reasoning", q or TelemetryQuery())

    def query_execution(self, q: TelemetryQuery | None = None) -> list[dict[str, Any]]:
        return self._db.query_telemetry("tel_execution", q or TelemetryQuery())

    def query_counterfactuals(self, q: TelemetryQuery | None = None) -> list[dict[str, Any]]:
        return self._db.query_telemetry("tel_counterfactuals", q or TelemetryQuery())

    def query_lifecycle(self, q: TelemetryQuery | None = None) -> list[dict[str, Any]]:
        return self._db.query_telemetry("tel_lifecycle", q or TelemetryQuery())

    def query_inference(self, q: TelemetryQuery | None = None) -> list[dict[str, Any]]:
        return self._db.query_telemetry("tel_inference", q or TelemetryQuery())

    # ── Convenience reads for downstream systems ───────────────────────────────

    def trades_for_symbol(
        self,
        symbol: str,
        model_version: str | None = None,
        since: float | None = None,
    ) -> list[dict[str, Any]]:
        """All trade outcomes for a symbol, newest first."""
        return self.query_trades(
            TelemetryQuery(model_version=model_version, symbol=symbol, since=since)
        )

    def decisions_for_symbol(
        self,
        symbol: str,
        model_version: str | None = None,
        since: float | None = None,
    ) -> list[dict[str, Any]]:
        """All decisions for a symbol."""
        return self.query_decisions(
            TelemetryQuery(model_version=model_version, symbol=symbol, since=since)
        )

    def failures_since(
        self, since: float, model_version: str | None = None
    ) -> list[dict[str, Any]]:
        """All telemetry failures after a given timestamp."""
        return self.query_failures(TelemetryQuery(model_version=model_version, since=since))

    def stream_counts(self) -> dict[str, int]:
        """Row counts for all nine telemetry tables."""
        return self._db.telemetry_stream_counts()
