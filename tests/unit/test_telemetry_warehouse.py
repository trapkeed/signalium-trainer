"""Unit tests for S20 Telemetry Ingestion API + Warehouse.

Covers:
- All eight stream ingestion paths (write → DB → read back)
- TelemetryWarehouse typed read methods
- DB-level idempotency (duplicate event_id is silently ignored)
- TelemetryQuery filters (symbol, model_version, since/until, limit)
- stream_counts() aggregation
- Invalid table name raises ValueError
- Pydantic request models via telemetry_server.py request classes
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.telemetry_models import (
    TelemetryQuery,
    TelemetryStream,
)
from trainer.research_lab.telemetry_warehouse import TelemetryWarehouse

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def db(tmp_path: Path) -> ResearchLabDatabase:
    """Fresh in-memory-ish DB for each test."""
    return ResearchLabDatabase(str(tmp_path / "test_telemetry.db"))


@pytest.fixture()
def warehouse(db: ResearchLabDatabase) -> TelemetryWarehouse:
    return TelemetryWarehouse(db)


# ── Helpers ────────────────────────────────────────────────────────────────────

NOW = time.time()
MV = "v1.0.0"
SYM = "XAUUSD"


def _base(symbol: str = SYM, model_version: str = MV) -> dict:
    return {"model_version": model_version, "symbol": symbol, "timestamp": NOW}


# ── Stream: decisions ─────────────────────────────────────────────────────────


class TestDecisionStream:
    def test_ingest_and_read_back(self, warehouse: TelemetryWarehouse) -> None:
        event = warehouse.ingest_decision(
            **_base(),
            direction="LONG",
            confidence=0.87,
            reasoning_summary="Trend continuation",
            entry_price=1950.0,
            stop_distance=5.0,
            take_profit_distance=15.0,
            position_size=0.1,
            session="LONDON",
            market_conditions={"spread": 0.3, "atr": 12.0},
            regime="TRENDING_BULL",
        )
        assert event.event_id.startswith("dec-")
        assert event.stream if hasattr(event, "stream") else True  # stored in dict
        d = event.to_dict()
        assert d["stream"] == TelemetryStream.DECISIONS.value
        assert d["direction"] == "LONG"
        assert d["confidence"] == pytest.approx(0.87)
        assert d["market_conditions"]["spread"] == pytest.approx(0.3)

    def test_read_back_from_warehouse(self, warehouse: TelemetryWarehouse) -> None:
        warehouse.ingest_decision(**_base(), direction="SHORT", confidence=0.65)
        results = warehouse.query_decisions(TelemetryQuery(symbol=SYM))
        assert len(results) == 1
        assert results[0]["direction"] == "SHORT"

    def test_decisions_for_symbol_helper(self, warehouse: TelemetryWarehouse) -> None:
        warehouse.ingest_decision(**_base(symbol="EURUSD"), direction="LONG", confidence=0.7)
        warehouse.ingest_decision(**_base(symbol="BTCUSD"), direction="SHORT", confidence=0.8)
        results = warehouse.decisions_for_symbol("EURUSD")
        assert len(results) == 1
        assert results[0]["symbol"] == "EURUSD"


# ── Stream: trades ────────────────────────────────────────────────────────────


class TestTradeStream:
    def test_ingest_and_read_back(self, warehouse: TelemetryWarehouse) -> None:
        event = warehouse.ingest_trade(
            **_base(),
            direction="LONG",
            entry_price=1950.0,
            exit_price=1965.0,
            size=0.1,
            pnl=150.0,
            pnl_pct=0.77,
            hold_duration_s=3600.0,
            exit_reason="TP_HIT",
            confidence_at_entry=0.87,
        )
        assert event.pnl == pytest.approx(150.0)
        assert event.to_dict()["exit_reason"] == "TP_HIT"

    def test_read_by_model_version(self, warehouse: TelemetryWarehouse) -> None:
        warehouse.ingest_trade(
            **_base(model_version="v1"),
            direction="LONG",
            entry_price=1.0,
            exit_price=1.1,
            size=1.0,
            pnl=100.0,
            pnl_pct=10.0,
            hold_duration_s=100.0,
        )
        warehouse.ingest_trade(
            **_base(model_version="v2"),
            direction="SHORT",
            entry_price=1.0,
            exit_price=0.9,
            size=1.0,
            pnl=100.0,
            pnl_pct=10.0,
            hold_duration_s=100.0,
        )
        v1 = warehouse.query_trades(TelemetryQuery(model_version="v1"))
        assert len(v1) == 1
        assert v1[0]["model_version"] == "v1"

    def test_trades_for_symbol_helper(self, warehouse: TelemetryWarehouse) -> None:
        warehouse.ingest_trade(
            **_base(),
            direction="LONG",
            entry_price=1.0,
            exit_price=1.1,
            size=1.0,
            pnl=10.0,
            pnl_pct=1.0,
            hold_duration_s=60.0,
        )
        trades = warehouse.trades_for_symbol(SYM)
        assert len(trades) == 1


# ── Stream: positions ─────────────────────────────────────────────────────────


class TestPositionStream:
    def test_ingest_and_read_back(self, warehouse: TelemetryWarehouse) -> None:
        event = warehouse.ingest_position(
            **_base(),
            action="OPEN",
            size=0.1,
            price=1950.0,
            confidence=0.87,
            cumulative_pnl=0.0,
        )
        assert event.action == "OPEN"
        results = warehouse.query_positions()
        assert len(results) == 1

    def test_position_actions_stored(self, warehouse: TelemetryWarehouse) -> None:
        actions = [{"ts": NOW - 100, "action": "OPEN", "size": 0.1}]
        event = warehouse.ingest_position(
            **_base(), action="SCALE_IN", size=0.05, price=1955.0, position_actions=actions
        )
        d = event.to_dict()
        assert len(d["position_actions"]) == 1


# ── Stream: failures ──────────────────────────────────────────────────────────


class TestFailureStream:
    def test_ingest_and_read_back(self, warehouse: TelemetryWarehouse) -> None:
        event = warehouse.ingest_failure(
            **_base(),
            failure_type="EXECUTION_REJECTION",
            severity="HIGH",
            details="Broker rejected order: margin insufficient",
            affected_order_id="ord-abc123",
        )
        assert event.failure_type == "EXECUTION_REJECTION"
        results = warehouse.query_failures()
        assert len(results) == 1
        assert results[0]["severity"] == "HIGH"

    def test_failures_since_helper(self, warehouse: TelemetryWarehouse) -> None:
        t_old = NOW - 1000
        t_new = NOW
        # Ingest failure with old timestamp (create manually via model)
        from trainer.research_lab.telemetry_models import TelemetryFailureEvent

        old_ev = TelemetryFailureEvent.new("v1", SYM, t_old, "TIMEOUT")
        warehouse._db.insert_telemetry_event("tel_failures", old_ev.event_id, old_ev.to_dict())
        # Ingest failure with new timestamp
        warehouse.ingest_failure(**_base(), failure_type="SLIPPAGE_EXCEEDED")

        results = warehouse.failures_since(since=NOW - 1)
        assert len(results) == 1
        assert results[0]["failure_type"] == "SLIPPAGE_EXCEEDED"


# ── Stream: reasoning ─────────────────────────────────────────────────────────


class TestReasoningStream:
    def test_ingest_and_read_back(self, warehouse: TelemetryWarehouse) -> None:
        event = warehouse.ingest_reasoning(
            **_base(),
            decision_id="did-abc",
            output_heads={"direction_logit": 0.9, "size_logit": 0.5},
            feature_values={"atr_ratio": 1.2, "spread_norm": 0.3},
            confidence_breakdown={"direction": 0.87, "risk": 0.71},
        )
        assert event.decision_id == "did-abc"
        d = event.to_dict()
        assert d["output_heads"]["direction_logit"] == pytest.approx(0.9)
        assert d["feature_values"]["atr_ratio"] == pytest.approx(1.2)
        results = warehouse.query_reasoning(TelemetryQuery(symbol=SYM))
        assert len(results) == 1


# ── Stream: execution ─────────────────────────────────────────────────────────


class TestExecutionStream:
    def test_ingest_and_read_back(self, warehouse: TelemetryWarehouse) -> None:
        event = warehouse.ingest_execution(
            **_base(),
            order_type="MARKET",
            requested_price=1950.0,
            executed_price=1950.3,
            size=0.1,
            execution_time_ms=12.5,
            venue="LMAX",
            status="FILLED",
        )
        assert event.slippage == pytest.approx(0.3)
        assert abs(event.slippage_pct - (0.3 / 1950.0 * 100)) < 1e-6

    def test_slippage_computed(self, warehouse: TelemetryWarehouse) -> None:
        event = warehouse.ingest_execution(
            **_base(),
            order_type="LIMIT",
            requested_price=2000.0,
            executed_price=2001.5,
            size=1.0,
        )
        assert event.slippage == pytest.approx(1.5)


# ── Stream: counterfactuals ───────────────────────────────────────────────────


class TestCounterfactualStream:
    def test_ingest_and_read_back(self, warehouse: TelemetryWarehouse) -> None:
        event = warehouse.ingest_counterfactual(
            **_base(),
            trade_id="tid-xyz",
            what_if_description="If we had held 2h longer",
            actual_pnl=50.0,
            counterfactual_pnl=120.0,
        )
        assert event.pnl_delta == pytest.approx(70.0)
        results = warehouse.query_counterfactuals()
        assert len(results) == 1
        assert results[0]["pnl_delta"] == pytest.approx(70.0)


# ── Stream: lifecycle ─────────────────────────────────────────────────────────


class TestLifecycleStream:
    def test_ingest_and_read_back(self, warehouse: TelemetryWarehouse) -> None:
        event = warehouse.ingest_lifecycle(
            **_base(),
            kind="POSITION_OPENED",
            previous_state="FLAT",
            new_state="LONG",
            trigger="QDE_SIGNAL",
            metadata={"confidence": 0.87},
        )
        assert event.kind == "POSITION_OPENED"
        results = warehouse.query_lifecycle(TelemetryQuery(symbol=SYM))
        assert len(results) == 1
        assert results[0]["trigger"] == "QDE_SIGNAL"


# ── Cross-stream: filters and counts ─────────────────────────────────────────


class TestWarehouseFilters:
    def test_symbol_filter(self, warehouse: TelemetryWarehouse) -> None:
        warehouse.ingest_decision(**_base(symbol="XAUUSD"), direction="LONG", confidence=0.8)
        warehouse.ingest_decision(**_base(symbol="EURUSD"), direction="SHORT", confidence=0.6)
        results = warehouse.query_decisions(TelemetryQuery(symbol="XAUUSD"))
        assert len(results) == 1
        assert results[0]["symbol"] == "XAUUSD"

    def test_model_version_filter(self, warehouse: TelemetryWarehouse) -> None:
        warehouse.ingest_decision(**_base(model_version="v1"), direction="LONG", confidence=0.7)
        warehouse.ingest_decision(**_base(model_version="v2"), direction="LONG", confidence=0.8)
        r1 = warehouse.query_decisions(TelemetryQuery(model_version="v1"))
        r2 = warehouse.query_decisions(TelemetryQuery(model_version="v2"))
        assert len(r1) == 1
        assert len(r2) == 1

    def test_since_until_filter(self, warehouse: TelemetryWarehouse) -> None:
        from trainer.research_lab.telemetry_models import DecisionEvent

        t1, t2, t3 = NOW - 200, NOW - 100, NOW
        for t in (t1, t2, t3):
            ev = DecisionEvent.new(MV, SYM, t, "LONG", 0.7)
            warehouse._db.insert_telemetry_event("tel_decisions", ev.event_id, ev.to_dict())

        results = warehouse.query_decisions(TelemetryQuery(since=NOW - 150, until=NOW - 50))
        assert len(results) == 1
        assert results[0]["timestamp"] == pytest.approx(t2)

    def test_limit_filter(self, warehouse: TelemetryWarehouse) -> None:
        from trainer.research_lab.telemetry_models import DecisionEvent

        for i in range(10):
            ev = DecisionEvent.new(MV, SYM, NOW + i, "LONG", 0.7)
            warehouse._db.insert_telemetry_event("tel_decisions", ev.event_id, ev.to_dict())

        results = warehouse.query_decisions(TelemetryQuery(limit=3))
        assert len(results) == 3

    def test_stream_counts(self, warehouse: TelemetryWarehouse) -> None:
        warehouse.ingest_decision(**_base(), direction="LONG", confidence=0.7)
        warehouse.ingest_trade(
            **_base(),
            direction="LONG",
            entry_price=1.0,
            exit_price=1.1,
            size=1.0,
            pnl=10.0,
            pnl_pct=1.0,
            hold_duration_s=60.0,
        )
        warehouse.ingest_lifecycle(**_base(), kind="SESSION_STARTED")
        counts = warehouse.stream_counts()
        assert counts["tel_decisions"] == 1
        assert counts["tel_trades"] == 1
        assert counts["tel_lifecycle"] == 1
        assert counts.get("tel_positions", 0) == 0

    def test_stats_includes_telemetry(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase
    ) -> None:
        warehouse.ingest_decision(**_base(), direction="LONG", confidence=0.7)
        stats = db.stats()
        assert "tel_decisions" in stats
        assert stats["tel_decisions"] == 1


# ── Idempotency ───────────────────────────────────────────────────────────────


class TestIdempotency:
    def test_duplicate_event_id_silently_ignored(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase
    ) -> None:
        """Re-delivering the same event twice must not duplicate it."""
        from trainer.research_lab.telemetry_models import DecisionEvent

        ev = DecisionEvent.new(MV, SYM, NOW, "LONG", 0.8)
        db.insert_telemetry_event("tel_decisions", ev.event_id, ev.to_dict())
        db.insert_telemetry_event("tel_decisions", ev.event_id, ev.to_dict())  # duplicate

        results = warehouse.query_decisions()
        assert len(results) == 1


# ── Safety: invalid table name ────────────────────────────────────────────────


class TestSafety:
    def test_invalid_table_raises(self, db: ResearchLabDatabase) -> None:
        with pytest.raises(ValueError, match="Unknown telemetry table"):
            db.insert_telemetry_event("injected_table", "ev-1", {})

    def test_invalid_query_table_raises(self, db: ResearchLabDatabase) -> None:
        with pytest.raises(ValueError, match="Unknown telemetry table"):
            db.query_telemetry("injected_table", TelemetryQuery())


# ── API-level idempotency ─────────────────────────────────────────────────────


class TestApiIdempotency:
    """Re-delivering the same event_id through the API must produce exactly one row."""

    def _make_app(self, warehouse: TelemetryWarehouse):
        """Build a minimal FastAPI test app with the telemetry router."""
        from fastapi import FastAPI

        from trainer.research_lab.telemetry_server import make_telemetry_router

        test_app = FastAPI()

        def _noop_auth() -> None:
            return None

        router = make_telemetry_router(warehouse, _noop_auth)
        test_app.include_router(router)
        return test_app

    def test_decision_deduped_when_same_event_id_posted_twice(
        self, warehouse: TelemetryWarehouse
    ) -> None:
        from starlette.testclient import TestClient

        client = TestClient(self._make_app(warehouse))
        payload = {
            "model_version": MV,
            "symbol": SYM,
            "timestamp": NOW,
            "direction": "LONG",
            "confidence": 0.87,
            "event_id": "dec-idempotency-test-001",
        }
        r1 = client.post("/telemetry/decisions", json=payload)
        r2 = client.post("/telemetry/decisions", json=payload)
        assert r1.status_code == 200
        assert r2.status_code == 200
        # Both calls succeed but only one row stored
        rows = warehouse.query_decisions()
        assert len(rows) == 1
        assert rows[0]["event_id"] == "dec-idempotency-test-001"

    def test_trade_deduped_when_same_event_id_posted_twice(
        self, warehouse: TelemetryWarehouse
    ) -> None:
        from starlette.testclient import TestClient

        client = TestClient(self._make_app(warehouse))
        payload = {
            "model_version": MV,
            "symbol": SYM,
            "timestamp": NOW,
            "direction": "LONG",
            "entry_price": 1950.0,
            "exit_price": 1965.0,
            "size": 0.1,
            "pnl": 150.0,
            "pnl_pct": 0.77,
            "hold_duration_s": 3600.0,
            "event_id": "trd-idempotency-test-001",
        }
        client.post("/telemetry/trades", json=payload)
        client.post("/telemetry/trades", json=payload)
        rows = warehouse.query_trades()
        assert len(rows) == 1

    def test_all_streams_accept_event_id_field(self, warehouse: TelemetryWarehouse) -> None:
        """Smoke-test that all eight streams accept an event_id without error."""
        from starlette.testclient import TestClient

        client = TestClient(self._make_app(warehouse))
        base = {"model_version": MV, "symbol": SYM, "timestamp": NOW}

        endpoints_payloads = [
            (
                "/telemetry/decisions",
                {**base, "direction": "LONG", "confidence": 0.8, "event_id": "eid-dec"},
            ),
            (
                "/telemetry/trades",
                {
                    **base,
                    "direction": "LONG",
                    "entry_price": 1.0,
                    "exit_price": 1.1,
                    "size": 1.0,
                    "pnl": 10.0,
                    "pnl_pct": 1.0,
                    "hold_duration_s": 60.0,
                    "event_id": "eid-trd",
                },
            ),
            (
                "/telemetry/positions",
                {**base, "action": "OPEN", "size": 0.1, "price": 1950.0, "event_id": "eid-pos"},
            ),
            ("/telemetry/failures", {**base, "failure_type": "TIMEOUT", "event_id": "eid-fail"}),
            ("/telemetry/reasoning", {**base, "decision_id": "did-1", "event_id": "eid-rsn"}),
            (
                "/telemetry/execution",
                {
                    **base,
                    "order_type": "MARKET",
                    "requested_price": 1950.0,
                    "executed_price": 1950.5,
                    "size": 0.1,
                    "event_id": "eid-exec",
                },
            ),
            (
                "/telemetry/counterfactuals",
                {
                    **base,
                    "trade_id": "tid-1",
                    "what_if_description": "test",
                    "actual_pnl": 10.0,
                    "counterfactual_pnl": 20.0,
                    "event_id": "eid-cf",
                },
            ),
            ("/telemetry/lifecycle", {**base, "kind": "SESSION_STARTED", "event_id": "eid-lce"}),
        ]
        for endpoint, payload in endpoints_payloads:
            resp = client.post(endpoint, json=payload)
            assert resp.status_code == 200, f"{endpoint} returned {resp.status_code}: {resp.text}"


# ── Auth enforcement ──────────────────────────────────────────────────────────


class TestAuthEnforcement:
    """All telemetry endpoints must be blocked without a valid admin token."""

    def _make_authenticated_app(self, warehouse: TelemetryWarehouse):
        """Build an app that enforces a real bearer token."""

        from fastapi import FastAPI, HTTPException, Security
        from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

        from trainer.research_lab.telemetry_server import make_telemetry_router

        test_app = FastAPI()
        _token = "test-admin-secret"
        _bearer = HTTPBearer(auto_error=False)

        def _require_admin(
            credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
        ) -> None:
            token = credentials.credentials if credentials else None
            if not token or token != _token:
                raise HTTPException(403, detail="Forbidden")

        router = make_telemetry_router(warehouse, _require_admin)
        test_app.include_router(router)
        return test_app, _token

    def test_missing_token_returns_403(self, warehouse: TelemetryWarehouse) -> None:
        from starlette.testclient import TestClient

        app, _ = self._make_authenticated_app(warehouse)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/telemetry/decisions",
            json={
                "model_version": MV,
                "symbol": SYM,
                "timestamp": NOW,
                "direction": "LONG",
                "confidence": 0.8,
            },
        )
        assert resp.status_code == 403

    def test_invalid_token_returns_403(self, warehouse: TelemetryWarehouse) -> None:
        from starlette.testclient import TestClient

        app, _ = self._make_authenticated_app(warehouse)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/telemetry/decisions",
            json={
                "model_version": MV,
                "symbol": SYM,
                "timestamp": NOW,
                "direction": "LONG",
                "confidence": 0.8,
            },
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 403

    def test_valid_token_succeeds(self, warehouse: TelemetryWarehouse) -> None:
        from starlette.testclient import TestClient

        app, token = self._make_authenticated_app(warehouse)
        client = TestClient(app)
        resp = client.post(
            "/telemetry/decisions",
            json={
                "model_version": MV,
                "symbol": SYM,
                "timestamp": NOW,
                "direction": "LONG",
                "confidence": 0.8,
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_status_endpoint_also_requires_auth(self, warehouse: TelemetryWarehouse) -> None:
        from starlette.testclient import TestClient

        app, _ = self._make_authenticated_app(warehouse)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/telemetry/status")
        assert resp.status_code == 403

    def test_all_post_endpoints_blocked_without_auth(self, warehouse: TelemetryWarehouse) -> None:
        """Verify every POST endpoint is auth-gated, not just decisions."""
        from starlette.testclient import TestClient

        app, _ = self._make_authenticated_app(warehouse)
        client = TestClient(app, raise_server_exceptions=False)

        endpoints_payloads = [
            (
                "/telemetry/decisions",
                {
                    "model_version": MV,
                    "symbol": SYM,
                    "timestamp": NOW,
                    "direction": "LONG",
                    "confidence": 0.8,
                },
            ),
            (
                "/telemetry/trades",
                {
                    "model_version": MV,
                    "symbol": SYM,
                    "timestamp": NOW,
                    "direction": "LONG",
                    "entry_price": 1.0,
                    "exit_price": 1.1,
                    "size": 1.0,
                    "pnl": 10.0,
                    "pnl_pct": 1.0,
                    "hold_duration_s": 60.0,
                },
            ),
            (
                "/telemetry/positions",
                {
                    "model_version": MV,
                    "symbol": SYM,
                    "timestamp": NOW,
                    "action": "OPEN",
                    "size": 0.1,
                    "price": 1950.0,
                },
            ),
            (
                "/telemetry/failures",
                {"model_version": MV, "symbol": SYM, "timestamp": NOW, "failure_type": "TIMEOUT"},
            ),
            (
                "/telemetry/reasoning",
                {"model_version": MV, "symbol": SYM, "timestamp": NOW, "decision_id": "did-1"},
            ),
            (
                "/telemetry/execution",
                {
                    "model_version": MV,
                    "symbol": SYM,
                    "timestamp": NOW,
                    "order_type": "MARKET",
                    "requested_price": 1950.0,
                    "executed_price": 1950.5,
                    "size": 0.1,
                },
            ),
            (
                "/telemetry/counterfactuals",
                {
                    "model_version": MV,
                    "symbol": SYM,
                    "timestamp": NOW,
                    "trade_id": "tid-1",
                    "what_if_description": "test",
                    "actual_pnl": 10.0,
                    "counterfactual_pnl": 20.0,
                },
            ),
            (
                "/telemetry/lifecycle",
                {"model_version": MV, "symbol": SYM, "timestamp": NOW, "kind": "SESSION_STARTED"},
            ),
        ]
        for endpoint, payload in endpoints_payloads:
            resp = client.post(endpoint, json=payload)
            assert (
                resp.status_code == 403
            ), f"{endpoint} should be 403 without auth, got {resp.status_code}"


# ── Pydantic request model validation ─────────────────────────────────────────


class TestRequestModels:
    def test_ingest_decision_request_validation(self) -> None:
        from trainer.research_lab.telemetry_server import IngestDecisionRequest

        req = IngestDecisionRequest(
            model_version="v1",
            symbol="XAUUSD",
            timestamp=NOW,
            direction="LONG",
            confidence=0.87,
        )
        assert req.direction == "LONG"
        assert req.market_conditions == {}

    def test_ingest_trade_request_requires_prices(self) -> None:
        from pydantic import ValidationError

        from trainer.research_lab.telemetry_server import IngestTradeRequest

        with pytest.raises(ValidationError):
            IngestTradeRequest(
                model_version="v1",
                symbol="XAUUSD",
                timestamp=NOW,
                direction="LONG",
                # missing required: entry_price, exit_price, size, pnl, pnl_pct, hold_duration_s
            )

    def test_all_request_models_instantiate(self) -> None:
        from trainer.research_lab.telemetry_server import (
            IngestCounterfactualRequest,
            IngestExecutionRequest,
            IngestFailureRequest,
            IngestLifecycleRequest,
            IngestPositionRequest,
            IngestReasoningRequest,
        )

        base = {"model_version": "v1", "symbol": "XAUUSD", "timestamp": NOW}
        IngestPositionRequest(**base, action="OPEN", size=0.1, price=1950.0)
        IngestFailureRequest(**base, failure_type="TIMEOUT")
        IngestReasoningRequest(**base, decision_id="did-1")
        IngestExecutionRequest(
            **base, order_type="MARKET", requested_price=1950.0, executed_price=1950.5, size=0.1
        )
        IngestCounterfactualRequest(
            **base,
            trade_id="tid-1",
            what_if_description="test",
            actual_pnl=10.0,
            counterfactual_pnl=20.0,
        )
        IngestLifecycleRequest(**base, kind="SESSION_STARTED")
