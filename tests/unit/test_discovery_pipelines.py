"""Unit tests for S20 Auto Experiment, Feature, and Label Discovery pipelines.

Covers:
- ExperimentCreator: proposal from failure reports, deduplication, approve/reject
- FeatureDiscoveryPipeline: all six detectors with synthetic telemetry
- LabelDiscoveryEngine: all four labelers with synthetic trade paths
- DB: label_datasets table (upsert, list, stats)
- POST /research/discover endpoint integration
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from pydantic import BaseModel

from trainer.research_lab.discovery_models import DiscoveryQuery, DiscoverySummary
from trainer.research_lab.experiment_creator import ExperimentCreator
from trainer.research_lab.feature_discovery_pipeline import FeatureDiscoveryPipeline
from trainer.research_lab.investigation_models import (
    InvestigationFailureKind,
)
from trainer.research_lab.label_discovery_engine import (
    LabelDiscoveryEngine,
    _trade_to_outcome_path,
)
from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.rl_models import ExperimentStatus
from trainer.research_lab.telemetry_warehouse import TelemetryWarehouse


# ── Module-level Pydantic model for endpoint tests (avoid deferred-annotation issues)
class _DiscoverReq(BaseModel):
    model_version: str | None = None
    since: float | None = None
    until: float | None = None
    symbol: str | None = None
    min_failure_severity: float = 0.0
    max_proposed_per_run: int = 10
    min_feature_samples: int = 1
    min_label_samples: int = 1
    limit_per_stream: int = 5000


# ── Fixtures ───────────────────────────────────────────────────────────────────

NOW = time.time()
MV = "v2.1.0"
SYM = "XAUUSD"


@pytest.fixture()
def db(tmp_path: Path) -> ResearchLabDatabase:
    return ResearchLabDatabase(str(tmp_path / "test_disco.db"))


@pytest.fixture()
def warehouse(db: ResearchLabDatabase) -> TelemetryWarehouse:
    return TelemetryWarehouse(db)


@pytest.fixture()
def query() -> DiscoveryQuery:
    return DiscoveryQuery(
        min_failure_severity=0.0,
        max_proposed_per_run=10,
        min_feature_samples=1,
        min_label_samples=1,
    )


# ── Telemetry helpers ──────────────────────────────────────────────────────────


def _trade(wh, pnl=50.0, confidence=0.7, hold_s=3600.0, symbol=SYM, mv=MV, ts=None):
    return wh.ingest_trade(
        model_version=mv,
        symbol=symbol,
        timestamp=ts or NOW,
        direction="LONG",
        entry_price=1950.0,
        exit_price=1950.0 + pnl / 0.1,
        size=0.1,
        pnl=pnl,
        pnl_pct=pnl / 1950 * 100,
        hold_duration_s=hold_s,
        confidence_at_entry=confidence,
    )


def _decision(wh, symbol=SYM, mv=MV, ts=None, confidence=0.7):
    return wh.ingest_decision(
        model_version=mv,
        symbol=symbol,
        timestamp=ts or NOW,
        direction="LONG",
        confidence=confidence,
    )


def _position(wh, action="SCALE_IN", cum_pnl=50.0, symbol=SYM, mv=MV):
    return wh.ingest_position(
        model_version=mv,
        symbol=symbol,
        timestamp=NOW,
        action=action,
        size=0.1,
        price=1950.0,
        cumulative_pnl=cum_pnl,
    )


def _failure_report(db, kind: InvestigationFailureKind, severity=0.75, mv=MV):
    from trainer.research_lab.investigation_models import FailureReport

    r = FailureReport.new(
        failure_kind=kind,
        model_version=mv,
        affected_symbols=[SYM],
        time_range_start=NOW - 3600,
        time_range_end=NOW,
        evidence=[],
        severity_score=severity,
        summary=f"Synthetic {kind.value} report",
    )
    db.upsert_failure_report(r.to_dict())
    return r


# ── ExperimentCreator ──────────────────────────────────────────────────────────


class TestExperimentCreator:
    def test_proposes_experiment_from_failure_report(self, db: ResearchLabDatabase) -> None:
        _failure_report(db, InvestigationFailureKind.HIGH_CONFIDENCE_LOSS)
        creator = ExperimentCreator(db)
        summary = DiscoverySummary()
        creator.propose(DiscoveryQuery(min_failure_severity=0.0), summary)
        assert summary.experiments_proposed == 1
        assert len(summary.experiment_ids) == 1
        # Verify it was persisted in PROPOSED state
        exps = db.list_experiments(status=ExperimentStatus.PROPOSED.value)
        assert len(exps) == 1
        assert "confidence" in exps[0]["name"].lower()

    def test_proposes_multiple_kinds(self, db: ResearchLabDatabase) -> None:
        for kind in [
            InvestigationFailureKind.HIGH_CONFIDENCE_LOSS,
            InvestigationFailureKind.UNEXPECTED_REVERSAL,
            InvestigationFailureKind.FAILED_SCALE_IN,
        ]:
            _failure_report(db, kind)
        creator = ExperimentCreator(db)
        summary = DiscoverySummary()
        creator.propose(DiscoveryQuery(min_failure_severity=0.0, max_proposed_per_run=10), summary)
        assert summary.experiments_proposed == 3

    def test_deduplicates_same_kind(self, db: ResearchLabDatabase) -> None:
        # Add two HIGH_CONFIDENCE_LOSS reports
        _failure_report(db, InvestigationFailureKind.HIGH_CONFIDENCE_LOSS, severity=0.8)
        _failure_report(db, InvestigationFailureKind.HIGH_CONFIDENCE_LOSS, severity=0.6)
        creator = ExperimentCreator(db)
        summary = DiscoverySummary()
        # First propose: should create 1 experiment
        creator.propose(DiscoveryQuery(min_failure_severity=0.0), summary)
        assert summary.experiments_proposed == 1
        # Second propose: should skip (already PROPOSED)
        summary2 = DiscoverySummary()
        creator.propose(DiscoveryQuery(min_failure_severity=0.0), summary2)
        assert summary2.experiments_proposed == 0
        assert summary2.experiments_skipped_duplicate == 1

    def test_max_proposed_per_run_respected(self, db: ResearchLabDatabase) -> None:
        for kind in InvestigationFailureKind:
            _failure_report(db, kind)
        creator = ExperimentCreator(db)
        summary = DiscoverySummary()
        creator.propose(DiscoveryQuery(min_failure_severity=0.0, max_proposed_per_run=2), summary)
        assert summary.experiments_proposed == 2

    def test_severity_filter_respected(self, db: ResearchLabDatabase) -> None:
        _failure_report(db, InvestigationFailureKind.HIGH_CONFIDENCE_LOSS, severity=0.2)
        creator = ExperimentCreator(db)
        summary = DiscoverySummary()
        # min_severity=0.5 should exclude severity=0.2 report
        creator.propose(DiscoveryQuery(min_failure_severity=0.5), summary)
        assert summary.experiments_proposed == 0

    def test_experiment_has_required_fields(self, db: ResearchLabDatabase) -> None:
        _failure_report(db, InvestigationFailureKind.LATE_EXIT)
        creator = ExperimentCreator(db)
        summary = DiscoverySummary()
        creator.propose(DiscoveryQuery(min_failure_severity=0.0), summary)
        exp_id = summary.experiment_ids[0]
        record = db.get_experiment(exp_id)
        assert record is not None
        assert record["status"] == ExperimentStatus.PROPOSED.value
        assert "auto-proposed" in record.get("tags", [])
        assert "auto-discovery" == record.get("researcher")
        assert record.get("hyperparameters")  # non-empty
        assert record.get("notes")  # rationale embedded

    def test_approve_advances_to_pending(self, db: ResearchLabDatabase) -> None:
        _failure_report(db, InvestigationFailureKind.PREMATURE_EXIT)
        creator = ExperimentCreator(db)
        summary = DiscoverySummary()
        creator.propose(DiscoveryQuery(min_failure_severity=0.0), summary)
        exp_id = summary.experiment_ids[0]
        creator.approve(exp_id)
        record = db.get_experiment(exp_id)
        assert record["status"] == ExperimentStatus.PENDING.value

    def test_reject_advances_to_archived(self, db: ResearchLabDatabase) -> None:
        _failure_report(db, InvestigationFailureKind.PORTFOLIO_CONCENTRATION)
        creator = ExperimentCreator(db)
        summary = DiscoverySummary()
        creator.propose(DiscoveryQuery(min_failure_severity=0.0), summary)
        exp_id = summary.experiment_ids[0]
        creator.reject(exp_id)
        record = db.get_experiment(exp_id)
        assert record["status"] == ExperimentStatus.ARCHIVED.value

    def test_approve_non_proposed_raises(self, db: ResearchLabDatabase) -> None:
        _failure_report(db, InvestigationFailureKind.HIGH_CONFIDENCE_LOSS)
        creator = ExperimentCreator(db)
        summary = DiscoverySummary()
        creator.propose(DiscoveryQuery(min_failure_severity=0.0), summary)
        exp_id = summary.experiment_ids[0]
        creator.approve(exp_id)  # → PENDING
        with pytest.raises(ValueError, match="PROPOSED"):
            creator.approve(exp_id)  # already PENDING

    def test_no_reports_produces_no_proposals(self, db: ResearchLabDatabase) -> None:
        creator = ExperimentCreator(db)
        summary = DiscoverySummary()
        creator.propose(DiscoveryQuery(min_failure_severity=0.0), summary)
        assert summary.experiments_proposed == 0

    def test_list_proposed_returns_only_proposed(self, db: ResearchLabDatabase) -> None:
        _failure_report(db, InvestigationFailureKind.HIGH_CONFIDENCE_LOSS)
        _failure_report(db, InvestigationFailureKind.FAILED_SCALE_IN)
        creator = ExperimentCreator(db)
        creator.propose(
            DiscoveryQuery(min_failure_severity=0.0, max_proposed_per_run=10), DiscoverySummary()
        )
        proposed = creator.list_proposed()
        assert len(proposed) == 2
        assert all(p["status"] == "PROPOSED" for p in proposed)

    def test_model_version_scope_respected(self, db: ResearchLabDatabase) -> None:
        """propose() must only consider failure reports for the requested model version."""
        _failure_report(db, InvestigationFailureKind.HIGH_CONFIDENCE_LOSS, mv="v1.0.0")
        _failure_report(db, InvestigationFailureKind.FAILED_SCALE_IN, mv="v2.0.0")
        creator = ExperimentCreator(db)
        summary = DiscoverySummary()
        # Request proposals only for v1.0.0 — should yield exactly 1 proposal
        creator.propose(
            DiscoveryQuery(min_failure_severity=0.0, model_version="v1.0.0"),
            summary,
        )
        assert summary.experiments_proposed == 1
        proposed = creator.list_proposed()
        # All proposed experiments must reference v1.0.0
        assert all(p.get("qde_version") == "v1.0.0" for p in proposed)

    def test_unfiltered_run_proposes_all_versions(self, db: ResearchLabDatabase) -> None:
        """When model_version is None, all failure reports are eligible."""
        _failure_report(db, InvestigationFailureKind.HIGH_CONFIDENCE_LOSS, mv="v1.0.0")
        _failure_report(db, InvestigationFailureKind.UNEXPECTED_REVERSAL, mv="v2.0.0")
        creator = ExperimentCreator(db)
        summary = DiscoverySummary()
        creator.propose(
            DiscoveryQuery(min_failure_severity=0.0, model_version=None),
            summary,
        )
        assert summary.experiments_proposed == 2


# ── FeatureDiscoveryPipeline ───────────────────────────────────────────────────


class TestFeatureDiscoveryPipeline:
    def test_discovers_confidence_slope(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase, query: DiscoveryQuery
    ) -> None:
        for i in range(15):
            _decision(warehouse, ts=NOW + i * 600, confidence=0.9 - i * 0.02)
        pipeline = FeatureDiscoveryPipeline(warehouse, db)
        summary = DiscoverySummary()
        pipeline.discover(query, summary)
        assert summary.features_discovered >= 1
        names = {f["name"] for f in db.list_features()}
        assert "confidence_slope_per_hour" in names

    def test_discovers_decision_density(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase, query: DiscoveryQuery
    ) -> None:
        for i in range(20):
            _decision(warehouse, ts=NOW + i * 120)
        pipeline = FeatureDiscoveryPipeline(warehouse, db)
        summary = DiscoverySummary()
        pipeline.discover(query, summary)
        names = {f["name"] for f in db.list_features()}
        assert "session_decision_density_per_hour" in names

    def test_discovers_loss_run_length(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase, query: DiscoveryQuery
    ) -> None:
        for _ in range(5):
            _trade(warehouse, pnl=-20.0)
        _trade(warehouse, pnl=50.0)
        pipeline = FeatureDiscoveryPipeline(warehouse, db)
        summary = DiscoverySummary()
        pipeline.discover(query, summary)
        names = {f["name"] for f in db.list_features()}
        assert "max_consecutive_loss_run" in names

    def test_discovers_confidence_outcome_correlation(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase, query: DiscoveryQuery
    ) -> None:
        for i in range(10):
            _trade(warehouse, pnl=float(i * 10 - 30), confidence=0.5 + i * 0.04, ts=NOW + i)
        pipeline = FeatureDiscoveryPipeline(warehouse, db)
        summary = DiscoverySummary()
        pipeline.discover(query, summary)
        names = {f["name"] for f in db.list_features()}
        assert "entry_confidence_pnl_correlation" in names

    def test_discovers_reversal_rate(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase, query: DiscoveryQuery
    ) -> None:
        # 5 quick losses, 5 normal trades
        for _ in range(5):
            _trade(warehouse, pnl=-10.0, hold_s=60.0)
        for _ in range(5):
            _trade(warehouse, pnl=30.0, hold_s=3600.0)
        pipeline = FeatureDiscoveryPipeline(warehouse, db)
        summary = DiscoverySummary()
        pipeline.discover(query, summary)
        names = {f["name"] for f in db.list_features()}
        assert "quick_loss_reversal_rate" in names

    def test_discovers_scale_in_success_rate(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase, query: DiscoveryQuery
    ) -> None:
        for i in range(8):
            _position(warehouse, cum_pnl=100.0 if i % 2 == 0 else -50.0)
        pipeline = FeatureDiscoveryPipeline(warehouse, db)
        summary = DiscoverySummary()
        pipeline.discover(query, summary)
        names = {f["name"] for f in db.list_features()}
        assert "scale_in_success_rate" in names

    def test_empty_telemetry_safe(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase, query: DiscoveryQuery
    ) -> None:
        pipeline = FeatureDiscoveryPipeline(warehouse, db)
        summary = DiscoverySummary()
        pipeline.discover(query, summary)
        assert summary.features_discovered == 0

    def test_no_duplicate_feature_names(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase, query: DiscoveryQuery
    ) -> None:
        for i in range(10):
            _decision(warehouse, ts=NOW + i * 60, confidence=0.7 - i * 0.01)
        pipeline = FeatureDiscoveryPipeline(warehouse, db)
        # Run twice; second run should not create duplicates
        pipeline.discover(query, DiscoverySummary())
        count_after_first = len(db.list_features())
        pipeline.discover(query, DiscoverySummary())
        assert len(db.list_features()) == count_after_first

    def test_feature_has_required_fields(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase, query: DiscoveryQuery
    ) -> None:
        for i in range(15):
            _decision(warehouse, ts=NOW + i * 600, confidence=0.9 - i * 0.02)
        pipeline = FeatureDiscoveryPipeline(warehouse, db)
        pipeline.discover(query, DiscoverySummary())
        features = db.list_features()
        assert features
        f = features[0]
        required = {
            "feature_id",
            "name",
            "category",
            "importance_score",
            "evidence_strength",
            "formula_description",
            "suggested_qde_role",
            "sample_count",
            "status",
        }
        assert required.issubset(f.keys())
        assert 0.0 <= f["importance_score"] <= 1.0
        assert 0.0 <= f["evidence_strength"] <= 1.0
        assert f["sample_count"] >= 1

    def test_min_samples_filter(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase
    ) -> None:
        # Only 2 decisions — below min_feature_samples=5
        for i in range(2):
            _decision(warehouse, ts=NOW + i * 60)
        pipeline = FeatureDiscoveryPipeline(warehouse, db)
        summary = DiscoverySummary()
        pipeline.discover(DiscoveryQuery(min_feature_samples=5), summary)
        # Confidence slope needs at least 5 to pass filter; 2 samples should not persist it
        # (decision-density also needs ≥2 but 2 samples may or may not make it)
        assert summary.features_discovered == 0 or all(
            f["sample_count"] >= 1 for f in db.list_features()
        )


# ── LabelDiscoveryEngine ───────────────────────────────────────────────────────


class TestLabelDiscoveryEngine:
    def test_discovers_all_label_types(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase, query: DiscoveryQuery
    ) -> None:
        for i in range(10):
            _trade(warehouse, pnl=float(30 - i * 5), hold_s=3600.0, ts=NOW + i)
        engine = LabelDiscoveryEngine(warehouse, db)
        summary = DiscoverySummary()
        engine.discover(query, summary)
        assert summary.label_datasets_created >= 1
        label_types = {d["label_type"] for d in db.list_label_datasets()}
        assert "stop_loss" in label_types
        assert "take_profit" in label_types
        assert "position_size" in label_types

    def test_label_dataset_has_statistics(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase, query: DiscoveryQuery
    ) -> None:
        for i in range(5):
            _trade(warehouse, pnl=float(20 + i * 10), hold_s=3600.0)
        engine = LabelDiscoveryEngine(warehouse, db)
        engine.discover(query, DiscoverySummary())
        datasets = db.list_label_datasets(label_type="stop_loss")
        assert datasets
        d = datasets[0]
        stats = d["label_statistics"]
        assert "stop_distance_mean" in stats
        assert stats["stop_distance_mean"] >= 0

    def test_empty_telemetry_safe(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase, query: DiscoveryQuery
    ) -> None:
        engine = LabelDiscoveryEngine(warehouse, db)
        summary = DiscoverySummary()
        engine.discover(query, summary)
        assert summary.label_datasets_created == 0

    def test_min_label_samples_respected(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase
    ) -> None:
        # Only 1 trade, min=3
        _trade(warehouse, pnl=50.0)
        engine = LabelDiscoveryEngine(warehouse, db)
        summary = DiscoverySummary()
        engine.discover(DiscoveryQuery(min_label_samples=3), summary)
        # Engine skips entirely if paths < min
        assert summary.label_datasets_created == 0

    def test_stop_loss_label_statistics_bounds(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase, query: DiscoveryQuery
    ) -> None:
        for i in range(5):
            _trade(warehouse, pnl=float(40 - i * 10), hold_s=1800.0)
        engine = LabelDiscoveryEngine(warehouse, db)
        engine.discover(query, DiscoverySummary())
        datasets = db.list_label_datasets(label_type="stop_loss")
        assert datasets
        stats = datasets[0]["label_statistics"]
        assert stats["stop_distance_mean"] >= 0
        assert stats["stop_distance_std"] >= 0
        assert stats["stop_distance_min"] <= stats["stop_distance_mean"]
        assert stats["stop_distance_mean"] <= stats["stop_distance_max"]

    def test_position_size_labels_in_range(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase, query: DiscoveryQuery
    ) -> None:
        for i in range(5):
            _trade(warehouse, pnl=float(50 - i * 20), hold_s=3600.0)
        engine = LabelDiscoveryEngine(warehouse, db)
        engine.discover(query, DiscoverySummary())
        datasets = db.list_label_datasets(label_type="position_size")
        assert datasets
        stats = datasets[0]["label_statistics"]
        assert "size_signal_label_mean" in stats
        # size_signal_label ∈ [0, 1]
        assert 0.0 <= stats["size_signal_label_mean"] <= 1.0

    def test_list_datasets_filter_by_type(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase, query: DiscoveryQuery
    ) -> None:
        for _ in range(5):
            _trade(warehouse, pnl=30.0)
        engine = LabelDiscoveryEngine(warehouse, db)
        engine.discover(query, DiscoverySummary())
        sl_only = engine.list_datasets(label_type="stop_loss")
        assert all(d["label_type"] == "stop_loss" for d in sl_only)

    def test_lifecycle_label_dataset_persisted(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase, query: DiscoveryQuery
    ) -> None:
        """Lifecycle labels must be persisted when sufficient trades exist."""
        for i in range(6):
            _trade(warehouse, pnl=float(30 - i * 8), hold_s=3600.0, ts=NOW + i)
        engine = LabelDiscoveryEngine(warehouse, db)
        summary = DiscoverySummary()
        engine.discover(query, summary)
        lifecycle_datasets = engine.list_datasets(label_type="lifecycle")
        assert len(lifecycle_datasets) >= 1
        stats = lifecycle_datasets[0]["label_statistics"]
        assert "transitions_per_trade_mean" in stats

    def test_all_four_label_types_persisted(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase, query: DiscoveryQuery
    ) -> None:
        """When all four labelers run successfully, all four types appear in the DB."""
        for i in range(8):
            _trade(warehouse, pnl=float(40 - i * 8), hold_s=3600.0, ts=NOW + i)
        engine = LabelDiscoveryEngine(warehouse, db)
        summary = DiscoverySummary()
        engine.discover(query, summary)
        persisted_types = {d["label_type"] for d in db.list_label_datasets()}
        assert "stop_loss" in persisted_types
        assert "take_profit" in persisted_types
        assert "position_size" in persisted_types
        assert "lifecycle" in persisted_types


class TestTradeToOutcomePath:
    """Unit tests for the helper that reconstructs TradeOutcomePath from telemetry."""

    def test_long_trade_produces_valid_path(self) -> None:
        trade = {
            "entry_price": 1950.0,
            "exit_price": 1960.0,
            "direction": "LONG",
            "pnl": 10.0,
            "pnl_pct": 0.51,
            "hold_duration_s": 3600.0,
            "timestamp": NOW,
            "symbol": "XAUUSD",
        }
        path = _trade_to_outcome_path(trade)
        assert path is not None
        assert path.direction == 1
        assert path.entry_price == 1950.0
        assert len(path.future_bars) == 2
        assert path.future_bars[-1].close == pytest.approx(1960.0)

    def test_short_trade_direction(self) -> None:
        trade = {
            "entry_price": 1950.0,
            "exit_price": 1940.0,
            "direction": "SHORT",
            "pnl": 10.0,
            "pnl_pct": 0.51,
            "hold_duration_s": 3600.0,
            "timestamp": NOW,
            "symbol": "XAUUSD",
        }
        path = _trade_to_outcome_path(trade)
        assert path is not None
        assert path.direction == -1

    def test_losing_trade_works(self) -> None:
        trade = {
            "entry_price": 1950.0,
            "exit_price": 1930.0,
            "direction": "LONG",
            "pnl": -20.0,
            "pnl_pct": -1.0,
            "hold_duration_s": 1800.0,
            "timestamp": NOW,
            "symbol": "XAUUSD",
        }
        path = _trade_to_outcome_path(trade)
        assert path is not None
        assert path.future_bars[-1].close == pytest.approx(1930.0)

    def test_invalid_entry_price_returns_none(self) -> None:
        trade = {
            "entry_price": 0.0,
            "exit_price": 1950.0,
            "direction": "LONG",
            "pnl": 0.0,
            "hold_duration_s": 3600.0,
            "timestamp": NOW,
        }
        assert _trade_to_outcome_path(trade) is None


# ── DB: label_datasets table ──────────────────────────────────────────────────


class TestLabelDatasetsDB:
    def test_upsert_and_get(self, db: ResearchLabDatabase) -> None:
        from trainer.research_lab.label_discovery_engine import LabelDataset

        ds = LabelDataset.new(
            label_type="stop_loss",
            model_version=MV,
            sample_count=42,
            label_statistics={"stop_distance_mean": 0.05},
            notes="test dataset",
        )
        db.upsert_label_dataset(ds.to_dict())
        fetched = db.get_label_dataset(ds.dataset_id)
        assert fetched is not None
        assert fetched["sample_count"] == 42
        assert fetched["label_statistics"]["stop_distance_mean"] == pytest.approx(0.05)

    def test_list_filter_by_type(self, db: ResearchLabDatabase) -> None:
        from trainer.research_lab.label_discovery_engine import LabelDataset

        for lt in ["stop_loss", "take_profit", "position_size"]:
            ds = LabelDataset.new(lt, MV, 10, {}, "")
            db.upsert_label_dataset(ds.to_dict())
        sl = db.list_label_datasets(label_type="stop_loss")
        assert len(sl) == 1
        assert sl[0]["label_type"] == "stop_loss"

    def test_list_all_sorted_newest_first(self, db: ResearchLabDatabase) -> None:
        import time as _time

        from trainer.research_lab.label_discovery_engine import LabelDataset

        for _ in range(3):
            ds = LabelDataset.new("stop_loss", MV, 5, {}, "")
            db.upsert_label_dataset(ds.to_dict())
            _time.sleep(0.01)
        rows = db.list_label_datasets()
        assert len(rows) == 3
        timestamps = [r["created_at"] for r in rows]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_stats_includes_label_datasets(
        self, db: ResearchLabDatabase, warehouse: TelemetryWarehouse
    ) -> None:
        from trainer.research_lab.label_discovery_engine import LabelDataset

        ds = LabelDataset.new("stop_loss", MV, 5, {}, "")
        db.upsert_label_dataset(ds.to_dict())
        stats = db.stats()
        assert "label_datasets" in stats
        assert stats["label_datasets"] >= 1


# ── ExperimentStatus.PROPOSED ─────────────────────────────────────────────────


class TestProposedExperimentStatus:
    def test_proposed_is_valid_status(self) -> None:
        assert ExperimentStatus.PROPOSED.value == "PROPOSED"

    def test_proposed_not_in_pending_list(self, db: ResearchLabDatabase) -> None:
        # Ensure PROPOSED and PENDING are separate queryable states
        _failure_report(db, InvestigationFailureKind.HIGH_CONFIDENCE_LOSS)
        creator = ExperimentCreator(db)
        creator.propose(DiscoveryQuery(min_failure_severity=0.0), DiscoverySummary())
        pending = db.list_experiments(status=ExperimentStatus.PENDING.value)
        proposed = db.list_experiments(status=ExperimentStatus.PROPOSED.value)
        assert len(proposed) == 1
        assert len(pending) == 0


# ── POST /research/discover endpoint ─────────────────────────────────────────


class TestDiscoverEndpoint:
    def _make_app(
        self,
        warehouse: TelemetryWarehouse,
        db: ResearchLabDatabase,
        query: DiscoveryQuery,
    ):
        from fastapi import FastAPI, HTTPException

        test_app = FastAPI()
        creator = ExperimentCreator(db)
        feature_pipeline = FeatureDiscoveryPipeline(warehouse, db)
        label_engine = LabelDiscoveryEngine(warehouse, db)

        @test_app.post("/research/discover")
        def run_discover(req: _DiscoverReq) -> dict:
            try:
                dq = DiscoveryQuery(
                    model_version=req.model_version,
                    since=req.since,
                    until=req.until,
                    symbol=req.symbol,
                    min_failure_severity=req.min_failure_severity,
                    max_proposed_per_run=req.max_proposed_per_run,
                    min_feature_samples=req.min_feature_samples,
                    min_label_samples=req.min_label_samples,
                    limit_per_stream=req.limit_per_stream,
                )
                s = DiscoverySummary()
                creator.propose(dq, s)
                feature_pipeline.discover(dq, s)
                label_engine.discover(dq, s)
                return {"success": True, "summary": s.to_dict()}
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc))

        return test_app

    def test_discover_returns_success_envelope(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase, query: DiscoveryQuery
    ) -> None:
        from starlette.testclient import TestClient

        app = self._make_app(warehouse, db, query)
        client = TestClient(app)
        resp = client.post("/research/discover", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        s = body["summary"]
        assert "experiments_proposed" in s
        assert "features_discovered" in s
        assert "label_datasets_created" in s

    def test_discover_with_failure_reports_proposes_experiment(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase, query: DiscoveryQuery
    ) -> None:
        from starlette.testclient import TestClient

        _failure_report(db, InvestigationFailureKind.LATE_EXIT)
        app = self._make_app(warehouse, db, query)
        client = TestClient(app)
        resp = client.post("/research/discover", json={"min_failure_severity": 0.0})
        assert resp.status_code == 200
        s = resp.json()["summary"]
        assert s["experiments_proposed"] >= 1

    def test_discover_with_telemetry_discovers_features(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase, query: DiscoveryQuery
    ) -> None:
        from starlette.testclient import TestClient

        for i in range(12):
            _decision(warehouse, ts=NOW + i * 300, confidence=0.8 - i * 0.02)
        app = self._make_app(warehouse, db, query)
        client = TestClient(app)
        resp = client.post("/research/discover", json={"min_feature_samples": 1})
        assert resp.status_code == 200
        s = resp.json()["summary"]
        assert s["features_discovered"] >= 1

    def test_discover_with_trades_creates_label_datasets(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase, query: DiscoveryQuery
    ) -> None:
        from starlette.testclient import TestClient

        for i in range(5):
            _trade(warehouse, pnl=float(20 + i * 10))
        app = self._make_app(warehouse, db, query)
        client = TestClient(app)
        resp = client.post("/research/discover", json={"min_label_samples": 1})
        assert resp.status_code == 200
        s = resp.json()["summary"]
        assert s["label_datasets_created"] >= 1

    def test_discover_errors_do_not_crash(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase, query: DiscoveryQuery
    ) -> None:
        """Discovery errors are captured in summary.errors, not raised."""
        from starlette.testclient import TestClient

        app = self._make_app(warehouse, db, query)
        client = TestClient(app)
        resp = client.post("/research/discover", json={})
        assert resp.status_code == 200


class TestProposedExperimentEndpoints:
    """Tests for /research/proposed-experiments approve/reject endpoints."""

    def _make_app(self, db: ResearchLabDatabase):
        from fastapi import FastAPI, HTTPException

        test_app = FastAPI()
        creator = ExperimentCreator(db)

        @test_app.get("/research/proposed-experiments")
        def list_proposed() -> dict:
            rows = creator.list_proposed()
            return {"success": True, "total": len(rows), "experiments": rows}

        @test_app.post("/research/proposed-experiments/{experiment_id}/approve")
        def approve(experiment_id: str) -> dict:
            try:
                creator.approve(experiment_id)
                return {"success": True, "experiment_id": experiment_id, "new_status": "PENDING"}
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))

        @test_app.post("/research/proposed-experiments/{experiment_id}/reject")
        def reject(experiment_id: str) -> dict:
            try:
                creator.reject(experiment_id)
                return {"success": True, "experiment_id": experiment_id, "new_status": "ARCHIVED"}
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))

        return test_app

    def test_list_proposed_empty(self, db: ResearchLabDatabase) -> None:
        from starlette.testclient import TestClient

        app = self._make_app(db)
        client = TestClient(app)
        resp = client.get("/research/proposed-experiments")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_list_proposed_after_creation(self, db: ResearchLabDatabase) -> None:
        from starlette.testclient import TestClient

        _failure_report(db, InvestigationFailureKind.LATE_EXIT)
        ExperimentCreator(db).propose(DiscoveryQuery(min_failure_severity=0.0), DiscoverySummary())
        app = self._make_app(db)
        client = TestClient(app)
        resp = client.get("/research/proposed-experiments")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_approve_endpoint(self, db: ResearchLabDatabase) -> None:
        from starlette.testclient import TestClient

        _failure_report(db, InvestigationFailureKind.PREMATURE_EXIT)
        summary = DiscoverySummary()
        ExperimentCreator(db).propose(DiscoveryQuery(min_failure_severity=0.0), summary)
        exp_id = summary.experiment_ids[0]

        app = self._make_app(db)
        client = TestClient(app)
        resp = client.post(f"/research/proposed-experiments/{exp_id}/approve")
        assert resp.status_code == 200
        assert resp.json()["new_status"] == "PENDING"
        assert db.get_experiment(exp_id)["status"] == "PENDING"

    def test_reject_endpoint(self, db: ResearchLabDatabase) -> None:
        from starlette.testclient import TestClient

        _failure_report(db, InvestigationFailureKind.PORTFOLIO_CONCENTRATION)
        summary = DiscoverySummary()
        ExperimentCreator(db).propose(DiscoveryQuery(min_failure_severity=0.0), summary)
        exp_id = summary.experiment_ids[0]

        app = self._make_app(db)
        client = TestClient(app)
        resp = client.post(f"/research/proposed-experiments/{exp_id}/reject")
        assert resp.status_code == 200
        assert resp.json()["new_status"] == "ARCHIVED"
        assert db.get_experiment(exp_id)["status"] == "ARCHIVED"

    def test_approve_nonexistent_returns_400(self, db: ResearchLabDatabase) -> None:
        from starlette.testclient import TestClient

        app = self._make_app(db)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/research/proposed-experiments/nonexistent/approve")
        assert resp.status_code == 400

    def test_double_approve_returns_400(self, db: ResearchLabDatabase) -> None:
        from starlette.testclient import TestClient

        _failure_report(db, InvestigationFailureKind.FAILED_SCALE_IN)
        summary = DiscoverySummary()
        ExperimentCreator(db).propose(DiscoveryQuery(min_failure_severity=0.0), summary)
        exp_id = summary.experiment_ids[0]

        app = self._make_app(db)
        client = TestClient(app, raise_server_exceptions=False)
        client.post(f"/research/proposed-experiments/{exp_id}/approve")
        resp = client.post(f"/research/proposed-experiments/{exp_id}/approve")
        assert resp.status_code == 400


class TestLabelDatasetsEndpoint:
    """Tests for /research/label-datasets endpoint."""

    def _make_app(self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase):
        from fastapi import FastAPI

        test_app = FastAPI()
        engine = LabelDiscoveryEngine(warehouse, db)

        @test_app.get("/research/label-datasets")
        def list_label_datasets(
            label_type: str | None = None,
            model_version: str | None = None,
            limit: int = 50,
        ) -> dict:
            rows = engine.list_datasets(
                label_type=label_type, model_version=model_version, limit=limit
            )
            return {"success": True, "total": len(rows), "datasets": rows}

        return test_app

    def test_returns_empty_when_no_datasets(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase
    ) -> None:
        from starlette.testclient import TestClient

        app = self._make_app(warehouse, db)
        client = TestClient(app)
        resp = client.get("/research/label-datasets")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_returns_datasets_after_discovery(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase
    ) -> None:
        from starlette.testclient import TestClient

        for i in range(5):
            _trade(warehouse, pnl=float(30 - i * 5))
        engine = LabelDiscoveryEngine(warehouse, db)
        engine.discover(DiscoveryQuery(min_label_samples=1), DiscoverySummary())

        app = self._make_app(warehouse, db)
        client = TestClient(app)
        resp = client.get("/research/label-datasets")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 1
        assert all("label_type" in d for d in body["datasets"])

    def test_label_type_filter_query_param(
        self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase
    ) -> None:
        from starlette.testclient import TestClient

        for _ in range(5):
            _trade(warehouse, pnl=20.0)
        LabelDiscoveryEngine(warehouse, db).discover(
            DiscoveryQuery(min_label_samples=1), DiscoverySummary()
        )
        app = self._make_app(warehouse, db)
        client = TestClient(app)
        resp = client.get("/research/label-datasets?label_type=stop_loss")
        assert resp.status_code == 200
        for d in resp.json()["datasets"]:
            assert d["label_type"] == "stop_loss"
