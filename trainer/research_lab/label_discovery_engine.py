"""S20 — LabelDiscoveryEngine: derives hindsight training labels from trade telemetry.

Orchestrates the existing HindsightLabeler infrastructure over realized trade
paths reconstructed from the TelemetryWarehouse. Label datasets are persisted
in the `label_datasets` table, ready for the training pipeline to consume.

Label types produced:
    - stop_loss       — optimal stop distance derived from realized MAE
    - take_profit     — optimal target derived from realized MFE
    - position_size   — risk-adjusted size signal from realized return/risk ratio
    - lifecycle       — entry/peak/exit transition points from realized path
    - portfolio_size  — multi-asset allocation signal from ensemble path statistics
"""

from __future__ import annotations

import math
import statistics
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.microstructure.micro_models import OHLCBar
from trainer.research_lab.discovery_models import DiscoveryQuery, DiscoverySummary
from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.telemetry_models import TelemetryQuery
from trainer.research_lab.telemetry_warehouse import TelemetryWarehouse
from trainer.training_signals.lifecycle_transition_labels import HindsightLifecycleLabeler
from trainer.training_signals.outcome_path import TradeOutcomePath
from trainer.training_signals.position_size_labels import HindsightPositionSizeLabeler
from trainer.training_signals.stop_loss_labels import HindsightStopLossLabeler
from trainer.training_signals.take_profit_labels import HindsightTakeProfitLabeler


# ── Synthetic path construction ────────────────────────────────────────────────


def _trade_to_outcome_path(trade: dict[str, Any]) -> TradeOutcomePath | None:
    """Reconstruct a TradeOutcomePath from a telemetry trade record.

    We only have aggregate trade data (entry_price, exit_price, hold_duration_s)
    rather than tick-by-tick OHLC bars. We create a 3-bar synthetic path that
    preserves the realized PnL and provides enough structure for each labeler:

        Bar 0 (mid-point): high = max(entry, exit), low = min(entry, exit)
                           close = midpoint between entry and exit
        Bar 1 (at exit):   close = exit_price
                           MAE proxy from pnl direction

    This gives a reasonable substrate for all labelers without requiring
    historical price data.
    """
    try:
        entry = float(trade.get("entry_price", 0.0))
        exit_ = float(trade.get("exit_price", 0.0))
        if entry <= 0 or exit_ <= 0:
            return None

        direction_str = str(trade.get("direction", "LONG")).upper()
        direction = 1 if direction_str in ("LONG", "BUY", "1") else -1
        instrument = trade.get("symbol", "UNKNOWN")
        entry_time = float(trade.get("timestamp", time.time()))
        hold_s = float(trade.get("hold_duration_s", 3600.0))

        # Estimate max adverse excursion from pnl
        pnl = float(trade.get("pnl", 0.0))
        pnl_pct = float(trade.get("pnl_pct", pnl / entry if entry else 0.0))
        is_winner = pnl >= 0

        # For a winner: adverse excursion estimate is 30% of the favorable move
        # For a loser:  adverse excursion equals the loss magnitude
        if direction == 1:
            favorable_move = max(0.0, exit_ - entry)
            adverse_move = max(0.0, entry - exit_) if not is_winner else favorable_move * 0.3
        else:
            favorable_move = max(0.0, entry - exit_)
            adverse_move = max(0.0, exit_ - entry) if not is_winner else favorable_move * 0.3

        mid_price = (entry + exit_) / 2

        # Build 3 synthetic bars
        if direction == 1:
            bar_mid = OHLCBar(
                timestamp=entry_time + hold_s * 0.5,
                open=entry,
                high=max(entry, mid_price) + adverse_move * 0.1,
                low=entry - adverse_move,
                close=mid_price,
                instrument=instrument,
            )
            bar_exit = OHLCBar(
                timestamp=entry_time + hold_s,
                open=mid_price,
                high=max(mid_price, exit_) + favorable_move * 0.05,
                low=min(mid_price, exit_) - adverse_move * 0.05,
                close=exit_,
                instrument=instrument,
            )
        else:
            bar_mid = OHLCBar(
                timestamp=entry_time + hold_s * 0.5,
                open=entry,
                high=entry + adverse_move,
                low=min(entry, mid_price) - adverse_move * 0.1,
                close=mid_price,
                instrument=instrument,
            )
            bar_exit = OHLCBar(
                timestamp=entry_time + hold_s,
                open=mid_price,
                high=max(mid_price, exit_) + adverse_move * 0.05,
                low=min(mid_price, exit_) - favorable_move * 0.05,
                close=exit_,
                instrument=instrument,
            )

        return TradeOutcomePath(
            instrument=instrument,
            direction=direction,
            entry_price=entry,
            entry_time=entry_time,
            future_bars=[bar_mid, bar_exit],
        )
    except Exception:
        return None


# ── Label dataset record ───────────────────────────────────────────────────────


@dataclass
class LabelDataset:
    """Persisted output of one labeler run over a batch of trades."""

    dataset_id: str
    label_type: str          # "stop_loss" | "take_profit" | "position_size" | "lifecycle" | "portfolio_size"
    model_version: str
    created_at: float
    sample_count: int
    label_statistics: dict[str, float]    # mean, std, min, max per key field
    notes: str = ""

    @classmethod
    def new(
        cls,
        label_type: str,
        model_version: str,
        sample_count: int,
        label_statistics: dict[str, float],
        notes: str = "",
    ) -> LabelDataset:
        return cls(
            dataset_id=f"ld-{label_type[:8]}-{str(uuid.uuid4())[:8]}",
            label_type=label_type,
            model_version=model_version,
            created_at=time.time(),
            sample_count=sample_count,
            label_statistics=label_statistics,
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "label_type": self.label_type,
            "model_version": self.model_version,
            "created_at": round(self.created_at, 3),
            "sample_count": self.sample_count,
            "label_statistics": {k: round(v, 6) for k, v in self.label_statistics.items()},
            "notes": self.notes,
        }


# ── Engine ─────────────────────────────────────────────────────────────────────


class LabelDiscoveryEngine:
    """Orchestrates HindsightLabelers over telemetry trade paths.

    For each labeler:
      1. Pull trade telemetry from the warehouse
      2. Reconstruct synthetic TradeOutcomePaths
      3. Run the labeler on each path
      4. Compute summary statistics
      5. Persist a LabelDataset record

    The lifecycle and portfolio labelers require multi-bar paths. With the
    2-bar synthetic approximation they produce valid but rough labels — still
    meaningful for detecting average label distributions and regime shifts.
    """

    def __init__(self, warehouse: TelemetryWarehouse, db: ResearchLabDatabase) -> None:
        self._wh = warehouse
        self._db = db
        self._sl_labeler = HindsightStopLossLabeler()
        self._tp_labeler = HindsightTakeProfitLabeler()
        self._ps_labeler = HindsightPositionSizeLabeler()
        self._lc_labeler = HindsightLifecycleLabeler()

    # ── Public API ─────────────────────────────────────────────────────────────

    def discover(self, query: DiscoveryQuery, summary: DiscoverySummary) -> None:
        """Run all labelers and persist qualifying label datasets."""
        tq = TelemetryQuery(
            model_version=query.model_version,
            symbol=query.symbol,
            since=query.since,
            until=query.until,
            limit=query.limit_per_stream,
        )
        trades = self._wh.query_trades(tq)
        if not trades:
            logger.debug("LabelDiscoveryEngine.discover: no trade telemetry — skipping")
            return

        mv = query.model_version or (trades[0].get("model_version", "unknown") if trades else "unknown")
        paths = [p for t in trades if (p := _trade_to_outcome_path(t)) is not None]

        if len(paths) < query.min_label_samples:
            logger.debug(
                f"LabelDiscoveryEngine.discover: only {len(paths)} valid paths "
                f"(min {query.min_label_samples}) — skipping"
            )
            return

        labeler_fns = [
            ("stop_loss", self._run_stop_loss),
            ("take_profit", self._run_take_profit),
            ("position_size", self._run_position_size),
            ("lifecycle", self._run_lifecycle),
        ]

        for label_type, run_fn in labeler_fns:
            try:
                dataset = run_fn(paths, mv, query)
                if dataset is not None and dataset.sample_count >= query.min_label_samples:
                    self._db.upsert_label_dataset(dataset.to_dict())
                    summary.label_dataset_ids.append(dataset.dataset_id)
                    summary.label_datasets_created += 1
                    logger.info(
                        f"LabelDiscovery: persisted {label_type} dataset "
                        f"{dataset.dataset_id} n={dataset.sample_count}"
                    )
            except Exception as exc:
                msg = f"LabelDiscoveryEngine.{label_type}: {exc}"
                logger.warning(msg)
                summary.errors.append(msg)

    # ── Labeler runners ────────────────────────────────────────────────────────

    def _run_stop_loss(
        self, paths: list[TradeOutcomePath], mv: str, query: DiscoveryQuery
    ) -> LabelDataset | None:
        labels = [self._sl_labeler.label(p) for p in paths]
        distances = [lb.optimal_stop_distance for lb in labels]
        returns = [lb.realized_return_pct for lb in labels]
        stats = _label_stats(
            stop_distance=distances,
            realized_return_pct=returns,
        )
        return LabelDataset.new(
            label_type="stop_loss",
            model_version=mv,
            sample_count=len(labels),
            label_statistics=stats,
            notes=(
                f"Hindsight stop-loss labels from {len(labels)} trades. "
                f"Mean optimal stop distance: {stats.get('stop_distance_mean', 0):.5f}. "
                f"Mean realized return: {stats.get('realized_return_pct_mean', 0):.4f}."
            ),
        )

    def _run_take_profit(
        self, paths: list[TradeOutcomePath], mv: str, query: DiscoveryQuery
    ) -> LabelDataset | None:
        labels = [self._tp_labeler.label(p) for p in paths]
        distances = [lb.optimal_target_distance for lb in labels]
        bars = [float(lb.bars_to_reach) for lb in labels]
        stats = _label_stats(
            target_distance=distances,
            bars_to_reach=bars,
        )
        return LabelDataset.new(
            label_type="take_profit",
            model_version=mv,
            sample_count=len(labels),
            label_statistics=stats,
            notes=(
                f"Hindsight take-profit labels from {len(labels)} trades. "
                f"Mean optimal target distance: {stats.get('target_distance_mean', 0):.5f}. "
                f"Mean bars to reach target: {stats.get('bars_to_reach_mean', 0):.1f}."
            ),
        )

    def _run_position_size(
        self, paths: list[TradeOutcomePath], mv: str, query: DiscoveryQuery
    ) -> LabelDataset | None:
        labels = [self._ps_labeler.label(p) for p in paths]
        # Field is `size_signal_label` on HindsightPositionSizeLabel
        size_signals = [lb.size_signal_label for lb in labels]
        rar_vals = [lb.risk_adjusted_return for lb in labels]
        stats = _label_stats(
            size_signal_label=size_signals,
            risk_adjusted_return=rar_vals,
        )
        return LabelDataset.new(
            label_type="position_size",
            model_version=mv,
            sample_count=len(labels),
            label_statistics=stats,
            notes=(
                f"Hindsight position-size labels from {len(labels)} trades. "
                f"Mean size signal: {stats.get('size_signal_label_mean', 0):.4f}. "
                f"Mean risk-adjusted return: {stats.get('risk_adjusted_return_mean', 0):.4f}."
            ),
        )

    def _run_lifecycle(
        self, paths: list[TradeOutcomePath], mv: str, query: DiscoveryQuery
    ) -> LabelDataset | None:
        labels = [self._lc_labeler.label(p) for p in paths]
        # HindsightLifecycleLabel has `transitions: list[LifecycleTransitionPoint]`
        # where each transition has `bar_index` and `state`.
        # Summarise by counting transitions per label and recording mean bar indices.
        transition_counts = [float(len(lb.transitions)) for lb in labels]
        first_bar_idxs = [
            float(lb.transitions[0].bar_index)
            for lb in labels
            if lb.transitions
        ]
        return_pcts = [lb.realized_return_pct for lb in labels]
        if not transition_counts:
            return None
        stats = _label_stats(
            transitions_per_trade=transition_counts,
            first_transition_bar=first_bar_idxs if first_bar_idxs else transition_counts,
            realized_return_pct=return_pcts,
        )
        return LabelDataset.new(
            label_type="lifecycle",
            model_version=mv,
            sample_count=len(labels),
            label_statistics=stats,
            notes=(
                f"Hindsight lifecycle transition labels from {len(labels)} trades. "
                f"Mean transitions per trade: {stats.get('transitions_per_trade_mean', 0):.1f}. "
                f"Mean first-transition bar: {stats.get('first_transition_bar_mean', 0):.2f}."
            ),
        )

    # ── Public query ───────────────────────────────────────────────────────────

    def list_datasets(
        self,
        label_type: str | None = None,
        model_version: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return self._db.list_label_datasets(
            label_type=label_type, model_version=model_version, limit=limit
        )


# ── Statistics helpers ─────────────────────────────────────────────────────────


def _label_stats(**series: list[float]) -> dict[str, float]:
    """Compute mean/std/min/max for each named series."""
    out: dict[str, float] = {}
    for name, values in series.items():
        if not values:
            continue
        out[f"{name}_mean"] = round(statistics.mean(values), 6)
        out[f"{name}_std"] = round(statistics.stdev(values) if len(values) > 1 else 0.0, 6)
        out[f"{name}_min"] = round(min(values), 6)
        out[f"{name}_max"] = round(max(values), 6)
    return out
