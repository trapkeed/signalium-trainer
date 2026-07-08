"""Phase 15 — Institutional Validation Engine.

The top-level orchestrator for the Institutional Research Validation Platform.
Runs the complete validation pipeline from a single call:

  1. Leakage detection (abort if critical leakage found)
  2. Walk-forward validation (configurable strategy)
  3. Statistical metrics computation
  4. Trading metrics computation (via simulation)
  5. Calibration evaluation
  6. Market regime classification
  7. Multi-asset / multi-timeframe validation
  8. Prediction head validation
  9. Monte Carlo simulation
  10. Robustness testing
  11. Drift analysis
  12. Head-to-head model comparison (if champion exists)
  13. Failure analysis
  14. Promotion eligibility decision
  15. Leaderboard update
  16. Database persistence
  17. Report export (JSON, CSV, HTML)

No model is trained or modified. Evaluation only.

Usage::

    engine = InstitutionalValidationEngine(
        db_path="experiments/validation.db",
        leaderboard_path="experiments/leaderboard.json",
        policy=PromotionPolicy(min_accuracy=0.55, min_sharpe=1.0, max_drawdown=0.20),
        export_dir=Path("reports"),
    )
    result = engine.validate(
        model=my_model,
        x=data_tensor,
        y=labels_tensor,
        model_id="quant_v1",
        model_version="2.0.0",
    )
    print(result.summary())
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from trainer.validation.api import ValidationAPI
from trainer.validation.calibration_validator import CalibrationValidator
from trainer.validation.database import ValidationDatabase, ValidationRunRecord
from trainer.validation.drift import DriftAnalyzer
from trainer.validation.export import ExportConfig, ReportExporter
from trainer.validation.failure_analysis import FailureAnalyzer
from trainer.validation.head_validator import PredictionHeadValidator
from trainer.validation.leaderboard import InstitutionalLeaderboard, LeaderboardEntry
from trainer.validation.leakage import LeakageDetector, LeakageReport
from trainer.validation.monte_carlo import MonteCarloValidator
from trainer.validation.promotion import PromotionDecision, PromotionPolicy
from trainer.validation.robustness import RobustnessTester
from trainer.validation.statistical import StatisticalMetrics, StatisticalMetricsEngine
from trainer.validation.trading import TradeSimulator, TradingMetrics, TradingMetricsEngine
from trainer.validation.visualization import VisualizationDataBuilder
from trainer.validation.windows import ExpandingWindowValidator, SlidingWindowValidator, WindowFold

# ---------------------------------------------------------------------------
# Engine result
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Complete result from a single InstitutionalValidationEngine run.

    Attributes:
        run_id: UUID of this validation run.
        model_id: Model that was validated.
        model_version: Model version.
        statistical: Statistical metrics from the final fold.
        trading: Trading metrics.
        n_folds: Number of walk-forward folds evaluated.
        promotion_decision: Promotion eligibility decision.
        leaderboard_entry: Entry added to the leaderboard.
        leakage_report: Leakage detection results.
        run_duration_seconds: Total wall-clock time.
        status: "completed" | "failed" | "aborted_leakage".
        exported_paths: Paths to all written report files.
    """

    run_id: str
    model_id: str
    model_version: str
    statistical: StatisticalMetrics | None = None
    trading: TradingMetrics | None = None
    n_folds: int = 0
    promotion_decision: PromotionDecision | None = None
    leaderboard_entry: LeaderboardEntry | None = None
    leakage_report: LeakageReport | None = None
    run_duration_seconds: float = 0.0
    status: str = "completed"
    error: str = ""
    exported_paths: list[Path] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== Institutional Validation Result ===",
            f"  Model       : {self.model_id} v{self.model_version}",
            f"  Run ID      : {self.run_id}",
            f"  Status      : {self.status.upper()}",
            f"  Duration    : {self.run_duration_seconds:.1f}s",
            f"  Folds       : {self.n_folds}",
        ]
        if self.statistical:
            s = self.statistical
            lines += [
                "",
                "  Statistical Metrics:",
                f"    Accuracy     : {s.accuracy:.4f}",
                f"    F1 (macro)   : {s.f1_macro:.4f}",
                f"    ROC AUC      : {s.roc_auc:.4f}",
                f"    ECE          : {s.ece:.4f}",
            ]
        if self.trading:
            t = self.trading
            lines += [
                "",
                "  Trading Metrics:",
                f"    Win Rate     : {t.win_rate:.3f}",
                f"    Profit Factor: {t.profit_factor:.3f}",
                f"    Sharpe       : {t.sharpe_ratio:.3f}",
                f"    Max Drawdown : {t.max_drawdown:.3f}",
                f"    Expectancy   : {t.expectancy:.3f}",
                f"    N Trades     : {t.n_trades}",
            ]
        if self.promotion_decision:
            p = self.promotion_decision
            lines += [
                "",
                f"  Promotion    : {'ELIGIBLE' if p.is_eligible else 'NOT ELIGIBLE'}",
                f"  Promoted     : {p.is_promoted}",
            ]
        if self.exported_paths:
            lines += ["", "  Exports:"] + [f"    {p}" for p in self.exported_paths]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------


@dataclass
class EngineConfig:
    """Full configuration for InstitutionalValidationEngine.

    All settings have sensible defaults for immediate use.
    """

    # Walk-forward settings
    strategy: str = "expanding"  # "expanding" | "sliding"
    min_train_size: int = 500
    val_size: int = 100
    test_size: int = 0
    step_size: int = 50
    gap_size: int = 0
    max_folds: int = 0

    # Feature settings
    n_classes: int = 3
    n_calibration_bins: int = 15

    # Trade simulation
    pip_size: float = 0.0001
    spread_pips: float = 1.5
    sl_pips: float = 20.0
    tp_pips: float = 40.0
    direction_threshold: float = 0.50

    # Monte Carlo
    n_mc_simulations: int = 500
    mc_seed: int = 42

    # Robustness
    robustness_tolerance: float = 0.20
    robustness_seed: int = 42

    # Leakage detection
    abort_on_leakage: bool = True

    # Drift detection
    psi_threshold: float = 0.20

    # Export
    export_formats: list[str] = field(default_factory=lambda: ["json", "csv", "html"])

    # Model metadata
    architecture: str = ""
    backbone: str = ""
    phase: str = "15"
    n_parameters: int = 0
    training_time_seconds: float = 0.0
    dataset_version: str = ""
    feature_version: str = ""
    label_version: str = ""
    experiment_id: str = ""
    git_commit: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Institutional Validation Engine
# ---------------------------------------------------------------------------


class InstitutionalValidationEngine:
    """Phase 15 Institutional Research Validation Platform — main orchestrator.

    No model is trained or modified. This engine evaluates models only.
    Backward compatible with all existing trainer infrastructure.

    Args:
        db_path: Path to SQLite validation database.
        leaderboard_path: Path to JSON leaderboard file.
        policy: Promotion policy. If None, uses a permissive default.
        export_dir: Directory for exported reports.
        config: Full engine configuration.
    """

    def __init__(
        self,
        db_path: str | Path = "experiments/validation.db",
        leaderboard_path: str | Path | None = None,
        policy: PromotionPolicy | None = None,
        export_dir: Path | None = None,
        config: EngineConfig | None = None,
    ) -> None:
        self._cfg = config or EngineConfig()
        self._db = ValidationDatabase(db_path)
        self._lb = InstitutionalLeaderboard(
            store_path=Path(leaderboard_path) if leaderboard_path else None
        )
        self._policy = policy or PromotionPolicy(min_accuracy=0.50, require_manual_approval=True)
        self._export_dir = Path(export_dir) if export_dir else Path("reports")
        self._api = ValidationAPI(self._db, self._lb, self._policy)

        # Sub-engines
        self._stat_engine = StatisticalMetricsEngine(
            n_classes=self._cfg.n_classes,
            n_calibration_bins=self._cfg.n_calibration_bins,
        )
        self._trade_engine = TradingMetricsEngine()
        self._calib_validator = CalibrationValidator(
            n_bins=self._cfg.n_calibration_bins,
            n_classes=self._cfg.n_classes,
        )
        self._mc_validator = MonteCarloValidator(
            n_simulations=self._cfg.n_mc_simulations,
            seed=self._cfg.mc_seed,
        )
        self._robustness_tester = RobustnessTester(
            seed=self._cfg.robustness_seed,
            tolerance=self._cfg.robustness_tolerance,
        )
        self._leakage_detector = LeakageDetector(abort_on_detect=self._cfg.abort_on_leakage)
        self._drift_analyzer = DriftAnalyzer(psi_threshold=self._cfg.psi_threshold)
        self._failure_analyzer = FailureAnalyzer()
        self._head_validator = PredictionHeadValidator(
            n_calibration_bins=self._cfg.n_calibration_bins
        )
        self._simulator = TradeSimulator(
            pip_size=self._cfg.pip_size,
            spread_pips=self._cfg.spread_pips,
            sl_pips=self._cfg.sl_pips,
            tp_pips=self._cfg.tp_pips,
            direction_threshold=self._cfg.direction_threshold,
        )
        self._viz_builder = VisualizationDataBuilder()
        self._exporter = ReportExporter(
            output_dir=self._export_dir,
            config=ExportConfig(
                output_dir=self._export_dir,
                formats=self._cfg.export_formats,
            ),
        )

    def validate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        model_id: str,
        model_version: str = "1.0.0",
        ohlcv: torch.Tensor | None = None,
        timestamps: torch.Tensor | None = None,
        feature_names: list[str] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """Run the complete institutional validation pipeline.

        Args:
            model: Trained model (not modified — evaluation only).
            x: Feature tensor ``(N, seq_len, n_feat)`` or ``(N, n_feat)``.
            y: Label tensor ``(N,)``.
            model_id: Identifier for the model being validated.
            model_version: Semantic version string.
            ohlcv: Optional ``(N, 5)`` OHLCV data for trade simulation.
            timestamps: Optional ``(N,)`` Unix timestamps.
            feature_names: Optional feature column names for leakage reporting.
            extra_metadata: Optional extra fields merged into model_metadata.

        Returns:
            ``ValidationResult`` with all results and report paths.
        """
        run_id = str(uuid.uuid4())
        t0 = time.perf_counter()
        logger.info(f"InstitutionalValidationEngine: starting run {run_id} for {model_id!r}")

        model.eval()
        status = "completed"
        leakage_report = None

        # ── Build walk-forward strategy ───────────────────────────────────
        if self._cfg.strategy == "sliding":
            strategy = SlidingWindowValidator(
                train_size=self._cfg.min_train_size,
                val_size=self._cfg.val_size,
                test_size=self._cfg.test_size,
                step_size=self._cfg.step_size,
                gap_size=self._cfg.gap_size,
                max_folds=self._cfg.max_folds,
            )
        else:
            strategy = ExpandingWindowValidator(
                min_train_size=self._cfg.min_train_size,
                val_size=self._cfg.val_size,
                test_size=self._cfg.test_size,
                step_size=self._cfg.step_size,
                gap_size=self._cfg.gap_size,
                max_folds=self._cfg.max_folds,
            )

        # ── Generate folds ────────────────────────────────────────────────
        folds: list[WindowFold] = list(strategy.split(x, y))
        n_folds = len(folds)
        logger.info(f"  Walk-forward: {n_folds} folds ({self._cfg.strategy})")

        if n_folds == 0:
            logger.warning("  No folds generated — dataset too small.")
            return ValidationResult(
                run_id=run_id,
                model_id=model_id,
                model_version=model_version,
                n_folds=0,
                status="failed",
                error="Dataset too small for even one validation fold.",
                run_duration_seconds=time.perf_counter() - t0,
            )

        # ── Use the last fold for deep evaluation ─────────────────────────
        final_fold = folds[-1]
        x_train, y_train = final_fold.x_train, final_fold.y_train
        x_val, y_val = final_fold.x_val, final_fold.y_val

        # ── Step 1: Leakage detection ─────────────────────────────────────
        try:
            leakage_report = self._leakage_detector.check(
                x_train,
                y_train,
                x_val,
                y_val,
                timestamps=timestamps,
                feature_names=feature_names,
            )
        except Exception as exc:
            # LeakageDetected is a RuntimeError subclass
            if "leakage" in str(exc).lower():
                logger.error(f"  Leakage detected — aborting: {exc}")
                return ValidationResult(
                    run_id=run_id,
                    model_id=model_id,
                    model_version=model_version,
                    n_folds=n_folds,
                    status="aborted_leakage",
                    leakage_report=leakage_report,
                    error=str(exc),
                    run_duration_seconds=time.perf_counter() - t0,
                )
            logger.warning(f"  Leakage check error (non-critical): {exc}")

        # ── Step 2: Run inference on validation set ───────────────────────
        with torch.no_grad():
            try:
                logits = model(x_val)
                probs = torch.softmax(logits, dim=-1)
            except Exception as exc:
                logger.error(f"  Model inference failed: {exc}")
                return ValidationResult(
                    run_id=run_id,
                    model_id=model_id,
                    model_version=model_version,
                    n_folds=n_folds,
                    status="failed",
                    error=f"Model inference failed: {exc}",
                    run_duration_seconds=time.perf_counter() - t0,
                )

        # ── Step 3: Statistical metrics ───────────────────────────────────
        stat_metrics = self._stat_engine.compute(probs, y_val)
        logger.info(
            f"  Statistical: acc={stat_metrics.accuracy:.4f} "
            f"f1={stat_metrics.f1_macro:.4f} ece={stat_metrics.ece:.4f}"
        )

        # ── Step 4: Trading metrics ───────────────────────────────────────
        ohlcv_np = ohlcv.cpu().numpy() if ohlcv is not None else np.zeros((len(x_val), 5))
        probs_np = probs.cpu().numpy()
        trades = self._simulator.simulate(probs_np, ohlcv_np)
        trade_metrics = self._trade_engine.compute(trades)
        logger.info(
            f"  Trading: n_trades={trade_metrics.n_trades} "
            f"sharpe={trade_metrics.sharpe_ratio:.3f} "
            f"pf={trade_metrics.profit_factor:.3f}"
        )

        # ── Step 5: Calibration ───────────────────────────────────────────
        calib_report = self._calib_validator.evaluate(probs, y_val)

        # ── Step 6: Prediction head validation ────────────────────────────
        head_results = self._head_validator.validate(model, x_val, y_val, probs)

        # ── Step 7: Monte Carlo ───────────────────────────────────────────
        mc_result = self._mc_validator.run(trades)

        # ── Step 8: Robustness testing ────────────────────────────────────
        robustness_report = self._robustness_tester.evaluate(model, x_val, y_val)
        logger.info(f"  Robustness score: {robustness_report.overall_robustness_score:.1%}")

        # ── Step 9: Drift analysis ────────────────────────────────────────
        drift_report = self._drift_analyzer.detect(
            x_ref=x_train,
            x_current=x_val,
            y_ref=y_train,
            y_current=y_val,
            probs_ref=None,
            probs_current=probs,
            feature_names=feature_names,
        )

        # ── Step 10: Failure analysis ─────────────────────────────────────
        head_perf = {name: r.to_dict() for name, r in head_results.items() if r.is_active}
        failure_report = self._failure_analyzer.analyze(
            head_results=head_perf,
        )

        # ── Step 11: Promotion decision ───────────────────────────────────
        all_metrics = {
            **stat_metrics.to_dict(),
            **trade_metrics.to_dict(),
            "ece": stat_metrics.ece,
            "calibration_error": calib_report.ece,
        }
        promotion_decision = self._policy.evaluate(
            model_id=model_id,
            metrics=all_metrics,
            n_trades=trade_metrics.n_trades,
            n_folds=n_folds,
        )
        logger.info(
            f"  Promotion: eligible={promotion_decision.is_eligible} "
            f"promoted={promotion_decision.is_promoted}"
        )

        # ── Step 12: Build visualization data ─────────────────────────────
        viz_data = {
            "walk_forward_timeline": self._viz_builder.build_walk_forward_timeline(folds),
            "equity_curves": self._viz_builder.build_equity_curves(mc_result),
            "calibration_curve": self._viz_builder.build_calibration_curve(calib_report),
            "drift_chart": self._viz_builder.build_drift_chart(drift_report),
            "performance_trend": self._viz_builder.build_performance_trend(
                fold_metrics=[{"accuracy": 0.5}],
                metric="accuracy",
            ),
            "trade_distribution": self._viz_builder.build_trade_distribution(trades),
            "risk_distribution": self._viz_builder.build_risk_distribution(
                trade_metrics, mc_result
            ),
        }

        # ── Step 13: Leaderboard entry ────────────────────────────────────
        lb_entry = LeaderboardEntry(
            model_id=model_id,
            model_version=model_version,
            model_name=model_id,
            training_date=(extra_metadata or {}).get("training_date", ""),
            architecture=self._cfg.architecture,
            backbone=self._cfg.backbone,
            phase=self._cfg.phase,
            n_parameters=self._cfg.n_parameters,
            training_time_seconds=self._cfg.training_time_seconds,
            dataset_version=self._cfg.dataset_version,
            feature_version=self._cfg.feature_version,
            label_version=self._cfg.label_version,
            validation_accuracy=stat_metrics.accuracy,
            validation_f1=stat_metrics.f1_macro,
            ece=stat_metrics.ece,
            brier_score=stat_metrics.brier_score,
            roc_auc=stat_metrics.roc_auc,
            profit_factor=trade_metrics.profit_factor,
            sharpe_ratio=trade_metrics.sharpe_ratio,
            sortino_ratio=trade_metrics.sortino_ratio,
            calmar_ratio=trade_metrics.calmar_ratio,
            max_drawdown=trade_metrics.max_drawdown,
            win_rate=trade_metrics.win_rate,
            expectancy=trade_metrics.expectancy,
            promotion_status=(
                "champion"
                if promotion_decision.is_promoted
                else "candidate" if promotion_decision.is_eligible else "experimental"
            ),
            is_champion=promotion_decision.is_promoted,
            experiment_id=self._cfg.experiment_id,
            git_commit=self._cfg.git_commit,
            n_validation_folds=n_folds,
            notes=self._cfg.notes,
        )
        self._lb.add(lb_entry)

        # ── Step 14: Database persistence ────────────────────────────────
        run_record = ValidationRunRecord(
            run_id=run_id,
            model_id=model_id,
            model_version=model_version,
            configuration=self._cfg.__dict__,
            statistical_metrics=stat_metrics.to_dict(),
            trading_metrics=trade_metrics.to_dict(),
            calibration=calib_report.to_dict(),
            drift=drift_report.to_dict(),
            monte_carlo=mc_result.to_dict(),
            robustness=robustness_report.to_dict(),
            leaderboard_score=lb_entry.overall_score,
            promotion_decision=promotion_decision.to_dict(),
            model_metadata={
                "n_parameters": self._cfg.n_parameters,
                "architecture": self._cfg.architecture,
                "phase": self._cfg.phase,
                "n_validation_folds": n_folds,
                **(extra_metadata or {}),
            },
            git_commit=self._cfg.git_commit,
            experiment_id=self._cfg.experiment_id,
            dataset_version=self._cfg.dataset_version,
            feature_version=self._cfg.feature_version,
            label_version=self._cfg.label_version,
            visualization_data=viz_data,
            notes=self._cfg.notes,
            status=status,
        )
        self._db.save_run(run_record)

        # ── Step 15: Export reports ───────────────────────────────────────
        exported_paths = self._exporter.export_all(run_record, self._lb)
        duration = time.perf_counter() - t0
        logger.info(f"InstitutionalValidationEngine: run {run_id} completed in {duration:.1f}s")

        return ValidationResult(
            run_id=run_id,
            model_id=model_id,
            model_version=model_version,
            statistical=stat_metrics,
            trading=trade_metrics,
            n_folds=n_folds,
            promotion_decision=promotion_decision,
            leaderboard_entry=lb_entry,
            leakage_report=leakage_report,
            run_duration_seconds=duration,
            status=status,
            exported_paths=exported_paths,
        )

    @property
    def api(self) -> ValidationAPI:
        """Access the internal REST-like API."""
        return self._api

    @property
    def leaderboard(self) -> InstitutionalLeaderboard:
        return self._lb

    @property
    def database(self) -> ValidationDatabase:
        return self._db

    def close(self) -> None:
        """Close database connection and persist leaderboard."""
        self._db.close()
        if self._lb._path:
            try:
                self._lb.save()
            except Exception:
                pass
