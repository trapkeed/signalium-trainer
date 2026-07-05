"""Phase 25 — Certification data models."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from trainer.cert.cert_types import (
    CertificationStatus, CertificationStage, CheckStatus,
    StressScenario, BenchmarkMetric, ValidationCheck,
    RiskLevel, ScorecardCategory,
)


# ── Primary Input ──────────────────────────────────────────────────────────────

@dataclass
class CertificationInput:
    """Canonical input flowing through all certification stages."""
    model_id:                str              = ""
    model_name:              str              = ""
    model_version:           str              = "1.0.0"
    git_commit:              str              = ""
    training_completed_at:   float            = field(default_factory=time.time)
    requester:               str              = ""
    notes:                   str              = ""
    tags:                    List[str]         = field(default_factory=list)

    # Research quality
    research_quality_score:  float            = 0.0   # 0-100

    # Validation metrics
    oos_sharpe:              float            = 0.0
    oos_win_rate:            float            = 0.5
    oos_max_drawdown:        float            = 0.5   # fraction (0-1)
    walk_forward_results:    List[Dict]       = field(default_factory=list)
    cross_regime_results:    List[Dict]       = field(default_factory=list)
    cross_instrument_results: List[Dict]      = field(default_factory=list)
    cross_timeframe_results: List[Dict]       = field(default_factory=list)
    generalization_score:    float            = 0.0   # 0-100
    stability_score:         float            = 0.0   # 0-100

    # Calibration
    calibration_score:       float            = 0.5   # 0-1
    brier_score:             float            = 0.25
    ece:                     float            = 0.10

    # Stress testing  (scenario name -> pass/fail/degradation dict)
    stress_results:          Dict[str, Dict]  = field(default_factory=dict)

    # Monte Carlo
    monte_carlo_results:     Dict[str, Any]   = field(default_factory=dict)

    # Benchmark (metric name -> value)
    benchmark_values:        Dict[str, float] = field(default_factory=dict)

    # Reproducibility (check name -> bool)
    reproducibility_checks:  Dict[str, bool]  = field(default_factory=dict)
    dataset_version:         str              = ""
    feature_version:         str              = ""
    label_version:           str              = ""
    training_config:         Dict[str, Any]   = field(default_factory=dict)
    random_seed:             Optional[int]    = None
    environment:             str              = ""
    package_versions:        Dict[str, str]   = field(default_factory=dict)
    checkpoint_hash:         str              = ""
    model_hash:              str              = ""

    # Security (check name -> bool)
    security_checks:         Dict[str, bool]  = field(default_factory=dict)

    # Operational (check name -> bool)
    operational_checks:      Dict[str, bool]  = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()
                if not isinstance(v, (list, dict))}


# ── Stage Results ──────────────────────────────────────────────────────────────

@dataclass
class StageResult:
    stage:     CertificationStage = CertificationStage.RESEARCH
    status:    CheckStatus        = CheckStatus.PASS
    score:     float              = 100.0   # 0-100
    findings:  List[str]         = field(default_factory=list)
    warnings:  List[str]         = field(default_factory=list)
    failures:  List[str]         = field(default_factory=list)
    duration_ms: float           = 0.0
    timestamp: float             = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "stage":       self.stage.value,
            "status":      self.status.value,
            "score":       round(self.score, 2),
            "findings":    self.findings,
            "warnings":    self.warnings,
            "failures":    self.failures,
            "duration_ms": round(self.duration_ms, 2),
        }


# ── Checklist ─────────────────────────────────────────────────────────────────

@dataclass
class ChecklistItem:
    check_id:   str              = ""
    category:   str              = ""
    description: str             = ""
    mandatory:  bool             = True
    status:     CheckStatus      = CheckStatus.NOT_APPLICABLE
    value:      Optional[Any]    = None
    threshold:  Optional[float]  = None
    rationale:  str              = ""
    risk_level: RiskLevel        = RiskLevel.MEDIUM

    def to_dict(self) -> Dict:
        return {
            "check_id":    self.check_id,
            "category":    self.category,
            "description": self.description,
            "mandatory":   self.mandatory,
            "status":      self.status.value,
            "value":       self.value,
            "threshold":   self.threshold,
            "rationale":   self.rationale,
            "risk_level":  self.risk_level.value,
        }


@dataclass
class ChecklistReport:
    items:            List[ChecklistItem] = field(default_factory=list)
    pass_count:       int                 = 0
    warning_count:    int                 = 0
    fail_count:       int                 = 0
    na_count:         int                 = 0
    mandatory_failed: int                 = 0
    can_proceed:      bool                = True
    timestamp:        float               = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "items":            [i.to_dict() for i in self.items],
            "pass_count":       self.pass_count,
            "warning_count":    self.warning_count,
            "fail_count":       self.fail_count,
            "na_count":         self.na_count,
            "mandatory_failed": self.mandatory_failed,
            "can_proceed":      self.can_proceed,
        }


# ── Validation ────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    check:          ValidationCheck = ValidationCheck.OUT_OF_SAMPLE
    status:         CheckStatus     = CheckStatus.PASS
    score:          float           = 0.0
    details:        Dict[str, Any]  = field(default_factory=dict)
    timestamp:      float           = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "check":   self.check.value,
            "status":  self.status.value,
            "score":   round(self.score, 2),
            "details": self.details,
        }


# ── Stress Testing ────────────────────────────────────────────────────────────

@dataclass
class StressTestResult:
    scenario:          StressScenario = StressScenario.HIGH_VOLATILITY
    status:            CheckStatus    = CheckStatus.PASS
    degradation_pct:   float          = 0.0   # % performance drop vs baseline
    max_drawdown:      float          = 0.0
    sharpe_under_stress: float        = 0.0
    details:           Dict[str, Any] = field(default_factory=dict)
    timestamp:         float          = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "scenario":           self.scenario.value,
            "status":             self.status.value,
            "degradation_pct":    round(self.degradation_pct, 2),
            "max_drawdown":       round(self.max_drawdown, 4),
            "sharpe_under_stress": round(self.sharpe_under_stress, 4),
            "details":            self.details,
        }


# ── Monte Carlo ───────────────────────────────────────────────────────────────

@dataclass
class MonteCarloResult:
    technique:         str           = ""
    iterations:        int           = 1000
    mean_return:       float         = 0.0
    std_return:        float         = 0.0
    ci_low:            float         = 0.0   # 5th percentile
    ci_high:           float         = 0.0   # 95th percentile
    prob_profitable:   float         = 0.5   # fraction of runs > 0
    status:            CheckStatus   = CheckStatus.PASS
    timestamp:         float         = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "technique":       self.technique,
            "iterations":      self.iterations,
            "mean_return":     round(self.mean_return, 4),
            "std_return":      round(self.std_return, 4),
            "ci_low":          round(self.ci_low, 4),
            "ci_high":         round(self.ci_high, 4),
            "prob_profitable": round(self.prob_profitable, 4),
            "status":          self.status.value,
        }


# ── Benchmarks ────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    metric:    BenchmarkMetric = BenchmarkMetric.INFERENCE_LATENCY
    value:     float           = 0.0
    unit:      str             = ""
    threshold: Optional[float] = None
    status:    CheckStatus     = CheckStatus.PASS
    timestamp: float           = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "metric":    self.metric.value,
            "value":     round(self.value, 4),
            "unit":      self.unit,
            "threshold": self.threshold,
            "status":    self.status.value,
        }


# ── Reproducibility ───────────────────────────────────────────────────────────

@dataclass
class ReproducibilityResult:
    check_name:  str        = ""
    status:      CheckStatus = CheckStatus.PASS
    expected:    str        = ""
    actual:      str        = ""
    details:     str        = ""
    timestamp:   float      = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "check_name": self.check_name,
            "status":     self.status.value,
            "expected":   self.expected,
            "actual":     self.actual,
            "details":    self.details,
        }


# ── Security ──────────────────────────────────────────────────────────────────

@dataclass
class SecurityResult:
    check_name:  str        = ""
    status:      CheckStatus = CheckStatus.PASS
    risk_level:  RiskLevel  = RiskLevel.LOW
    finding:     str        = ""
    remediation: str        = ""
    timestamp:   float      = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "check_name":  self.check_name,
            "status":      self.status.value,
            "risk_level":  self.risk_level.value,
            "finding":     self.finding,
            "remediation": self.remediation,
        }


# ── Operational ───────────────────────────────────────────────────────────────

@dataclass
class OperationalResult:
    check_name: str        = ""
    status:     CheckStatus = CheckStatus.PASS
    details:    str        = ""
    action:     str        = ""
    timestamp:  float      = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "check_name": self.check_name,
            "status":     self.status.value,
            "details":    self.details,
            "action":     self.action,
        }


# ── Scorecard ─────────────────────────────────────────────────────────────────

@dataclass
class ScorecardResult:
    research_quality:    float = 0.0
    validation:          float = 0.0
    performance:         float = 0.0
    calibration:         float = 0.0
    robustness:          float = 0.0
    security:            float = 0.0
    operational:         float = 0.0
    documentation:       float = 0.0
    reproducibility:     float = 0.0
    overall:             float = 0.0
    grade:               str   = "F"
    timestamp:           float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "research_quality": round(self.research_quality, 2),
            "validation":       round(self.validation, 2),
            "performance":      round(self.performance, 2),
            "calibration":      round(self.calibration, 2),
            "robustness":       round(self.robustness, 2),
            "security":         round(self.security, 2),
            "operational":      round(self.operational, 2),
            "documentation":    round(self.documentation, 2),
            "reproducibility":  round(self.reproducibility, 2),
            "overall":          round(self.overall, 2),
            "grade":            self.grade,
        }

    def category_scores(self) -> Dict[str, float]:
        return {
            ScorecardCategory.RESEARCH_QUALITY.value: self.research_quality,
            ScorecardCategory.VALIDATION.value:       self.validation,
            ScorecardCategory.PERFORMANCE.value:      self.performance,
            ScorecardCategory.CALIBRATION.value:      self.calibration,
            ScorecardCategory.ROBUSTNESS.value:       self.robustness,
            ScorecardCategory.SECURITY.value:         self.security,
            ScorecardCategory.OPERATIONAL.value:      self.operational,
            ScorecardCategory.DOCUMENTATION.value:    self.documentation,
            ScorecardCategory.REPRODUCIBILITY.value:  self.reproducibility,
        }


# ── Certification Decision ────────────────────────────────────────────────────

@dataclass
class CertificationDecision:
    status:       CertificationStatus = CertificationStatus.REJECTED
    rationale:    str                 = ""
    conditions:   List[str]           = field(default_factory=list)
    restrictions: List[str]           = field(default_factory=list)
    expires_at:   Optional[float]     = None
    approved_by:  str                 = "AUTOMATED"
    timestamp:    float               = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "status":       self.status.value,
            "rationale":    self.rationale,
            "conditions":   self.conditions,
            "restrictions": self.restrictions,
            "expires_at":   self.expires_at,
            "approved_by":  self.approved_by,
        }


# ── Audit Package ─────────────────────────────────────────────────────────────

@dataclass
class AuditPackage:
    run_id:               str          = field(default_factory=lambda: str(uuid.uuid4())[:16])
    certification_report: str          = ""
    validation_report:    str          = ""
    benchmark_report:     str          = ""
    stress_test_report:   str          = ""
    security_report:      str          = ""
    operational_checklist: str         = ""
    executive_summary:    str          = ""
    generated_at:         float        = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "run_id":               self.run_id,
            "certification_report": self.certification_report,
            "validation_report":    self.validation_report,
            "benchmark_report":     self.benchmark_report,
            "stress_test_report":   self.stress_test_report,
            "security_report":      self.security_report,
            "operational_checklist": self.operational_checklist,
            "executive_summary":    self.executive_summary,
            "generated_at":         self.generated_at,
        }


# ── Full Certification Run ────────────────────────────────────────────────────

@dataclass
class CertificationRun:
    """Unified output of a complete certification pipeline run."""
    run_id:          str                               = field(default_factory=lambda: str(uuid.uuid4())[:16])
    model_id:        str                               = ""
    model_name:      str                               = ""
    model_version:   str                               = "1.0.0"
    started_at:      float                             = field(default_factory=time.time)
    completed_at:    Optional[float]                   = None
    stages:          List[StageResult]                 = field(default_factory=list)
    checklist:       Optional[ChecklistReport]         = None
    validations:     List[ValidationResult]            = field(default_factory=list)
    stress_tests:    List[StressTestResult]            = field(default_factory=list)
    monte_carlos:    List[MonteCarloResult]            = field(default_factory=list)
    benchmarks:      List[BenchmarkResult]             = field(default_factory=list)
    reproducibility: List[ReproducibilityResult]       = field(default_factory=list)
    security:        List[SecurityResult]              = field(default_factory=list)
    operational:     List[OperationalResult]           = field(default_factory=list)
    scorecard:       Optional[ScorecardResult]         = None
    decision:        Optional[CertificationDecision]   = None
    audit:           Optional[AuditPackage]            = None

    def to_dict(self) -> Dict:
        return {
            "run_id":        self.run_id,
            "model_id":      self.model_id,
            "model_name":    self.model_name,
            "model_version": self.model_version,
            "started_at":    self.started_at,
            "completed_at":  self.completed_at,
            "decision":      self.decision.to_dict() if self.decision else None,
            "scorecard":     self.scorecard.to_dict() if self.scorecard else None,
            "stages":        [s.to_dict() for s in self.stages],
        }
