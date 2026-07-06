"""Phase 25 — Certification types and enumerations."""

from __future__ import annotations

from enum import Enum
from typing import Dict


class CertificationStatus(str, Enum):
    CERTIFIED = "CERTIFIED"
    CERTIFIED_WITH_CONDITIONS = "CERTIFIED_WITH_CONDITIONS"
    PILOT_ONLY = "PILOT_ONLY"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    REJECTED = "REJECTED"


class CertificationStage(str, Enum):
    RESEARCH = "RESEARCH"
    TRAINING = "TRAINING"
    VALIDATION = "VALIDATION"
    WALK_FORWARD = "WALK_FORWARD"
    CALIBRATION = "CALIBRATION"
    STRESS_TESTING = "STRESS_TESTING"
    MONTE_CARLO = "MONTE_CARLO"
    REPRODUCIBILITY = "REPRODUCIBILITY"
    BENCHMARK = "BENCHMARK"
    SECURITY_REVIEW = "SECURITY_REVIEW"
    OPERATIONAL_REVIEW = "OPERATIONAL_REVIEW"
    PERFORMANCE_REVIEW = "PERFORMANCE_REVIEW"
    CERTIFICATION_DECISION = "CERTIFICATION_DECISION"


class CheckStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class StressScenario(str, Enum):
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    FLASH_CRASH = "FLASH_CRASH"
    SPREAD_EXPANSION = "SPREAD_EXPANSION"
    LIQUIDITY_REDUCTION = "LIQUIDITY_REDUCTION"
    DATA_GAPS = "DATA_GAPS"
    DELAYED_DATA = "DELAYED_DATA"
    MODEL_LATENCY = "MODEL_LATENCY"
    SERVER_RESTART = "SERVER_RESTART"
    MISSING_FEATURES = "MISSING_FEATURES"


class BenchmarkMetric(str, Enum):
    INFERENCE_LATENCY = "INFERENCE_LATENCY"
    TRAINING_TIME = "TRAINING_TIME"
    GPU_USAGE = "GPU_USAGE"
    CPU_USAGE = "CPU_USAGE"
    MEMORY_USAGE = "MEMORY_USAGE"
    DISK_USAGE = "DISK_USAGE"
    MODEL_SIZE = "MODEL_SIZE"
    API_THROUGHPUT = "API_THROUGHPUT"
    RESPONSE_TIME = "RESPONSE_TIME"


class ValidationCheck(str, Enum):
    WALK_FORWARD = "WALK_FORWARD"
    OUT_OF_SAMPLE = "OUT_OF_SAMPLE"
    CROSS_REGIME = "CROSS_REGIME"
    CROSS_INSTRUMENT = "CROSS_INSTRUMENT"
    CROSS_TIMEFRAME = "CROSS_TIMEFRAME"
    CALIBRATION = "CALIBRATION"
    GENERALIZATION = "GENERALIZATION"
    STABILITY = "STABILITY"


class RiskLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class ScorecardCategory(str, Enum):
    RESEARCH_QUALITY = "RESEARCH_QUALITY"
    VALIDATION = "VALIDATION"
    PERFORMANCE = "PERFORMANCE"
    CALIBRATION = "CALIBRATION"
    ROBUSTNESS = "ROBUSTNESS"
    SECURITY = "SECURITY"
    OPERATIONAL = "OPERATIONAL"
    DOCUMENTATION = "DOCUMENTATION"
    REPRODUCIBILITY = "REPRODUCIBILITY"


class DashboardType(str, Enum):
    CERTIFICATION = "CERTIFICATION"
    READINESS = "READINESS"
    VALIDATION = "VALIDATION"
    BENCHMARK = "BENCHMARK"
    AUDIT = "AUDIT"


# ── Thresholds for automated certification decisions ──────────────────────────

CERT_THRESHOLDS: Dict[str, float] = {
    "min_overall_score": 75.0,  # minimum scorecard overall to certify
    "min_research_quality": 60.0,
    "min_validation": 65.0,
    "min_performance": 60.0,
    "min_calibration": 60.0,
    "min_robustness": 55.0,
    "min_security": 70.0,
    "min_operational": 65.0,
    "min_documentation": 50.0,
    "min_reproducibility": 60.0,
    "pilot_overall": 55.0,  # minimum for PILOT_ONLY
    "research_overall": 35.0,  # minimum for RESEARCH_ONLY
    "conditions_gap": 10.0,  # gap below cert threshold for WITH_CONDITIONS
}

STAGE_WEIGHTS: Dict[str, float] = {
    CertificationStage.RESEARCH: 0.08,
    CertificationStage.TRAINING: 0.06,
    CertificationStage.VALIDATION: 0.15,
    CertificationStage.WALK_FORWARD: 0.10,
    CertificationStage.CALIBRATION: 0.08,
    CertificationStage.STRESS_TESTING: 0.10,
    CertificationStage.MONTE_CARLO: 0.08,
    CertificationStage.REPRODUCIBILITY: 0.07,
    CertificationStage.BENCHMARK: 0.08,
    CertificationStage.SECURITY_REVIEW: 0.08,
    CertificationStage.OPERATIONAL_REVIEW: 0.07,
    CertificationStage.PERFORMANCE_REVIEW: 0.05,
    CertificationStage.CERTIFICATION_DECISION: 0.00,
}

SCORECARD_WEIGHTS: Dict[str, float] = {
    ScorecardCategory.RESEARCH_QUALITY: 0.08,
    ScorecardCategory.VALIDATION: 0.20,
    ScorecardCategory.PERFORMANCE: 0.12,
    ScorecardCategory.CALIBRATION: 0.12,
    ScorecardCategory.ROBUSTNESS: 0.15,
    ScorecardCategory.SECURITY: 0.12,
    ScorecardCategory.OPERATIONAL: 0.10,
    ScorecardCategory.DOCUMENTATION: 0.06,
    ScorecardCategory.REPRODUCIBILITY: 0.05,
}


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def cert_status_label(status: CertificationStatus) -> str:
    return {
        CertificationStatus.CERTIFIED: "✓ Certified for Production",
        CertificationStatus.CERTIFIED_WITH_CONDITIONS: "⚠ Certified with Conditions",
        CertificationStatus.PILOT_ONLY: "⬡ Pilot Deployment Only",
        CertificationStatus.RESEARCH_ONLY: "◌ Research Only",
        CertificationStatus.REJECTED: "✗ Rejected",
    }.get(status, "Unknown")
