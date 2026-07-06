"""Probability calibration for QuantModelV1 — Phase 14.6.

Supports:
  - Temperature Scaling  (post-hoc, single parameter)
  - Platt Scaling        (post-hoc, logistic regression on logits)
  - Isotonic Regression  (post-hoc, non-parametric)
"""

from trainer.calibration.isotonic import IsotonicCalibration
from trainer.calibration.metrics import (
    compute_confidence_accuracy,
    compute_ece,
    compute_reliability_data,
)
from trainer.calibration.platt import PlattScaling
from trainer.calibration.temperature import TemperatureScaling

__all__ = [
    "TemperatureScaling",
    "PlattScaling",
    "IsotonicCalibration",
    "compute_ece",
    "compute_reliability_data",
    "compute_confidence_accuracy",
]
