"""S33 — Opportunity Quality Scoring.

Assigns every observed setup an opportunity score from 0-100 answering a
single question: *is this worth capital allocation?* — never *which
direction?* Direction is decided elsewhere (QDE's own heads / decision
translation); this module deliberately has no direction input or output.

Additive subsystem: it does not replace any existing QDE observer,
Setup Memory Engine, or Decision Memory logic. It *consumes* their outputs
(readiness/quality sub-scores, setup-memory confidence history, decision
benchmarking win-rates) as inputs and persists its own score history,
integrating with the memory/research systems via optional hooks rather
than hard dependencies.
"""

from trainer.qde.opportunity_scoring.engine import OpportunityScoringEngine
from trainer.qde.opportunity_scoring.models import (
    OpportunityBand,
    OpportunityScoreInputs,
    OpportunityScoreResult,
    OpportunityScoreWeights,
)

__all__ = [
    "OpportunityScoringEngine",
    "OpportunityBand",
    "OpportunityScoreInputs",
    "OpportunityScoreResult",
    "OpportunityScoreWeights",
]
