"""Evaluation framework package."""

from trainer.evaluation.evaluator import StandardEvaluator
from trainer.evaluation.reporter import EvaluationReporter

__all__ = ["EvaluationReporter", "StandardEvaluator"]
