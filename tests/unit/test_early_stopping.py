"""Unit tests for EarlyStopping callback."""

from __future__ import annotations

import pytest

from trainer.training.early_stopping import EarlyStopping


def test_triggers_after_patience_epochs() -> None:
    es = EarlyStopping(patience=3, mode="min")
    es.step(1.0)   # improved
    es.step(1.1)   # no improvement
    es.step(1.2)   # no improvement
    assert not es.should_stop
    es.step(1.3)   # 3 non-improvements → trigger
    assert es.should_stop


def test_improvement_resets_counter() -> None:
    es = EarlyStopping(patience=3, mode="min")
    es.step(1.0)
    es.step(1.1)
    es.step(0.9)   # improvement — counter resets
    es.step(1.1)
    es.step(1.2)
    assert not es.should_stop


def test_mode_max() -> None:
    es = EarlyStopping(patience=2, mode="max")
    es.step(0.5)   # improved
    es.step(0.4)   # no improvement
    es.step(0.3)   # no improvement → trigger
    assert es.should_stop


def test_best_score_tracked() -> None:
    es = EarlyStopping(patience=5, mode="min")
    es.step(1.0)
    es.step(0.8)
    es.step(0.9)
    assert es.best_score == pytest.approx(0.8)


def test_reset_clears_state() -> None:
    es = EarlyStopping(patience=1, mode="min")
    es.step(1.0)
    es.step(1.1)   # triggers
    assert es.should_stop
    es.reset()
    assert not es.should_stop
    assert es.best_score is None


def test_min_delta_respected() -> None:
    es = EarlyStopping(patience=2, min_delta=0.1, mode="min")
    es.step(1.0)
    es.step(0.95)  # improvement < min_delta → not counted
    es.step(0.90)  # improvement < min_delta → not counted
    assert es.should_stop
