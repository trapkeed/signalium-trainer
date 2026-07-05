"""Unit tests for MetricsTracker."""

from __future__ import annotations

from trainer.metrics.tracker import MetricsTracker


def test_update_and_history() -> None:
    tracker = MetricsTracker()
    tracker.update({"loss": 1.0, "val_loss": 0.9})
    tracker.update({"loss": 0.8, "val_loss": 0.7})
    history = tracker.history()
    assert len(history) == 2
    assert history[0]["loss"] == 1.0
    assert history[1]["val_loss"] == 0.7


def test_best_min() -> None:
    tracker = MetricsTracker()
    tracker.update({"val_loss": 0.5})
    tracker.update({"val_loss": 0.3})
    tracker.update({"val_loss": 0.4})
    assert tracker.best("val_loss", mode="min") == 0.3


def test_best_max() -> None:
    tracker = MetricsTracker()
    tracker.update({"accuracy": 0.7})
    tracker.update({"accuracy": 0.9})
    assert tracker.best("accuracy", mode="max") == 0.9


def test_last() -> None:
    tracker = MetricsTracker()
    tracker.update({"loss": 1.0})
    tracker.update({"loss": 0.5})
    assert tracker.last("loss") == 0.5


def test_best_returns_none_for_unknown_metric() -> None:
    tracker = MetricsTracker()
    tracker.update({"loss": 1.0})
    assert tracker.best("nonexistent") is None


def test_to_dict() -> None:
    tracker = MetricsTracker()
    tracker.update({"loss": 1.0, "acc": 0.5})
    tracker.update({"loss": 0.8, "acc": 0.7})
    d = tracker.to_dict()
    assert d["loss"] == [1.0, 0.8]
    assert d["acc"] == [0.5, 0.7]
