"""Unit tests for Phase S2 — the 10-head QDE output schema."""

from __future__ import annotations

import unittest

from trainer.qde.qde_outputs import OUTPUT_SCHEMA_VERSION, Direction, QDEOutputHeads

_HEAD_NAMES = (
    "confidence",
    "size_signal",
    "risk_signal",
    "entry_urgency",
    "exit_horizon",
    "scale_signal",
    "reduce_signal",
    "management_confidence",
    "attention_focus",
)


class TestQDEOutputHeadsSchema(unittest.TestCase):
    def test_ten_heads_present(self):
        heads = QDEOutputHeads(instrument="EURUSD", direction=Direction.LONG)
        # direction + 9 signal heads = 10 output heads total.
        self.assertEqual(len(_HEAD_NAMES) + 1, 10)
        for name in _HEAD_NAMES:
            self.assertTrue(hasattr(heads, name))

    def test_defaults_are_valid(self):
        heads = QDEOutputHeads(instrument="EURUSD", direction=Direction.FLAT)
        self.assertEqual(heads.validate(), [])

    def test_validate_flags_out_of_range(self):
        heads = QDEOutputHeads(instrument="EURUSD", direction=Direction.LONG, confidence=1.5)
        errors = heads.validate()
        self.assertEqual(len(errors), 1)
        self.assertIn("confidence", errors[0])

    def test_direction_is_categorical_not_a_signal(self):
        heads = QDEOutputHeads(instrument="EURUSD", direction=Direction.SHORT)
        self.assertIsInstance(heads.direction, Direction)
        self.assertNotIsInstance(heads.direction, float)

    def test_to_dict_round_trips_all_heads(self):
        heads = QDEOutputHeads(
            instrument="EURUSD",
            direction=Direction.LONG,
            confidence=0.8,
            size_signal=0.5,
            risk_signal=0.2,
            entry_urgency=0.7,
            exit_horizon=0.4,
            scale_signal=0.1,
            reduce_signal=0.0,
            management_confidence=0.6,
            attention_focus=0.3,
        )
        d = heads.to_dict()
        for name in _HEAD_NAMES:
            self.assertIn(name, d)
        self.assertEqual(d["direction"], "LONG")
        self.assertEqual(d["schema_version"], OUTPUT_SCHEMA_VERSION)

    def test_schema_version_no_strategy_name(self):
        # Guardrail: the schema version identifier must not embed any named
        # trading strategy (e.g. "trend_following", "mean_reversion").
        forbidden_substrings = ("trend", "reversion", "breakout", "scalp", "swing")
        for token in forbidden_substrings:
            self.assertNotIn(token, OUTPUT_SCHEMA_VERSION.lower())


if __name__ == "__main__":
    unittest.main()
