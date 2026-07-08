"""Phase S2 — QDE 10-head output schema.

Per the "One Brain" architecture, QDE's raw model output is redefined from a
single BUY/SELL/NO_TRADE classification into 10 continuous/categorical output
heads. Every head is a *signal*, not a strategy label — none of them encode a
named trading strategy, and no runtime branch in this module or its
consumers switches behavior based on a strategy name/id. See
`docs/qde/phase_s2_schema_contract.md` for the full field-by-field contract.

`QDEOutputHeads` is what a trained QDE model actually produces. `QDEDecision`
(`trainer.qde.qde_inference`) remains the execution-facing translation of
those heads into the entry/exit/scale/size vocabulary `ExecutionPlanner`
already knows how to execute — `derive_decision_fields()` is the one place
that translation happens, so ExecutionPlanner never needs to know the heads
exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

OUTPUT_SCHEMA_VERSION = "s2_v1_10head"


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class QDEOutputHeads:
    """The 10 raw output heads produced by an active QDE model.

    All signal heads are floats in ``[0.0, 1.0]`` except ``direction``
    (categorical) — this keeps every head independently interpretable and
    serializable (see the protobuf contract) without any strategy-specific
    encoding.
    """

    instrument: str
    direction: Direction

    # 1. direction            — categorical, above
    # 2. confidence           — overall conviction in `direction`
    confidence: float = 0.0
    # 3. size_signal          — how large a position to take, 0=none, 1=max
    size_signal: float = 0.0
    # 4. risk_signal          — how much risk/caution this call carries,
    #                           0=low risk, 1=high risk (advisory; combined
    #                           with, never overriding, the capital
    #                           circuit-breakers from Phase S1)
    risk_signal: float = 0.0
    # 5. entry_urgency        — 0=wait/no rush, 1=act immediately
    entry_urgency: float = 0.0
    # 6. exit_horizon         — expected holding-duration signal,
    #                           0=very short-term, 1=very long-term
    exit_horizon: float = 0.5
    # 7. scale_signal         — propensity to add to an existing position,
    #                           0=never scale in, 1=strongly scale in
    scale_signal: float = 0.0
    # 8. reduce_signal        — propensity to reduce/exit an existing
    #                           position, 0=hold fully, 1=exit fully
    reduce_signal: float = 0.0
    # 9. management_confidence — confidence in the ongoing position-management
    #                            read (used by Phase S3's live management loop)
    management_confidence: float = 0.0
    # 10. attention_focus     — how much monitoring attention this
    #                           instrument/position needs right now,
    #                           0=routine, 1=needs immediate attention
    attention_focus: float = 0.0

    model_version_id: str = ""
    schema_version: str = OUTPUT_SCHEMA_VERSION

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty list means valid)."""
        errors: list[str] = []
        for name in (
            "confidence",
            "size_signal",
            "risk_signal",
            "entry_urgency",
            "exit_horizon",
            "scale_signal",
            "reduce_signal",
            "management_confidence",
            "attention_focus",
        ):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                errors.append(f"{name}={value!r} is out of range [0.0, 1.0]")
        return errors

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "direction": self.direction.value,
            "confidence": round(self.confidence, 4),
            "size_signal": round(self.size_signal, 4),
            "risk_signal": round(self.risk_signal, 4),
            "entry_urgency": round(self.entry_urgency, 4),
            "exit_horizon": round(self.exit_horizon, 4),
            "scale_signal": round(self.scale_signal, 4),
            "reduce_signal": round(self.reduce_signal, 4),
            "management_confidence": round(self.management_confidence, 4),
            "attention_focus": round(self.attention_focus, 4),
            "model_version_id": self.model_version_id,
            "schema_version": self.schema_version,
        }
