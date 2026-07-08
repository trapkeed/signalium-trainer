"""Phase S4 — Training Signal Generators.

`execution/`, `icari/`, `portfolio/`, `lifecycle/`, and `mie/` are runtime
info-providers/context-only modules (see `decision-engines-context-providers`
in project memory) — QDE is the sole runtime decision maker, per Phase S1.

This package is the Phase S4 companion: it turns *historical, already-realized*
price/outcome data into hindsight-optimal labels for training QDE's output
heads (`trainer.qde.qde_outputs.QDEOutputHeads`). Every labeler here answers
"what would the best possible call have been, now that we can see what
actually happened?" — never "what heuristic rule should fire right now?".

None of these labelers are invoked at runtime/inference time. They are a
training-data pipeline concern only. See `docs/qde/phase_s4_migration_report.md`
for the full reframing of execution/icari/portfolio/lifecycle/mie as label,
feature, evaluation-dimension, and telemetry sources rather than decision
makers.
"""

from trainer.training_signals.lifecycle_transition_labels import (
    HindsightLifecycleLabel,
    HindsightLifecycleLabeler,
    LifecycleTransitionPoint,
)
from trainer.training_signals.outcome_path import TradeOutcomePath
from trainer.training_signals.portfolio_allocation_labels import (
    HindsightAllocationLabel,
    HindsightAllocationLabeler,
)
from trainer.training_signals.position_size_labels import (
    HindsightPositionSizeLabel,
    HindsightPositionSizeLabeler,
)
from trainer.training_signals.stop_loss_labels import (
    HindsightStopLossLabel,
    HindsightStopLossLabeler,
)
from trainer.training_signals.take_profit_labels import (
    HindsightTakeProfitLabel,
    HindsightTakeProfitLabeler,
)

__all__ = [
    "TradeOutcomePath",
    "HindsightStopLossLabel",
    "HindsightStopLossLabeler",
    "HindsightTakeProfitLabel",
    "HindsightTakeProfitLabeler",
    "HindsightPositionSizeLabel",
    "HindsightPositionSizeLabeler",
    "HindsightAllocationLabel",
    "HindsightAllocationLabeler",
    "HindsightLifecycleLabel",
    "HindsightLifecycleLabeler",
    "LifecycleTransitionPoint",
]
