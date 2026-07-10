---
name: Hypothesis maturity vs narrative status
description: Why QDE Brain has two hypothesis status vocabularies (narrative HypothesisStatus vs execution-facing HypothesisMaturity) and how they relate without duplicating storage.
---

## The two vocabularies

`trainer/qde/brain/brain_types.py::HypothesisStatus` (FORMING/ACTIVE/
REINFORCED/WEAKENING/INVALIDATED/CONFIRMED/EXPIRED) describes the
*narrative* lifecycle of a belief — how QDE's internal story about a
symbol is evolving tick to tick. It is mutated directly by
`hypotheses.HypothesisTracker`.

`trainer/qde/brain/hypothesis_lifecycle.py::HypothesisMaturity`
(OBSERVING/BUILDING/HIGH_CONVICTION/EXECUTABLE/INVALIDATED/EXPIRED)
answers a different, execution-facing question: is this thesis mature
enough that a trade should emerge from it, rather than from an isolated
per-tick prediction? It is a **pure projection function**
(`compute_maturity`) over the existing `Hypothesis` fields (confidence,
reinforcement_count, contradicting/supporting evidence counts, age) —
it introduces no second dataclass and no second storage location.
`Hypothesis` objects still live only in `BrainState.hypotheses`.

**Why:** collapsing both into one enum would force every narrative
status change to also imply something about execution readiness, which
isn't true (e.g. a REINFORCED hypothesis with thin evidence shouldn't
auto-qualify as tradeable). Keeping maturity as a stateless projection
means it's always consistent with the underlying hypothesis and never
needs its own persistence/migration path.

**How to apply:** any new "is this ready to act on" concept layered on
top of an existing tracked entity should default to a pure projection
function first, before reaching for a new dataclass/table. Only add new
storage when the concept genuinely can't be derived from what's already
persisted (e.g. it needs its own independent audit trail).

## Execution integration point

`hypothesis_registry.HypothesisRegistry.executable()` /
`hypothesis_engine.HypothesisEngine.executable_hypotheses()` /
`.mature_direction()` are the intended read surface for a decision layer
that wants "trade execution emerges from mature hypotheses" — they do
not place orders themselves. The actual order-placing threshold logic
stays solely in `qde_inference.derive_decision_fields`; this is
additional context a caller may consult, following the same
never-duplicate-the-execution-boundary convention as the rest of the
brain package (see `qde-capability-boundary.md`).
