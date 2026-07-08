# Phase S2 Inference Contract — `QDEDecisionModel.decide()`

**Date:** 2026-07-08
**Module:** `trainer/qde/qde_inference.py`
**Related:** [`phase_s2_schema_contract.md`](phase_s2_schema_contract.md), [`unified_qde_architecture.md`](unified_qde_architecture.md).

## Interface (unchanged signature, changed internal contract)

```python
class QDEDecisionModel(ABC):
    version_id: str = ""

    @abstractmethod
    def decide(self, inputs: QDEInferenceInput) -> QDEDecision: ...
```

`decide()` still takes a `QDEInferenceInput` and still returns a `QDEDecision` — no caller (`ExecutionPlanner`, tests, API routes) needs to change. What changed is what happens *inside* a compliant implementation:

1. Compute a `QDEOutputHeads` instance — the model's raw, strategy-agnostic 10-signal read of the current instrument/position/account/market state.
2. Translate those heads into the execution-facing `QDEDecision` fields via `derive_decision_fields(heads, has_position, free_margin_pct)`.
3. Attach the heads instance to `QDEDecision.heads` so downstream consumers (shadow logging, memory, future S3 loop) can access the full signal vector, not just the derived action.

`HeuristicQDEDecisionModel` (the interim baseline, `trainer/qde/qde_inference.py`) has been updated to follow this two-step pattern (`_compute_heads()` then `_decision_from_heads()`) as the reference implementation. Any future trained-model implementation of `QDEDecisionModel` should follow the same split: **produce heads, then derive**, never skip straight to execution-facing fields, so the heads remain a genuine, independently-loggable model output rather than an afterthought wrapper.

## `derive_decision_fields()` contract

```python
def derive_decision_fields(
    heads: QDEOutputHeads,
    has_position: bool,
    free_margin_pct: float,
) -> tuple[EntryAction, ExitAction, ScaleAction, list[ManagementAction]]:
```

This is the **only** place a heads vector is translated into `EntryAction`/`ExitAction`/`ScaleAction`/`ManagementAction`. It is pure and stateless (no side effects, no I/O), deliberately threshold-based rather than learned, and reads only from the 10 heads plus the two hard-constraint inputs (`has_position`, `free_margin_pct`) — no strategy name or id is consulted. This isolation means the mapping itself can later be replaced (e.g. with a learned or configurable mapping) without touching `QDEOutputHeads`, `ExecutionPlanner`, or any other consumer.

## Backward compatibility guarantee

- `QDEDecision`'s pre-existing fields (`entry_action`, `exit_action`, `scale_action`, `position_size_lots`, `risk_allocation_pct`, `management_actions`, `confidence`, `model_version_id`, `rationale`, `decided_at`) are unchanged in name, type, and semantics.
- `heads` is additive and optional (`QDEOutputHeads | None = None`), so any test or caller that constructs a bare `QDEDecision(...)` without heads (as prior tests do) continues to work unmodified.
- `HeuristicQDEDecisionModel`'s entry/exit/scale thresholds were preserved exactly (0.55 enter, 0.45 wait, 0.65/0.4 reduce, 0.75 scale-in-equivalent), verified by the full pre-existing `test_qde_inference.py` suite (60 tests) passing unchanged.

## Test evidence

`tests/unit/test_qde_inference.py` (60, regression-only — no test changes needed), `tests/unit/test_qde_outputs.py` (7, new), `tests/unit/test_qde_shadow.py` (6, regression). Combined 73 passed via `python -m pytest tests/unit/test_qde_outputs.py tests/unit/test_qde_inference.py tests/unit/test_qde_shadow.py tests/unit/test_qde_rollback.py --noconftest -o addopts="" -q`.
