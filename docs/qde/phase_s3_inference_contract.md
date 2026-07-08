# Phase S3 Inference Contract — `QDEPositionManager.review()`

**Date:** 2026-07-08
**Module:** `trainer/qde/qde_position_manager.py`
**Related:** [`phase_s2_inference_contract.md`](phase_s2_inference_contract.md), [`phase_s3_telemetry_contract.md`](phase_s3_telemetry_contract.md), [`phase_s3_lifecycle_contract.md`](phase_s3_lifecycle_contract.md).

## Interface

```python
class QDEPositionManager:
    def due_for_review(self, telemetry: PositionTelemetry) -> bool: ...
    def review(self, telemetry: PositionTelemetry) -> PositionManagementDecision: ...
```

`review()` is the S3 equivalent of `QDEDecisionModel.decide()`: one call in, one decision out, pure and stateless (no I/O, no side effects). It is intended to be invoked once per open position per review cycle (default cadence: 30s, `REVIEW_INTERVAL_SECONDS`).

## Output: `PositionManagementDecision`

```python
@dataclass
class PositionManagementDecision:
    instrument: str
    action: PositionManagementAction   # HOLD | REDUCE | ADD | CLOSE |
                                        # MOVE_TO_BREAKEVEN | TRAIL | LOCK_PROFIT
    confidence: float
    reasoning: str
    management_confidence: float
    risk_signal: float
    model_version_id: str = ""
    schema_version: str = OUTPUT_SCHEMA_VERSION
    decided_at: float = <now>
```

This matches the Phase S3 spec exactly: one of the 7 named actions, plus `confidence`, `reasoning`, `management_confidence`, and `risk_signal`. `confidence` and `management_confidence` are passed straight through from the cycle's `QDEOutputHeads` (no re-derivation) — the position manager's job is to pick an *action* given the heads, not to reinterpret the heads' own confidence semantics. `risk_signal` is likewise passed through unchanged, preserving Phase S2's definition (advisory; combined with, never overriding, the Phase S1 circuit-breakers).

## Decision logic (threshold-based, mirrors Phase S2's `derive_decision_fields()`)

Evaluated in order, first match wins — deliberately threshold-based rather than learned, for the same reason `derive_decision_fields()` is (auditability, and a stable target for a future learned-mapping replacement):

1. **Hard safety override** — `market.news_risk > 0.85` → `CLOSE`, `confidence=1.0`, regardless of any head value. This is a Phase-S1-style safety condition, not a signal-driven call, so it is checked before any head is read. (Note this is independent of, and does not replace, the Phase S1 `DrawdownManager`/`BrokerValidation` safety systems — those still operate on the resulting order at the execution layer.)
2. `reduce_signal >= 0.65` → `CLOSE` (full exit warranted).
3. `reduce_signal >= 0.35` → `REDUCE` (partial reduction warranted).
4. `scale_signal >= 0.65` **and** `free_margin_pct > 0.3` **and** `unrealised_pct >= 0` → `ADD`. The margin and non-negative-P&L guards prevent adding to a position that is both already losing and already margin-constrained — a scale_signal alone is never sufficient.
5. Profit-protection branch (only reachable once none of the reduce/scale conditions above have fired):
   - `unrealised_pct >= 0.02` and `confidence < 0.55` and `reduce_signal < 0.2` → `LOCK_PROFIT` (meaningful profit, but conviction has faded — protect the gain).
   - `unrealised_pct > 0` and `exit_horizon < 0.4` → `MOVE_TO_BREAKEVEN` (in profit, but the model expects this move to end soon — protect the entry).
   - `unrealised_pct > 0` and `management_confidence >= 0.5` → `TRAIL` (in profit, model still has ongoing conviction in managing the trade).
6. Otherwise → `HOLD`.

## Why thresholds live here and not in `derive_decision_fields()`

`derive_decision_fields()` (Phase S2) answers "given a fresh signal read with no open position yet, what entry/exit/scale/management action should `ExecutionPlanner` take?" `QDEPositionManager.review()` answers a related but distinct question: "given a signal read *and* the live state of an already-open position (P&L, time in trade, margin headroom), what single management action should happen next?" Position P&L and margin headroom are S3-only inputs that don't exist at entry time, so this needed its own decision function rather than reusing `derive_decision_fields()` — but it deliberately mirrors that function's shape (heads-in, enum-out, pure, first-match-wins) so the two stay easy to reason about side by side.

## Backward / forward compatibility

- `QDEOutputHeads` is unchanged — S3 consumes the same 10-head schema (`OUTPUT_SCHEMA_VERSION = "s2_v1_10head"`) introduced in Phase S2. No schema bump was required for S3, as anticipated in `phase_s2_deployment_contract.md`'s "Forward compatibility with Phase S3" section.
- `QDEPositionManager` and `PositionMonitor` (`trainer/lifecycle/position_monitor.py`) currently coexist: `PositionMonitor` is a `LifecycleState`-driven advisory evaluator (pre-existing, Phase 21.5), while `QDEPositionManager` is the new QDE-signal-driven reasoner introduced here. Wiring the live execution loop to prefer `QDEPositionManager`'s decision (with `PositionMonitor` demoted to an advisory/cross-check role, matching the Phase S1 "advisory systems never decide" pattern) is a live-integration task for the service that owns the execution loop, not a change to either reasoning module.

## Test evidence

`tests/unit/test_qde_position_manager.py` (13 tests) — covers cadence gating, the hard safety override, each of the 7 actions individually reachable, margin-gating on `ADD`, and full-field serialization via `to_dict()`.
