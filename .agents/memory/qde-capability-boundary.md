---
name: QDE capability boundary
description: Architectural rules for the Phase 28.3 QDE capability system — what capabilities may and may not do, and how to verify the boundary holds.
---

## Rule
Only `qde_intelligence.py` may construct `QDEDecision`. Capability modules are advisory — they return typed output objects, never decisions.

## Why
The single-intelligence principle: the QDE is one reasoning system with 11 internal faculties, not 11 independent engines. This prevents conflicting decisions and keeps accountability in one place.

## How to apply
- Never import `QDEDecision` from inside `trainer/qde/capabilities/*.py` (except `cap_models.py` where it is defined, and `__init__.py` which re-exports it for the package API).
- Capability outputs extend `CapabilityOutput` and carry `confidence` + `warnings`.
- Capabilities call no other capabilities directly — all wiring is in `QDEIntelligence.reason()`.

## Verification (use these exact grep patterns — comments will false-positive on bare grep)
```bash
# No capability imports QDEDecision (import lines only)
grep -rn "^from.*import.*QDEDecision\|^import.*QDEDecision" \
  trainer/qde/capabilities/ --include="*.py" | grep -v "cap_models.py\|__init__.py"
# Expected: zero results

# QDEDecision only constructed in qde_intelligence.py
grep -rn "QDEDecision(" trainer/ --include="*.py" | grep -v "qde_intelligence\|cap_models"
# Expected: zero results
```

## File layout (Phase 28.3)
- `trainer/qde/capabilities/cap_models.py` — all shared types including QDEDecision (defined here)
- `trainer/qde/capabilities/__init__.py` — re-exports for package API
- `trainer/qde/capabilities/{11 modules}.py` — one per capability, advisory only
- `trainer/qde/qde_intelligence.py` — single brain, only place that builds QDEDecision
- `docs/qde/intelligence_architecture.md` — data flow, 13 responsibilities, 6-stage cycle
- `docs/qde/capability_contract.md` — what every capability must/must not do
- `docs/qde/decision_boundary.md` — formal proof with verified grep commands
