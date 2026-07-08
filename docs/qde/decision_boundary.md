# QDE Decision Boundary — Formal Proof

## Claim

> **The QDE is a single intelligence. Capabilities do not independently produce decisions. Only QDE decisions leave the trainer.**

This document proves the claim through three independent lines of evidence:
type system, call graph, and module import graph.

---

## Proof 1 — Type System

`QDEDecision` is the only type whose fields encode actionable trade instructions. It contains:
- `entry: EntrySignal | None` — instructs an entry with price, size, stop, TP
- `exits: list[ExitSignal]` — instructs exits with urgency
- `scaling: list[ScalingSignal]` — instructs position increases
- `partial_closes: list[PartialCloseSignal]` — instructs partial profit taking
- `break_evens: list[BreakEvenSignal]` — instructs stop adjustment

All other types in the system (`MarketContext`, `OpportunityReport`, `EVScore`, `RiskAssessment`, `ExposureReport`, `TradeManagementSignals`, `PositionSizeRecommendation`, `PreservationStatus`, `MemoryContext`, `ReflectionInsight`, `LearningUpdate`) contain only:
- Scored numeric values
- Flagged trade IDs (not instructions — the QDE decides what to do with them)
- Descriptive enumerations
- Confidence scores and warnings

**None of these types contain `EntrySignal`, `ExitSignal`, `ScalingSignal`, `PartialCloseSignal`, or `BreakEvenSignal`.** They cannot instruct a trade. They can only inform.

**Conclusion:** The type system structurally prevents capabilities from encoding trade instructions.

---

## Proof 2 — Construction Site

`QDEDecision` is instantiated in exactly two places, both inside `qde_intelligence.py`:

| Method | Condition | Decision Type |
|---|---|---|
| `QDEDecision.preserve()` | Capital Preservation circuit triggered | `PRESERVE` |
| `QDEDecision(...)` constructor | Stage 6 synthesis after full reasoning cycle | All other types |

No other file in the `trainer/` directory constructs a `QDEDecision`. This is verifiable by grep:

```bash
grep -r "QDEDecision(" trainer/ --include="*.py"
# Expected: only matches in trainer/qde/qde_intelligence.py
#           and trainer/qde/capabilities/cap_models.py (class definition only)
```

**Conclusion:** `QDEDecision` is only ever built in one place. Capabilities cannot build it — they do not import it.

---

## Proof 3 — Import Graph

The import rules are:

| Module | May import from cap_models.py | May import QDEDecision | May import QDEIntelligence |
|---|---|---|---|
| `market_understanding.py` | YES (advisory types only) | NO | NO |
| `opportunity_discovery.py` | YES (advisory types only) | NO | NO |
| `expected_value_reasoning.py` | YES (advisory types only) | NO | NO |
| `risk_reasoning.py` | YES (advisory types only) | NO | NO |
| `exposure_reasoning.py` | YES (advisory types only) | NO | NO |
| `trade_management.py` | YES (advisory types only) | NO | NO |
| `position_management.py` | YES (advisory types only) | NO | NO |
| `capital_preservation.py` | YES (advisory types only) | NO | NO |
| `market_memory.py` | YES (advisory types only) | NO | NO |
| `reflection.py` | YES (advisory types only) | NO | NO |
| `learning.py` | YES (advisory types only) | NO | NO |
| `qde_intelligence.py` | YES (all types) | YES | — (is QDEIntelligence) |

This is verifiable by grep:

```bash
grep -r "from trainer.qde.capabilities.cap_models import" trainer/qde/capabilities/ | grep QDEDecision
# Expected: zero results
```

**Conclusion:** The import graph prevents any capability from accessing the `QDEDecision` constructor.

---

## Proof 4 — Call Graph Acyclicity

The reasoning call graph is a directed acyclic graph (DAG):

```
Snapshot
  └─► CapitalPreservationCapability  (Stage 1)
  └─► MarketUnderstandingCapability  (Stage 2)
  └─► MarketMemoryCapability         (Stage 2, 4)
  └─► TradeManagementCapability      (Stage 3)
  └─► OpportunityDiscoveryCapability (Stage 4)
  └─► RiskReasoningCapability        (Stage 4)
  └─► ExpectedValueReasoningCapability (Stage 4)
  └─► ExposureReasoningCapability    (Stage 5)
  └─► PositionManagementCapability   (Stage 4–6)
  └─► QDEIntelligence.reason()  →  QDEDecision
```

There are no cycles. No capability calls another capability. No capability calls `QDEIntelligence`. The graph is strictly:

```
Snapshot → Capabilities (parallel, staged) → QDEIntelligence → QDEDecision
```

**Conclusion:** The call graph makes it structurally impossible for capabilities to produce decisions or to trigger additional reasoning cycles.

---

## Proof 5 — The Only Exit Point

All consumers of the QDE receive `QDEDecision` objects. The only place these are produced is `QDEIntelligence.reason()`. Therefore:

```
Anything that leaves the trainer is a QDEDecision.
QDEDecision is only produced by QDEIntelligence.reason().
Therefore: only QDE decisions leave the trainer.
```

Capability outputs (`MarketContext`, `EVScore`, etc.) are internal. They are created inside `reason()`, consumed inside `reason()`, and never returned to the caller. The caller receives only the final `QDEDecision`.

---

## Summary Table

| Claim | Mechanism | Evidence |
|---|---|---|
| QDE is a single intelligence | One class (`QDEIntelligence`) owns all 13 responsibilities | Architecture, call graph |
| Capabilities do not produce decisions | `QDEDecision` not importable from capability modules | Import graph, type system |
| Only QDE decisions leave the trainer | `reason()` is the only public method returning `QDEDecision` | Call graph, construction sites |

---

## Verification Commands

```bash
# Prove no capability module imports QDEDecision
# (grep for real import lines only — excludes comments and docstrings)
grep -rn "^from.*import.*QDEDecision\|^import.*QDEDecision" \
  trainer/qde/capabilities/ --include="*.py" | grep -v "cap_models.py\|__init__.py"
# Expected: zero results

# Prove QDEDecision is only constructed in qde_intelligence.py
grep -rn "QDEDecision(" trainer/ --include="*.py" | grep -v "qde_intelligence\|cap_models"
# Expected: zero results

# Prove capabilities don't import qde_intelligence
grep -rn "^from.*qde_intelligence\|^import.*qde_intelligence" \
  trainer/qde/capabilities/ --include="*.py"
# Expected: zero results

# Prove only reason() is the external-facing decision method
grep -rn "def reason" trainer/qde/ --include="*.py"
# Expected: one result — qde_intelligence.py
```

All four commands must return zero (or one, for `def reason`) results to confirm the architecture is intact.
