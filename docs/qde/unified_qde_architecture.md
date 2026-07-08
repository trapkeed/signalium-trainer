# Unified Quant Decision Engine (QDE) — Formal Architecture

**Date:** 2026-07-08
**Status:** Target architecture definition
**Related documents:**
- `docs/qde/hardcoded_behavior_audit.md` — inventory of hardcoded behavior to eliminate
- `docs/qde/learning_dimensions_architecture.md` — conversion of legacy engines into evaluation dimensions

---

## 1. First Principle

> **The QDE is one model, one intelligence, one trader, one decision maker.**

There is no ensemble of independent rule engines each owning a slice of the decision (regime detection, risk sizing, exposure limits, trade management, exits). All of that collapses into a single learned system. The QDE receives market state and produces the complete trading decision end-to-end. Nothing downstream of the QDE re-decides, overrides, or re-filters its output through hardcoded logic.

This is an architectural boundary, not a suggestion: any code that computes a threshold-based decision and substitutes it for the QDE's output is, by definition, outside this architecture.

---

## 2. What the QDE Learns

The QDE is responsible for learning all of the following as a single coherent policy, not as separate modules bolted together:

| Domain | Learned Behavior |
|---|---|
| Market context | Reading current market state/regime from raw data, without a hand-coded classifier |
| Risk management | How much risk to take, and how to size it, per situation |
| Capital preservation | When and how aggressively to protect capital |
| Trade management | How to manage a position across its lifecycle |
| Position management | Ongoing adjustments to open positions |
| Profit maximization | Capturing upside without a fixed reward target |
| Drawdown minimization | Limiting losses without a fixed drawdown rule |
| Reversal detection | Recognizing when a move is turning |
| Continuation detection | Recognizing when a move is likely to persist |
| Exposure management | How much of the book to commit to a given position/asset class |
| Portfolio management | Managing exposure across the whole portfolio, not just one trade |
| Holding duration decisions | How long to stay in a trade |
| Scaling decisions | Whether/when to add to a position |
| Partial close decisions | Whether/when to reduce a position |
| Exit decisions | When to fully close a position |

These are not fifteen separate models or fifteen separate rule engines — they are **facets of one policy's output space**. The QDE takes market state in and produces a unified decision that simultaneously reflects all of these considerations, the same way a human trader doesn't consciously run fifteen separate mental subroutines — they just decide.

---

## 3. System Diagram

```
                         ┌─────────────────────────────┐
                         │         Market Data          │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │   Unified QDE (one model)     │
                         │                               │
                         │  Learns jointly:               │
                         │   - market context             │
                         │   - risk & capital preservation │
                         │   - trade/position management   │
                         │   - profit/drawdown balance      │
                         │   - reversal/continuation read   │
                         │   - exposure/portfolio management│
                         │   - holding/scaling/partial/exit  │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │      Unified Decision         │
                         │  (single output: what to do,   │
                         │   how much, how long, how to    │
                         │   manage it)                    │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                                   Execution


              ═══════════════════ Trainer (outside the decision path) ═══════════════════

  ┌──────────┐   ┌───────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
  │   Data    │   │ Feedback  │   │ Evaluation  │   │ Validation  │   │ Deployment  │
  │ pipelines │──▶│  signals  │──▶│  (6 dims,   │──▶│ (promotion  │──▶│  (rollout,  │
  │  & labels │   │(objectives)│   │ see below)  │   │  gates)     │   │  rollback)  │
  └──────────┘   └───────────┘   └────────────┘   └────────────┘   └────────────┘
                                                                            │
                                                                            ▼
                                                              feeds back into next
                                                              training iteration
                                                              (never into the live
                                                               decision path)
```

The trainer sits **beside** the decision path, never **inside** it. It shapes what the QDE becomes across training iterations; it never intercepts or overrides a live decision.

---

## 4. Division of Responsibility

### 4.1 What the QDE owns
- The entire decision: enter/hold/scale/reduce/exit, sizing, timing, and risk posture — as one inseparable output.
- Its own read of market context (no upstream rule-based regime classifier feeding it a label it must obey).
- Its own risk and exposure posture (no upstream fixed caps that clip its output post-hoc as normal operation — see §6 for the one exception).

### 4.2 What the trainer provides

| Responsibility | Description |
|---|---|
| **Data** | Historical and live market data, cleaned, labeled, and versioned into training/benchmark datasets |
| **Feedback** | Reward/loss signals derived from realized outcomes (P&L, drawdown, calibration) that shape the QDE's learning — not instructions on what to do next |
| **Evaluation** | Scoring the QDE's decisions across the six evaluation dimensions defined in `learning_dimensions_architecture.md` (Risk Reasoning, Exposure Management, Trade Management, Expected Value, Decision Confidence, Market Understanding) |
| **Validation** | Backtesting, shadow testing, and statistical comparison against benchmark datasets and prior QDE versions before promotion |
| **Deployment** | Versioning, promotion, rollout, and rollback of QDE model artifacts (`trainer/qde/qde_registry.py`, `qde_promotion.py`, `qde_rollback.py`) |

### 4.3 What the trainer must never provide
- **Strategies** — no hand-designed trading strategy the QDE is wrapped around or falls back to.
- **Rules** — no if/else, threshold, or lookup-table logic that determines trading behavior in production.
- **Signals** — no hardcoded indicator-derived buy/sell/hold signal injected into or substituted for the QDE's decision.
- **Decision overrides** — no post-hoc correction, clamp, or veto of the QDE's output based on fixed logic. If the QDE's behavior is wrong, the fix is retraining (better data/objective/evaluation), never a patch downstream of the decision.

This is the hard boundary: the trainer can change what the QDE *is* (via training) and can *judge* what the QDE *did* (via evaluation), but it can never *replace* what the QDE *decides* at inference time.

---

## 5. Relationship to Existing Code Structure

| Existing Component | Role Under Unified Architecture |
|---|---|
| `trainer/qde/qde_pipeline.py` | Orchestrates the training → evaluation → validation → promotion loop around the single QDE model |
| `trainer/qde/qde_models.py` | Defines QDE version/model metadata; promotion floors here move from hardcoded rule gates (see audit) to thresholds validated against the six evaluation dimensions |
| `trainer/qde/qde_registry.py` | Version registry for QDE model artifacts — supports the "one model" principle by tracking exactly one lineage of the unified model, not per-domain sub-models |
| `trainer/qde/qde_promotion.py` | Applies validation results to decide whether a new QDE version replaces the current production version |
| `trainer/qde/qde_ab_testing.py` | Compares QDE versions against each other under real/shadow traffic — evaluation, not decision-time branching |
| `trainer/qde/qde_rollback.py` | Deployment safety net — reverts to a prior QDE version if the new one underperforms; this is a deployment operation, not a decision-time override |
| `trainer/qde/qde_server.py`, `trainer/qde/qde_api.py` | Serve the single QDE model's decisions; must not contain business logic that reshapes those decisions |
| Legacy engines (Risk/Portfolio/Lifecycle/EV/Conviction/Context — see `learning_dimensions_architecture.md`) | Retired from the decision path; their historical logic becomes benchmark baselines used only inside the Evaluation stage |

---

## 6. The One Necessary Exception: Hard Safety Rails

A fully learned decision-maker still operates inside a brokerage/account with real capital constraints (margin calls, exchange position limits, catastrophic-loss circuit breakers). These are **not trading rules** — they are physical/legal/account-level constraints the QDE operates within, analogous to a car's physical top speed rather than a driver's decision on how fast to go.

- Hard rails must be minimal, infrastructural, and clearly separated from decision logic (e.g. "reject an order that would exceed exchange position limits" or "halt trading if account equity hits a broker-mandated margin call level").
- Hard rails must never be dressed up as trading strategy (e.g. a "0.5x confidence-based size reduction" is a rule and is disallowed; "reject any order the broker's API itself would reject" is an infrastructure constraint and is allowed).
- Every hard rail must be documented and justified as an external constraint, not a trading decision, and reviewed periodically to ensure none of them have quietly become a proxy for a trading rule.

---

## 7. Success Criteria for This Architecture

The architecture is correctly implemented when:
1. There is exactly one trained model producing the full trading decision — no per-domain sub-engines contributing independent rule-based decisions in production.
2. Every one of the fifteen learned behaviors in §2 can be traced to training data, a training objective, or an evaluation metric — not to a line of hardcoded logic.
3. The trainer's five responsibilities (data, feedback, evaluation, validation, deployment) are the only ways the trainer affects trading behavior — there is no code path where the trainer injects a strategy, rule, signal, or override into a live decision.
4. Promotion/rollback decisions in `trainer/qde/` are driven by evaluation-dimension scores against benchmark datasets, not by hardcoded floors.
5. The only non-learned logic present is the minimal, documented set of hard safety rails described in §6.

---

## 8. Non-Goals

This document defines the target architecture. It does not itself migrate code. Migration should proceed per the plan in `docs/qde/learning_dimensions_architecture.md` (§5), using `docs/qde/hardcoded_behavior_audit.md` as the checklist of logic to remove from the live decision path.
