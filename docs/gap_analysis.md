# Signalium Trainer — Gap Analysis

**Date:** July 2026  
**Scope:** Audit of existing trainer against 11 target domains.  
**Constraint:** Refine and extend only. No redesign.

---

## How to Read This Document

Each domain lists:
- **Exists** — what is already implemented and usable
- **Gaps** — specific missing components, named precisely
- **Priority** — `P1` (blocking), `P2` (important), `P3` (enhancement)

A gap is only listed if it is absent or incomplete in the current codebase. Nothing here is speculative redesign.

---

## 1. Unified QDE Architecture

### Exists
- `trainer/qde/` — full lifecycle FSM: `REGISTERED → TRAINING → VALIDATION → BACKTESTING → SHADOW_TESTING → DEPLOYMENT_CANDIDATE → ACTIVE → RETIRED`
- `QDERegistry` — JSON-persisted version store with lineage (`parent_version_id`)
- `QDEPromotionCriteria` — five-dimension promotion gates (Sharpe, drawdown, consistency, capital efficiency, decision quality)
- `QDEPipeline` — sequential stage orchestration with pass/fail/skip
- `QDEABTest` — shadow, canary, and side-by-side test modes
- `QDERollback` — immutable audit trail, auto-selection of most recent retired version
- `trainer/mlops/` — second registry (`ModelRegistry`) with semantic versioning, drift detection, governance, monitoring

### Gaps

| # | Gap | Priority |
|---|---|---|
| G1.1 | **Dual registry unification** — `QDERegistry` and `MLOps ModelRegistry` are parallel, disconnected registries with no synchronisation. A QDE version promotion does not update the MLOps registry and vice versa. One of them must become the authoritative source of truth, or a bridge must be added. | P1 |
| G1.2 | **Training trigger** — `QDEPipeline.advance()` changes stage to `TRAINING` but issues no actual training job. There is no call to `trainer/training/engine.py` from the pipeline. The pipeline is a state tracker, not an orchestrator. | P1 |
| G1.3 | **Artifact loading** — `QDEVersion.artifact_uri` is a bare string. There is no loader that resolves the URI to a live model object. Hot-swapping the active version in the execution layer is not implemented. | P1 |
| G1.4 | **Dataset version capture** — When a pipeline run starts, the dataset version used for training is not automatically recorded on `QDEVersion`. `QDEVersion` has no `dataset_version` field. The lineage breaks at training time. | P2 |
| G1.5 | **Execution hot-swap** — Nothing in `trainer/execution/` listens to the `QDERegistry`. When a new version is promoted to `ACTIVE`, the execution layer continues using whatever model was loaded at startup. | P1 |
| G1.6 | **MLOps `auto_promote_enabled` is false** — The governance flag exists but is disabled. No automated promotion path exists end-to-end; every transition requires a named approver. | P2 |

---

## 2. Continuous Learning

### Exists
- `trainer/memory/self_evaluation.py` — generates institutional report cards: strengths, weaknesses, overconfidence patterns, regime-specific failure modes
- `trainer/memory/pattern_discovery.py` — detects performance clusters (e.g., "high-confidence London session trades underperform")
- `trainer/memory/outcome_tracker.py` — links PnL and hit/miss outcomes back to original decisions
- `trainer/research/retraining_advisor.py` — generates `RetrainingRecommendation` objects based on model degradation signals
- `trainer/mlops/retraining.py` — evaluates thresholds and produces advisory recommendations

### Gaps

| # | Gap | Priority |
|---|---|---|
| G2.1 | **Retraining trigger** — Both `retraining_advisor.py` and `mlops/retraining.py` are explicitly advisory-only. No automated worker consumes `RetrainingRecommendation` objects and calls `trainer/training/engine.py`. The loop is open. | P1 |
| G2.2 | **Online ensemble weight adjustment** — Patterns discovered in memory (e.g., "Expert A is failing in high volatility") are not fed back into `trainer/ensemble/routing.py` or `trainer/ensemble/weighting.py` in real time or between training cycles. | P2 |
| G2.3 | **Failure-to-feature pipeline** — When `SelfEvaluationEngine` identifies a recurring failure mode (e.g., "indicator X precedes 80% of losses"), there is no mechanism to surface that finding as a candidate feature or as a retraining signal to the Research Lab's `FeatureDiscovery`. | P2 |
| G2.4 | **Indicator trust decay** — The system records "primary reason" for each trade decision but has no loop to lower trust in a specific indicator globally when that indicator is consistently associated with failed reasoning. | P3 |

---

## 3. Autonomous Trading

### Exists
- Full execution planning stack in `trainer/execution/` (planner, risk engine, sizer, manager, timing, stop/TP engines, order strategy)
- Confidence-gated scale-in logic (`confidence > 0.75`)
- `ICARI DrawdownManager` — adaptive risk reduction state machine (`NORMAL → SOFT_LIMIT → HARD_LIMIT → EMERGENCY_LOCKDOWN`)
- `Portfolio.recommend()` — unified portfolio signal aggregation

### Gaps

| # | Gap | Priority |
|---|---|---|
| G3.1 | **Broker adapter layer** — The entire `trainer/execution/` stack produces `ExecutionPlan` objects but has no connectivity layer to any broker, exchange, or order management system. Plans are never submitted. | P1 |
| G3.2 | **Active model wiring** — The `ExecutionPlanner` has no reference to the currently active QDE version. It cannot invoke the model to produce forecasts; it consumes pre-computed signals. The inference call is missing. | P1 |
| G3.3 | **Market data ingestion** — There is no live market data feed adapter that produces `MarketSnapshot` objects for the execution pipeline to consume in real time. | P1 |
| G3.4 | **Risk engine uses static hit probability** — `risk_engine.py` uses a forecasted hit probability parameter. It does not query `trainer/memory/outcome_tracker.py` for the model's empirical hit rate in the current regime. | P2 |
| G3.5 | **Cross-asset sizing gap** — `position_sizer.py` enforces portfolio caps but does not adjust per-instrument sizing based on current cross-asset correlation from `trainer/icari/correlation_engine.py` or `trainer/portfolio/correlation_engine.py`. | P2 |

---

## 4. Trade Lifecycle Learning

### Exists
- `trainer/lifecycle/` — 20-state FSM with full forward and terminal paths
- `LifecycleOptimizer` — heuristic analysis: breakeven timing, trailing stop activation, scaling efficiency, risk/timeout rates
- `LifecycleAnalytics` — aggregate transition matrices and systemic pattern detection
- `AiReasoning` — confidence narrative tracking across the trade's life
- `PositionMonitor` — real-time scaling, stop-move, and exposure reduction recommendations

### Gaps

| # | Gap | Priority |
|---|---|---|
| G4.1 | **Static transition rules** — All transition thresholds (e.g., `confidence >= 0.65` for `ENTRY_READY → PENDING_ENTRY`) are hardcoded in `state_machine.py`. There is no feedback loop that adjusts these thresholds based on historical win rates per regime. | P1 |
| G4.2 | **MFE/MAE unused** — `PositionMetrics` tracks Maximum Favourable Excursion and Maximum Adverse Excursion per trade, but `LifecycleOptimizer` does not use this data to mathematically optimise trailing stop distance or partial exit targets. The data is collected but not actioned. | P2 |
| G4.3 | **Intra-trade adaptation** — The memory system is post-mortem only. There is no mechanism to update the trade's reasoning or confidence mid-trade as new market data arrives while a position is open. | P2 |
| G4.4 | **Execution slippage feedback** — Lifecycle tracks planned entries but does not receive actual fill data (slippage, partial fills). The gap between planned and actual execution is not fed back to refine future planning. | P2 |
| G4.5 | **Lifecycle-to-training signal** — `LifecycleOptimizer` findings (e.g., "scaling succeeds 40% less often in VOLATILE regime") are stored analytically but not converted into training labels or retraining triggers. | P3 |

---

## 5. Position Management Learning

### Exists
- `trainer/execution/trade_manager.py` — explicit lifecycle rules for breakeven, trailing stop, partial close, and scale-in
- `ManagementPlan` — structured dataclass with trigger prices and trailing step distances
- `ExecutionDatabase.management_plans` table — persists every management decision
- `ExecutionDatabase.optimization_history` — stores past outcomes for offline analysis

### Gaps

| # | Gap | Priority |
|---|---|---|
| G5.1 | **No feedback loop from DB to engine** — `management_plans` and `optimization_history` are written to the database but nothing reads them back to tune `TradeManager` coefficients. The database is write-only from a learning perspective. | P1 |
| G5.2 | **Static scaling confidence threshold** — The scale-in trigger (`confidence > 0.75`) is hardcoded. It does not adapt based on how previous scale-ins in similar market conditions performed. | P2 |
| G5.3 | **Breakeven distance is fixed** — Breakeven is moved at a fixed ATR multiple (1.0x). There is no per-instrument, per-regime calibration of when breakeven movement actually improves outcomes vs. prematurely exits profitable trades. | P2 |
| G5.4 | **Trailing stop is fixed** — Trailing stop step is 1x ATR regardless of volatility regime, instrument, or confidence level. Historical `optimization_history` data is available but not used to tune this. | P2 |
| G5.5 | **No cross-module position context** — `TradeManager` does not query `ICARI DrawdownManager` for current equity state. It can recommend scale-ins during an active drawdown lockdown. | P2 |

---

## 6. Portfolio Reasoning Learning

### Exists
- `trainer/portfolio/optimizer.py` — multi-objective grid-search optimiser (max Sharpe, min vol, max diversification)
- `trainer/portfolio/regime_detector.py` — rolling volatility / trend-strength regime classification with position-size multipliers
- `trainer/portfolio/correlation_engine.py` — Pearson, Spearman, tail, and EWMA correlation matrices
- `trainer/portfolio/exposure_engine.py` — over-concentration, currency imbalance, and conflicting signal auditing
- `trainer/icari/allocation_engine.py` — composite-factor capital allocation (15+ factors: research score, execution quality, ensemble reliability, etc.)
- `trainer/icari/drawdown_manager.py` — adaptive risk reduction state machine

### Gaps

| # | Gap | Priority |
|---|---|---|
| G6.1 | **Hardcoded allocation weights** — `allocation_engine.py` uses static `_WEIGHTS` (e.g., `research_score: 0.12`). There is no mechanism to learn these weights from realised PnL or capital efficiency outcomes. | P1 |
| G6.2 | **No portfolio-level training signal** — Models are trained on per-instrument labels. There is no portfolio-regret signal from `PortfolioScorer`, `ExposureEngine`, or `ICARI` that penalises models for generating over-concentrated or tail-correlated predictions. | P2 |
| G6.3 | **Regime does not feed ensemble routing** — `regime_detector.py` adjusts position-size multipliers but does not feed back into `trainer/ensemble/routing.py` to select which expert models are active in each regime. | P2 |
| G6.4 | **Reasoning not used as future features** — The `AllocationRationale` generated per allocation (e.g., "composite_confidence=0.82") is stored in the database but never used as a feature in future training cycles. | P3 |
| G6.5 | **Optimiser not differentiable** — `optimizer.py` uses discrete grid search. Portfolio-level constraints (correlation, risk budget) cannot be backpropagated into neural network weights to encourage portfolio-friendly predictions at training time. | P3 |

---

## 7. Reflection Learning

### Exists
- `trainer/memory/self_evaluation.py` — per-model report cards: strengths, weaknesses, overconfidence, regime failure modes
- `trainer/memory/counterfactual_engine.py` — what-if scenario simulation for every completed decision
- `trainer/memory/decision_lineage.py` — full traceability: Decision → Forecast → Model → Dataset → Training Run → Experiment → Git Commit
- `trainer/memory/pattern_discovery.py` — automatic performance cluster detection

### Gaps

| # | Gap | Priority |
|---|---|---|
| G7.1 | **No cross-model reflection** — `SelfEvaluationEngine` evaluates the current champion in isolation. There is no "competitive reflection" that asks: where would the challenger have beaten the champion in the last N days, and why? | P2 |
| G7.2 | **Reasoning quality not scored** — The system records `DecisionReasoning.primary_reason` and `supporting_factors` but never scores whether those reasons were causally correct. A reason that preceded losses is not downweighted. | P2 |
| G7.3 | **Counterfactuals not fed to optimiser** — `CounterfactualEngine` computes delta-PnL for alternative SL/TP scenarios but the results are stored analytically. They are not fed to `TradeManager` or `StopLossEngine` to adjust future plans. | P2 |
| G7.4 | **Reflection not time-gated** — `SelfEvaluationEngine` runs on-demand. There is no scheduled reflection cycle (e.g., end-of-week) that systematically reviews the last N decisions and produces a structured finding for the Research Lab. | P3 |

---

## 8. Market Memory

### Exists
- `trainer/memory/knowledge_base.py` — searchable index: decisions → trades → experiments → research reports
- `trainer/memory/decision_memory.py` — `DecisionSnapshot` captures regime, session, volatility context at decision time
- `trainer/portfolio/regime_detector.py` — rolling regime classification (trending, volatile, risk-on/off)
- `trainer/mie/market_regime_engine.py` — market intelligence regime engine
- `trainer/microstructure/state_classifier.py` — microstructure state classification

### Gaps

| # | Gap | Priority |
|---|---|---|
| G8.1 | **Regime transition memory** — Regimes are treated as static point-in-time labels. There is no memory of the *sequence* of conditions that historically precede a regime change (e.g., "rising ATR + narrowing spread breadth = trending-to-volatile transition in 3–5 sessions"). | P2 |
| G8.2 | **Global event memory** — No "world state" memory that correlates macro events across all instruments simultaneously. An external shock (e.g., a broad risk-off event) affecting 12 instruments simultaneously is treated as 12 separate unrelated events. | P2 |
| G8.3 | **Regime-keyed recall** — `KnowledgeBase` search is general-purpose. There is no regime-indexed query mode that retrieves "what did the model do the last 20 times it entered a VOLATILE + RISK_OFF regime" as a first-class operation. | P2 |
| G8.4 | **Memory does not decay** — There is no time-decay or recency weighting in the knowledge base. A pattern from 18 months ago carries the same weight as one from last week. | P3 |
| G8.5 | **No inter-instrument correlation memory** — The knowledge base links decisions to their own instrument but does not record what other instruments were doing at the same timestamp. Systemic correlation events are invisible to the memory system. | P3 |

---

## 9. Version Management

### Exists
- `trainer/qde/qde_registry.py` — QDE-specific registry with JSON persistence, stage FSM, lineage (parent ID)
- `trainer/mlops/registry.py` — MLOps `ModelRegistry` with semantic versioning, dataset/feature/label version pinning, full dependency tracking
- `trainer/mlops/lifecycle.py` — stage FSM: `DRAFT → TRAINING → VALIDATING → CANDIDATE → APPROVED → CHAMPION → PRODUCTION`
- `trainer/datasets/versioning.py` — dataset version management
- `trainer/research/lineage.py` and `lineage_api.py` — experiment-to-git-commit traceability

### Gaps

| # | Gap | Priority |
|---|---|---|
| G9.1 | **Two disconnected registries** — `QDERegistry` and `MLOps ModelRegistry` are parallel systems with overlapping responsibilities (both track champion/challenger, both have stage FSMs, both store lineage). Neither is aware of the other. A model can be `ACTIVE` in QDE and `CANDIDATE` in MLOps simultaneously. | P1 |
| G9.2 | **Dataset version not recorded at pipeline start** — `QDEVersion` has no `dataset_version` field. When a QDE pipeline run starts training, the dataset version consumed is not captured on the version record. | P2 |
| G9.3 | **No automated artifact registration** — After training completes, the model artifact is not automatically registered in either registry. The `artifact_uri` on `QDEVersion` must be set manually. | P2 |
| G9.4 | **Research Lab uses in-memory stores** — `ResearchLabAPI` stores all experiments, datasets, and shadow test results in Python dictionaries. A server restart loses all administrative research state. Unlike `research/database.py` and `mlops/database.py`, `research_lab` has no persistence layer. | P1 |
| G9.5 | **MLOps promotion requires named approver** — `governance.py` has `auto_promote_enabled` but it is `False`. Full automation requires this path to be wired and tested. | P2 |

---

## 10. Shadow Deployment

### Exists
- `trainer/qde/qde_ab_testing.py` — `QDEABTest` model with `SHADOW`, `CANARY`, `SIDE_BY_SIDE` types; status and metric tracking
- `trainer/qde/qde_models.py` — `QDEStage.SHADOW_TESTING` stage in the version FSM
- `trainer/mlops/champion.py` — `ChampionChallengerManager` with metric-based side-by-side comparison
- `trainer/mlops/governance.py` — `SHADOW` state in `LifecycleState` enum
- `trainer/research_lab/rl_models.py` — `ShadowTestRun` with challenger vs. champion decision comparison, agreement rate, per-regime breakdown, and promotion recommendation

### Gaps

| # | Gap | Priority |
|---|---|---|
| G10.1 | **No traffic splitter** — `QDEABTest` and `MLOps governance` define shadow states but neither contains an actual data fork. There is no component that receives a live market snapshot and routes it simultaneously to both the champion and challenger for parallel inference. | P1 |
| G10.2 | **Research Lab shadow test is manual** — `ShadowTestRun` requires decisions to be posted via `POST /research/shadow-tests/:id/decisions`. There is no automated bridge that feeds live decisions from the QDE execution layer into the Research Lab's shadow test recorder. | P1 |
| G10.3 | **Canary traffic split not enforced** — `QDEABTest` records a `canary_percentage` but there is no enforcement layer that actually routes that fraction of decisions to the challenger and the remainder to the champion. | P2 |
| G10.4 | **No latency comparison** — Shadow and canary tests compare decision types and confidence but do not record inference latency for each version. A challenger that is 10x slower than the champion would pass the shadow test undetected. | P3 |
| G10.5 | **Shadow result does not trigger promotion** — When a `ShadowTestRun` concludes with recommendation `PROMOTE`, this result is not automatically forwarded to the QDE pipeline to advance the version to `DEPLOYMENT_CANDIDATE`. The recommendation is an inert record. | P2 |

---

## 11. Research Lab

### Exists
- `trainer/research_lab/research_lab_api.py` — 13 responsibilities: datasets, features, experiments, training runs, validation gates, backtesting, shadow testing, version promotion, failure analysis, deployment approval, parameter search, model comparison, regime analysis
- `trainer/research_lab/rl_models.py` — full data model layer: `ExperimentRecord`, `ResearchDataset`, `DiscoveredFeature`, `ShadowTestRun`, `DeploymentGateResult`, `FailureEvent`, `ParameterSearchRun`
- `trainer/research_lab/research_lab_server.py` — admin-only FastAPI server (port 8002), no public docs, no CORS
- `trainer/research/` — intelligence layer: `ResearchEngine`, `MarketEvolutionMonitor`, `LabelQualityMonitor`, `RetrainingAdvisor`, experiment tracker, lineage, champion/challenger registry

### Gaps

| # | Gap | Priority |
|---|---|---|
| G11.1 | **No database persistence** — `ResearchLabAPI` stores all state in Python in-memory dictionaries. A server restart loses all datasets, experiments, shadow tests, and failure events. Every other comparable module (`research/database.py`, `mlops/database.py`, `execution/execution_database.py`, `memory/memory_database.py`) uses SQLite or equivalent. | P1 |
| G11.2 | **No automated trigger from `research/`** — `research/retraining_advisor.py` generates `RetrainingRecommendation` objects when model performance degrades, but these are never forwarded to `ResearchLabAPI` to automatically create a new experiment or initiate a shadow test. The governance layer and the intelligence layer are disconnected. | P1 |
| G11.3 | **Approval does not trigger deployment** — `approve_deployment()` sets `ExperimentRecord.deployment_status = ACTIVE` in memory but issues no command to `QDERegistry` or `MLOps ModelRegistry` to actually activate the version in the live pipeline. The approval is an administrative record, not an operational command. | P1 |
| G11.4 | **Failure events do not feed feature discovery** — When a `FailureEvent` is investigated and a `root_cause` is identified, this finding is not automatically surfaced to `FeatureDiscovery` as a candidate signal or to `ParameterSearch` as a constraint. The failure analysis is a closed record. | P2 |
| G11.5 | **No scheduled reflection cycle** — The Research Lab is entirely reactive (admin posts events). There is no scheduled job that queries `self_evaluation.py` at the end of each day/week, converts findings into `FailureEvent` records, and assigns them for investigation. | P2 |
| G11.6 | **Shadow test not wired to QDE pipeline** — `conclude_shadow_test()` produces a recommendation but does not call `qde_pipeline.advance()` or write to `QDERegistry`. A `PROMOTE` recommendation is not acted upon. | P2 |

---

## Consolidated Gap Register

Below is a flat prioritised list of all gaps, for use as a work backlog.

### P1 — Blocking (system cannot function end-to-end without these)

| ID | Gap | Domain |
|---|---|---|
| G1.1 | Unify QDE and MLOps registries into one authoritative source | QDE Architecture |
| G1.2 | Wire QDE pipeline stage `TRAINING` to `training/engine.py` | QDE Architecture |
| G1.3 | Implement artifact loader — resolve `artifact_uri` to live model | QDE Architecture |
| G1.5 | Implement execution hot-swap on QDE active version change | QDE Architecture |
| G2.1 | Implement retraining trigger worker — consume `RetrainingRecommendation` → call training engine | Continuous Learning |
| G3.1 | Broker adapter layer — submit `ExecutionPlan` to an OMS/broker | Autonomous Trading |
| G3.2 | Wire active QDE version to `ExecutionPlanner` for live inference | Autonomous Trading |
| G3.3 | Live market data ingestion — produce `MarketSnapshot` in real time | Autonomous Trading |
| G4.1 | Replace hardcoded lifecycle transition thresholds with learned per-regime values | Trade Lifecycle Learning |
| G9.1 | Consolidate dual registries (QDE + MLOps) | Version Management |
| G9.4 | Add database persistence to `ResearchLabAPI` | Version Management / Research Lab |
| G10.1 | Implement traffic splitter for shadow and canary deployment | Shadow Deployment |
| G10.2 | Automate live decision feed into `ShadowTestRun` recorder | Shadow Deployment |
| G11.1 | Add database persistence to `ResearchLabAPI` | Research Lab |
| G11.2 | Bridge `research/retraining_advisor.py` → `ResearchLabAPI.create_experiment()` | Research Lab |
| G11.3 | Wire `approve_deployment()` → `QDERegistry.promote()` | Research Lab |

### P2 — Important (system works but without these learning is manual or degraded)

| ID | Gap | Domain |
|---|---|---|
| G1.4 | Capture dataset version on `QDEVersion` at pipeline start | QDE Architecture |
| G1.6 | Enable and test `auto_promote_enabled` path in MLOps governance | QDE Architecture |
| G2.2 | Online ensemble weight adjustment from memory patterns | Continuous Learning |
| G2.3 | Failure-to-feature pipeline: self-evaluation → feature discovery | Continuous Learning |
| G3.4 | Replace static hit probability with empirical rate from `outcome_tracker` | Autonomous Trading |
| G3.5 | Cross-asset correlation adjustment in `position_sizer.py` | Autonomous Trading |
| G4.2 | Use MFE/MAE data in `LifecycleOptimizer` to tune stop/TP distances | Trade Lifecycle Learning |
| G4.3 | Intra-trade adaptation — update reasoning mid-trade on new data | Trade Lifecycle Learning |
| G4.4 | Feed actual fill data (slippage, partial fills) into lifecycle | Trade Lifecycle Learning |
| G5.1 | Read `management_plans` DB back into `TradeManager` for coefficient tuning | Position Management Learning |
| G5.2 | Adaptive scale-in confidence threshold (per regime, per instrument) | Position Management Learning |
| G5.3 | Per-regime, per-instrument breakeven distance calibration | Position Management Learning |
| G5.4 | Per-regime trailing stop distance calibration | Position Management Learning |
| G5.5 | Wire `TradeManager` to `ICARI DrawdownManager` for lockdown awareness | Position Management Learning |
| G6.1 | Learn `allocation_engine.py` factor weights from realised outcomes | Portfolio Reasoning Learning |
| G6.2 | Portfolio-level training signal — penalise over-concentration at training time | Portfolio Reasoning Learning |
| G6.3 | Regime-aware expert selection — feed `regime_detector` into `ensemble/routing.py` | Portfolio Reasoning Learning |
| G7.1 | Cross-model reflection — champion vs. challenger comparative analysis | Reflection Learning |
| G7.2 | Score reasoning quality — downweight reasons that precede losses | Reflection Learning |
| G7.3 | Feed `CounterfactualEngine` results to `TradeManager` / `StopLossEngine` | Reflection Learning |
| G8.1 | Regime transition memory — sequence signatures preceding regime changes | Market Memory |
| G8.2 | Global event memory — correlate macro shocks across all instruments | Market Memory |
| G8.3 | Regime-keyed knowledge base recall | Market Memory |
| G9.2 | Record dataset version on `QDEVersion` at pipeline start | Version Management |
| G9.3 | Automated artifact registration after training completes | Version Management |
| G9.5 | Enable `auto_promote_enabled` in MLOps governance | Version Management |
| G10.3 | Enforce canary traffic split percentage at inference time | Shadow Deployment |
| G10.5 | Wire shadow test `PROMOTE` recommendation → QDE pipeline advance | Shadow Deployment |
| G11.4 | Wire `FailureEvent.root_cause` → `FeatureDiscovery` / `ParameterSearch` | Research Lab |
| G11.5 | Scheduled reflection cycle: self-evaluation → `FailureEvent` creation | Research Lab |
| G11.6 | Wire `conclude_shadow_test(PROMOTE)` → `qde_pipeline.advance()` | Research Lab |

### P3 — Enhancement

| ID | Gap | Domain |
|---|---|---|
| G2.4 | Indicator trust decay based on decision history | Continuous Learning |
| G4.5 | Convert lifecycle optimizer findings into training labels | Trade Lifecycle Learning |
| G6.4 | Use `AllocationRationale` fields as future training features | Portfolio Reasoning Learning |
| G6.5 | Differentiable portfolio optimiser for end-to-end training | Portfolio Reasoning Learning |
| G7.4 | Scheduled reflection cycle (end-of-week systematic review) | Reflection Learning |
| G8.4 | Time-decay / recency weighting in the knowledge base | Market Memory |
| G8.5 | Inter-instrument correlation memory at decision time | Market Memory |
| G10.4 | Latency comparison in shadow and canary tests | Shadow Deployment |

---

## Cross-Cutting Observations

**The trainer has two parallel universes.** `QDERegistry` and `MLOps ModelRegistry` evolved independently and now both claim to manage the model lifecycle. Before implementing any new learning feature, this must be resolved (G1.1 / G9.1). Every new capability built on top of a split registry will have to be built twice.

**Every learning loop is open.** The memory system, self-evaluation, counterfactual engine, lifecycle optimiser, and portfolio regime detector all produce valuable analytical outputs. None of them are wired to an actuator — nothing triggers a retrain, adjusts a coefficient, or calls a training job. The intelligence exists; the actuation does not.

**Research Lab needs a database before it needs anything else.** Adding the remaining 10 Research Lab responsibilities on top of in-memory storage (G11.1) means any server restart wipes all administrative history. This is the highest-leverage single fix in the Research Lab domain.

**Shadow deployment is the FSM without the engine.** The QDE, MLOps, and Research Lab all define shadow test states and models. The gap is not conceptual — it is one missing component: a data-forking proxy that sends the same `MarketSnapshot` to two models simultaneously (G10.1).
