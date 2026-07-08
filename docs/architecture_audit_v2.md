# Signalium Trainer 2.0 — Architecture Audit
## Unified Quant Decision Engine Migration Plan

**Repository:** signalium-trainer  
**Date:** 2026-07-08  
**Scope:** All 34 trainer modules, 427 Python files  
**Standard:** One Model. One Brain. One Trader. One Decision Maker.

---

## 1. Current State

The trainer is built as a **federation of independent decision engines** coordinated by the QDE lifecycle layer. The QDE is not the trading intelligence — it is a version manager sitting above a collection of rule systems.

The actual decisions are made by:

| Layer | What decides | Where |
|---|---|---|
| Regime Classification | Market state (TRENDING, VOLATILE, etc.) | `mie/market_regime_engine.py` |
| Expert Selection | Which model fires | `ensemble/routing.py` |
| Signal Combination | How models vote | `ensemble/strategies.py` (8 strategies) |
| Exit Placement | Where stops go | `execution/stop_loss_engine.py` |
| Target Placement | Where take profits go | `execution/take_profit_engine.py` |
| Order Type | MARKET vs LIMIT vs STOP | `execution/order_strategy.py` |
| Position Size | How much capital | `icari/position_sizing_engine.py` (10 methods) |
| Size Adjustment | Dynamic scaling | `icari/dynamic_sizer.py` (10 factors) |
| Portfolio Allocation | Capital across instruments | `icari/allocation_engine.py` |
| Risk Budget | Exposure limits | `icari/risk_budget_engine.py` |
| Trade Transitions | When to exit or hold | `lifecycle/transition_engine.py` |
| Meta-Override | Override all experts | `ensemble/meta_model.py` |

The QDE model is consulted for one thing only: **buy, sell, or no-trade**.  
Every other dimension of the trading decision is handled by a rule system.

### Module Inventory

| Module | Files | Role | Classification |
|---|---|---|---|
| `research_lab/` | 4 | Admin: experiments, backtesting, validation, rollbacks | **KEEP** |
| `datasets/` | 5 | Dataset pipeline, versioning, validation | **KEEP** |
| `dataloaders/` | 2 | Data loading factory | **KEEP** |
| `sampling/` | 2 | Sampling strategies | **KEEP** |
| `training/` | 8 | Training engine, scheduler, reproducibility | **KEEP** |
| `losses/` | 6 | Loss functions (focal, multi-task, weighted CE) | **KEEP** |
| `calibration/` | 4 | Output calibration (Platt, isotonic, temperature) | **KEEP** |
| `automl/` | 10 | Hyperparameter search for one model | **KEEP** |
| `validation/` | 17 | Validation pipelines, drift, robustness, walk-forward | **KEEP** |
| `evaluation/` | 2 | Model evaluation infrastructure | **KEEP** |
| `metrics/` | 4 | Metrics computation | **KEEP** |
| `cert/` | 13 | Certification suite (stress, security, Monte Carlo, operational) | **KEEP** |
| `checkpoints/` | 2 | Checkpoint manager | **KEEP** |
| `mlops/` | 13 | Registry, monitoring, retraining policy, health, governance | **KEEP** |
| `qde/` | 8 | Version lifecycle, pipeline, promotion, rollback | **KEEP** (partial — see §3) |
| `traceability/` | 6 | Reproducibility, metadata, lineage | **KEEP** |
| `exporters/` | 5 | ONNX, TorchScript, manifest | **KEEP** |
| `framework/` | 5 | Model factory, inference, traced inference | **KEEP** |
| `models/` | 2 | Model base classes | **KEEP** |
| `research/` | 38 | HRSA research, observatory, experiment engine, lineage | **KEEP** |
| `analyst/` | 14 | Admin: model comparator, research chat, citations | **KEEP** |
| `explainability/` | 7 | Attention, attribution, permutation importance | **KEEP** |
| `memory/` | 12 | Decision memory, outcome tracking, pattern discovery | **KEEP** |
| `logging/` | 2 | Logging setup | **KEEP** |
| `utils/` | 4 | Device, GPU, git | **KEEP** |
| `config/` | 2 | Config schema | **KEEP** |
| `api/` | 4 | Trainer API client | **KEEP** |
| `cli/` | 2 | Command-line interface | **KEEP** |
| `plugins/` | 2 | Plugin registry | **KEEP** |
| `interfaces/` | 3 | Base interfaces, future encoders | **KEEP** |
| `registry/` | 2 | Model registry | **KEEP** |
| `microstructure/` | 15 | Tick interface, order flow, market microstructure features | **KEEP** |
| `mie/` | 23 | Intelligence engines + feature store + database | **KEEP + CONVERT** |
| `execution/` | 17 | Execution engines (stop, TP, risk, order, sizing) | **KEEP + CONVERT** |
| `icari/` | 17 | Capital allocation, risk budget, sizing engines | **KEEP + CONVERT** |
| `portfolio/` | 12 | Portfolio engines, regime detector, conflict resolver | **KEEP + CONVERT** |
| `lifecycle/` | 13 | Trade lifecycle state machine, transition engine | **KEEP + CONVERT** |
| `ensemble/` | 14 | Ensemble voting, expert registry, meta-model, routing | **REMOVE** |

---

## 2. Target State

```
Market Data / Features
         │
         ▼
  ┌─────────────────┐
  │   One QDE Model  │   ◄── Trained on all dimensions:
  │  (one artifact,  │       direction · size · entry timing
  │   one version)   │       exit level · order type
  └─────────────────┘       risk per trade · hold duration
         │
         ▼
  One Decision Output
  {
    direction:    BUY | SELL | NO_TRADE
    confidence:   0.0 – 1.0
    size_signal:  0.0 – 1.0   (learned fraction of budget)
    entry_urgency: 0.0 – 1.0  (learned: now vs. wait)
    exit_horizon: bars (learned hold duration)
    risk_signal:  0.0 – 1.0   (learned risk appetite)
  }
         │
    ┌────┴────────────────────────────────────────────────┐
    │             Supporting Infrastructure                 │
    │  Research Lab · Training · Validation · MLOps         │
    │  Feature Warehouse · Memory · Telemetry · Cert        │
    └───────────────────────────────────────────────────────┘
```

The model is trained with **multi-dimensional objectives** — each CONVERT module becomes a training target or evaluation dimension rather than a runtime rule engine:

| Former Rule Engine | Target State |
|---|---|
| `StopLossEngine` | Training objective: "where should this trade exit given these features?" |
| `TakeProfitEngine` | Training objective: "what return is achievable in N bars?" |
| `PositionSizingEngine` | Training objective: "what fraction of budget has positive EV here?" |
| `OrderStrategyEngine` | Training objective: "should this trade execute now or wait?" |
| `MarketRegimeEngine` | Training input feature (not a classifier — raw features go to the model) |
| `RiskEngine` | Evaluation dimension: model is scored on whether its risk signal predicts outcomes |
| `TransitionEngine` | Training objective: "what is the optimal action at each trade state?" |
| `AllocationEngine` | Training objective: "given portfolio state, what is the optimal exposure?" |

---

## 3. Component Classification

### 3.1 KEEP — No changes required

These modules are infrastructure. They serve the model — they do not replace it.

#### Research Lab
| Component | Keep Rationale |
|---|---|
| `research_lab/rl_database.py` | SQLite persistence for all research entities |
| `research_lab/research_lab_api.py` | All 8 research capabilities (experiment tracking, backtesting, validation, dataset generation, failure analysis, version comparisons, promotion approval, rollback approval) |
| `research_lab/research_lab_server.py` | Admin HTTP server |
| `research_lab/rl_models.py` | Data models for research entities |

#### Dataset & Feature Pipeline
| Component | Keep Rationale |
|---|---|
| `datasets/` | Versioned dataset construction, parquet loading, multi-task labels, validation |
| `dataloaders/` | Batch loading for training |
| `sampling/` | Temporal and stratified sampling |
| `microstructure/` | Tick data interface, order flow, microstructure feature extraction. These are **inputs** to the model, not decision makers. |
| `mie/feature_store.py` | Caches computed MIE features for training |
| `mie/mie_database.py` | Persists MIE signals (become training features) |
| `mie/mie_models.py` | Data classes for market inputs and results |
| `mie/mie_types.py` | Enumerations for market states (inputs, not decisions) |
| `mie/mie_worker.py` | Async worker for MIE signal computation |
| `mie/ai_commentary.py` | AI-generated market commentary for admin dashboards |
| `mie/behavior_similarity.py` | Similarity matching for pattern-based training data labelling |
| `mie/calibration_engine.py` | MIE signal calibration |
| `mie/prediction_reliability.py` | Prediction reliability scoring |

#### Training Infrastructure
| Component | Keep Rationale |
|---|---|
| `training/engine.py` | Core training loop |
| `training/multi_task_engine.py` | Multi-task training (required for CONVERT outputs) |
| `training/early_stopping.py` | Early stopping policy |
| `training/scheduler.py` | LR scheduler |
| `training/reproducibility.py` | Seed management and reproducibility |
| `training/mlflow_tracker.py` | Experiment tracking integration |
| `losses/` | All loss functions: focal, multi-task, weighted cross-entropy, advanced |
| `calibration/` | Platt scaling, isotonic, temperature scaling for output calibration |
| `automl/` | Hyperparameter search over the one model architecture |
| `checkpoints/` | Checkpoint saving and loading |
| `framework/` | Model factory and inference framework |
| `models/` | Model base classes and interfaces |

#### Validation & Certification
| Component | Keep Rationale |
|---|---|
| `validation/engine.py` | Institutional validation pipeline |
| `validation/drift.py` | Feature and label drift detection |
| `validation/robustness.py` | Robustness testing (noise, adversarial) |
| `validation/monte_carlo.py` | Monte Carlo simulation |
| `validation/windows.py` | Walk-forward validation windows |
| `validation/statistical.py` | Statistical significance tests |
| `validation/trading.py` | Trading metrics (Sharpe, drawdown, win rate) |
| `validation/promotion.py` | Promotion gate evaluation |
| `validation/regime.py` | Per-regime performance breakdown |
| `validation/failure_analysis.py` | Failure mode analysis |
| `cert/` | All 8 certification engines (benchmark, stress, security, operational, Monte Carlo, reproducibility, checklist, scorecard) |
| `evaluation/` | Evaluator and reporter |
| `metrics/` | Evaluation metrics, multi-task metrics |

#### MLOps & Model Management
| Component | Keep Rationale |
|---|---|
| `mlops/engine.py` | MLOps orchestration |
| `mlops/registry.py` | Model registry |
| `mlops/retraining.py` | Retraining trigger evaluation (recommendations only — never auto-retrain) |
| `mlops/monitoring.py` | Production monitoring |
| `mlops/health.py` | Model health scoring |
| `mlops/governance.py` | Governance and audit trail |
| `mlops/alerting.py` | Alert management |
| `mlops/champion.py` | Champion model tracking |
| `mlops/lifecycle.py` | Model lifecycle management |
| `mlops/export.py` | Model export utilities |
| `mlops/audit.py` | Audit logging |
| `qde/qde_pipeline.py` | 6-stage training pipeline orchestration |
| `qde/qde_registry.py` | QDE version registry and persistence |
| `qde/qde_models.py` | QDE version data structures (not the hardcoded promotion thresholds) |
| `qde/qde_promotion.py` | Promotion workflow and approval |
| `qde/qde_rollback.py` | Rollback infrastructure |
| `qde/qde_api.py` | Public QDE lifecycle API |
| `qde/qde_server.py` | QDE FastAPI server |

**Note on `qde/qde_ab_testing.py`:** The shadow observation infrastructure (recording challenger decisions without executing them) is KEEP. The CANARY mode which routes actual live decisions to a challenger is REMOVE — only one model makes live decisions.

#### Telemetry & Memory
| Component | Keep Rationale |
|---|---|
| `memory/decision_memory.py` | Records every decision with full context — primary training data feed |
| `memory/outcome_tracker.py` | Records trade outcomes — primary supervised learning signal |
| `memory/pattern_discovery.py` | Discovers patterns in historical decisions — training data enrichment |
| `memory/decision_benchmarking.py` | Benchmarks decisions against baselines |
| `memory/decision_lineage.py` | Decision provenance tracking |
| `memory/knowledge_base.py` | Structured knowledge base |
| `memory/memory_api.py` | Memory API |
| `memory/memory_database.py` | Memory persistence |
| `memory/self_evaluation.py` | Model self-evaluation against past decisions |
| `memory/quality_engine.py` | Decision quality scoring |
| `memory/counterfactual_engine.py` | Counterfactual analysis |
| `lifecycle/lifecycle_states.py` | Trade state definitions |
| `lifecycle/lifecycle_models.py` | Trade lifecycle data models |
| `lifecycle/lifecycle_database.py` | Lifecycle persistence |
| `lifecycle/state_machine.py` | Trade state machine |
| `lifecycle/event_engine.py` | Event recording |
| `lifecycle/lifecycle_analytics.py` | Lifecycle analytics |
| `lifecycle/lifecycle_dashboard.py` | Lifecycle dashboard |
| `lifecycle/lifecycle_api.py` | Lifecycle API |
| `lifecycle/lifecycle_optimizer.py` | Lifecycle optimization analysis |
| `execution/execution_database.py` | Execution telemetry persistence |
| `execution/execution_dashboard.py` | Execution observability dashboard |
| `execution/execution_simulator.py` | Backtesting simulator — generates training labels |
| `execution/execution_types.py` | Execution data types |
| `portfolio/portfolio_database.py` | Portfolio state persistence |
| `portfolio/portfolio_dashboard.py` | Portfolio observability |
| `portfolio/portfolio_instruments.py` | Instrument registry |
| `portfolio/dependency_graph.py` | Instrument dependency graph |
| `portfolio/conflict_resolver.py` | Detects conflicting recommendations (telemetry, not decision) |
| `icari/icari_database.py` | Capital allocation telemetry |
| `icari/icari_dashboard.py` | Capital allocation dashboard |
| `icari/icari_models.py` | Allocation data models |
| `icari/icari_types.py` | Allocation type definitions |
| `icari/drawdown_manager.py` | Drawdown state tracking (telemetry) |

#### Research & Admin
| Component | Keep Rationale |
|---|---|
| `research/` | All 38 files — HRSA research system, observatory, lineage, dataset evolution, knowledge graph, experiment recommendations. Admin tooling only, no live decisions. |
| `analyst/` | All 14 files — Research assistant, model comparator, citation engine, LLM-powered knowledge retrieval. Admin only. |
| `explainability/` | Attention maps, feature attribution, permutation importance — used for model understanding and training data quality |
| `traceability/` | Traced outputs, metadata, reproducibility hashing |
| `exporters/` | ONNX, TorchScript, manifest exports |
| `logging/`, `utils/`, `config/`, `api/`, `cli/`, `plugins/`, `interfaces/`, `registry/` | Platform infrastructure |

---

### 3.2 CONVERT — From Decision Engine to Training Objective / Evaluation Dimension / Telemetry

These modules contain genuine domain knowledge that the model needs to learn. They must not be deleted. Their signal must be preserved — but as **training targets** or **evaluation dimensions**, not as runtime rule engines that override or supplement the model's output.

#### Execution — Decision Engines → Training Objectives

| Component | Current role | Target role | Training signal |
|---|---|---|---|
| `execution/stop_loss_engine.py` | Computes 6 ATR-multiple stop levels at runtime | **Training objective** | Label: "optimal exit given what happened next" — generated by `execution_simulator.py` in hindsight |
| `execution/take_profit_engine.py` | Computes 5 TP levels at runtime | **Training objective** | Label: "max favourable excursion in N bars" |
| `execution/risk_engine.py` | Computes EV and tail risk at runtime | **Evaluation dimension** | The model's `risk_signal` output is scored against actual trade outcomes |
| `execution/order_strategy.py` | Selects MARKET/LIMIT/STOP via if/else rules | **Training objective** | Label: "which order type achieved best fill quality" from simulator |
| `execution/entry_zones.py` | Detects S/R zones for entry placement | **Training feature** | Zone proximity is a model input feature, not an override |
| `execution/entry_quality.py` | Scores entry quality 0–100 | **Evaluation dimension** | Entry quality score is used to weight training examples |
| `execution/position_sizer.py` | Wrapper calling 10 sizing methods | **Training objective** | Label: "what size produced best risk-adjusted return" |
| `execution/execution_planner.py` | Plans full execution from rule outputs | **Evaluation dimension** | Plan quality evaluated against actual outcomes |
| `execution/execution_timing.py` | Recommends timing via rules | **Training objective** | Label: "sessions/times with historically better execution" |
| `execution/trade_manager.py` | Manages active trade recommendations | **Telemetry** | State transitions recorded as training context |
| `execution/trade_scorer.py` | Scores trades by rule criteria | **Evaluation dimension** | Scoring dimensions become model evaluation axes |

#### ICARI — Capital Allocation Engines → Training Objectives / Evaluation Dimensions

| Component | Current role | Target role | Training signal |
|---|---|---|---|
| `icari/position_sizing_engine.py` | 10 sizing formulas at runtime | **Training objective** | Label: "optimal size given outcomes" via counterfactual analysis |
| `icari/dynamic_sizer.py` | 10-factor geometric mean adjustment | **Training objective** | The composite factor becomes a training label |
| `icari/allocation_engine.py` | Allocates capital across instruments via rules | **Training objective** | Label: "allocation that maximised portfolio Sharpe over N bars" |
| `icari/risk_budget_engine.py` | Enforces risk budget via thresholds | **Evaluation dimension** | Budget compliance is measured, not enforced by rules |
| `icari/portfolio_risk_engine.py` | Computes portfolio-level risk via rules | **Evaluation dimension** | Portfolio risk is a dimension the model is evaluated against |
| `icari/correlation_engine.py` | Computes pairwise correlations | **Training feature** | Correlation matrix is a model input, not an override |
| `icari/scenario_engine.py` | Runs scenarios against rule-based sizing | **Validation** | Scenario outcomes validate model generalisation |
| `icari/capital_health.py` | Computes capital health by rules | **Telemetry** | Capital health is a conditioning signal for the model |
| `icari/capital_efficiency.py` | Computes capital efficiency by rules | **Evaluation dimension** | Efficiency is an evaluation axis, not a runtime adjuster |
| `icari/allocation_optimizer.py` | Optimises allocation by rules | **Training objective** | Optimal allocation becomes a training target |

#### MIE — Intelligence Engines → Training Features / Evaluation Dimensions

The MIE engines contain genuine signal. The signal is real. The mistake is that they are **decision systems** rather than **feature producers**. Convert them: strip the classification and threshold logic, keep the raw computations as features fed to the model.

| Component | Current role | Target role | What changes |
|---|---|---|---|
| `mie/market_regime_engine.py` | Classifies 16 regimes via ADX/vol/RSI thresholds | **Training feature producer** | Output raw: ADX value, vol_ratio, RSI, spread. Remove regime enum. Model learns what these values mean. |
| `mie/momentum_intelligence.py` | Classifies momentum into 5 states with lookup-table quality scores | **Training feature producer** | Output raw: momentum_strength, momentum_acceleration, rsi_signal. Remove MomentumState enum. |
| `mie/trend_intelligence.py` | Classifies trend into 5 states with hardcoded scoring | **Training feature producer** | Output raw: ADX, EMA cross gap, HH/HL bool, LL/LH bool. Remove TrendState enum and quality formula. |
| `mie/volatility_intelligence.py` | Classifies volatility regime | **Training feature producer** | Output raw: vol_ratio, ATR, current_vol, historical_vol. Remove VolatilityState enum. |
| `mie/liquidity_intelligence.py` | Classifies liquidity regime | **Training feature producer** | Output raw: avg_volume, bid_ask_spread, volume_ratio. Remove LiquidityState enum. |
| `mie/risk_intelligence.py` | Classifies risk intelligence state | **Training feature + evaluation dimension** | Raw risk metrics as features; risk outcomes as evaluation axis. |
| `mie/session_intelligence.py` | Classifies session with hardcoded session multipliers | **Training feature producer** | Output raw: session_id, hours_to_close, session_vol_hist. Remove session multipliers. |
| `mie/news_intelligence.py` | Classifies news impact | **Training feature producer** | Output raw: news_count, high_impact_count, news_sentiment. Remove impact classification. |
| `mie/execution_intelligence.py` | Classifies execution quality | **Training feature + telemetry** | Raw execution quality signals as inputs. |
| `mie/market_quality_engine.py` | Scores market quality 0–100 via rules | **Training feature producer** | Raw quality signal components as features. |
| `mie/tradeability_engine.py` | Rules-based tradeability gate | **Training feature producer** | Raw tradeability components as features. Model learns the gate. |

#### Portfolio — Portfolio Engines → Training Objectives / Evaluation Dimensions

| Component | Current role | Target role | Training signal |
|---|---|---|---|
| `portfolio/regime_detector.py` | Detects 8 portfolio regimes; applies hardcoded position-size multipliers (1.3× TRENDING, 0.6× HIGH_VOL, etc.) | **Training feature producer** | Raw: portfolio_vol, trend_R2, risk_sentiment_spread. Remove `REGIME_ADJUSTMENTS` lookup table. |
| `portfolio/risk_budget.py` | Enforces risk budget by thresholds | **Evaluation dimension** | Budget compliance is measured; model is trained to respect it |
| `portfolio/exposure_engine.py` | Computes and caps exposure by rules | **Telemetry + training context** | Current exposure is a conditioning input to the model |
| `portfolio/correlation_engine.py` | Computes cross-instrument correlations | **Training feature** | Correlation matrix is a model input |
| `portfolio/optimizer.py` | Optimises portfolio by mean-variance rules | **Training objective** | Optimal weights become training targets |
| `portfolio/scenario_analysis.py` | Runs scenarios against rule-based portfolio | **Validation** | Scenario outcomes validate model generalisation |
| `portfolio/portfolio_scorer.py` | Scores portfolio quality by rules | **Evaluation dimension** | Scoring dimensions become evaluation axes |
| `portfolio/position_allocator.py` | Allocates position sizes by rules | **Training objective** | Optimal allocation becomes training target |

#### Lifecycle — Transition Engine → Training Objectives / Telemetry

| Component | Current role | Target role | Training signal |
|---|---|---|---|
| `lifecycle/transition_engine.py` | Recommends HOLD/ADD/TRIM/EXIT/STOP via rule conditions | **Training objective** | Label: "transition that maximised outcome from this state" |
| `lifecycle/position_monitor.py` | Monitors position and recommends management via rules | **Telemetry + training context** | Position state is a model conditioning input |
| `lifecycle/lifecycle_confidence.py` | Computes confidence via rule formula | **Evaluation dimension** | Lifecycle confidence is an evaluation axis |
| `lifecycle/ai_reasoning.py` | Generates text reasoning about trade states | **Training output** | Model learns to produce reasoning as a supervised output |
| `lifecycle/trade_lifecycle.py` | Coordinates all lifecycle sub-engines | **Telemetry coordinator** | State transitions recorded; no rules override model |

---

### 3.3 REMOVE — Architecturally Incompatible

These modules implement the multi-model voting architecture. They cannot coexist with One Decision Maker. No conversion path exists — their function assumes multiple competing models.

| Component | Why REMOVE |
|---|---|
| `ensemble/engine.py` — `EnsembleEngine` | The central orchestrator of N competing models. Embodies everything the architecture prohibits. |
| `ensemble/expert.py` — `ExpertModel`, `ExpertRegistry` | The multi-model registry. "Expert 1 is never removed" is incompatible with One Model. |
| `ensemble/routing.py` — `ExpertRouter` | Routes prediction to different experts based on regime. Assumes N models exist. |
| `ensemble/strategies.py` — 8 strategy classes | `SimpleAveraging`, `WeightedAveraging`, `HardVoting`, `SoftVoting`, `MixtureOfExperts`, `Stacking`, `Blending`, `PerformanceWeighted`. Voting assumes multiple outputs to combine. |
| `ensemble/meta_model.py` — `EnsembleMetaModel` | A model that overrides other models. Violates One Decision Maker. |
| `ensemble/weighting.py` — `DynamicWeightEngine` | Computes weights across N experts. No purpose with one model. |
| `ensemble/calibration.py` — `ExpertCalibrator` | Per-expert calibration. With one model, use `trainer/calibration/` directly. |
| `ensemble/failure_analysis.py` — `ExpertFailureAnalyzer` | Tracks which expert failed on which regime. No experts means no role. |
| `ensemble/online_scoring.py` — `OnlineScorer` | Scores N experts online. Use `mlops/monitoring.py` for one model. |
| `ensemble/leaderboard.py` — `EnsembleLeaderboard` | Expert ranking system. Incompatible with One Model. |
| `ensemble/database.py` — `EnsembleDatabase` | Persistence for multi-expert voting history. |
| `ensemble/api.py` — `EnsembleAPI` | API for multi-expert dashboards. |
| `ensemble/export.py` — `EnsembleExporter` | Exports ensemble of N models. Use `exporters/` for one model. |
| `ensemble/visualization.py` — `EnsembleVisualizationBuilder` | Expert comparison dashboards. |
| `qde/qde_ab_testing.py` — CANARY mode | CANARY routes live decisions to a challenger. Only SHADOW observation mode is retained. |

---

### 3.4 ARCHIVE — No longer needed but contains historical knowledge

These components are not architecturally incompatible, but their function is fully superseded by CONVERT outputs or KEEP infrastructure. Archive before deletion to preserve institutional knowledge.

| Component | Archive Reason |
|---|---|
| `portfolio/regime_detector.py` REGIME_ADJUSTMENTS table | Contains domain knowledge about regime effects. Archive the table as a reference for training label generation before the class is converted. |
| `execution/stop_loss_engine.py` ATR multiplier constants | The 1.5×/2.0×/3.5× ATR heuristics represent domain knowledge about stop placement. Archive as baseline targets for training label generation. |
| `execution/take_profit_engine.py` RR constants | The 1:1, 1.5:1, 2:1 RR levels are initial label seeds for TP training objectives. |
| `icari/position_sizing_engine.py` 10 sizing formulas | Each formula represents a validated sizing heuristic. Archive as candidate training targets for the sizing objective. |
| `ensemble/` entire module | Archive before removal. The leaderboard data, voting history, and expert performance records may be useful as historical training data. |

---

## 4. Migration Complexity

| Component Group | Files | Complexity | Notes |
|---|---|---|---|
| Remove `ensemble/` | 14 | **Low** | No consumers in KEEP modules. Isolated. |
| Remove CANARY from `qde_ab_testing.py` | 1 | **Low** | Shadow mode retained, canary routing removed |
| Convert `mie/` engines to feature producers | 11 | **Medium** | Each engine's raw outputs must be defined and wired into the feature pipeline. Threshold logic removed, raw computations kept. |
| Convert `execution/` engines | 10 | **High** | Requires training labels generated by `execution_simulator.py` in hindsight mode. New multi-task output heads needed. |
| Convert `icari/` sizing engines | 8 | **High** | Requires counterfactual analysis to generate size labels. New model output head needed. |
| Convert `portfolio/` engines | 6 | **Medium** | Portfolio context becomes conditioning input; allocation becomes training target. |
| Convert `lifecycle/transition_engine.py` | 1 | **High** | Requires state-aware training — the model must receive trade state as input and predict optimal transition. |
| Add multi-task output heads to QDE model | 1 | **High** | Model architecture change: single head → N heads (direction, size, urgency, risk, hold, exit). |
| Retrain QDE on multi-dimensional labels | — | **Very High** | Requires: label generation pipeline, multi-task losses, multi-task validation, cross-head calibration. |
| Update QDE promotion criteria | 1 | **Low** | Remove hardcoded floors; promotion criteria become configurable + distribution-relative. |

---

## 5. Dependencies

### What REMOVE depends on (must remove first)

```
ensemble/engine.py
  └─ ensemble/expert.py
  └─ ensemble/routing.py
  └─ ensemble/strategies.py
  └─ ensemble/meta_model.py
  └─ ensemble/weighting.py
  └─ ensemble/calibration.py
  └─ ensemble/failure_analysis.py
  └─ ensemble/online_scoring.py
  └─ ensemble/leaderboard.py
  └─ ensemble/database.py
  └─ ensemble/api.py
  └─ ensemble/export.py
  └─ ensemble/visualization.py
```

The ensemble module is self-contained. Grep confirms no KEEP module imports from `trainer.ensemble`. The QDE pipeline currently does not import ensemble — the ensemble was designed as a parallel system.

### What CONVERT engines currently feed (dependencies to migrate)

```
mie/market_regime_engine.py
  → ensemble/routing.py (REMOVE — this dependency dissolves)
  → portfolio/regime_detector.py (CONVERT — becomes feature input)
  → mie/mie_worker.py (KEEP — worker continues, outputs change to raw features)

execution/stop_loss_engine.py
  → execution/take_profit_engine.py (CONVERT)
  → execution/risk_engine.py (CONVERT)
  → execution/execution_planner.py (CONVERT)

icari/position_sizing_engine.py
  → icari/dynamic_sizer.py (CONVERT)
  → icari/allocation_engine.py (CONVERT)

lifecycle/transition_engine.py
  → lifecycle/trade_lifecycle.py (KEEP as telemetry coordinator)
  → lifecycle/position_monitor.py (CONVERT)
```

### What KEEP modules depend on CONVERT modules (temporary coupling to manage)

| KEEP module | CONVERT dependency | Migration action |
|---|---|---|
| `research_lab/research_lab_api.py` | References `execution/` types for backtest registration | Add abstract types layer; decouple from engine classes |
| `validation/trading.py` | Uses `execution/trade_scorer.py` for metric computation | Migrate scoring to pure metric functions |
| `cert/benchmark_engine.py` | May reference `execution/` for benchmark definitions | Decouple via interface |
| `lifecycle/trade_lifecycle.py` | Currently instantiates `TransitionEngine`, `PositionMonitor` | Refactor: lifecycle records telemetry, does not call engines |

---

## 6. Risk Assessment

### R1 — Training Label Quality [HIGH]

**Risk:** The CONVERT engines currently encode the output of many years of domain knowledge (ATR stop multipliers, Kelly sizing, etc.). When these become training targets, the quality of the training labels determines whether the model learns better or worse behaviour than the rules it replaces.

**Mitigation:** Use `execution_simulator.py` to generate hindsight-optimal labels (e.g., "what stop level would have maximised PnL given what actually happened?") rather than using the rule outputs as labels. Compare simulator-generated labels against rule-engine labels before training.

### R2 — Model Architecture Scope [HIGH]

**Risk:** Adding 5–6 output heads to the QDE model (direction, size, urgency, exit, risk, hold duration) significantly increases model complexity, training time, and risk of head interference.

**Mitigation:** Introduce heads progressively. Start with direction + size. Validate each head independently before adding the next. Use gradient uncertainty weighting (e.g., GradNorm or uncertainty weighting) to balance multi-task losses.

### R3 — Ensemble Removal Before Model is Ready [CRITICAL]

**Risk:** If the ensemble is removed before the one model is trained to handle sizing, exit, and risk, live trading will have no sizing or exit logic at all.

**Mitigation:** Removal is the final migration step. All CONVERT training must complete and validate before `ensemble/` is removed from production paths. Run in parallel (one model + ensemble both running in shadow) during transition.

### R4 — MIE Feature Contract [MEDIUM]

**Risk:** Removing regime labels from MIE engines and replacing with raw features changes the downstream feature contract. Any system that expects `MarketRegime.STRONG_TREND` instead of raw `adx=47.2, vol_ratio=0.8` will break.

**Mitigation:** Define a `RawMarketFeatures` dataclass that replaces `MarketRegimeResult`. Update `mie_worker.py` to produce it. Update the feature store schema. All consumers receive raw features.

### R5 — Promotion Criteria Regression [LOW]

**Risk:** Removing hardcoded promotion gates (`min_sharpe=0.5`, `min_win_rate=0.45`) without replacing them may allow a broken model to promote.

**Mitigation:** The Research Lab promotion workflow requires explicit admin approval regardless of gate values. Gates become advisory, not blocking. Admin retains veto.

### R6 — Data Drift in Historical Decision Memory [MEDIUM]

**Risk:** The `memory/` module contains decision records generated by the ensemble system. These records have `decision_type`, `reasoning`, and `outcome` fields that reflect ensemble logic, not single-model logic. Using these as training data will teach the new model ensemble behaviour.

**Mitigation:** Tag all pre-migration decision records with `source=ensemble_era`. Exclude them from training by default. Use only post-migration records for supervised training of the new model.

---

## 7. Migration Order

### Phase 1 — Prepare (no live impact)

1. **Archive** `ensemble/` module data (export leaderboard, voting history, expert performance records to Research Lab as historical datasets)
2. **Archive** hardcoded constants from CONVERT engines to `docs/archived_heuristics.md` (ATR multipliers, RR ratios, sizing formulas, regime adjustment tables)
3. **Define** `RawMarketFeatures` dataclass to replace regime enums throughout MIE
4. **Define** multi-task output schema for the new QDE model heads
5. **Generate** hindsight training labels using `execution_simulator.py` in hindsight mode over historical data

### Phase 2 — Convert MIE (low risk, high value)

6. **Convert** `mie/market_regime_engine.py` → raw feature producer. Remove regime enum. Retain ADX, vol_ratio, RSI, spread as outputs.
7. **Convert** `mie/momentum_intelligence.py` → raw feature producer. Remove `MomentumState` enum. Retain strength, acceleration, rsi_signal.
8. **Convert** `mie/trend_intelligence.py` → raw feature producer. Remove `TrendState` enum. Retain ADX value, EMA gap, HH/HL booleans.
9. **Convert** remaining 8 MIE intelligence engines to raw feature producers.
10. **Update** `mie/feature_store.py` to store `RawMarketFeatures`.
11. **Update** `portfolio/regime_detector.py` to produce raw vol/trend/sentiment features; remove `REGIME_ADJUSTMENTS` table.
12. **Validate** that all features round-trip correctly through the updated feature store.

### Phase 3 — Train Multi-Task QDE (research environment only)

13. **Add** multi-task output heads to QDE model architecture: direction · size\_signal · entry\_urgency · exit\_horizon · risk\_signal.
14. **Add** multi-task loss functions using `losses/multi_task.py`.
15. **Train** QDE v2 on direction + size objectives first. Validate against incumbent using Research Lab backtesting.
16. **Add** exit/timing/risk heads progressively. Validate each before adding the next.
17. **Run** full certification suite (`cert/`) against multi-task QDE v2.
18. **Shadow-deploy** QDE v2 alongside ensemble (observation only, no live execution impact).

### Phase 4 — Convert Execution & ICARI (after Phase 3 validates)

19. **Convert** `execution/stop_loss_engine.py` → training objective only. Remove from runtime inference path.
20. **Convert** `execution/take_profit_engine.py` → training objective only. Remove from runtime inference path.
21. **Convert** `execution/order_strategy.py` → training objective only. Remove from runtime inference path.
22. **Convert** `icari/position_sizing_engine.py` → training objective only. Remove from runtime inference path.
23. **Convert** `icari/dynamic_sizer.py` → training objective only. Remove from runtime inference path.
24. **Convert** `lifecycle/transition_engine.py` → training objective only.
25. QDE v2 now handles all outputs that were previously handled by rule engines.

### Phase 5 — Remove Ensemble (after Phase 4 validates in shadow)

26. **Confirm** QDE v2 shadow performance meets or exceeds ensemble on all dimensions: direction accuracy, sizing quality, exit quality, portfolio risk.
27. **Promote** QDE v2 to ACTIVE through Research Lab promotion workflow (admin approval required).
28. **Remove** `ensemble/` module entirely (14 files).
29. **Remove** CANARY mode from `qde/qde_ab_testing.py`.
30. **Remove** ensemble imports from any remaining consumers.

### Phase 6 — Stabilise

31. **Update** QDE promotion criteria — remove hardcoded floors; make all thresholds configurable per-deployment.
32. **Tag** pre-migration decision records in `memory/` as `source=ensemble_era` to prevent contamination of future training.
33. **Run** full certification suite against the clean single-model architecture.
34. **Update** all documentation, API contracts, and admin dashboards to reflect the single-model architecture.

---

## Summary

| Classification | Module count | File count |
|---|---|---|
| **KEEP** | 25 modules | ~310 files |
| **CONVERT** | 7 modules (partial) | ~70 files |
| **REMOVE** | 1 module (`ensemble/`) + CANARY mode | 14 files |
| **ARCHIVE** | Cross-cutting constants and historical data | — |

The migration is achievable in 6 phases. The highest-risk step is Phase 3 (training the multi-task model) and Phase 5 (removing the ensemble). Both are gated on explicit validation in shadow before any live impact.

The ensemble removal is the endpoint, not the starting point. Infrastructure first. Labels second. Training third. Validation fourth. Promotion fifth. Removal last.
