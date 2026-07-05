# Execution Intelligence Guide

## Overview

Phase 21 introduces Signalium's Institutional Execution Intelligence Platform — a pure analytics layer that converts high-quality forecasts into institutional-grade execution recommendations.

**Critical rule:** This system does NOT place trades. It does NOT connect to brokers. All outputs are recommendations.

## Architecture

```
ExecutionForecast
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  ExecutionPlanner (Orchestrator)                    │
│                                                     │
│  EntryZoneEngine  →  EntryZones                    │
│  EntryQualityEngine → EntryQualityResult           │
│  StopLossEngine   →  StopLossRecommendation        │
│  TakeProfitEngine →  TakeProfitRecommendation      │
│  PositionSizer    →  SizingResult                  │
│  TradeManager     →  ManagementPlan                │
│  ExecutionTiming  →  TimingAssessment              │
│  RiskEngine       →  RiskProfile                   │
│  OrderStrategyEngine → OrderRecommendation         │
│  TradeScorer      →  TradeScore                    │
│  ExecutionSimulator → SimulationResult             │
│                                                     │
│         ExecutionPlan (complete output)            │
└─────────────────────────────────────────────────────┘
       │
       ▼
ExecutionDatabase  →  SQLite persistence
ExecutionDashboard →  8 chart views
ExecutionApi       →  16 endpoints
```

## Core Input: ExecutionForecast

```python
ExecutionForecast(
    instrument      = "EURUSD",
    direction       = "BUY",          # BUY or SELL
    confidence      = 0.75,           # 0-1
    expected_return = 0.08,           # 8%
    expected_risk   = 0.10,           # 10%
    current_price   = 1.0850,         # current market price
    atr             = 0.0012,         # Average True Range in price units
    spread          = 0.0001,         # bid-ask spread in price units
    session         = Session.LONDON, # current trading session
    volatility      = 0.009,          # realised volatility
    momentum        = 0.30,           # -1 to +1
    news_risk       = 0.10,           # 0 = none, 1 = extreme
    account_size    = 10_000.0,       # USD account size
)
```

## Quick Start

```python
from trainer.execution import ExecutionApi

api = ExecutionApi(db_path="data/execution.db")

resp = api.plan({
    "instrument":      "EURUSD",
    "direction":       "BUY",
    "confidence":      0.75,
    "expected_return": 0.08,
    "expected_risk":   0.10,
    "current_price":   1.0850,
    "atr":             0.0012,
    "spread":          0.0001,
    "session":         "LONDON",
    "volatility":      0.009,
    "momentum":        0.30,
    "news_risk":       0.10,
    "account_size":    10000.0,
})

summary = resp["data"]["executive_summary"]
# {instrument, direction, executable, timing, entry_grade,
#  trade_grade, trade_score, primary_entry, initial_stop,
#  tp1, tp2, runner, lots, risk_pct, risk_reward, ...}
```

## Execution Workflow

1. **Entry Zones** — ATR-based pullback zones (Aggressive → Conservative)
2. **Entry Quality** — 5-factor scoring (spread, session, momentum, vol, zone)
3. **Stop Loss** — 6 variants (Initial, Adaptive, ATR, Structure, Volatility, Emergency)
4. **Take Profit** — TP1/TP2/TP3/Runner with hit probabilities
5. **Position Sizing** — 4 methods with risk limit enforcement
6. **Trade Management** — breakeven, trail, partial close, scale-in rules
7. **Execution Timing** — EXECUTE_NOW / WAIT / AVOID recommendation
8. **Risk Profile** — P(success), P(failure), RR, EV, tail risk
9. **Order Strategy** — Market / Limit / Stop / Conditional selection
10. **Trade Score** — 8-factor composite grade (EXCELLENT → REJECT)
11. **Simulation** — fill quality, slippage, scenario outcomes

## Module List (16 modules)

| Module | Purpose |
|--------|---------|
| `execution_types.py` | Core enums and dataclasses |
| `entry_zones.py` | Multi-level entry zone calculation |
| `entry_quality.py` | Entry quality classification |
| `stop_loss_engine.py` | Full stop loss suite |
| `take_profit_engine.py` | Structured TP levels |
| `position_sizer.py` | 4 sizing methods + risk limits |
| `trade_manager.py` | Active trade management rules |
| `execution_timing.py` | Timing evaluation |
| `risk_engine.py` | Risk profile analysis |
| `order_strategy.py` | Order type selection |
| `trade_scorer.py` | Execution quality score |
| `execution_simulator.py` | Fill and outcome simulation |
| `execution_planner.py` | Central orchestrator |
| `execution_database.py` | SQLite persistence |
| `execution_dashboard.py` | 8 chart views |
| `execution_api.py` | 16 API endpoints |
