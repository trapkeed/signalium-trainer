---
name: Quant Model Phase Architecture
description: Key design decisions for Phase 14.5 and 14.6 multi-task head extensions to QuantModelV1
---

## Phase 14.5 (5 optional heads)
ExpectedMove, StopLoss, TakeProfit, HoldingTime, TPProbability

## Phase 14.6 (15 institutional heads)
MFE, MAEExcursion, TradeEfficiency, RewardRisk, TradeSurvival,
SLResilience, TPAchievability, EntryQuality, ExitQuality,
MarketStability, TradeDecay, ProfitDistribution, LossDistribution,
PositionSizeQuality, Uncertainty

## Critical invariants
- `forward()` always returns direction logits only (unchanged from pre-14.5)
- `forward_multi_task()` returns a dict; all heads disabled by default
- All heads share a single `_encode()` backbone pass — zero re-computation
- Non-negative heads use Softplus (not ReLU — avoids dead neurons)
- Score heads (0-100) use 100 × Sigmoid
- MarketStability forward() returns logits; predict_proba() returns softmax
- PositionSizeQuality is a quality score ONLY — never a lot size

## Loop-based instantiation pattern
model.py uses `_all_optional: list[tuple[Optional[nn.Module], str]]` with
a single `for head, key in [*_p14_5, *_p14_6]` loop in forward_multi_task.
This is correct even though grep shows 1 match for `out[key]`.

## MultiTaskLossOutput field count
23 total: total, direction, + 21 optional (7 from 14.5, 14 from 14.6).
Wait — 7 from 14.5 (conf, move, sl, tp, ht, tp_prob) + 15 from 14.6 = 21 optional.

## Calibration
trainer/calibration/ package: TemperatureScaling (LBFGS), PlattScaling (Adam),
IsotonicCalibration (sklearn wrapper). CalibrationLoss in losses/advanced.py.
