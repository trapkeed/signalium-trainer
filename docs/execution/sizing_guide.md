# Position Sizing Guide

## Overview

The `PositionSizer` supports four sizing methods, all governed by configurable `RiskLimits` and `BrokerConstraints`.

## Methods

### 1. Fixed Risk (default)

Risk a fixed dollar amount per trade. The lot size is derived from the stop distance.

```
risk_amount = account_size × risk_pct_per_trade
lots = risk_amount / (units_per_lot × stop_distance)
```

### 2. Fixed Fractional

Risk 1% of account regardless of stop size. Equivalent to fixed risk with a fixed 1% rate.

### 3. Volatility Adjusted

Scale lot size inversely to current volatility. Higher volatility → smaller position.

```
base_lots = risk_amount / (units_per_lot × price × base_vol)
lots = base_lots × (base_vol / current_vol)   [capped 0.1×–3.0×]
```

### 4. Kelly Fraction

Uses the Kelly criterion to size based on edge and odds, with a 50% Kelly discount for safety.

```
Kelly_f = p − q / RR    where p = win_prob, q = 1−p, RR = expected_return / expected_risk
lots = fixed_risk(account × Kelly_f × 0.5)
```

## Risk Limits

```yaml
max_risk_per_trade_pct: 0.02    # 2% hard cap per trade
max_daily_risk_pct:     0.06    # 6% per calendar day
max_weekly_risk_pct:    0.12    # 12% per rolling week
max_portfolio_risk_pct: 0.20    # 20% total open risk
```

The effective risk per trade is `min(per_trade, available_daily, available_weekly, available_portfolio)`.

## Broker Validation

After calculating the raw lot size, the result is clipped and rounded to broker step:

```python
lots = max(min_lot, min(max_lot, lots))
lots = round(round(lots / lot_step) * lot_step, 4)
```

## Usage

```python
from trainer.execution import PositionSizer, RiskLimits, SizingMethod

sizer = PositionSizer(risk_limits=RiskLimits(max_risk_per_trade_pct=0.01))

result = sizer.calculate(
    forecast  = fc,
    stop_loss = stop,
    method    = SizingMethod.FIXED_RISK,
)

print(result.lots)        # e.g. 0.05
print(result.risk_pct)    # e.g. 0.0087 (0.87%)
print(result.kelly_fraction)  # informational
```

## Output: SizingResult

| Field | Description |
|-------|-------------|
| `lots` | Final validated lot size |
| `units` | lots × 100,000 |
| `risk_amount` | USD at risk |
| `risk_pct` | Fraction of account |
| `stop_distance` | Distance to initial stop |
| `kelly_fraction` | Raw Kelly fraction (info only) |
| `margin_required` | Margin required at broker margin rate |
| `commission_cost` | Round-trip commission estimate |
| `size_reduced` | True if broker/limits forced a reduction |
