---
name: Structural vs flow liquidity split
description: The two distinct liquidity subsystems and why they must stay separate.
---

## Rule
`mie/liquidity_intelligence.py` = **flow liquidity** (is the market liquid enough to trade *right now*?).  
`trainer/qde/brain/liquidity/` = **structural liquidity** (where do price levels exist, where have they been taken, where may they exist next?).

Never merge or conflate the two. `SymbolObservationInput.liquidity_state/quality` carries flow-liquidity values from the MIE engine. The `liq_*` brain_context keys carry structural-liquidity values from `LiquidityMapEngine`.

**Why:** Flow liquidity affects whether an entry is executable now (spread/volume). Structural liquidity affects where the price is likely to react, where stops may cluster, and which levels have been resolved. Conflating them would mis-route both sets of signals.

**How to apply:** Any new feature that asks "is there enough volume to trade?" → touch `LiquidityIntelligenceEngine`. Any feature that asks "where are the structural levels?" → touch `LiquidityMapEngine`.
