---
name: Cert operational scoring
description: How OperationalEngine.overall_score() weights mandatory vs non-mandatory failures
---

## Rule
In `OperationalEngine._check()`, a False operational check becomes:
- `CheckStatus.FAIL` if the check is mandatory (`_MANDATORY[name] == True`)
- `CheckStatus.WARNING` if the check is non-mandatory

In `overall_score()`, mandatory FAIL → weight 0.0 (hard block). Non-mandatory WARNING → weight 55.0.

## Why
Non-mandatory failures (BACKUPS_VERIFIED, DOCUMENTATION_COMPLETE, RECOVERY_PROCEDURES_VERIFIED) are important but not deployment blockers. Setting them to WARNING instead of FAIL allows the overall score to reflect "mostly ready with gaps" rather than binary rejection.

## How to apply
When writing test assertions for `overall_score` with all-False checks:
- Mandatory (5): weight 0 → contribute 0
- Non-mandatory (3): weight 55 → contribute 165
- Score = 165/8 = **20.625** — NOT ≤ 10.0
- Safe assertion: `assertLessEqual(score, 30.0)` and compare against all-pass score

## All-pass score
With all 8 checks passing → score = 100.0 (weight 100 each). Gap between all-fail and all-pass is always > 60 points.
