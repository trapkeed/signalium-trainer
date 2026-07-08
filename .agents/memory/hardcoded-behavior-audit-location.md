---
name: Hardcoded trading behavior audit location
description: Where the canonical audit of hardcoded trading rules (thresholds, multipliers, allocation weights) lives and how to keep it current
---

The canonical "hardcoded trading behavior" audit deliverable is `docs/qde/hardcoded_behavior_audit.md`.
It catalogs hardcoded confidence thresholds, ATR/stop/TP multipliers, breakeven/trailing rules,
position sizing, regime/indicator thresholds, and allocation-weight formulas across `trainer/`,
along with a conversion plan (which trainer artifact — training label, training objective,
evaluation metric, benchmark — should replace each hardcoded rule) and its relationship to the
`trainer/learning_loop/actuators.py` layer.

**Why:** When asked to (re-)audit hardcoded trading behavior, this file already exists and is
comprehensive — re-running a full codebase audit from scratch duplicates work and risks drift
between two competing "source of truth" documents.

**How to apply:** Read this file first, verify/spot-check claims against current code (line
numbers and constants can drift as the codebase changes), and extend it in place with new
findings rather than creating a second audit file.
