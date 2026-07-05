---
name: Phase 14.9B infrastructure gaps
description: Critical gaps found and fixed in Phase 14.9B; durable constraints for future backbone/optimizer/scheduler additions.
---

## Critical gap: trainer/checkpoints/ was missing
Both TrainingEngine and MultiTaskTrainingEngine imported `from trainer.checkpoints.manager import CheckpointManager` but the directory didn't exist. Created trainer/checkpoints/__init__.py + manager.py. The load() signature makes optimizer and scheduler Optional to preserve backward compat with existing test_checkpoint_manager.py.

**Why:** The engines were written expecting the package to exist, but it was never created.

## Gradient accumulation was in schema but unwired
`TrainingConfig.accumulate_grad_batches` existed in schema.py but neither engine used it. Fix: divide loss by accum_steps, zero_grad at window start, optimizer.step() at window end or last batch.

**Why:** Loss scale must be invariant to accum_steps; gradient equivalence to full batch verified by test.

## warmup_epochs was in SchedulerConfig but ignored
The engines built schedulers ignoring warmup_epochs. Now engines check cfg.warmup_epochs > 0 and wrap with WarmupScheduler. WarmupScheduler has is_plateau flag for ReduceLROnPlateau compat.

## Backbone factory pattern
build_backbone(input_dim, config) in backbone.py is the only place to add new backbones. BackboneType enum in quant_v1/config.py must also be extended. Transformer and Hybrid support gradient_checkpointing via BackboneConfig flag — handled at backbone level, not engine level.

## Python not available in Nix shell
`python3 -c` and `python -c` fail with ModuleNotFoundError for torch. Use static analysis (grep, read) for verification in this environment.
