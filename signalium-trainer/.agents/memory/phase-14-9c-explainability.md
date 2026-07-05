---
name: Phase 14.9C explainability design
description: Key design decisions and gotchas in the explainability/plugin layer built in Phase 14.9C.
---

## Attribution methods — grad management
All attribution classes use `_grad_mode(model)` context manager (in attribution.py) which: (1) sets eval mode, (2) enables grad via `torch.enable_grad()`, (3) restores training state in finally. This means they work correctly even when called inside a `torch.no_grad()` block since `enable_grad` overrides it in PyTorch.

**Why:** Standard pattern for torch.autograd.grad() — it needs grad enabled on the tensors being differentiated, not just on the model.

## Attention hooks — leak prevention
`AttentionExtractor` uses `register_forward_hook` on every `nn.MultiheadAttention` sub-module. Hooks are always removed in a `finally` block inside the context manager. For LSTM/GRU/TCN models, no hooks are registered and an empty `AttentionMap` is returned — never raises.

**Why:** Leaked hooks accumulate on the module and cause memory leaks plus double-capture in subsequent calls.

## PluginRegistry is class-level
`PluginRegistry._plugins` is a class-level dict (not instance). Tests MUST use `PluginRegistry.clear()` in an autouse fixture to prevent cross-test pollution. The decorator executes at import time.

**Why:** Class-level allows decorator registration without instantiation; autouse fixture isolation is the correct test pattern.

## ModelInfo extension
Added 5 new fields (phase, backbone_type, active_streams, fusion_strategy, encoder_versions) to ModelInfo with safe defaults. All have default values so `ModelInfo()` still works without arguments.

## future_encoders — interface stubs only
All 9 future encoder classes in `trainer/interfaces/future_encoders.py` are abstract stubs — no implementation. They inherit BaseExternalEncoder which is ABC, so instantiation raises TypeError. The `name` property is a non-abstract concrete property on these stubs (returns a fixed string). The abstract methods are the domain-specific encode_*() methods.
