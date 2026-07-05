---
name: Traceability Layer (Phase 14.7)
description: Pure-decorator prediction metadata system; TracedInferenceAdapter wraps any adapter without mutating predictions.
---

## Rule
`TracedInferenceAdapter` is a pure decorator over any adapter with `predict(x) -> dict`.
It returns `TracedPrediction(predictions=<original dict unchanged>, metadata=PredictionMetadata)`.
Prediction values are NEVER touched.

**Why:** The spec requires zero prediction mutation. The decorator pattern isolates
metadata logic from model and inference code completely.

## How to apply
- Wrap at the adapter level, not in the model.
- `TraceabilityConfig` carries all static fields (versions, IDs, hashes).
- `InferenceContext` captures dynamic environment (git hash, pytorch/python version, device) once at construction time.
- `MetadataManifestBuilder` (subclass of ManifestBuilder) embeds traceability into export manifests — backward-compatible.

## trainer_version
`TraceabilityConfig.trainer_version` defaults to `"0.1.0"` to match `trainer.__version__`.
Keep these in sync when the package version is bumped.
