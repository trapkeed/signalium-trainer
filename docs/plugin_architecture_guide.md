# Signalium Quant — Plugin Architecture Guide

**Phase 14.9C** — Encoder Plugin System

---

## Overview

The plugin architecture allows new data-stream encoders to be registered and used in the model without modifying any core model code. This is the mechanism by which future data sources (news, macro, order book, sentiment, etc.) will be integrated.

**Key principle:** The core model's `_encode()`, `forward()`, and all 22 prediction heads are never touched. New encoders plug in as additional streams to the existing `AdaptiveFusionLayer`.

---

## How the Plugin System Works

```
PluginRegistry
    │
    ├── register("news_encoder")  ← decorator on encoder class
    ├── get("news_encoder")       ← retrieve EncoderPlugin wrapper
    └── instantiate(**kwargs)     ← create encoder instance
                │
                ▼
        BaseExternalEncoder  (trainer/interfaces/base.py)
                │
                ├── name        ← unique registry key
                ├── output_dim  ← must match fusion.latent_dim
                ├── forward(**kwargs) → (batch, output_dim)
                └── get_metadata()   → {version, status, ...}
                            │
                            ▼
               HybridEncoder.fusion  (AdaptiveFusionLayer)
                            │
                            ▼
                   Shared Latent Representation
                            │
                   All 22 Prediction Heads (unchanged)
```

---

## Registering a Plugin

### Using the Decorator (recommended)

```python
from trainer.plugins.registry import PluginRegistry
from trainer.interfaces.base import BaseExternalEncoder
import torch

@PluginRegistry.register(
    name="news_encoder",
    version="1.0.0",
    description="FinBERT-based news sentiment encoder",
    status="experimental",
    tags=["nlp", "sentiment", "news"],
)
class NewsEncoder(BaseExternalEncoder):

    def __init__(self, hidden_dim: int = 128) -> None:
        self._output_dim = hidden_dim
        # ... initialise FinBERT or embedding model ...

    @property
    def name(self) -> str:
        return "news_encoder"

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        token_ids: torch.Tensor,      # (batch, n_articles, seq_len)
        attention_mask: torch.Tensor, # (batch, n_articles, seq_len)
        **kwargs,
    ) -> torch.Tensor:
        # ... encode news → (batch, output_dim) ...
        raise NotImplementedError("News encoder not yet implemented")

    def get_metadata(self) -> dict:
        return {
            "version": "1.0.0",
            "status": "experimental",
            "backbone": "finbert-base",
            "input_schema": {"token_ids": "(B, N, L)", "attention_mask": "(B, N, L)"},
        }
```

### Using the Explicit Registration Form

```python
PluginRegistry.register_class(
    encoder_class=MyEncoder,
    name="my_encoder",
    version="2.0.0",
    status="production",
)
```

---

## Discovering and Using Plugins

```python
from trainer.plugins import PluginRegistry

# List all registered plugins
print(PluginRegistry.list_plugins())
# ["cross_asset", "news_encoder", "order_book"]

# Filter by status
print(PluginRegistry.list_by_status("production"))

# Inspect metadata
info = PluginRegistry.describe("news_encoder")
# {"name": "news_encoder", "version": "1.0.0", "class": "NewsEncoder", ...}

# Check existence before use
if PluginRegistry.is_registered("news_encoder"):
    plugin = PluginRegistry.get("news_encoder")
    encoder = plugin.instantiate(hidden_dim=256)

# Validate encoder output shape before passing to fusion
embedding = encoder.forward(token_ids, attention_mask)
assert encoder.validate_output(embedding, batch_size=x.shape[0])
```

---

## Output Shape Contract

Every plugin MUST output a tensor of shape `(batch_size, output_dim)` where:
- `batch_size` matches the input batch
- `output_dim` matches `encoder.output_dim` (validated by `validate_output()`)
- `output_dim` must equal the `fusion.latent_dim` configured in `HybridEncoderConfig`

```python
# Correct: encoder contracts are enforced at boundary
embedding = encoder.forward(data)
if not encoder.validate_output(embedding, batch_size):
    raise ValueError(f"Encoder {encoder.name!r} returned wrong shape: {embedding.shape}")
```

---

## Plugin Lifecycle States

| Status | Meaning |
|---|---|
| `"future"` | Interface stub — no implementation |
| `"experimental"` | Research prototype — not production-ready |
| `"production"` | Fully tested and deployed |

---

## EncoderPlugin Metadata

Each registration produces an `EncoderPlugin` wrapper:

```python
plugin = PluginRegistry.get("news_encoder")

plugin.name          # "news_encoder"
plugin.version       # "1.0.0"
plugin.encoder_class # NewsEncoder
plugin.description   # human-readable description
plugin.status        # "experimental"
plugin.tags          # ["nlp", "sentiment"]
plugin.registered_at # ISO-8601 UTC timestamp

plugin.instantiate(hidden_dim=128)  # → NewsEncoder instance
plugin.to_dict()                     # JSON-safe metadata dict
```

---

## Adding a Plugin to Training

Once registered, a plugin encoder is connected to the model by adding its output as a stream to the `HybridEncoder.fusion` layer. This requires a configuration update — not a model code change:

```yaml
# conf/model/quant_v1_hybrid_with_news.yaml
hybrid_encoder:
  enabled: true
  # ... existing streams ...
  fusion:
    strategy: attention        # or weighted or concat
    latent_dim: 256            # must match plugin.output_dim
    # Future: plugin streams listed here
```

The plugin encoder is called before the fusion layer and its output is appended to `stream_outputs`. No changes to `engine.py`, `model.py`, or any prediction head.

---

## Testing a Plugin

```python
import torch
from trainer.plugins.registry import PluginRegistry

# Register
@PluginRegistry.register("test_enc", version="0.1.0")
class TestEncoder(BaseExternalEncoder):
    @property
    def name(self): return "test_enc"
    @property
    def output_dim(self): return 64
    def forward(self, x, **kw): return torch.zeros(x.shape[0], 64)
    def get_metadata(self): return {"version": "0.1.0", "status": "test"}

# Smoke test
enc = PluginRegistry.get("test_enc").instantiate()
dummy = torch.randn(4, 10, 5)
out = enc.forward(dummy)
assert enc.validate_output(out, batch_size=4), "Shape mismatch"
print("Plugin smoke test passed")

# Clean up test registry
PluginRegistry.unregister("test_enc")
```
