# Signalium Quant — Future Expansion Guide

**Phase 14.9C** — Future-Ready Architecture Interfaces

---

## Overview

Phase 14.9C prepares the architecture for future research phases by defining abstract interface contracts for nine future encoder types. No implementations are provided — only the contracts that future implementations must satisfy.

**Critical constraint:** Do NOT implement reinforcement learning, graph neural networks, or macroeconomic models in the current codebase. These interfaces are stubs only.

---

## Available Future Interfaces

All interfaces live in `trainer/interfaces/future_encoders.py` and inherit from `BaseExternalEncoder` (`trainer/interfaces/base.py`).

| Interface | Registry Key | Planned Data Source |
|---|---|---|
| `BaseReinforcementLearningInterface` | `reinforcement_learning` | RL policy / value embedding |
| `BaseCrossAssetEncoder` | `cross_asset` | Multi-asset price + correlations |
| `BaseNewsEncoder` | `news` | NLP-encoded financial news |
| `BaseMacroEncoder` | `macro` | Macroeconomic indicators (CPI, rates, GDP) |
| `BaseAlternativeDataEncoder` | `alternative_data` | Satellite, card, foot traffic |
| `BaseOrderBookEncoder` | `order_book` | Level 2 bid/ask depth |
| `BaseOptionsDataEncoder` | `options_data` | Volatility surface, Greeks |
| `BaseSentimentEncoder` | `sentiment` | Aggregated market sentiment |
| `BaseGraphNeuralNetworkInterface` | `graph_neural_network` | Market relationship graph |

---

## How to Implement a Future Encoder

When a future phase implements one of these interfaces:

### 1. Inherit the correct interface

```python
from trainer.interfaces.future_encoders import BaseNewsEncoder
from trainer.plugins.registry import PluginRegistry
import torch

@PluginRegistry.register("news_v1", version="1.0.0", status="experimental")
class NewsEncoderV1(BaseNewsEncoder):

    def __init__(self, hidden_dim: int = 128, model_name: str = "finbert-base"):
        self._output_dim = hidden_dim
        # ... load model ...

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def encode_articles(self, token_ids, attention_mask=None, timestamps=None):
        # ... real implementation ...
        return torch.zeros(token_ids.shape[0], self._output_dim)

    def forward(self, token_ids, attention_mask=None, **kwargs):
        return self.encode_articles(token_ids, attention_mask)

    def get_metadata(self):
        return {
            "version": "1.0.0",
            "status": "experimental",
            "model_backbone": "finbert-base",
        }
```

### 2. Validate the output contract

```python
enc = NewsEncoderV1(hidden_dim=128)
dummy = torch.randn(4, 10, 50)  # (batch, n_articles, seq_len)
out = enc.forward(dummy)
assert enc.validate_output(out, batch_size=4)
```

### 3. Register with PluginRegistry

The `@PluginRegistry.register(...)` decorator handles this automatically.

### 4. Write tests

```python
# tests/unit/test_news_encoder.py
def test_news_encoder_output_shape():
    enc = NewsEncoderV1(hidden_dim=128)
    dummy = torch.randn(2, 5, 128)
    out = enc.forward(dummy)
    assert out.shape == (2, 128)
```

---

## Future Interface Details

### Reinforcement Learning Interface

`BaseReinforcementLearningInterface` in `trainer/interfaces/future_encoders.py`

**Important:** This does NOT implement RL training. It provides a socket for connecting an already-trained external RL policy's embeddings as an additional context stream.

Planned inputs:
- `state_tensor: (batch, state_dim)` — current market state representation
- `reward_history: (batch, T)` — optional historical reward signal

Planned output: `(batch, output_dim)` — policy embedding

### Cross-Asset Encoder

`BaseCrossAssetEncoder` — encodes price and correlation data from related instruments.

Planned inputs:
- `asset_sequences: dict[symbol → (batch, seq_len, features)]`
- `correlation_matrix: (batch, n_assets, n_assets)`

Planned output: `(batch, output_dim)` — cross-asset context embedding

### News Encoder

`BaseNewsEncoder` — FinBERT or similar language model for financial news.

Planned inputs:
- `token_ids: (batch, n_articles, seq_len)`
- `attention_mask: (batch, n_articles, seq_len)`
- `timestamps: (batch, n_articles)` — article recency weighting

### Macroeconomic Encoder

`BaseMacroEncoder` — encodes interest rates, CPI, employment, GDP, etc.

**Important:** Do NOT implement econometric models. This encoder only maps pre-processed macro indicator values to a latent vector.

Planned inputs:
- `macro_indicators: (batch, n_indicators)` — standardised indicator values
- `indicator_names: list[str]`

### Alternative Data Encoder

`BaseAlternativeDataEncoder` — satellite imagery analytics, credit card flows, web traffic signals.

Planned inputs:
- `alt_data: (batch, n_signals)` — normalised alternative data signals
- `source_ids: (batch, n_signals)` — data source identifiers

### Order Book Encoder

`BaseOrderBookEncoder` — Level 2 market microstructure.

Planned inputs:
- `bid_prices, bid_sizes, ask_prices, ask_sizes: (batch, n_levels)` each
- `timestamps: (batch,)` — optional snapshot timestamps

### Options Data Encoder

`BaseOptionsDataEncoder` — volatility surface and options positioning.

Planned inputs:
- `strikes: (batch, n_strikes)`
- `expiries: (batch, n_expiries)` — days to expiry
- `ivs: (batch, n_strikes, n_expiries)` — implied volatility surface
- `greeks: (batch, n_strikes, n_expiries, n_greeks)` — optional delta/gamma/vega
- `open_interest: (batch, n_strikes, n_expiries)` — optional

### Sentiment Encoder

`BaseSentimentEncoder` — aggregates put/call ratio, social sentiment, analyst ratings.

Planned inputs:
- `sentiment_scores: (batch, n_sources)` — normalised per-source sentiment
- `source_weights: (n_sources,)` — reliability weights
- `timestamps: (batch, n_sources)` — signal recency

### Graph Neural Network Interface

`BaseGraphNeuralNetworkInterface` — models relationships between market participants, instruments, and sectors.

**Important:** Do NOT implement GNN layers here. This interface connects an already-trained external GNN's graph embeddings as an additional stream.

Planned inputs:
- `node_features: (batch, n_nodes, feat_dim)` — per-node attributes
- `edge_index: (2, n_edges)` — COO-format adjacency
- `edge_weights: (n_edges,)` — optional edge weights

---

## Backward Compatibility Guarantee

All future encoder implementations MUST:

1. Not change `forward(tensor)` return type or shape
2. Not change `predict()` or `forward_multi_task()` outputs
3. Not modify any existing prediction head
4. Register via `PluginRegistry` — never by editing `model.py`
5. Pass `validate_output()` before the embedding enters the fusion layer
6. Expose `get_metadata()` with at least `{"version": str, "status": str}`

---

## Future Architecture Readiness Checklist

- [x] `BaseExternalEncoder` abstract contract defined
- [x] `PluginRegistry` implemented and tested
- [x] 9 future encoder interface stubs created
- [x] `AdaptiveFusionLayer` already accepts variable-length stream lists
- [x] `HybridEncoder.enabled_streams` is dynamic — no hardcoded stream count
- [x] `get_export_metadata()` includes `encoder_versions` and `active_streams`
- [ ] HybridEncoder wiring to accept registered plugin streams (future phase)
- [ ] Data pipeline adapters for each encoder type (future phase)
- [ ] Pre-trained weights for news/macro/sentiment encoders (future phase)
