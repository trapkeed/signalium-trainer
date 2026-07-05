"""QuantModelV1 ModelSpec registration.

Import this module once (e.g. in an application entry-point or test suite)
to attach the full ModelSpec to the QuantModelV1 registry entry.

The model class itself is registered via the ``@ModelRegistry.register("quant_v1")``
decorator in ``model.py``. This module supplements that with rich metadata
so callers can inspect the model's input contract, output schema, and
capabilities without instantiating it.
"""

from __future__ import annotations

from trainer.framework.contracts.model_spec import FeatureSchema, ModelSpec, OutputSchema
from trainer.registry.registry import ModelRegistry


def register_quant_v1_spec() -> None:
    """Attach the full ModelSpec to the 'quant_v1' registry entry.

    Safe to call multiple times — it checks whether the spec is already
    registered before overwriting.
    """
    if ModelRegistry.has_spec("quant_v1"):
        return

    spec = ModelSpec(
        name="quant_v1",
        version="1.0.0",
        architecture="LSTM-2L-256H",
        description=(
            "Signalium Quant Model V1 — supervised BUY/SELL/NO TRADE classifier "
            "with LSTM backbone, confidence head, and configurable loss."
        ),
        input_schema=FeatureSchema(
            feature_dim=128,
            seq_len=60,
            feature_groups=[
                "ohlc",
                "technical_indicators",
                "market_memory",
                "volatility",
                "trend",
                "liquidity",
                "session",
                "time",
                "price_structure",
                "feature_store",
                "dataset_metadata",
            ],
            supported_feature_versions=["1.0.0"],
        ),
        output_schema=OutputSchema(
            num_classes=3,
            class_labels=["BUY", "SELL", "NO_TRADE"],
            has_confidence=True,
            auxiliary_outputs=["buy_prob", "sell_prob", "no_trade_prob"],
        ),
        export_formats=["torchscript", "onnx"],
        tags={
            "backbone": "lstm",
            "task": "classification",
            "domain": "quantitative_finance",
            "phase": "2",
        },
    )

    # Attach to the already-registered class entry
    ModelRegistry._specs["quant_v1"] = spec
