"""ModelFactory — configuration-driven model instantiation.

The factory is the single point of entry for creating any registered model.
It decouples callers from concrete model classes and supports future
dynamic loading from YAML / dict configs.
"""

from __future__ import annotations

from typing import Any, Optional, Type

from trainer.config.schema import TrainerConfig
from trainer.models.base import BaseQuantModel


class ModelFactory:
    """Creates model instances from a registry name or configuration dict.

    Usage::

        # By name (most common)
        model = ModelFactory.create("quant_v1", trainer_config)

        # By name + model-specific config
        from trainer.models.quant_v1 import QuantV1Config
        model = ModelFactory.create("quant_v1", trainer_config, model_config=QuantV1Config())

        # From a plain dict (e.g. loaded from YAML)
        model = ModelFactory.from_dict({
            "name": "quant_v1",
            "backbone": {"type": "gru", "hidden_size": 512},
        }, trainer_config)

    Supported model families (registered via ``@ModelRegistry.register``):
        - ``quant_v1`` — LSTM-backbone supervised classifier (implemented)
        - Future: ``quant_v2``, ``quant_tcn``, ``quant_transformer``, etc.
    """

    @classmethod
    def create(
        cls,
        name: str,
        trainer_config: TrainerConfig,
        model_config: Optional[Any] = None,
    ) -> BaseQuantModel:
        """Instantiate a registered model by name.

        Args:
            name: Registry key (e.g. ``"quant_v1"``).
            trainer_config: Global TrainerConfig for the training run.
            model_config: Optional model-specific config dataclass.
                          If omitted, the model uses its built-in defaults.

        Returns:
            Instantiated and initialised ``BaseQuantModel``.

        Raises:
            KeyError: If the model name is not registered.
        """
        from trainer.registry.registry import ModelRegistry

        model_cls = ModelRegistry.get(name)
        if model_config is not None:
            return model_cls(trainer_config, model_config)
        return model_cls(trainer_config)

    @classmethod
    def from_dict(
        cls,
        config_dict: dict[str, Any],
        trainer_config: TrainerConfig,
    ) -> BaseQuantModel:
        """Instantiate a model from a plain dict (e.g. parsed from YAML).

        The dict must contain a ``"name"`` key matching a registered model.
        Remaining keys are forwarded to the model-specific config builder.

        Args:
            config_dict: Dict with at minimum ``{"name": "<model_name>"}``.
            trainer_config: Global TrainerConfig.

        Returns:
            Instantiated ``BaseQuantModel``.

        Raises:
            KeyError: If ``"name"`` is missing or the model is not registered.
            ValueError: If the config dict has an unrecognised structure.
        """
        if "name" not in config_dict:
            raise KeyError(
                "config_dict must contain a 'name' key identifying the model to create. "
                f"Got keys: {list(config_dict.keys())}"
            )

        name = config_dict["name"]
        model_config = cls._build_model_config(name, config_dict)
        return cls.create(name, trainer_config, model_config)

    @classmethod
    def list_available(cls) -> list[str]:
        """Return a sorted list of all available (registered) model names.

        Returns:
            Sorted list of model name strings.
        """
        from trainer.registry.registry import ModelRegistry
        return ModelRegistry.list_models()

    @classmethod
    def _build_model_config(
        cls, name: str, config_dict: dict[str, Any]
    ) -> Optional[Any]:
        """Build a model-specific config from a raw dict.

        Each model family handles its own config construction. This keeps
        the factory open for extension without modifying existing model code.

        Args:
            name: Model name (used to dispatch to the correct config class).
            config_dict: Raw dict (may contain nested backbone/encoder keys).

        Returns:
            Model-specific config object, or None to use defaults.
        """
        if name == "quant_v1":
            return cls._build_quant_v1_config(config_dict)
        # Future model configs are added here.
        # Unrecognised models fall back to None (model uses its own defaults).
        return None

    @classmethod
    def _build_quant_v1_config(cls, d: dict[str, Any]) -> Optional[Any]:
        """Build a QuantV1Config from a flat/nested dict.

        Args:
            d: Config dict, may contain 'backbone', 'encoder', 'loss' sub-dicts.

        Returns:
            QuantV1Config with overridden fields, or None for pure defaults.
        """
        from trainer.models.quant_v1.config import (
            BackboneConfig,
            BackboneType,
            ClassificationHeadConfig,
            ConfidenceHeadConfig,
            InputEncoderConfig,
            LossConfig,
            LossType,
            QuantV1Config,
        )

        cfg = QuantV1Config()

        if "encoder" in d:
            enc = d["encoder"]
            cfg.encoder = InputEncoderConfig(
                input_dim=enc.get("input_dim", cfg.encoder.input_dim),
                hidden_dim=enc.get("hidden_dim", cfg.encoder.hidden_dim),
                use_layer_norm=enc.get("use_layer_norm", cfg.encoder.use_layer_norm),
                use_projection=enc.get("use_projection", cfg.encoder.use_projection),
                projection_dropout=enc.get("projection_dropout", cfg.encoder.projection_dropout),
            )

        if "backbone" in d:
            bb = d["backbone"]
            backbone_type_str = bb.get("type", cfg.backbone.type.value)
            cfg.backbone = BackboneConfig(
                type=BackboneType(backbone_type_str),
                hidden_size=bb.get("hidden_size", cfg.backbone.hidden_size),
                num_layers=bb.get("num_layers", cfg.backbone.num_layers),
                dropout=bb.get("dropout", cfg.backbone.dropout),
                bidirectional=bb.get("bidirectional", cfg.backbone.bidirectional),
                window_length=bb.get("window_length", cfg.backbone.window_length),
            )

        if "loss" in d:
            ls = d["loss"]
            loss_type_str = ls.get("type", cfg.loss.type.value)
            cfg.loss = LossConfig(
                type=LossType(loss_type_str),
                focal_gamma=ls.get("focal_gamma", cfg.loss.focal_gamma),
                label_smoothing=ls.get("label_smoothing", cfg.loss.label_smoothing),
                class_weights=ls.get("class_weights", cfg.loss.class_weights),
            )

        return cfg
