"""Phase 17 — AEIP Ensemble Export.

Exports the ensemble, individual experts, and meta model for deployment.

Export formats:
  - JSON manifest (inference contract)
  - Per-expert metadata dicts
  - Meta model state dict
  - Full ensemble bundle (all experts + meta + config)

Usage::

    exporter = EnsembleExporter(registry, meta_model)
    bundle = exporter.export_bundle()
    exporter.save_bundle(bundle, "experiments/ensemble/exports/bundle_v1.json")

    contract = exporter.export_inference_contract()
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from trainer.ensemble.expert import ExpertRegistry
from trainer.ensemble.meta_model import EnsembleMetaModel


class EnsembleExporter:
    """Exports the ensemble for deployment and reproducibility.

    Args:
        registry: ExpertRegistry with all experts.
        meta_model: EnsembleMetaModel instance.
        strategy_name: Name of the ensemble strategy in use.
        ensemble_version: Version tag for the exported ensemble.
    """

    def __init__(
        self,
        registry: ExpertRegistry,
        meta_model: EnsembleMetaModel | None = None,
        strategy_name: str = "weighted_averaging",
        ensemble_version: str = "1.0.0",
    ) -> None:
        self.registry = registry
        self.meta_model = meta_model
        self.strategy_name = strategy_name
        self.ensemble_version = ensemble_version

    def export_expert(self, expert_id: str) -> dict[str, Any]:
        """Export one expert's metadata and configuration."""
        expert = self.registry.get_or_raise(expert_id)
        return {
            "export_type": "expert",
            "expert_id": expert.expert_id,
            "name": expert.name,
            "model_family": expert.model_family,
            "specializations": expert.specializations,
            "instruments": expert.instruments,
            "timeframes": expert.timeframes,
            "version": expert.version,
            "checkpoint_path": expert.checkpoint_path,
            "model_version": expert.model_version,
            "phase": expert.phase,
            "status": expert.status,
            "weight": expert.weight,
            "recent_accuracy": expert.recent_accuracy,
            "recent_sharpe": expert.recent_sharpe,
            "calibration_error": expert.calibration_error,
            "n_predictions": expert.n_predictions,
            "metadata": expert.metadata,
            "exported_at": datetime.now(UTC).isoformat(),
        }

    def export_all_experts(self) -> list[dict[str, Any]]:
        """Export all experts."""
        return [self.export_expert(e.expert_id) for e in self.registry.all_experts()]

    def export_meta_model(self) -> dict[str, Any] | None:
        """Export meta model state."""
        if self.meta_model is None:
            return None
        return {
            "export_type": "meta_model",
            "exported_at": datetime.now(UTC).isoformat(),
            **self.meta_model.to_dict(),
        }

    def export_bundle(self) -> dict[str, Any]:
        """Export the complete ensemble bundle.

        Returns a JSON-serializable dict containing:
          - Ensemble configuration
          - All expert metadata
          - Meta model state
          - Inference contract
          - Registry summary
        """
        bundle_id = str(uuid.uuid4())
        bundle = {
            "bundle_id": bundle_id,
            "bundle_type": "ensemble_bundle",
            "ensemble_version": self.ensemble_version,
            "strategy": self.strategy_name,
            "n_experts": self.registry.n_experts,
            "n_active": self.registry.n_active,
            "experts": self.export_all_experts(),
            "meta_model": self.export_meta_model(),
            "registry_summary": self.registry.summary(),
            "inference_contract": self.export_inference_contract(),
            "exported_at": datetime.now(UTC).isoformat(),
        }
        logger.info(
            f"Ensemble bundle {bundle_id} exported: "
            f"{self.registry.n_experts} experts, strategy={self.strategy_name}"
        )
        return bundle

    def export_inference_contract(self) -> dict[str, Any]:
        """Export the inference contract for deployment.

        The inference contract documents the expected input/output interface,
        model versions, and validation requirements.
        """
        experts = self.registry.all_experts()
        active = self.registry.active_experts()
        return {
            "contract_type": "ensemble_inference_contract",
            "ensemble_version": self.ensemble_version,
            "strategy": self.strategy_name,
            "n_experts_total": len(experts),
            "n_experts_active": len(active),
            "expert_ids": [e.expert_id for e in active],
            "model_families": list({e.model_family for e in active}),
            "input_schema": {
                "type": "object",
                "required": ["features"],
                "properties": {
                    "features": {"type": "object"},
                    "regime": {"type": "string", "default": "unknown"},
                    "instrument": {"type": "string", "default": ""},
                    "timeframe": {"type": "string", "default": ""},
                },
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "buy_prob": {"type": "number", "minimum": 0, "maximum": 1},
                    "sell_prob": {"type": "number", "minimum": 0, "maximum": 1},
                    "no_trade_prob": {"type": "number", "minimum": 0, "maximum": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "uncertainty": {"type": "number", "minimum": 0, "maximum": 1},
                    "predicted_class": {"type": "string", "enum": ["buy", "sell", "no_trade"]},
                    "institutional_forecast": {"type": "string"},
                    "strategy": {"type": "string"},
                    "n_experts": {"type": "integer"},
                    "agreement_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "weight_map": {"type": "object"},
                },
            },
            "meta_model_active": self.meta_model is not None
            and (self.meta_model.is_trained if self.meta_model else False),
            "generated_at": datetime.now(UTC).isoformat(),
        }

    def save_bundle(self, bundle: dict[str, Any], path: str) -> str:
        """Save bundle dict to a JSON file. Returns the absolute path."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2, default=str)
        logger.info(f"Ensemble bundle saved to {out_path}")
        return str(out_path.absolute())

    @staticmethod
    def load_bundle(path: str) -> dict[str, Any]:
        """Load a previously saved bundle from JSON."""
        with open(path, encoding="utf-8") as f:
            return json.load(f)
