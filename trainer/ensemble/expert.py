"""Phase 17 — AEIP Expert Model Registry.

Defines the core expert model data structures for the Adaptive Ensemble
Intelligence Platform.

Every model in Signalium is an "expert" with:
  - Its own weights, checkpoints, validation, metrics, version, metadata
  - A specialization (trending, ranging, high-vol, instrument, timeframe, etc.)
  - A prediction interface that returns ExpertPrediction
  - An independent calibration record

The existing Signalium Quant Model automatically becomes Expert 1.
All future models become additional experts.

Usage::

    registry = ExpertRegistry()
    expert = ExpertModel(
        expert_id="expert_01",
        name="Signalium Quant v1",
        model_family="lstm",
        specializations=[SpecializationType.TRENDING],
    )
    registry.register(expert)
    pred = registry.predict("expert_01", features)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Specialization types
# ---------------------------------------------------------------------------


class SpecializationType:
    """Expert specialization categories."""

    TRENDING = "trending"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    PULLBACK = "pullback"
    SCALPING = "scalping"
    INTRADAY = "intraday"
    SWING = "swing"
    INSTRUMENT = "instrument"
    TIMEFRAME = "timeframe"
    GENERAL = "general"

    ALL = [
        TRENDING,
        RANGING,
        HIGH_VOLATILITY,
        LOW_VOLATILITY,
        BREAKOUT,
        PULLBACK,
        SCALPING,
        INTRADAY,
        SWING,
        INSTRUMENT,
        TIMEFRAME,
        GENERAL,
    ]


# ---------------------------------------------------------------------------
# Model families
# ---------------------------------------------------------------------------


class ModelFamily:
    """Supported backbone model families."""

    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER_ENCODER = "transformer_encoder"
    TEMPORAL_CNN = "temporal_cnn"
    HYBRID_LSTM_TRANSFORMER = "hybrid_lstm_transformer"
    HYBRID_GRU_TRANSFORMER = "hybrid_gru_transformer"
    CUSTOM = "custom"

    ALL = [
        LSTM,
        GRU,
        TRANSFORMER_ENCODER,
        TEMPORAL_CNN,
        HYBRID_LSTM_TRANSFORMER,
        HYBRID_GRU_TRANSFORMER,
        CUSTOM,
    ]


# ---------------------------------------------------------------------------
# Expert prediction
# ---------------------------------------------------------------------------


@dataclass
class ExpertPrediction:
    """Output of a single expert model for one inference request.

    Attributes:
        expert_id: ID of the producing expert.
        buy_prob: Probability of a BUY signal [0, 1].
        sell_prob: Probability of a SELL signal [0, 1].
        no_trade_prob: Probability of NO TRADE signal [0, 1].
        confidence: Expert's confidence in this prediction [0, 1].
        uncertainty: Epistemic + aleatoric uncertainty estimate [0, 1].
        regime: Detected market regime (e.g. "trending_up").
        instrument: Instrument code (e.g. "EURUSD").
        timeframe: Timeframe code (e.g. "H1").
        latency_ms: Inference latency in milliseconds.
        metadata: Any additional expert-specific output.
        timestamp: UTC ISO-8601 prediction timestamp.
    """

    expert_id: str
    buy_prob: float = 0.33
    sell_prob: float = 0.33
    no_trade_prob: float = 0.34
    confidence: float = 0.5
    uncertainty: float = 0.5
    regime: str = "unknown"
    instrument: str = ""
    timeframe: str = ""
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        total = self.buy_prob + self.sell_prob + self.no_trade_prob
        if total > 0 and abs(total - 1.0) > 1e-4:
            self.buy_prob /= total
            self.sell_prob /= total
            self.no_trade_prob /= total

    @property
    def predicted_class(self) -> str:
        """Return the highest-probability class."""
        probs = {
            "buy": self.buy_prob,
            "sell": self.sell_prob,
            "no_trade": self.no_trade_prob,
        }
        return max(probs, key=probs.get)

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.7 and self.uncertainty <= 0.3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert_id": self.expert_id,
            "buy_prob": self.buy_prob,
            "sell_prob": self.sell_prob,
            "no_trade_prob": self.no_trade_prob,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "regime": self.regime,
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "latency_ms": self.latency_ms,
            "predicted_class": self.predicted_class,
            "is_high_confidence": self.is_high_confidence,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ExpertPrediction":
        return ExpertPrediction(
            expert_id=d["expert_id"],
            buy_prob=d.get("buy_prob", 0.33),
            sell_prob=d.get("sell_prob", 0.33),
            no_trade_prob=d.get("no_trade_prob", 0.34),
            confidence=d.get("confidence", 0.5),
            uncertainty=d.get("uncertainty", 0.5),
            regime=d.get("regime", "unknown"),
            instrument=d.get("instrument", ""),
            timeframe=d.get("timeframe", ""),
            latency_ms=d.get("latency_ms", 0.0),
            metadata=d.get("metadata", {}),
            timestamp=d.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )


# ---------------------------------------------------------------------------
# Expert model
# ---------------------------------------------------------------------------


@dataclass
class ExpertModel:
    """An expert model in the ensemble.

    Represents one independently trained forecasting model, with its own
    weights, metrics, version, and specialization.

    Attributes:
        expert_id: Unique expert UUID.
        name: Human-readable expert name.
        model_family: Architecture family (lstm, gru, transformer, etc.).
        specializations: List of SpecializationType values.
        instruments: Instruments this expert specializes in (empty = all).
        timeframes: Timeframes this expert specializes in (empty = all).
        version: Semantic version string.
        checkpoint_path: Path to model weights.
        model_version: Model version tag from registry.
        phase: Training phase this expert was introduced in.
        status: "active" | "retired" | "experimental" | "champion".
        weight: Current ensemble weight [0, 1].
        recent_accuracy: Rolling accuracy over recent predictions.
        recent_sharpe: Rolling Sharpe ratio.
        calibration_error: Current calibration error.
        n_predictions: Total predictions made.
        n_correct: Total correct predictions.
        created_at: UTC creation timestamp.
        updated_at: UTC last-updated timestamp.
        metadata: Arbitrary metadata.
        predict_fn: Optional callable for live inference.
            Signature: (features: dict) → ExpertPrediction.
    """

    expert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Expert"
    model_family: str = ModelFamily.LSTM
    specializations: List[str] = field(default_factory=list)
    instruments: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    checkpoint_path: str = ""
    model_version: str = ""
    phase: str = ""
    status: str = "active"
    weight: float = 1.0
    recent_accuracy: float = 0.0
    recent_sharpe: float = 0.0
    calibration_error: float = 0.0
    n_predictions: int = 0
    n_correct: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    predict_fn: Optional[Callable] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.specializations:
            self.specializations = [SpecializationType.GENERAL]

    @property
    def accuracy(self) -> float:
        if self.n_predictions == 0:
            return 0.0
        return self.n_correct / self.n_predictions

    @property
    def is_active(self) -> bool:
        return self.status in ("active", "champion", "experimental")

    def predict(self, features: Dict[str, Any]) -> ExpertPrediction:
        """Run live inference via the injected predict_fn."""
        if self.predict_fn is not None:
            return self.predict_fn(features)
        return ExpertPrediction(
            expert_id=self.expert_id,
            buy_prob=0.33,
            sell_prob=0.33,
            no_trade_prob=0.34,
            confidence=0.5,
            uncertainty=0.5,
            regime=features.get("regime", "unknown"),
            instrument=features.get("instrument", ""),
            timeframe=features.get("timeframe", ""),
        )

    def record_prediction(self, correct: bool) -> None:
        self.n_predictions += 1
        if correct:
            self.n_correct += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def retire(self) -> None:
        self.status = "retired"
        self.updated_at = datetime.now(timezone.utc).isoformat()
        logger.info(f"Expert {self.expert_id} ({self.name}) retired.")

    def promote_to_champion(self) -> None:
        self.status = "champion"
        self.updated_at = datetime.now(timezone.utc).isoformat()
        logger.info(f"Expert {self.expert_id} ({self.name}) promoted to champion.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert_id": self.expert_id,
            "name": self.name,
            "model_family": self.model_family,
            "specializations": self.specializations,
            "instruments": self.instruments,
            "timeframes": self.timeframes,
            "version": self.version,
            "checkpoint_path": self.checkpoint_path,
            "model_version": self.model_version,
            "phase": self.phase,
            "status": self.status,
            "weight": self.weight,
            "recent_accuracy": self.recent_accuracy,
            "recent_sharpe": self.recent_sharpe,
            "calibration_error": self.calibration_error,
            "n_predictions": self.n_predictions,
            "n_correct": self.n_correct,
            "accuracy": self.accuracy,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ExpertModel":
        return ExpertModel(
            expert_id=d.get("expert_id", str(uuid.uuid4())),
            name=d.get("name", "Expert"),
            model_family=d.get("model_family", ModelFamily.LSTM),
            specializations=d.get("specializations", []),
            instruments=d.get("instruments", []),
            timeframes=d.get("timeframes", []),
            version=d.get("version", "1.0.0"),
            checkpoint_path=d.get("checkpoint_path", ""),
            model_version=d.get("model_version", ""),
            phase=d.get("phase", ""),
            status=d.get("status", "active"),
            weight=d.get("weight", 1.0),
            recent_accuracy=d.get("recent_accuracy", 0.0),
            recent_sharpe=d.get("recent_sharpe", 0.0),
            calibration_error=d.get("calibration_error", 0.0),
            n_predictions=d.get("n_predictions", 0),
            n_correct=d.get("n_correct", 0),
            created_at=d.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=d.get("updated_at", datetime.now(timezone.utc).isoformat()),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Expert registry
# ---------------------------------------------------------------------------


class ExpertRegistry:
    """Central registry of all expert models in the ensemble.

    Maintains the ordered list of experts, their metadata, and provides
    methods for filtering by specialization, instrument, timeframe, or status.

    The existing Signalium Quant Model is registered as Expert 1 automatically.
    """

    def __init__(self) -> None:
        self._experts: Dict[str, ExpertModel] = {}
        self._order: List[str] = []
        self._champion_id: Optional[str] = None

    def register(self, expert: ExpertModel) -> str:
        """Register an expert. Returns expert_id."""
        self._experts[expert.expert_id] = expert
        if expert.expert_id not in self._order:
            self._order.append(expert.expert_id)
        logger.info(f"Registered expert {expert.expert_id} ({expert.name})")
        return expert.expert_id

    def get(self, expert_id: str) -> Optional[ExpertModel]:
        return self._experts.get(expert_id)

    def get_or_raise(self, expert_id: str) -> ExpertModel:
        expert = self._experts.get(expert_id)
        if expert is None:
            raise KeyError(f"Expert {expert_id!r} not found in registry.")
        return expert

    def remove(self, expert_id: str) -> bool:
        if expert_id in self._experts:
            del self._experts[expert_id]
            self._order = [e for e in self._order if e != expert_id]
            return True
        return False

    def all_experts(self) -> List[ExpertModel]:
        """Return all experts in registration order."""
        return [self._experts[eid] for eid in self._order if eid in self._experts]

    def active_experts(self) -> List[ExpertModel]:
        return [e for e in self.all_experts() if e.is_active]

    def by_specialization(self, spec: str) -> List[ExpertModel]:
        return [
            e
            for e in self.active_experts()
            if spec in e.specializations or SpecializationType.GENERAL in e.specializations
        ]

    def by_instrument(self, instrument: str) -> List[ExpertModel]:
        return [
            e for e in self.active_experts() if not e.instruments or instrument in e.instruments
        ]

    def by_timeframe(self, timeframe: str) -> List[ExpertModel]:
        return [e for e in self.active_experts() if not e.timeframes or timeframe in e.timeframes]

    def by_family(self, family: str) -> List[ExpertModel]:
        return [e for e in self.active_experts() if e.model_family == family]

    def champion(self) -> Optional[ExpertModel]:
        if self._champion_id:
            return self._experts.get(self._champion_id)
        champions = [e for e in self.active_experts() if e.status == "champion"]
        return champions[0] if champions else None

    def set_champion(self, expert_id: str) -> None:
        for e in self.active_experts():
            if e.status == "champion" and e.expert_id != expert_id:
                e.status = "active"
        expert = self.get_or_raise(expert_id)
        expert.promote_to_champion()
        self._champion_id = expert_id

    def retire(self, expert_id: str) -> None:
        self.get_or_raise(expert_id).retire()

    @property
    def n_experts(self) -> int:
        return len(self._experts)

    @property
    def n_active(self) -> int:
        return len(self.active_experts())

    def summary(self) -> Dict[str, Any]:
        return {
            "n_experts": self.n_experts,
            "n_active": self.n_active,
            "n_retired": sum(1 for e in self.all_experts() if e.status == "retired"),
            "champion_id": self._champion_id,
            "families": list({e.model_family for e in self.all_experts()}),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experts": {eid: e.to_dict() for eid, e in self._experts.items()},
            "order": self._order,
            "champion_id": self._champion_id,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ExpertRegistry":
        registry = ExpertRegistry()
        registry._champion_id = d.get("champion_id")
        for eid in d.get("order", []):
            expert_data = d.get("experts", {}).get(eid)
            if expert_data:
                registry.register(ExpertModel.from_dict(expert_data))
        return registry


def build_default_registry() -> ExpertRegistry:
    """Build the default expert registry with Expert 1 (Signalium Quant v1)."""
    registry = ExpertRegistry()
    registry.register(
        ExpertModel(
            expert_id="expert_01",
            name="Signalium Quant v1",
            model_family=ModelFamily.LSTM,
            specializations=[SpecializationType.GENERAL],
            version="1.0.0",
            phase="phase_1_14",
            status="champion",
            metadata={"description": "The original Signalium Quant model — Expert 1."},
        )
    )
    return registry
