"""Phase 16 — AutoML Search Space.

Defines configurable hyperparameter search spaces supporting:
  - FloatParameter   : continuous range with optional log scale
  - IntParameter     : integer range
  - CategoricalParameter : discrete choices
  - ConditionalParameter : active only when a parent parameter equals a value

All parameter types support YAML round-trip serialization.

Usage::

    space = SearchSpace()
    space.add(FloatParameter("learning_rate", low=1e-5, high=1e-2, log=True))
    space.add(IntParameter("batch_size", low=16, high=256, step=16))
    space.add(CategoricalParameter("optimizer", choices=["adam", "sgd", "adamw"]))
    space.add(ConditionalParameter(
        "momentum", parent="optimizer", parent_value="sgd",
        parameter=FloatParameter("momentum", low=0.8, high=0.99),
    ))
    sample = space.sample()
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Base parameter
# ---------------------------------------------------------------------------


@dataclass
class BaseParameter:
    """Abstract base for all parameter types."""

    name: str

    def sample(self, rng: Optional[random.Random] = None) -> Any:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "BaseParameter":
        kind = d.get("type")
        if kind == "float":
            return FloatParameter.from_dict(d)
        elif kind == "int":
            return IntParameter.from_dict(d)
        elif kind == "categorical":
            return CategoricalParameter.from_dict(d)
        elif kind == "conditional":
            return ConditionalParameter.from_dict(d)
        raise ValueError(f"Unknown parameter type: {kind}")


# ---------------------------------------------------------------------------
# Float parameter
# ---------------------------------------------------------------------------


@dataclass
class FloatParameter(BaseParameter):
    """Continuous floating-point hyperparameter.

    Attributes:
        name: Parameter name.
        low: Minimum value (inclusive).
        high: Maximum value (inclusive).
        log: If True, sample in log space.
        step: Optional quantization step (None = continuous).
    """

    low: float = 0.0
    high: float = 1.0
    log: bool = False
    step: Optional[float] = None

    def sample(self, rng: Optional[random.Random] = None) -> float:
        r = rng or random
        if self.log:
            log_low = math.log(self.low)
            log_high = math.log(self.high)
            val = math.exp(r.uniform(log_low, log_high))
        else:
            val = r.uniform(self.low, self.high)
        if self.step is not None:
            val = round(val / self.step) * self.step
        return float(val)

    def grid_values(self, n: int = 5) -> List[float]:
        """Return n evenly spaced values across the range."""
        if self.log:
            low = math.log(self.low)
            high = math.log(self.high)
            return [math.exp(low + i * (high - low) / max(n - 1, 1)) for i in range(n)]
        return [self.low + i * (self.high - self.low) / max(n - 1, 1) for i in range(n)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "float",
            "name": self.name,
            "low": self.low,
            "high": self.high,
            "log": self.log,
            "step": self.step,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FloatParameter":
        return FloatParameter(
            name=d["name"],
            low=d["low"],
            high=d["high"],
            log=d.get("log", False),
            step=d.get("step"),
        )


# ---------------------------------------------------------------------------
# Integer parameter
# ---------------------------------------------------------------------------


@dataclass
class IntParameter(BaseParameter):
    """Integer hyperparameter.

    Attributes:
        name: Parameter name.
        low: Minimum value (inclusive).
        high: Maximum value (inclusive).
        step: Sampling step size (default 1).
        log: If True, sample in log space then round.
    """

    low: int = 1
    high: int = 10
    step: int = 1
    log: bool = False

    def sample(self, rng: Optional[random.Random] = None) -> int:
        r = rng or random
        if self.log:
            log_low = math.log(max(self.low, 1))
            log_high = math.log(self.high)
            val = int(round(math.exp(r.uniform(log_low, log_high))))
        else:
            choices = list(range(self.low, self.high + 1, self.step))
            val = r.choice(choices)
        return int(val)

    def grid_values(self, n: Optional[int] = None) -> List[int]:
        values = list(range(self.low, self.high + 1, self.step))
        if n is not None and len(values) > n:
            step = max(1, len(values) // n)
            values = values[::step][:n]
        return values

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "int",
            "name": self.name,
            "low": self.low,
            "high": self.high,
            "step": self.step,
            "log": self.log,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "IntParameter":
        return IntParameter(
            name=d["name"],
            low=d["low"],
            high=d["high"],
            step=d.get("step", 1),
            log=d.get("log", False),
        )


# ---------------------------------------------------------------------------
# Categorical parameter
# ---------------------------------------------------------------------------


@dataclass
class CategoricalParameter(BaseParameter):
    """Discrete categorical hyperparameter.

    Attributes:
        name: Parameter name.
        choices: List of possible values (any JSON-serializable type).
        weights: Optional sampling weights (same length as choices).
    """

    choices: List[Any] = field(default_factory=list)
    weights: Optional[List[float]] = None

    def sample(self, rng: Optional[random.Random] = None) -> Any:
        r = rng or random
        if self.weights:
            total = sum(self.weights)
            dart = r.uniform(0, total)
            cumulative = 0.0
            for choice, w in zip(self.choices, self.weights):
                cumulative += w
                if dart <= cumulative:
                    return choice
        return r.choice(self.choices)

    def grid_values(self) -> List[Any]:
        return list(self.choices)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "categorical",
            "name": self.name,
            "choices": self.choices,
            "weights": self.weights,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CategoricalParameter":
        return CategoricalParameter(
            name=d["name"],
            choices=d["choices"],
            weights=d.get("weights"),
        )


# ---------------------------------------------------------------------------
# Conditional parameter
# ---------------------------------------------------------------------------


@dataclass
class ConditionalParameter(BaseParameter):
    """Hyperparameter active only when a parent parameter equals a specific value.

    Attributes:
        name: Parameter name (same as wrapped parameter's name).
        parent: Name of the parent parameter.
        parent_value: Value the parent must equal to activate this parameter.
        parameter: The underlying parameter to sample when active.
        default: Value to return when inactive (None = omit from sample).
    """

    parent: str = ""
    parent_value: Any = None
    parameter: Optional[BaseParameter] = None
    default: Any = None

    def is_active(self, params: Dict[str, Any]) -> bool:
        return params.get(self.parent) == self.parent_value

    def sample(self, rng: Optional[random.Random] = None) -> Any:
        if self.parameter is None:
            return self.default
        return self.parameter.sample(rng)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "conditional",
            "name": self.name,
            "parent": self.parent,
            "parent_value": self.parent_value,
            "parameter": self.parameter.to_dict() if self.parameter else None,
            "default": self.default,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ConditionalParameter":
        param = None
        if d.get("parameter"):
            param = BaseParameter.from_dict(d["parameter"])
        return ConditionalParameter(
            name=d["name"],
            parent=d["parent"],
            parent_value=d["parent_value"],
            parameter=param,
            default=d.get("default"),
        )


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

ParameterType = Union[FloatParameter, IntParameter, CategoricalParameter, ConditionalParameter]


class SearchSpace:
    """A complete hyperparameter search space.

    Supports random sampling, grid enumeration, and YAML round-trip.

    Args:
        name: Optional name for this search space.
    """

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._params: Dict[str, ParameterType] = {}
        self._conditionals: List[ConditionalParameter] = []

    def add(self, param: ParameterType) -> "SearchSpace":
        """Add a parameter to the space. Returns self for chaining."""
        if isinstance(param, ConditionalParameter):
            self._conditionals.append(param)
        else:
            self._params[param.name] = param
        return self

    def sample(self, rng: Optional[random.Random] = None) -> Dict[str, Any]:
        """Sample one configuration from the search space."""
        r = rng or random
        result: Dict[str, Any] = {}
        for name, param in self._params.items():
            result[name] = param.sample(r)
        for cond in self._conditionals:
            if cond.is_active(result):
                result[cond.name] = cond.sample(r)
            elif cond.default is not None:
                result[cond.name] = cond.default
        return result

    def grid(self, max_per_param: int = 5) -> List[Dict[str, Any]]:
        """Enumerate a grid of configurations.

        Only non-conditional parameters are enumerated. Returns up to
        ``product(min(choices, max_per_param))`` configurations.
        """
        import itertools

        param_values: List[List[Any]] = []
        param_names: List[str] = []
        for name, param in self._params.items():
            param_names.append(name)
            if isinstance(param, FloatParameter):
                param_values.append(param.grid_values(max_per_param))
            elif isinstance(param, IntParameter):
                param_values.append(param.grid_values(max_per_param))
            elif isinstance(param, CategoricalParameter):
                param_values.append(param.grid_values())
            else:
                param_values.append([param.sample()])
        configs = []
        for combo in itertools.product(*param_values):
            configs.append(dict(zip(param_names, combo)))
        return configs

    def param_names(self) -> List[str]:
        names = list(self._params.keys())
        names += [c.name for c in self._conditionals]
        return names

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "parameters": [p.to_dict() for p in self._params.values()],
            "conditionals": [c.to_dict() for c in self._conditionals],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SearchSpace":
        space = SearchSpace(name=d.get("name", "default"))
        for pd in d.get("parameters", []):
            space.add(BaseParameter.from_dict(pd))
        for cd in d.get("conditionals", []):
            space.add(ConditionalParameter.from_dict(cd))
        return space

    def __len__(self) -> int:
        return len(self._params) + len(self._conditionals)

    def __repr__(self) -> str:
        return f"SearchSpace(name={self.name!r}, n_params={len(self)})"


# ---------------------------------------------------------------------------
# Predefined standard search space for Signalium models
# ---------------------------------------------------------------------------


def build_standard_space() -> SearchSpace:
    """Return the standard Signalium hyperparameter search space.

    Covers all optimizable parameters from the Phase 16 specification.
    """
    space = SearchSpace("signalium_standard")

    space.add(FloatParameter("learning_rate", low=1e-5, high=1e-2, log=True))
    space.add(IntParameter("batch_size", low=16, high=256, step=16))
    space.add(CategoricalParameter("optimizer", choices=["adam", "adamw", "sgd", "rmsprop"]))
    space.add(FloatParameter("weight_decay", low=1e-6, high=1e-2, log=True))
    space.add(FloatParameter("dropout", low=0.0, high=0.5, step=0.05))
    space.add(IntParameter("hidden_size", low=64, high=512, step=64))
    space.add(IntParameter("embedding_size", low=32, high=256, step=32))
    space.add(IntParameter("num_layers", low=1, high=6))
    space.add(IntParameter("sequence_length", low=20, high=200, step=20))
    space.add(
        CategoricalParameter(
            "backbone",
            choices=[
                "lstm",
                "gru",
                "transformer",
                "temporal_cnn",
                "hybrid_lstm_cnn",
                "hybrid_transformer_cnn",
            ],
        )
    )
    space.add(
        CategoricalParameter("fusion_strategy", choices=["concat", "attention", "mean", "max"])
    )
    space.add(FloatParameter("gradient_clipping", low=0.1, high=5.0))
    space.add(IntParameter("warmup_steps", low=0, high=1000, step=100))
    space.add(
        CategoricalParameter(
            "scheduler", choices=["cosine", "linear", "constant", "onecycle", "plateau"]
        )
    )
    space.add(CategoricalParameter("activation", choices=["relu", "gelu", "silu", "tanh"]))
    space.add(
        CategoricalParameter(
            "normalization", choices=["layer_norm", "batch_norm", "group_norm", "none"]
        )
    )

    space.add(
        ConditionalParameter(
            name="attention_heads",
            parent="backbone",
            parent_value="transformer",
            parameter=IntParameter("attention_heads", low=2, high=16, step=2),
            default=4,
        )
    )
    space.add(
        ConditionalParameter(
            name="momentum",
            parent="optimizer",
            parent_value="sgd",
            parameter=FloatParameter("momentum", low=0.8, high=0.99),
            default=0.9,
        )
    )

    return space
