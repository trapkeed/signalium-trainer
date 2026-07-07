"""Phase 16 — AutoML Search Strategies.

Implements hyperparameter search strategies:

  - ManualStrategy         : run a predefined list of configurations
  - GridStrategy           : exhaustive grid enumeration
  - RandomStrategy         : random sampling with optional seed
  - BayesianStrategy       : Gaussian-process-surrogate Bayesian optimization
                             (pure-Python, no external library required)

All strategies implement the same interface::

    strategy = RandomStrategy(search_space, n_trials=50, seed=42)
    config = strategy.suggest(trial_id, completed_trials)
    strategy.report(trial_id, score)

Usage::

    space = build_standard_space()
    strategy = BayesianStrategy(space, n_trials=30, n_initial_random=5)
    for i in range(30):
        cfg = strategy.suggest(f"trial-{i}", completed)
        score = evaluate(cfg)
        strategy.report(f"trial-{i}", score)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from trainer.automl.search_space import SearchSpace

# ---------------------------------------------------------------------------
# Trial suggestion
# ---------------------------------------------------------------------------


@dataclass
class Suggestion:
    """A suggested hyperparameter configuration.

    Attributes:
        trial_id: The trial for which this suggestion was generated.
        config: Hyperparameter dict.
        strategy: Name of the strategy that generated the suggestion.
        iteration: Suggestion number (0-indexed).
    """

    trial_id: str
    config: dict[str, Any]
    strategy: str
    iteration: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "config": self.config,
            "strategy": self.strategy,
            "iteration": self.iteration,
        }


# ---------------------------------------------------------------------------
# Base strategy
# ---------------------------------------------------------------------------


class BaseStrategy:
    """Abstract base for search strategies."""

    name: str = "base"

    def __init__(
        self, search_space: SearchSpace, n_trials: int = 20, seed: int | None = None
    ) -> None:
        self.search_space = search_space
        self.n_trials = n_trials
        self.seed = seed
        self._rng = random.Random(seed)
        self._iteration = 0
        self._scores: dict[str, float] = {}

    def suggest(self, trial_id: str, completed: list[dict[str, Any]] | None = None) -> Suggestion:
        raise NotImplementedError

    def report(self, trial_id: str, score: float) -> None:
        """Record a completed trial score for the strategy to learn from."""
        self._scores[trial_id] = score

    def is_exhausted(self) -> bool:
        """Return True when the strategy has no more suggestions."""
        return self._iteration >= self.n_trials

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.name, "n_trials": self.n_trials, "seed": self.seed}


# ---------------------------------------------------------------------------
# Manual strategy
# ---------------------------------------------------------------------------


class ManualStrategy(BaseStrategy):
    """Run a predefined list of configurations in order.

    Args:
        search_space: Search space (used for metadata only).
        configs: List of hyperparameter dicts to evaluate.
    """

    name = "manual"

    def __init__(self, search_space: SearchSpace, configs: list[dict[str, Any]]) -> None:
        super().__init__(search_space, n_trials=len(configs))
        self._configs = list(configs)

    def suggest(self, trial_id: str, completed: list[dict[str, Any]] | None = None) -> Suggestion:
        if self._iteration >= len(self._configs):
            raise StopIteration("ManualStrategy exhausted all configurations.")
        cfg = self._configs[self._iteration]
        sug = Suggestion(
            trial_id=trial_id, config=cfg, strategy=self.name, iteration=self._iteration
        )
        self._iteration += 1
        return sug

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.name, "configs": self._configs}


# ---------------------------------------------------------------------------
# Grid strategy
# ---------------------------------------------------------------------------


class GridStrategy(BaseStrategy):
    """Enumerate all grid configurations exhaustively.

    Args:
        search_space: Search space to enumerate.
        max_per_param: Max values per float/int parameter (default 5).
        seed: Not used (grid is deterministic).
    """

    name = "grid"

    def __init__(
        self, search_space: SearchSpace, max_per_param: int = 5, seed: int | None = None
    ) -> None:
        self._grid = search_space.grid(max_per_param)
        super().__init__(search_space, n_trials=len(self._grid), seed=seed)

    def suggest(self, trial_id: str, completed: list[dict[str, Any]] | None = None) -> Suggestion:
        if self._iteration >= len(self._grid):
            raise StopIteration("GridStrategy exhausted all configurations.")
        cfg = self._grid[self._iteration]
        sug = Suggestion(
            trial_id=trial_id, config=cfg, strategy=self.name, iteration=self._iteration
        )
        self._iteration += 1
        return sug

    def is_exhausted(self) -> bool:
        return self._iteration >= len(self._grid)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.name, "n_grid": len(self._grid)}


# ---------------------------------------------------------------------------
# Random strategy
# ---------------------------------------------------------------------------


class RandomStrategy(BaseStrategy):
    """Random search over the hyperparameter space.

    Args:
        search_space: Search space to sample from.
        n_trials: Number of random configurations to try.
        seed: Random seed for reproducibility.
    """

    name = "random"

    def suggest(self, trial_id: str, completed: list[dict[str, Any]] | None = None) -> Suggestion:
        if self.is_exhausted():
            raise StopIteration("RandomStrategy exhausted all trials.")
        cfg = self.search_space.sample(self._rng)
        sug = Suggestion(
            trial_id=trial_id, config=cfg, strategy=self.name, iteration=self._iteration
        )
        self._iteration += 1
        return sug


# ---------------------------------------------------------------------------
# Bayesian strategy (GP surrogate, pure Python)
# ---------------------------------------------------------------------------


class BayesianStrategy(BaseStrategy):
    """Bayesian optimization using a simple Gaussian Process surrogate.

    Uses expected improvement (EI) acquisition with a radial basis function
    (RBF) kernel. Falls back to random search for the first n_initial_random
    trials while building the surrogate.

    Args:
        search_space: Search space to optimize.
        n_trials: Total number of trials.
        n_initial_random: Number of random trials before Bayesian kicks in.
        n_candidates: Candidates evaluated per acquisition step.
        seed: Random seed.
        exploration: Exploration factor xi in EI formula.
    """

    name = "bayesian"

    def __init__(
        self,
        search_space: SearchSpace,
        n_trials: int = 30,
        n_initial_random: int = 5,
        n_candidates: int = 64,
        seed: int | None = None,
        exploration: float = 0.01,
    ) -> None:
        super().__init__(search_space, n_trials=n_trials, seed=seed)
        self.n_initial_random = n_initial_random
        self.n_candidates = n_candidates
        self.exploration = exploration
        self._observed_configs: list[dict[str, Any]] = []
        self._observed_scores: list[float] = []

    def report(self, trial_id: str, score: float) -> None:
        super().report(trial_id, score)
        self._observed_scores.append(score)

    def _vectorize(self, config: dict[str, Any]) -> list[float]:
        """Convert a config dict to a numeric vector for GP."""
        vec = []
        for name, param in self.search_space._params.items():
            val = config.get(name, 0)
            if isinstance(val, bool) or isinstance(val, (int, float)):
                vec.append(float(val))
            else:
                choices = getattr(param, "choices", [val])
                idx = choices.index(val) if val in choices else 0
                vec.append(float(idx) / max(1, len(choices) - 1))
        return vec

    def _rbf_kernel(self, x1: list[float], x2: list[float], length_scale: float = 1.0) -> float:
        if not x1 or not x2:
            return 1.0
        sq_dist = sum((a - b) ** 2 for a, b in zip(x1, x2))
        return math.exp(-0.5 * sq_dist / (length_scale**2))

    def _gp_predict(self, x_test: list[float]) -> tuple[float, float]:
        """Return (mean, std) of GP posterior at x_test."""
        if not self._observed_configs:
            return 0.0, 1.0

        X = [self._vectorize(c) for c in self._observed_configs]
        y = self._observed_scores
        n = len(X)
        noise = 1e-6

        K = [
            [self._rbf_kernel(X[i], X[j]) + (noise if i == j else 0) for j in range(n)]
            for i in range(n)
        ]

        try:
            L = _cholesky(K)
            alpha = _solve_triangular(L, y)
            k_star = [self._rbf_kernel(x_test, X[i]) for i in range(n)]
            mean = sum(k_star[i] * alpha[i] for i in range(n))
            v = _forward_solve(L, k_star)
            var = max(0.0, self._rbf_kernel(x_test, x_test) - sum(vi**2 for vi in v))
            return mean, math.sqrt(var)
        except Exception:
            return float(sum(y) / len(y)) if y else 0.0, 1.0

    def _expected_improvement(self, mean: float, std: float, best: float) -> float:
        """Compute Expected Improvement."""
        if std < 1e-10:
            return 0.0
        z = (mean - best - self.exploration) / std
        return (mean - best - self.exploration) * _normal_cdf(z) + std * _normal_pdf(z)

    def suggest(self, trial_id: str, completed: list[dict[str, Any]] | None = None) -> Suggestion:
        if self.is_exhausted():
            raise StopIteration("BayesianStrategy exhausted all trials.")

        if len(self._observed_scores) < self.n_initial_random:
            cfg = self.search_space.sample(self._rng)
        else:
            best = max(self._observed_scores)
            best_cfg: dict[str, Any] = {}
            best_ei = -1.0

            candidates = [self.search_space.sample(self._rng) for _ in range(self.n_candidates)]
            for cand in candidates:
                x = self._vectorize(cand)
                mu, sigma = self._gp_predict(x)
                ei = self._expected_improvement(mu, sigma, best)
                if ei > best_ei:
                    best_ei = ei
                    best_cfg = cand

            cfg = best_cfg if best_cfg else self.search_space.sample(self._rng)

        self._observed_configs.append(cfg)
        sug = Suggestion(
            trial_id=trial_id, config=cfg, strategy=self.name, iteration=self._iteration
        )
        self._iteration += 1
        return sug

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "n_trials": self.n_trials,
            "n_initial_random": self.n_initial_random,
            "n_candidates": self.n_candidates,
            "exploration": self.exploration,
            "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# Pure-Python linear algebra helpers for GP
# ---------------------------------------------------------------------------


def _cholesky(A: list[list[float]]) -> list[list[float]]:
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(max(s, 1e-12))
            else:
                L[i][j] = s / L[j][j]
    return L


def _solve_triangular(L: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    x = [0.0] * n
    for i in range(n):
        x[i] = (b[i] - sum(L[i][j] * x[j] for j in range(i))) / L[i][i]
    return x


def _forward_solve(L: list[list[float]], b: list[float]) -> list[float]:
    return _solve_triangular(L, b)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def _normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------


def build_strategy(config: dict[str, Any], search_space: SearchSpace) -> BaseStrategy:
    """Instantiate a strategy from a config dict.

    Config keys:
        type: "manual" | "grid" | "random" | "bayesian"
        n_trials: Number of trials (random/bayesian).
        seed: Random seed.
        max_per_param: Grid values per parameter.
        n_initial_random: Bayesian warm-up trials.
        configs: List of dicts (manual only).
    """
    stype = config.get("type", "random")
    n_trials = config.get("n_trials", 20)
    seed = config.get("seed")

    if stype == "manual":
        return ManualStrategy(search_space, configs=config["configs"])
    elif stype == "grid":
        return GridStrategy(search_space, max_per_param=config.get("max_per_param", 5), seed=seed)
    elif stype == "random":
        return RandomStrategy(search_space, n_trials=n_trials, seed=seed)
    elif stype == "bayesian":
        return BayesianStrategy(
            search_space,
            n_trials=n_trials,
            n_initial_random=config.get("n_initial_random", 5),
            n_candidates=config.get("n_candidates", 64),
            seed=seed,
            exploration=config.get("exploration", 0.01),
        )
    raise ValueError(f"Unknown strategy type: {stype!r}")
