"""Phase 16 — Institutional AutoML and Research Optimization Platform.

Public API for the AutoML package. Import from here for all external usage.

Search Space::

    from trainer.automl import (
        SearchSpace, FloatParameter, IntParameter, CategoricalParameter,
        ConditionalParameter, build_standard_space,
    )

Objectives::

    from trainer.automl import (
        Objective, ObjectiveScorer, build_research_scorer,
        compute_composite_research_score, STANDARD_OBJECTIVES,
    )

Pruning::

    from trainer.automl import (
        BasePruner, NoPruner, MedianPruner, SuccessiveHalvingPruner,
        PruningDecision, build_pruner,
    )

Search Strategies::

    from trainer.automl import (
        BaseStrategy, ManualStrategy, GridStrategy, RandomStrategy,
        BayesianStrategy, Suggestion, build_strategy,
    )

Experiment Tracking::

    from trainer.automl import (
        Trial, TrialStatus, SearchJob, ExperimentRecord,
    )

Resource Management::

    from trainer.automl import (
        ResourceManager, ResourceSnapshot, WorkerSlot,
    )

Backbone Search::

    from trainer.automl import (
        BackboneSpec, BackboneCatalog, BackboneSearcher, BackboneEvalResult,
    )

Ranking::

    from trainer.automl import (
        ExperimentRanker, RankingEntry, RankingResult,
    )

Database::

    from trainer.automl import AutoMLDatabase

Visualization::

    from trainer.automl import AutoMLVisualizationBuilder

API::

    from trainer.automl import AutoMLAPI

Engine::

    from trainer.automl import (
        AutoMLEngine, AutoMLConfig, AutoMLResult,
    )
"""

from trainer.automl.api import AutoMLAPI
from trainer.automl.backbone_search import (
    BackboneCatalog,
    BackboneEvalResult,
    BackboneSearcher,
    BackboneSpec,
)
from trainer.automl.database import AutoMLDatabase
from trainer.automl.engine import (
    AutoMLConfig,
    AutoMLEngine,
    AutoMLResult,
)
from trainer.automl.experiment import (
    ExperimentRecord,
    SearchJob,
    Trial,
    TrialStatus,
)
from trainer.automl.objectives import (
    STANDARD_OBJECTIVES,
    Objective,
    ObjectiveScorer,
    build_research_scorer,
    compute_composite_research_score,
    get_objective,
)
from trainer.automl.pruning import (
    BasePruner,
    MedianPruner,
    NoPruner,
    PruningDecision,
    SuccessiveHalvingPruner,
    build_pruner,
)
from trainer.automl.ranking import (
    ExperimentRanker,
    RankingEntry,
    RankingResult,
)
from trainer.automl.resource_manager import (
    ResourceManager,
    ResourceSnapshot,
    WorkerSlot,
)
from trainer.automl.search_space import (
    BaseParameter,
    CategoricalParameter,
    ConditionalParameter,
    FloatParameter,
    IntParameter,
    SearchSpace,
    build_standard_space,
)
from trainer.automl.strategies import (
    BaseStrategy,
    BayesianStrategy,
    GridStrategy,
    ManualStrategy,
    RandomStrategy,
    Suggestion,
    build_strategy,
)
from trainer.automl.visualization import AutoMLVisualizationBuilder

__all__ = [
    # search_space
    "BaseParameter",
    "FloatParameter",
    "IntParameter",
    "CategoricalParameter",
    "ConditionalParameter",
    "SearchSpace",
    "build_standard_space",
    # objectives
    "Objective",
    "ObjectiveScorer",
    "STANDARD_OBJECTIVES",
    "get_objective",
    "compute_composite_research_score",
    "build_research_scorer",
    # pruning
    "BasePruner",
    "NoPruner",
    "MedianPruner",
    "SuccessiveHalvingPruner",
    "PruningDecision",
    "build_pruner",
    # strategies
    "BaseStrategy",
    "Suggestion",
    "ManualStrategy",
    "GridStrategy",
    "RandomStrategy",
    "BayesianStrategy",
    "build_strategy",
    # experiment
    "TrialStatus",
    "Trial",
    "SearchJob",
    "ExperimentRecord",
    # resource
    "ResourceSnapshot",
    "WorkerSlot",
    "ResourceManager",
    # backbone
    "BackboneSpec",
    "BackboneCatalog",
    "BackboneSearcher",
    "BackboneEvalResult",
    # ranking
    "RankingEntry",
    "RankingResult",
    "ExperimentRanker",
    # database
    "AutoMLDatabase",
    # visualization
    "AutoMLVisualizationBuilder",
    # api
    "AutoMLAPI",
    # engine
    "AutoMLConfig",
    "AutoMLResult",
    "AutoMLEngine",
]
