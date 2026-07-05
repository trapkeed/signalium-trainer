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

from trainer.automl.search_space import (
    BaseParameter,
    FloatParameter,
    IntParameter,
    CategoricalParameter,
    ConditionalParameter,
    SearchSpace,
    build_standard_space,
)

from trainer.automl.objectives import (
    Objective,
    ObjectiveScorer,
    STANDARD_OBJECTIVES,
    get_objective,
    compute_composite_research_score,
    build_research_scorer,
)

from trainer.automl.pruning import (
    BasePruner,
    NoPruner,
    MedianPruner,
    SuccessiveHalvingPruner,
    PruningDecision,
    build_pruner,
)

from trainer.automl.strategies import (
    BaseStrategy,
    Suggestion,
    ManualStrategy,
    GridStrategy,
    RandomStrategy,
    BayesianStrategy,
    build_strategy,
)

from trainer.automl.experiment import (
    TrialStatus,
    Trial,
    SearchJob,
    ExperimentRecord,
)

from trainer.automl.resource_manager import (
    ResourceSnapshot,
    WorkerSlot,
    ResourceManager,
)

from trainer.automl.backbone_search import (
    BackboneSpec,
    BackboneCatalog,
    BackboneSearcher,
    BackboneEvalResult,
)

from trainer.automl.ranking import (
    RankingEntry,
    RankingResult,
    ExperimentRanker,
)

from trainer.automl.database import AutoMLDatabase

from trainer.automl.visualization import AutoMLVisualizationBuilder

from trainer.automl.api import AutoMLAPI

from trainer.automl.engine import (
    AutoMLConfig,
    AutoMLResult,
    AutoMLEngine,
)

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
