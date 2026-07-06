"""Champion/Challenger promotion framework."""

from trainer.research.champion.registry import ChampionRegistry, ModelEntry
from trainer.research.champion.status import ModelStatus

__all__ = ["ChampionRegistry", "ModelEntry", "ModelStatus"]
