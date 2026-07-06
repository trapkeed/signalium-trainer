"""General utilities package."""

from trainer.utils.device import resolve_device
from trainer.utils.git import get_git_commit
from trainer.utils.gpu import get_gpu_info

__all__ = ["get_git_commit", "get_gpu_info", "resolve_device"]
