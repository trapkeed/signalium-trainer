"""Git utilities for experiment reproducibility."""

from __future__ import annotations

import subprocess
from typing import Optional


def get_git_commit() -> Optional[str]:
    """Return the current HEAD commit hash.

    Returns:
        40-character SHA string, or None if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def is_dirty() -> bool:
    """Return True if the working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
