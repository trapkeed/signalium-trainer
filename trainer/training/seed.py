"""Reproducibility utilities — deterministic seeding for all RNG sources."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from loguru import logger


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed all RNG sources for reproducible training.

    Args:
        seed: Integer seed value.
        deterministic: If True, enable fully deterministic CuDNN ops
                       (disables cuDNN benchmarking, may slow training).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True

    logger.info("RNG seeded with seed={}, deterministic={}", seed, deterministic)
