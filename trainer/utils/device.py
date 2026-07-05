"""Device resolution utilities."""

from __future__ import annotations

import torch
from loguru import logger


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a device string to a torch.device.

    "auto" selects CUDA if available, then MPS, then CPU.

    Args:
        device: "auto", "cuda", "cuda:N", "mps", or "cpu".

    Returns:
        Resolved torch.device.
    """
    if device == "auto":
        if torch.cuda.is_available():
            resolved = torch.device("cuda")
        elif torch.backends.mps.is_available():
            resolved = torch.device("mps")
        else:
            resolved = torch.device("cpu")
    else:
        resolved = torch.device(device)

    logger.info("Using device: {}", resolved)
    return resolved
