"""GPU information utilities."""

from __future__ import annotations

from typing import Any

import torch
from loguru import logger


def get_gpu_info() -> dict[str, Any]:
    """Collect GPU hardware and utilisation information.

    Returns:
        Dict with device count, names, memory (free/total), and CUDA version.
        Returns empty dict on CPU-only machines.
    """
    if not torch.cuda.is_available():
        logger.debug("No CUDA GPUs available")
        return {"available": False}

    info: dict[str, Any] = {
        "available": True,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
        "devices": [],
    }

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        info["devices"].append(
            {
                "index": i,
                "name": props.name,
                "total_memory_gb": round(total / 1024**3, 2),
                "free_memory_gb": round(free / 1024**3, 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            }
        )

    return info


def log_gpu_info() -> None:
    """Log GPU information at INFO level."""
    info = get_gpu_info()
    if not info.get("available"):
        logger.info("No GPU available — running on CPU")
        return
    logger.info("CUDA version: {}", info["cuda_version"])
    for dev in info["devices"]:
        logger.info(
            "GPU {}: {} | {:.1f} GB total | {:.1f} GB free",
            dev["index"],
            dev["name"],
            dev["total_memory_gb"],
            dev["free_memory_gb"],
        )
