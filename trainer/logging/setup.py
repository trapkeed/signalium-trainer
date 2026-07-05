"""Loguru-based structured logging setup."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger

from trainer.config.schema import LoggingConfig


def configure_logging(config: LoggingConfig) -> None:
    """Configure Loguru for the trainer process.

    Removes the default handler and sets up console + rotating file sinks.
    JSON format is used in non-development environments.

    Args:
        config: LoggingConfig instance.
    """
    logger.remove()

    log_format = _get_format(config.format)

    # Console sink
    logger.add(
        sys.stderr,
        level=config.level,
        format=log_format,
        colorize=config.format != "json",
        backtrace=config.backtrace,
        diagnose=config.diagnose,
        enqueue=True,
    )

    # File sink
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "trainer_{time}.log",
        level=config.level,
        format=log_format,
        rotation=config.rotation,
        retention=config.retention,
        compression="gz",
        backtrace=config.backtrace,
        diagnose=config.diagnose,
        enqueue=True,
        serialize=config.format == "json",
    )


def _get_format(fmt: str) -> str:
    """Return Loguru format string based on the requested format."""
    if fmt == "json":
        return (
            '{{"time":"{time:YYYY-MM-DDTHH:mm:ss.SSSZ}",'
            '"level":"{level}",'
            '"name":"{name}",'
            '"function":"{function}",'
            '"line":{line},'
            '"message":"{message}"}}'
        )
    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    )


def get_logger(name: str) -> Any:
    """Return a bound logger with a module name context.

    Args:
        name: Module or component name to bind.

    Returns:
        Loguru logger with ``name`` bound.
    """
    return logger.bind(name=name)
