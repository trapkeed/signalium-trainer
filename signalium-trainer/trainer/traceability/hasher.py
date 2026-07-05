"""Config and schema hashing utilities for reproducibility tracking.

Provides deterministic, stable SHA-256 hashes of:
  - Training configurations (dataclass or dict)
  - Inference configuration dicts
  - Feature / label schema dicts

All hashes are produced from sorted JSON serialisation to ensure
field-order independence and cross-platform stability.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any


def _to_serialisable(obj: Any) -> Any:
    """Recursively convert an object to a JSON-serialisable type."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_serialisable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(v) for v in obj]
    if hasattr(obj, "value"):  # Enum
        return obj.value
    return obj


def hash_config(config: Any, length: int = 12) -> str:
    """Compute a stable SHA-256 prefix hash of a configuration object.

    Accepts dataclasses, dicts, or any JSON-serialisable object.

    Args:
        config: The configuration to hash.
        length: Number of hex characters to return. Default: 12.

    Returns:
        A ``length``-character hex string.
    """
    serialisable = _to_serialisable(config)
    json_bytes = json.dumps(serialisable, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(json_bytes).hexdigest()
    return digest[:length]


def hash_schema(schema: dict[str, Any], length: int = 12) -> str:
    """Compute a stable SHA-256 prefix hash of a schema dict.

    Args:
        schema: The schema dictionary to hash.
        length: Number of hex characters to return. Default: 12.

    Returns:
        A ``length``-character hex string.
    """
    return hash_config(schema, length=length)


def hash_string(text: str, length: int = 12) -> str:
    """Compute a stable SHA-256 prefix hash of a plain string.

    Args:
        text: The string to hash.
        length: Number of hex characters to return. Default: 12.

    Returns:
        A ``length``-character hex string.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:length]
