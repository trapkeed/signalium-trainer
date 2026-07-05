"""EncoderPluginRegistry — dynamic registration for future encoder plugins.

Enables future data-stream encoders to be added to the Signalium Quant
architecture WITHOUT modifying any core model code.  Registration is done
via a class decorator or an explicit call.

Design:
  - Class-level dict ``_plugins`` stores all registered plugins.
  - Plugins are keyed by a unique string name (lowercase, underscored).
  - The registry enforces the ``BaseExternalEncoder`` interface contract.
  - ``PluginRegistry.register`` works as a decorator OR a regular function.

Usage (future implementation)::

    from trainer.plugins.registry import PluginRegistry
    from trainer.interfaces.base import BaseExternalEncoder

    # As a decorator:
    @PluginRegistry.register("news_encoder", version="1.0.0")
    class NewsEncoder(BaseExternalEncoder):
        @property
        def name(self) -> str:
            return "news_encoder"

        @property
        def output_dim(self) -> int:
            return 128

        def forward(self, *args, **kwargs):
            ...

        def get_metadata(self):
            return {"version": "1.0.0", "status": "experimental"}

    # Introspect:
    print(PluginRegistry.list_plugins())   # ["news_encoder"]
    info = PluginRegistry.describe("news_encoder")

    # Instantiate:
    plugin = PluginRegistry.get("news_encoder")
    encoder = plugin.instantiate(hidden_dim=128)

    # Use in training:
    embedding = encoder.forward(news_token_ids)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Type

from loguru import logger


# ---------------------------------------------------------------------------
# EncoderPlugin — metadata wrapper around a registered class
# ---------------------------------------------------------------------------


@dataclass
class EncoderPlugin:
    """Metadata and factory wrapper for a registered encoder class.

    Attributes:
        name: Unique registry key (lowercase, underscored).
        version: Semantic version string (e.g. ``"1.0.0"``).
        encoder_class: The registered class — must inherit from
            ``BaseExternalEncoder``.
        description: Human-readable description of what the encoder does.
        status: ``"future"`` | ``"experimental"`` | ``"production"``.
        registered_at: UTC timestamp of registration.
        tags: Optional list of descriptive tags for filtering.
    """

    name: str
    version: str
    encoder_class: type
    description: str = ""
    status: str = "future"
    registered_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    tags: list[str] = field(default_factory=list)

    def instantiate(self, **kwargs: Any) -> Any:
        """Instantiate the encoder class with the provided keyword arguments.

        Args:
            **kwargs: Constructor arguments forwarded to the encoder class.

        Returns:
            A new instance of the encoder.
        """
        try:
            return self.encoder_class(**kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to instantiate plugin {self.name!r} "
                f"({self.encoder_class.__name__}): {exc}"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        """Serialise plugin metadata to a JSON-safe dict."""
        return {
            "name": self.name,
            "version": self.version,
            "class": self.encoder_class.__name__,
            "description": self.description,
            "status": self.status,
            "registered_at": self.registered_at,
            "tags": self.tags,
        }


# ---------------------------------------------------------------------------
# PluginRegistry — class-level singleton registry
# ---------------------------------------------------------------------------


class PluginRegistry:
    """Class-level registry for future encoder plugins.

    All state is stored at the class level — no instance is needed.

    Thread safety: Registration during import is safe.  Concurrent
    registration from multiple threads is not supported and not expected
    in normal usage.
    """

    _plugins: dict[str, EncoderPlugin] = {}

    # ── Registration ─────────────────────────────────────────────────────

    @classmethod
    def register(
        cls,
        name: str,
        version: str = "1.0.0",
        description: str = "",
        status: str = "experimental",
        tags: Optional[list[str]] = None,
    ) -> Callable[[type], type]:
        """Class decorator that registers an encoder class.

        Args:
            name: Unique registry key (lowercase, underscored recommended).
            version: Semantic version string.
            description: Human-readable description.
            status: ``"future"`` | ``"experimental"`` | ``"production"``.
            tags: Optional tags for filtering / discovery.

        Returns:
            A decorator that registers the class and returns it unchanged.

        Raises:
            ValueError: When ``name`` is already registered.

        Usage::

            @PluginRegistry.register("my_encoder", version="1.0.0")
            class MyEncoder(BaseExternalEncoder):
                ...
        """
        def decorator(encoder_class: type) -> type:
            if name in cls._plugins:
                raise ValueError(
                    f"PluginRegistry: a plugin named {name!r} is already registered "
                    f"by {cls._plugins[name].encoder_class.__name__}. "
                    f"Use a unique name or call PluginRegistry.unregister({name!r}) first."
                )
            plugin = EncoderPlugin(
                name=name,
                version=version,
                encoder_class=encoder_class,
                description=description,
                status=status,
                tags=tags or [],
            )
            cls._plugins[name] = plugin
            logger.debug(
                f"PluginRegistry: registered {name!r} v{version} "
                f"({encoder_class.__name__}) [status={status}]"
            )
            return encoder_class
        return decorator

    @classmethod
    def register_class(
        cls,
        encoder_class: type,
        name: str,
        version: str = "1.0.0",
        description: str = "",
        status: str = "experimental",
        tags: Optional[list[str]] = None,
    ) -> EncoderPlugin:
        """Register an encoder class explicitly (non-decorator form).

        Useful when the class is defined in external code and the decorator
        cannot be applied.

        Args:
            encoder_class: Class to register.
            name: Unique registry key.
            version: Semantic version string.
            description: Human-readable description.
            status: ``"future"`` | ``"experimental"`` | ``"production"``.
            tags: Optional tags.

        Returns:
            The created ``EncoderPlugin`` entry.

        Raises:
            ValueError: When ``name`` is already registered.
        """
        # Re-use the decorator logic by calling it directly
        cls.register(
            name=name,
            version=version,
            description=description,
            status=status,
            tags=tags,
        )(encoder_class)
        return cls._plugins[name]

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a plugin from the registry.

        Args:
            name: Registry key to remove.

        Raises:
            KeyError: When the plugin is not found.
        """
        if name not in cls._plugins:
            raise KeyError(f"PluginRegistry: no plugin named {name!r}")
        del cls._plugins[name]
        logger.debug(f"PluginRegistry: unregistered {name!r}")

    # ── Retrieval ─────────────────────────────────────────────────────────

    @classmethod
    def get(cls, name: str) -> EncoderPlugin:
        """Retrieve an ``EncoderPlugin`` by name.

        Args:
            name: Registry key.

        Returns:
            ``EncoderPlugin`` metadata and factory.

        Raises:
            KeyError: When the plugin is not found.
        """
        if name not in cls._plugins:
            available = sorted(cls._plugins.keys())
            raise KeyError(
                f"PluginRegistry: no plugin named {name!r}. "
                f"Registered plugins: {available}"
            )
        return cls._plugins[name]

    @classmethod
    def list_plugins(cls) -> list[str]:
        """Return sorted list of all registered plugin names."""
        return sorted(cls._plugins.keys())

    @classmethod
    def list_by_status(cls, status: str) -> list[str]:
        """Return plugin names filtered by status.

        Args:
            status: ``"future"`` | ``"experimental"`` | ``"production"``.

        Returns:
            Sorted list of matching plugin names.
        """
        return sorted(
            name for name, p in cls._plugins.items() if p.status == status
        )

    @classmethod
    def describe(cls, name: str) -> dict[str, Any]:
        """Return JSON-safe metadata dict for a plugin.

        Args:
            name: Registry key.

        Returns:
            Dict from ``EncoderPlugin.to_dict()``.
        """
        return cls.get(name).to_dict()

    @classmethod
    def describe_all(cls) -> list[dict[str, Any]]:
        """Return metadata dicts for all registered plugins (sorted by name)."""
        return [cls._plugins[n].to_dict() for n in cls.list_plugins()]

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Return True if a plugin with the given name is registered."""
        return name in cls._plugins

    @classmethod
    def clear(cls) -> None:
        """Remove all registered plugins.

        Intended for use in tests only — clears the global registry.
        """
        cls._plugins.clear()
        logger.debug("PluginRegistry: cleared all plugins")
