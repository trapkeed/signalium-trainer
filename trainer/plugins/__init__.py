"""trainer.plugins — Phase 14.9C encoder plugin architecture.

Allows future encoders to be registered and loaded without modifying any
core model code.

Public API::

    from trainer.plugins import PluginRegistry, EncoderPlugin

    # Register (typically via decorator on the encoder class):
    @PluginRegistry.register("my_encoder", version="1.0.0")
    class MyEncoder(BaseExternalEncoder):
        ...

    # Introspect:
    print(PluginRegistry.list_plugins())
    info = PluginRegistry.describe("my_encoder")

    # Instantiate:
    plugin = PluginRegistry.get("my_encoder")
    encoder = plugin.instantiate(hidden_dim=128)
"""

from trainer.plugins.registry import EncoderPlugin, PluginRegistry

__all__ = ["PluginRegistry", "EncoderPlugin"]
