"""S32 — Setup Memory Engine.

Tracks market opportunities over time as a persistent per-(symbol,
timeframe) state machine, instead of the model reasoning from a single
isolated inference pass. See ``trainer/setup_memory/engine.py`` for the
public API (``SetupMemoryEngine``).

This is a new, additive subsystem — it does not replace
``trainer/qde/qde_observer.py`` (QDE's own advisory lifecycle) or anything
in ``trainer/memory/`` (Decision Memory / Knowledge Base). It is intended
to sit "in front of" model inference: callers query
``SetupMemoryEngine.feature_vector(symbol, timeframe)`` for a compact,
persisted summary of "how mature is this setup" to feed the model as an
input feature, and call ``strengthen``/``weaken``/``invalidate``/``advance``
as new evidence arrives each bar/tick.
"""

from trainer.setup_memory.engine import SetupMemoryEngine
from trainer.setup_memory.models import SetupRecord, SetupState

__all__ = ["SetupMemoryEngine", "SetupRecord", "SetupState"]
