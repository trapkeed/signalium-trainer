"""Root-level conftest.py — stubs missing optional heavy dependencies.

torch's compiled extension (.so) is not available in this environment.
Stubbing it here (before any other conftest is loaded) prevents ImportErrors
in test modules that transitively import trainer code which imports torch.
"""

from __future__ import annotations

import sys
import types


def _stub_module(name: str) -> types.ModuleType:
    """Create a minimal stub module and register it in sys.modules."""
    mod = types.ModuleType(name)
    mod.__spec__ = None  # type: ignore[assignment]
    sys.modules[name] = mod
    return mod


# Stub torch and its subpackages only if not already installed.
if "torch" not in sys.modules:
    _torch = _stub_module("torch")

    # Minimal tensor / dtype stubs
    _torch.Tensor = object  # type: ignore[attr-defined]
    _torch.float32 = "float32"  # type: ignore[attr-defined]
    _torch.float64 = "float64"  # type: ignore[attr-defined]
    _torch.long = "long"  # type: ignore[attr-defined]

    def _no_op(*a: object, **kw: object) -> None:
        pass

    _torch.no_grad = _no_op  # type: ignore[attr-defined]

    # torch.nn
    _nn = _stub_module("torch.nn")

    class _Module:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

    _nn.Module = _Module  # type: ignore[attr-defined]
    _nn.Linear = _Module  # type: ignore[attr-defined]
    _nn.functional = _stub_module("torch.nn.functional")
    _torch.nn = _nn  # type: ignore[attr-defined]

    # torch.optim
    _optim = _stub_module("torch.optim")
    _torch.optim = _optim  # type: ignore[attr-defined]

    # torch.utils.data — Dataset is commonly imported
    _utils = _stub_module("torch.utils")
    _utils_data = _stub_module("torch.utils.data")

    class _Dataset:
        pass

    class _DataLoader:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

    _utils_data.Dataset = _Dataset  # type: ignore[attr-defined]
    _utils_data.DataLoader = _DataLoader  # type: ignore[attr-defined]
    _utils.data = _utils_data  # type: ignore[attr-defined]
    _torch.utils = _utils  # type: ignore[attr-defined]

    # Other common submodules
    _stub_module("torch.cuda")
    _stub_module("torch.jit")
    _stub_module("torch.onnx")
    _stub_module("torch.distributed")
    _stub_module("torch.multiprocessing")
