"""Root-level conftest.py — stubs missing optional heavy dependencies.

In some environments torch's compiled extension (.so) is unavailable, which
breaks any import chain that transitively imports torch. When that happens,
this stubs a minimal fake ``torch`` (and submodules) into sys.modules before
any other conftest/test module runs, so those imports don't hard-fail.

If a real, fully-functional torch is installed, we leave it alone — stubbing
on top of a working torch install silently breaks anything (e.g. torchmetrics)
that relies on real attributes (like ``torch.dtype``) the stub doesn't define.

Tests that require actual tensor computation (real forward passes, loss
calculations, gradient flow) should be marked with
``@pytest.mark.requires_torch``.  When torch is unavailable the
``pytest_collection_modifyitems`` hook below skips them automatically.
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


def _real_torch_is_usable() -> bool:
    """Check whether a genuine, fully-functional torch install is present."""
    if "torch" in sys.modules:
        existing = sys.modules["torch"]
        try:
            existing.tensor([0])  # type: ignore[attr-defined]
            return True
        except Exception:
            return False
    try:
        import torch as _t  # noqa: PLC0415

        # Exercise the compiled extension; this is what actually fails when
        # torch's .so is missing/broken (ImportError/AttributeError/OSError).
        _t.tensor([0])
        return True
    except Exception:
        # Remove any partially-imported torch modules so the stub below
        # can install cleanly.
        for mod_name in list(sys.modules):
            if mod_name == "torch" or mod_name.startswith("torch."):
                del sys.modules[mod_name]
        return False


# Exposed so test modules can reference it for conditional skips.
HAS_REAL_TORCH: bool = _real_torch_is_usable()

# ---------------------------------------------------------------------------
# Stub torch and its subpackages only if a real torch isn't usable here.
# ---------------------------------------------------------------------------
if not HAS_REAL_TORCH:
    _torch = _stub_module("torch")

    # ── Decorator factory / context manager stub ─────────────────────────────
    # Needed for patterns like:
    #   @torch.no_grad()          — called with parens, returns a decorator
    #   with torch.no_grad(): …   — used as a context manager
    # The plain ``_no_op`` used previously returned None, which caused
    # ``TypeError: 'NoneType' object is not callable`` when Python tried to
    # apply the returned value as a decorator.
    class _NoGrad:
        def __call__(self, fn: object = None) -> object:  # type: ignore[override]
            if fn is not None and callable(fn):
                # @torch.no_grad (no parentheses) — fn is the wrapped function
                return fn
            # @torch.no_grad() (with parentheses) — return a pass-through decorator
            return lambda f: f

        def __enter__(self) -> "_NoGrad":
            return self

        def __exit__(self, *args: object) -> None:
            pass

    _torch.no_grad = _NoGrad()  # type: ignore[attr-defined]
    _torch.inference_mode = _NoGrad()  # type: ignore[attr-defined]
    _torch.enable_grad = _NoGrad()  # type: ignore[attr-defined]

    # ── Tensor type / dtype stubs ─────────────────────────────────────────────
    _torch.Tensor = object  # type: ignore[attr-defined]
    _torch.float32 = "float32"  # type: ignore[attr-defined]
    _torch.float64 = "float64"  # type: ignore[attr-defined]
    _torch.float16 = "float16"  # type: ignore[attr-defined]
    _torch.bfloat16 = "bfloat16"  # type: ignore[attr-defined]
    _torch.long = "long"  # type: ignore[attr-defined]
    _torch.int32 = "int32"  # type: ignore[attr-defined]
    _torch.int64 = "int64"  # type: ignore[attr-defined]
    _torch.bool = "bool"  # type: ignore[attr-defined]

    # ── Common tensor-creation / math functions ───────────────────────────────
    def _noop(*args: object, **kwargs: object) -> None:  # type: ignore[return]
        return None

    _torch.randn = _noop  # type: ignore[attr-defined]
    _torch.zeros = _noop  # type: ignore[attr-defined]
    _torch.ones = _noop  # type: ignore[attr-defined]
    _torch.tensor = _noop  # type: ignore[attr-defined]
    _torch.randint = _noop  # type: ignore[attr-defined]
    _torch.arange = _noop  # type: ignore[attr-defined]
    _torch.linspace = _noop  # type: ignore[attr-defined]
    _torch.cat = _noop  # type: ignore[attr-defined]
    _torch.stack = _noop  # type: ignore[attr-defined]
    _torch.sigmoid = _noop  # type: ignore[attr-defined]
    _torch.softmax = _noop  # type: ignore[attr-defined]
    _torch.relu = _noop  # type: ignore[attr-defined]
    _torch.clamp = _noop  # type: ignore[attr-defined]
    _torch.mean = _noop  # type: ignore[attr-defined]
    _torch.sum = _noop  # type: ignore[attr-defined]
    _torch.max = _noop  # type: ignore[attr-defined]
    _torch.min = _noop  # type: ignore[attr-defined]
    _torch.abs = _noop  # type: ignore[attr-defined]
    _torch.exp = _noop  # type: ignore[attr-defined]
    _torch.log = _noop  # type: ignore[attr-defined]
    _torch.sqrt = _noop  # type: ignore[attr-defined]
    _torch.pow = _noop  # type: ignore[attr-defined]
    _torch.where = _noop  # type: ignore[attr-defined]
    _torch.manual_seed = _noop  # type: ignore[attr-defined]
    _torch.save = _noop  # type: ignore[attr-defined]
    _torch.load = lambda *a, **kw: {}  # type: ignore[attr-defined]
    _torch.device = lambda *a, **kw: "cpu"  # type: ignore[attr-defined]
    _torch.get_rng_state = lambda: None  # type: ignore[attr-defined]
    _torch.set_rng_state = _noop  # type: ignore[attr-defined]
    _torch.rand = _noop  # type: ignore[attr-defined]
    _torch.randperm = _noop  # type: ignore[attr-defined]
    _torch.full = _noop  # type: ignore[attr-defined]
    _torch.eye = _noop  # type: ignore[attr-defined]
    _torch.norm = _noop  # type: ignore[attr-defined]
    _torch.flatten = _noop  # type: ignore[attr-defined]
    _torch.reshape = _noop  # type: ignore[attr-defined]
    _torch.squeeze = _noop  # type: ignore[attr-defined]
    _torch.unsqueeze = _noop  # type: ignore[attr-defined]
    _torch.transpose = _noop  # type: ignore[attr-defined]
    _torch.matmul = _noop  # type: ignore[attr-defined]
    _torch.bmm = _noop  # type: ignore[attr-defined]
    _torch.einsum = _noop  # type: ignore[attr-defined]
    _torch.isnan = _noop  # type: ignore[attr-defined]
    _torch.isinf = _noop  # type: ignore[attr-defined]
    _torch.isfinite = _noop  # type: ignore[attr-defined]
    _torch.any = _noop  # type: ignore[attr-defined]
    _torch.all = _noop  # type: ignore[attr-defined]
    _torch.argmax = _noop  # type: ignore[attr-defined]
    _torch.argmin = _noop  # type: ignore[attr-defined]
    _torch.topk = _noop  # type: ignore[attr-defined]
    _torch.sort = _noop  # type: ignore[attr-defined]
    _torch.unique = _noop  # type: ignore[attr-defined]
    _torch.zeros_like = _noop  # type: ignore[attr-defined]
    _torch.ones_like = _noop  # type: ignore[attr-defined]
    _torch.rand_like = _noop  # type: ignore[attr-defined]
    _torch.randn_like = _noop  # type: ignore[attr-defined]
    _torch.from_numpy = _noop  # type: ignore[attr-defined]
    _torch.no_op = _noop  # type: ignore[attr-defined]

    # ── torch.cuda ────────────────────────────────────────────────────────────
    _cuda = _stub_module("torch.cuda")
    _cuda.is_available = lambda: False  # type: ignore[attr-defined]
    _cuda.manual_seed_all = _noop  # type: ignore[attr-defined]
    _cuda.device_count = lambda: 0  # type: ignore[attr-defined]
    _cuda.get_rng_state_all = lambda: []  # type: ignore[attr-defined]
    _cuda.set_rng_state_all = _noop  # type: ignore[attr-defined]
    _cuda.empty_cache = _noop  # type: ignore[attr-defined]
    _cuda.synchronize = _noop  # type: ignore[attr-defined]
    _cuda.amp = _stub_module("torch.cuda.amp")
    _torch.cuda = _cuda  # type: ignore[attr-defined]

    # ── torch.nn ─────────────────────────────────────────────────────────────
    _nn = _stub_module("torch.nn")

    class _Module:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

        def __call__(self, *a: object, **kw: object) -> None:
            return None

        def parameters(self, recurse: bool = True):
            return iter([])

        def named_parameters(self, *a: object, **kw: object):
            return iter([])

        def named_modules(self, *a: object, **kw: object):
            return iter([])

        def to(self, *a: object, **kw: object) -> "_Module":
            return self

        def eval(self) -> "_Module":
            return self

        def train(self, mode: bool = True) -> "_Module":
            return self

        def state_dict(self) -> dict:
            return {}

        def load_state_dict(self, *a: object, **kw: object) -> None:
            pass

        def forward(self, *a: object, **kw: object) -> None:
            return None

        def zero_grad(self, set_to_none: bool = True) -> None:
            pass

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}(stub)"

    _nn.Module = _Module  # type: ignore[attr-defined]
    _nn.Linear = _Module  # type: ignore[attr-defined]
    _nn.Sequential = _Module  # type: ignore[attr-defined]
    _nn.Dropout = _Module  # type: ignore[attr-defined]
    _nn.BatchNorm1d = _Module  # type: ignore[attr-defined]
    _nn.BatchNorm2d = _Module  # type: ignore[attr-defined]
    _nn.LayerNorm = _Module  # type: ignore[attr-defined]
    _nn.MultiheadAttention = _Module  # type: ignore[attr-defined]
    _nn.Embedding = _Module  # type: ignore[attr-defined]
    _nn.LSTM = _Module  # type: ignore[attr-defined]
    _nn.GRU = _Module  # type: ignore[attr-defined]
    _nn.ReLU = _Module  # type: ignore[attr-defined]
    _nn.GELU = _Module  # type: ignore[attr-defined]
    _nn.Sigmoid = _Module  # type: ignore[attr-defined]
    _nn.Softmax = _Module  # type: ignore[attr-defined]
    _nn.LogSoftmax = _Module  # type: ignore[attr-defined]
    _nn.Tanh = _Module  # type: ignore[attr-defined]
    _nn.CrossEntropyLoss = _Module  # type: ignore[attr-defined]
    _nn.MSELoss = _Module  # type: ignore[attr-defined]
    _nn.BCELoss = _Module  # type: ignore[attr-defined]
    _nn.BCEWithLogitsLoss = _Module  # type: ignore[attr-defined]
    _nn.NLLLoss = _Module  # type: ignore[attr-defined]
    _nn.Softplus = _Module  # type: ignore[attr-defined]
    _nn.ELU = _Module  # type: ignore[attr-defined]
    _nn.LeakyReLU = _Module  # type: ignore[attr-defined]
    _nn.ModuleList = _Module  # type: ignore[attr-defined]
    _nn.ModuleDict = _Module  # type: ignore[attr-defined]
    _nn.ParameterList = _Module  # type: ignore[attr-defined]
    _nn.Conv1d = _Module  # type: ignore[attr-defined]
    _nn.Conv2d = _Module  # type: ignore[attr-defined]
    _nn.MaxPool1d = _Module  # type: ignore[attr-defined]
    _nn.AvgPool1d = _Module  # type: ignore[attr-defined]
    _nn.AdaptiveAvgPool1d = _Module  # type: ignore[attr-defined]
    _nn.TransformerEncoder = _Module  # type: ignore[attr-defined]
    _nn.TransformerEncoderLayer = _Module  # type: ignore[attr-defined]
    _nn.Parameter = lambda data=None, requires_grad=True: data  # type: ignore[attr-defined]
    _nn_functional = _stub_module("torch.nn.functional")
    _nn_functional.cross_entropy = _noop  # type: ignore[attr-defined]
    _nn_functional.mse_loss = _noop  # type: ignore[attr-defined]
    _nn_functional.binary_cross_entropy = _noop  # type: ignore[attr-defined]
    _nn_functional.binary_cross_entropy_with_logits = _noop  # type: ignore[attr-defined]
    _nn_functional.relu = _noop  # type: ignore[attr-defined]
    _nn_functional.softmax = _noop  # type: ignore[attr-defined]
    _nn_functional.log_softmax = _noop  # type: ignore[attr-defined]
    _nn_functional.sigmoid = _noop  # type: ignore[attr-defined]
    _nn_functional.dropout = _noop  # type: ignore[attr-defined]
    _nn.functional = _nn_functional  # type: ignore[attr-defined]
    _nn_init = _stub_module("torch.nn.init")
    _nn_init.xavier_uniform_ = _noop  # type: ignore[attr-defined]
    _nn_init.xavier_normal_ = _noop  # type: ignore[attr-defined]
    _nn_init.kaiming_uniform_ = _noop  # type: ignore[attr-defined]
    _nn_init.kaiming_normal_ = _noop  # type: ignore[attr-defined]
    _nn_init.zeros_ = _noop  # type: ignore[attr-defined]
    _nn_init.ones_ = _noop  # type: ignore[attr-defined]
    _nn_init.constant_ = _noop  # type: ignore[attr-defined]
    _nn.init = _nn_init  # type: ignore[attr-defined]
    _torch.nn = _nn  # type: ignore[attr-defined]

    # ── torch.optim ───────────────────────────────────────────────────────────
    _optim = _stub_module("torch.optim")

    class _OptimizerStub:
        def __init__(self, *a: object, **kw: object) -> None:
            self.param_groups: list = [{"lr": 1e-3}]

        def step(self, *a: object, **kw: object) -> None:
            pass

        def zero_grad(self, *a: object, **kw: object) -> None:
            pass

        def state_dict(self) -> dict:
            return {}

        def load_state_dict(self, *a: object, **kw: object) -> None:
            pass

    _optim.Adam = _OptimizerStub  # type: ignore[attr-defined]
    _optim.AdamW = _OptimizerStub  # type: ignore[attr-defined]
    _optim.SGD = _OptimizerStub  # type: ignore[attr-defined]
    _optim.RMSprop = _OptimizerStub  # type: ignore[attr-defined]

    _optim_lr = _stub_module("torch.optim.lr_scheduler")
    _optim_lr.ReduceLROnPlateau = _Module  # type: ignore[attr-defined]
    _optim_lr.CosineAnnealingLR = _Module  # type: ignore[attr-defined]
    _optim_lr.OneCycleLR = _Module  # type: ignore[attr-defined]
    _optim_lr.StepLR = _Module  # type: ignore[attr-defined]
    _optim_lr.LinearLR = _Module  # type: ignore[attr-defined]
    _optim_lr.SequentialLR = _Module  # type: ignore[attr-defined]
    _optim.lr_scheduler = _optim_lr  # type: ignore[attr-defined]
    _torch.optim = _optim  # type: ignore[attr-defined]

    # ── torch.utils / torch.utils.data / torch.utils.checkpoint ──────────────
    _utils = _stub_module("torch.utils")

    _utils_data = _stub_module("torch.utils.data")

    class _Dataset:
        pass

    class _DataLoader:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

        def __iter__(self):
            return iter([])

        def __len__(self) -> int:
            return 0

    class _WeightedRandomSampler:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

        def __iter__(self):
            return iter([])

        def __len__(self) -> int:
            return 0

    _utils_data.Dataset = _Dataset  # type: ignore[attr-defined]
    _utils_data.DataLoader = _DataLoader  # type: ignore[attr-defined]
    _utils_data.WeightedRandomSampler = _WeightedRandomSampler  # type: ignore[attr-defined]
    _utils_data.Sampler = object  # type: ignore[attr-defined]
    _utils_data.RandomSampler = object  # type: ignore[attr-defined]
    _utils_data.SequentialSampler = object  # type: ignore[attr-defined]
    _utils_data.random_split = _noop  # type: ignore[attr-defined]
    _utils.data = _utils_data  # type: ignore[attr-defined]

    # torch.utils.checkpoint — needed by backbone.py:
    #   import torch.utils.checkpoint as gradient_checkpoint
    # The plain stub above makes torch.utils a non-package ModuleType, so
    # importing a sub-attribute fails unless we register it in sys.modules.
    _utils_checkpoint = _stub_module("torch.utils.checkpoint")
    _utils_checkpoint.checkpoint = lambda fn, *args, use_reentrant=True, **kw: fn(*args, **kw)  # type: ignore[attr-defined]
    _utils.checkpoint = _utils_checkpoint  # type: ignore[attr-defined]

    _torch.utils = _utils  # type: ignore[attr-defined]

    # ── Other common submodules ───────────────────────────────────────────────
    _stub_module("torch.jit")
    _stub_module("torch.onnx")
    _stub_module("torch.distributed")
    _stub_module("torch.multiprocessing")
    _autograd = _stub_module("torch.autograd")
    _autograd.Function = object  # type: ignore[attr-defined]
    _torch.autograd = _autograd  # type: ignore[attr-defined]
    _backends = _stub_module("torch.backends")
    _backends.cudnn = _stub_module("torch.backends.cudnn")
    _torch.backends = _backends  # type: ignore[attr-defined]
    _stub_module("torch.amp")

    # ── torch.cuda.amp (GradScaler used by training engines) ─────────────────
    _cuda_amp = _stub_module("torch.cuda.amp")

    class _GradScaler:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

        def scale(self, loss: object) -> object:
            return loss

        def step(self, optimizer: object, *a: object, **kw: object) -> None:
            pass

        def update(self) -> None:
            pass

        def get_scale(self) -> float:
            return 1.0

        def state_dict(self) -> dict:
            return {}

        def load_state_dict(self, *a: object, **kw: object) -> None:
            pass

    _cuda_amp.GradScaler = _GradScaler  # type: ignore[attr-defined]
    _cuda_amp.autocast = _NoGrad  # type: ignore[attr-defined]
    _cuda.amp = _cuda_amp  # type: ignore[attr-defined]

    # ── torchmetrics stub (used by trainer/metrics/evaluation.py) ────────────
    _torchmetrics = _stub_module("torchmetrics")

    class _MetricStub:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

        def __call__(self, *a: object, **kw: object) -> None:
            return None

        def to(self, *a: object, **kw: object) -> "_MetricStub":
            return self

        def update(self, *a: object, **kw: object) -> None:
            pass

        def compute(self) -> None:
            return None

        def reset(self) -> None:
            pass

    _torchmetrics.Accuracy = _MetricStub  # type: ignore[attr-defined]
    _torchmetrics.Precision = _MetricStub  # type: ignore[attr-defined]
    _torchmetrics.Recall = _MetricStub  # type: ignore[attr-defined]
    _torchmetrics.F1Score = _MetricStub  # type: ignore[attr-defined]
    _torchmetrics.AUROC = _MetricStub  # type: ignore[attr-defined]
    _torchmetrics.MeanSquaredError = _MetricStub  # type: ignore[attr-defined]
    _torchmetrics.MeanAbsoluteError = _MetricStub  # type: ignore[attr-defined]
    _torchmetrics.R2Score = _MetricStub  # type: ignore[attr-defined]
    _torchmetrics.MetricCollection = _MetricStub  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# pytest marker + auto-skip hook
# ---------------------------------------------------------------------------
import pytest  # noqa: E402 — must come after the torch stub block


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_torch: mark test as requiring a working PyTorch installation "
        "with native extensions (skipped automatically when torch is unavailable)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip any test marked ``requires_torch`` when torch is not usable."""
    if HAS_REAL_TORCH:
        return
    skip_no_torch = pytest.mark.skip(
        reason=(
            "requires a working PyTorch installation — "
            "torch native extensions are unavailable in this environment"
        )
    )
    for item in items:
        if item.get_closest_marker("requires_torch"):
            item.add_marker(skip_no_torch)
