"""CPU-only macOS shims for python/sam3."""
import sys, types
import torch

# Stub triton-using module
edt = types.ModuleType("sam3.model.edt")
def edt_triton(*args, **kwargs):
    raise RuntimeError("edt_triton not available on macOS (stub)")
edt.edt_triton = edt_triton
sys.modules["sam3.model.edt"] = edt

# Fake CUDA props
class _FakeCudaProps:
    major, minor, total_memory, multi_processor_count = 0, 0, 0, 0
torch.cuda.get_device_properties = lambda idx=0: _FakeCudaProps()

# Redirect any cuda device spec to CPU
_CPU = torch.device("cpu")
def _redirect(d):
    if d is None: return d
    if isinstance(d, str) and d.startswith("cuda"): return "cpu"
    if hasattr(d, "type") and getattr(d, "type", None) == "cuda": return _CPU
    return d

def _wrap_creator(fn):
    def wrapped(*args, **kwargs):
        if "device" in kwargs: kwargs["device"] = _redirect(kwargs["device"])
        return fn(*args, **kwargs)
    return wrapped
for name in ("zeros", "ones", "empty", "full", "tensor", "rand", "randn",
             "arange", "eye", "linspace", "as_tensor", "zeros_like", "ones_like",
             "empty_like", "full_like"):
    if hasattr(torch, name):
        setattr(torch, name, _wrap_creator(getattr(torch, name)))

# Tensor.to / .cuda — handle both str args and torch.device args
_orig_tto = torch.Tensor.to
def _patched_tto(self, *args, **kwargs):
    new_args = tuple(_redirect(a) for a in args)
    if "device" in kwargs: kwargs["device"] = _redirect(kwargs["device"])
    return _orig_tto(self, *new_args, **kwargs)
torch.Tensor.to = _patched_tto
torch.Tensor.cuda = lambda self, *a, **k: self

# Module.to / .cuda
import torch.nn as _nn
_orig_mto = _nn.Module.to
def _patched_mto(self, *args, **kwargs):
    new_args = tuple(_redirect(a) for a in args)
    if "device" in kwargs: kwargs["device"] = _redirect(kwargs["device"])
    return _orig_mto(self, *new_args, **kwargs)
_nn.Module.to = _patched_mto
_nn.Module.cuda = lambda self, *a, **k: self

# autocast device_type=cuda → cpu
_orig_autocast = torch.autocast
class _PatchedAutocast(_orig_autocast):
    def __init__(self, device_type=None, *args, **kwargs):
        if device_type == "cuda":
            device_type = "cpu"
            kwargs.pop("dtype", None)
        super().__init__(device_type, *args, **kwargs)
torch.autocast = _PatchedAutocast

# pin_memory uses cuda; no-op on cpu
torch.Tensor.pin_memory = lambda self, *a, **k: self

print("[setup_stubs] cuda → cpu shims applied")

# Disable autocast entirely on CPU — many ops don't support cpu autocast.
class _NoAutocast:
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def __call__(self, fn): return fn
torch.autocast = _NoAutocast
torch.amp.autocast = _NoAutocast
torch.cuda.amp.autocast = _NoAutocast
print("[setup_stubs] autocast disabled")

# Force everything to fp32 — block bf16 conversions so model + inputs match.
import torch as _t
def _is_bf16(x):
    return x is _t.bfloat16
_orig_tto2 = _t.Tensor.to
def _patched_tto2(self, *args, **kwargs):
    args = tuple("cpu" if (isinstance(a, str) and a.startswith("cuda")) else a for a in args)
    args = tuple(a for a in args if not _is_bf16(a))  # drop bf16 dtype args
    if "device" in kwargs:
        d = kwargs["device"]
        if isinstance(d, str) and d.startswith("cuda"): kwargs["device"] = "cpu"
        elif hasattr(d, "type") and d.type == "cuda": kwargs["device"] = _t.device("cpu")
    if kwargs.get("dtype") is _t.bfloat16:
        kwargs.pop("dtype")
    return _orig_tto2(self, *args, **kwargs)
_t.Tensor.to = _patched_tto2
_t.Tensor.bfloat16 = lambda self, *a, **k: self  # no-op

print("[setup_stubs] forced fp32 (bf16 casts dropped)")

# Drop pin_memory=True from tensor creators (only valid with CUDA).
for _name in ("zeros", "ones", "empty", "full", "tensor", "rand", "randn",
              "arange", "eye", "linspace", "as_tensor"):
    _orig = getattr(_t, _name)
    def _make(orig):
        def w(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return orig(*args, **kwargs)
        return w
    setattr(_t, _name, _make(_orig))
print("[setup_stubs] pin_memory dropped")
