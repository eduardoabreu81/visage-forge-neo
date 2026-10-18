"""ONNX Runtime session management.

Visage only *uses* whatever onnxruntime distribution is installed; it never
swaps one build for another at runtime (that is how other face swap
extensions broke Forge installs). Provider choice is automatic unless the
user pins one in settings.
"""

from __future__ import annotations

import gc
import os
import threading

from . import models

_sessions: dict[tuple[str, str], object] = {}
_lock = threading.Lock()
provider_preference = "auto"

# Order matters: first available wins in "auto" mode.
_AUTO_ORDER = [
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "MIGraphXExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
]

PROVIDER_CHOICES = {
    "auto": None,
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "directml": "DmlExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


def ort():
    try:
        import onnxruntime
    except ImportError as e:  # pragma: no cover - surfaced in the UI
        raise RuntimeError(
            "onnxruntime is not installed. Restart Forge so Visage's installer can add it, "
            "or install onnxruntime-gpu (NVIDIA) / onnxruntime-directml (AMD, Intel on Windows) manually."
        ) from e
    if not hasattr(onnxruntime, "InferenceSession"):
        raise RuntimeError(
            "onnxruntime is broken in this venv (no InferenceSession). This usually means two onnxruntime "
            "builds were installed on top of each other. Uninstall every onnxruntime* package and install one."
        )
    return onnxruntime


def available_providers() -> list[str]:
    try:
        return list(ort().get_available_providers())
    except RuntimeError:
        return []


def resolve_providers() -> list:
    available = available_providers()
    pinned = PROVIDER_CHOICES.get(provider_preference)
    chosen: list = []
    if pinned and pinned in available:
        chosen.append(pinned)
    else:
        for name in _AUTO_ORDER:
            if name in available:
                chosen.append(name)
                break
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return [_with_options(p) for p in chosen]


def _with_options(name: str):
    if name == "CUDAExecutionProvider":
        return (name, {"cudnn_conv_algo_search": "HEURISTIC", "arena_extend_strategy": "kSameAsRequested"})
    if name == "TensorrtExecutionProvider":
        return (name, {"trt_engine_cache_enable": True, "trt_engine_cache_path": models.models_dir() + "/trt_cache"})
    return name


def active_provider() -> str:
    p = resolve_providers()[0]
    return p[0] if isinstance(p, tuple) else p


def ort_gpu_spec(cuda_major: int | None) -> str:
    """onnxruntime-gpu on PyPI is built for CUDA 13 from 1.27 on, CUDA 12 before that."""
    return "onnxruntime-gpu<1.27" if cuda_major is not None and cuda_major < 13 else "onnxruntime-gpu>=1.27"


def gpu_hint(nvidia: bool, cuda_major: int | None = None) -> str | None:
    """Advice for when an NVIDIA GPU is there but the installed onnxruntime can only use the CPU."""
    available = available_providers()
    if not nvidia or not available or {"CUDAExecutionProvider", "TensorrtExecutionProvider"} & set(available):
        return None
    py = r"venv\Scripts\python" if os.name == "nt" else "venv/bin/python"
    return (
        "Visage is running on the CPU: this machine has an NVIDIA GPU, but the installed onnxruntime has no CUDA support. "
        f"Close Forge and run, in the Forge folder: `{py} -m pip uninstall -y onnxruntime` and then "
        f"`{py} -m pip install \"{ort_gpu_spec(cuda_major)}\"`. Visage never changes onnxruntime on its own."
    )


_cuda_dlls_ready = False


def _prepare_cuda():
    """Let onnxruntime find the CUDA / cuDNN DLLs that PyTorch already ships (onnxruntime >= 1.21)."""
    global _cuda_dlls_ready
    if _cuda_dlls_ready:
        return
    _cuda_dlls_ready = True
    preload = getattr(ort(), "preload_dlls", None)
    if preload is not None:
        try:
            preload()
        except Exception:
            pass


def session(key: str, progress=None):
    path = models.ensure(key, progress)
    provider = active_provider()
    cache_key = (path, provider)
    with _lock:
        sess = _sessions.get(cache_key)
        if sess is None:
            rt = ort()
            if provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
                _prepare_cuda()
            opts = rt.SessionOptions()
            opts.log_severity_level = 3
            opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = rt.InferenceSession(path, sess_options=opts, providers=resolve_providers())
            _sessions[cache_key] = sess
        return sess


def loaded() -> list[str]:
    return sorted({p for p, _ in _sessions})


def unload(kind: str | None = None) -> int:
    """Drop cached sessions (all, or only those of one model kind). Returns how many."""
    with _lock:
        if kind is None:
            dropped = len(_sessions)
            _sessions.clear()
        else:
            paths = {models.local_path(k) for k in models.keys_of(kind)}
            victims = [k for k in _sessions if k[0] in paths]
            for k in victims:
                del _sessions[k]
            dropped = len(victims)
    gc.collect()
    return dropped


def input_names(sess) -> list[str]:
    return [i.name for i in sess.get_inputs()]
