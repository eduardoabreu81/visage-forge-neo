"""ONNX Runtime session management.

Visage never adds, removes or replaces packages in Forge's venv. It runs on the CPU by
default, with whatever onnxruntime the venv has (or its own CPU build when the venv has none).
When the user picks the GPU in settings:

  * if the venv's onnxruntime already has a GPU provider, it is used in Forge's process;
  * otherwise Visage's own GPU build (``ortenv``) runs in a separate worker process
    (``ortworker``), so it never shares an interpreter with the venv's onnxruntime.

Anything that goes wrong on the GPU path falls back to the CPU and is reported in ``note()``.
"""

from __future__ import annotations

import gc
import os
import secrets
import subprocess
import sys
import threading

from . import models, ortenv

DEVICES = ("CPU", "GPU")
device = "CPU"
gpu_package: str | None = None  # set by the Forge glue from the GPU it sees; None = no GPU build to offer

_sessions: dict[tuple[str, str], object] = {}
_lock = threading.Lock()
_notes: dict[str, str] = {}

# first available wins
GPU_PROVIDERS = [
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "MIGraphXExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
]


def _with_options(name: str):
    if name == "CUDAExecutionProvider":
        return (name, {"cudnn_conv_algo_search": "HEURISTIC", "arena_extend_strategy": "kSameAsRequested"})
    return name


def _gpu_first(available: list[str]) -> str | None:
    return next((p for p in GPU_PROVIDERS if p in available), None)


# --- the venv's onnxruntime, in Forge's process ---------------------------------------

def ort():
    try:
        import onnxruntime
    except ImportError as e:
        if not ortenv.installed("cpu"):
            raise RuntimeError(
                "onnxruntime is not available. Restart Forge so Visage can add its own CPU build "
                "(it goes into the extension folder, not into Forge's venv)."
            ) from e
        sys.path.append(ortenv.path("cpu"))  # appended: a build Forge installs later still wins
        import onnxruntime
    if not hasattr(onnxruntime, "InferenceSession"):
        raise RuntimeError(
            "onnxruntime is broken in this venv (no InferenceSession). This usually means two onnxruntime "
            "builds were installed on top of each other."
        )
    return onnxruntime


def _local_providers() -> list[str]:
    try:
        return list(ort().get_available_providers())
    except RuntimeError:
        return []


def _local_session(path: str, gpu: str | None):
    key = (path, gpu or "CPUExecutionProvider")
    sess = _sessions.get(key)
    if sess is None:
        rt = ort()
        if gpu in ("CUDAExecutionProvider",) and hasattr(rt, "preload_dlls"):
            try:
                rt.preload_dlls()  # reuses torch's CUDA / cuDNN when torch is already imported
            except Exception:
                pass
        opts = rt.SessionOptions()
        opts.log_severity_level = 3
        opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ([_with_options(gpu)] if gpu else []) + ["CPUExecutionProvider"]
        sess = rt.InferenceSession(path, sess_options=opts, providers=providers)
        _sessions[key] = sess
    return sess


# --- Visage's own GPU build, in a worker process -------------------------------------

class _IO:
    __slots__ = ("name", "shape", "type")

    def __init__(self, name, shape, type_):
        self.name, self.shape, self.type = name, shape, type_


class RemoteSession:
    """Stands in for ``onnxruntime.InferenceSession``; runs in the worker process."""

    def __init__(self, worker: "_Worker", path: str, meta: dict):
        self._worker, self._path = worker, path
        self._inputs = [_IO(*i) for i in meta["inputs"]]
        self._outputs = [_IO(*o) for o in meta["outputs"]]
        self._providers = meta["providers"]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_providers(self):
        return self._providers

    def run(self, output_names, feed):
        return self._worker.call("run", self._path, output_names, feed)


class _Worker:
    def __init__(self, runtime_dir: str):
        from multiprocessing.connection import Listener

        family = "AF_PIPE" if os.name == "nt" else "AF_UNIX"
        key = secrets.token_bytes(32)
        self._lock = threading.Lock()
        listener = Listener(family=family, authkey=key)
        env = dict(os.environ, VISAGE_WORKER_KEY=key.hex())
        if os.name != "nt":
            from .ortworker import cuda_lib_dirs

            env["LD_LIBRARY_PATH"] = os.pathsep.join(cuda_lib_dirs() + [env.get("LD_LIBRARY_PATH", "")])
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ortworker.py")
        self._proc = subprocess.Popen([sys.executable, "-I", script, listener.address, family, runtime_dir], env=env,
                                      creationflags=0x08000000 if os.name == "nt" else 0)
        try:
            self._conn = _accept(listener, self._proc)
        finally:
            listener.close()
        status, info = self._conn.recv()
        if status != "ready":
            self.stop()
            raise RuntimeError(info)
        self.version, self.providers = info["version"], info["providers"]

    def alive(self) -> bool:
        return self._proc.poll() is None

    def call(self, *msg):
        with self._lock:
            try:
                self._conn.send(msg)
                status, value = self._conn.recv()
            except (EOFError, OSError) as e:
                raise RuntimeError(f"the Visage GPU worker stopped ({e or 'no reply'})") from e
        if status != "ok":
            raise RuntimeError(value)
        return value

    def stop(self):
        try:
            if self.alive():
                self.call("exit")
        except Exception:
            pass
        try:
            self._proc.wait(timeout=10)
        except Exception:
            self._proc.kill()


def _accept(listener, proc, timeout: float = 120.0):
    """Wait for the worker to connect, giving up if it exits or takes longer than ``timeout``."""
    result: dict = {}

    def run():
        try:
            result["conn"] = listener.accept()
        except Exception as e:  # listener closed below
            result["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    waited = 0.0
    while t.is_alive() and proc.poll() is None and waited < timeout:
        t.join(0.25)
        waited += 0.25
    t.join(0.5)  # the worker may have connected right before exiting with an error message
    if "conn" not in result:
        proc.kill()
        raise RuntimeError(f"the Visage GPU worker did not start (exit code {proc.poll()})")
    return result["conn"]


_worker: _Worker | None = None
_worker_failed = False


def _get_worker() -> _Worker | None:
    global _worker, _worker_failed
    if _worker is not None and _worker.alive():
        return _worker
    if _worker is not None:  # crashed: drop its sessions and do not retry until settings change
        _drop_remote()
        _worker = None
        _worker_failed = True
        _notes["gpu"] = "the GPU worker stopped unexpectedly; Visage is using the CPU until the GPU setting is applied again"
    if _worker_failed or not gpu_package:
        return None
    if not ortenv.installed("gpu", gpu_package):
        ortenv.install_in_background("gpu", gpu_package)
        state = ortenv.status("gpu") or "installing"
        _notes["gpu"] = (f"{gpu_package} is being installed for Visage; the CPU is used until it is ready"
                         if state == "installing" else f"{gpu_package} could not be installed ({state}); using the CPU")
        return None
    try:
        _worker = _Worker(ortenv.path("gpu"))
    except Exception as e:
        _worker_failed = True
        _notes["gpu"] = f"the GPU worker could not start ({e}); using the CPU"
        return None
    if _gpu_first(_worker.providers) is None:
        _notes["gpu"] = f"{gpu_package} {_worker.version} sees no GPU (providers: {', '.join(_worker.providers)}); using the CPU"
        _worker.stop()
        _worker, _worker_failed = None, True
        return None
    _notes.pop("gpu", None)
    return _worker


def _remote_session(path: str):
    key = (path, "worker")
    sess = _sessions.get(key)
    if sess is None:
        worker = _get_worker()
        if worker is None:
            return None
        gpu = _gpu_first(worker.providers)
        meta = worker.call("load", path, [_with_options(gpu), "CPUExecutionProvider"])
        sess = RemoteSession(worker, path, meta)
        _sessions[key] = sess
    return sess


def _drop_remote():
    for k in [k for k in _sessions if k[1] == "worker"]:
        del _sessions[k]


def stop_worker():
    global _worker
    with _lock:
        _drop_remote()
        if _worker is not None:
            _worker.stop()
            _worker = None


# --- public API --------------------------------------------------------------------------

def configure(new_device: str, package: str | None) -> None:
    """Apply the device setting. Switching to CPU stops the worker, so its VRAM goes back to Forge."""
    global device, gpu_package, _worker_failed
    new_device = new_device if new_device in DEVICES else "CPU"
    if (new_device, package) != (device, gpu_package):
        _worker_failed = False
        _notes.pop("gpu", None)
        if new_device == "CPU" or package != gpu_package:
            stop_worker()
    device, gpu_package = new_device, package


def session(key: str, progress=None):
    path = models.ensure(key, progress)
    with _lock:
        if device == "GPU":
            local_gpu = _gpu_first(_local_providers())
            if local_gpu:
                return _local_session(path, local_gpu)
            try:
                remote = _remote_session(path)
            except Exception as e:
                _notes["gpu"] = f"the GPU worker failed ({e}); using the CPU"
                remote = None
            if remote is not None:
                return remote
        return _local_session(path, None)


def available_providers() -> list[str]:
    """Providers of the backend Visage uses with the current setting."""
    if device == "GPU":
        local = _local_providers()
        if _gpu_first(local):
            return local
        if _worker is not None and _worker.alive():
            return list(_worker.providers)
    return _local_providers()


def active_provider() -> str:
    if device == "GPU":
        gpu = _gpu_first(available_providers())
        if gpu:
            return gpu
    if not _local_providers():
        raise RuntimeError("onnxruntime not available")
    return "CPUExecutionProvider"


def backend() -> str:
    """Where Visage runs right now, for the status line."""
    if device == "GPU":
        if _gpu_first(_local_providers()):
            return "GPU, Forge's onnxruntime"
        if _worker is not None and _worker.alive():
            return f"GPU, Visage's {gpu_package} {_worker.version} (separate process)"
    return "CPU"


def note(nvidia: bool = False) -> str | None:
    """Something the user should know about the device choice, or None."""
    if device == "GPU" and "gpu" in _notes:
        return "Visage: " + _notes["gpu"] + "."
    if device == "GPU" and not gpu_package:
        return "Visage: no supported GPU was found; using the CPU."
    if device == "CPU" and nvidia:
        return "Visage runs on the CPU. To use the GPU, set Settings → Visage → Device to GPU."
    return None


def loaded() -> list[str]:
    return sorted({p for p, _ in _sessions})


def unload(kind: str | None = None) -> int:
    """Drop cached sessions (all, or only those of one model kind). Returns how many.

    Unloading everything also ends the GPU worker, which frees all of its VRAM."""
    with _lock:
        if kind is None:
            dropped = len(_sessions)
            _sessions.clear()
            worker = _worker
        else:
            paths = {models.local_path(k) for k in models.keys_of(kind)}
            victims = [k for k in _sessions if k[0] in paths]
            for k in victims:
                del _sessions[k]
            dropped = len(victims)
            if _worker is not None and _worker.alive():
                try:
                    _worker.call("unload", sorted(paths))
                except Exception:
                    pass
            worker = None
    if worker is not None:
        stop_worker()
    gc.collect()
    return dropped


def input_names(sess) -> list[str]:
    return [i.name for i in sess.get_inputs()]
