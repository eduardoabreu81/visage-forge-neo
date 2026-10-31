"""Separate process that runs Visage's own GPU onnxruntime build.

Started by `runtime.py` as ``python -I ortworker.py <address> <family> <runtime dir>``. Only the
standard library and numpy are imported before onnxruntime, so the worker never loads the
onnxruntime that lives in Forge's venv: ``<runtime dir>`` is put first on ``sys.path``.

Protocol (multiprocessing.connection, pickled tuples):
    ("load", path, providers)        -> {"inputs": [(name, shape, type)], "outputs": [...], "providers": [...]}
    ("run", path, output_names, feed) -> list of numpy arrays
    ("unload", paths_or_None)         -> number of sessions dropped
    ("exit",)                         -> None, then the worker ends
Each reply is ("ok", value) or ("error", message). The first message the worker sends is
("ready", info) or ("error", message).
"""

from __future__ import annotations

import glob
import os
import sys


def cuda_lib_dirs() -> list[str]:
    """Folders with the CUDA / cuDNN libraries PyTorch already ships, found without importing torch."""
    from importlib import util

    dirs = []
    spec = util.find_spec("torch")
    if spec and spec.origin:
        lib = os.path.join(os.path.dirname(spec.origin), "lib")
        if os.path.isdir(lib):
            dirs.append(lib)
    spec = util.find_spec("nvidia")  # nvidia-*-cu1x wheels (Linux, and some Windows torch builds)
    for root in (spec.submodule_search_locations or []) if spec else []:
        for sub in ("bin", "lib"):
            dirs.extend(d for d in glob.glob(os.path.join(root, "*", sub)) if os.path.isdir(d))
    return dirs


def _prepare_dlls(dirs: list[str]) -> None:
    os.environ["PATH"] = os.pathsep.join(dirs + [os.environ.get("PATH", "")])
    if os.name == "nt":
        for d in dirs:
            try:
                os.add_dll_directory(d)
            except OSError:
                pass


def _io(items) -> list[tuple]:
    return [(i.name, list(i.shape) if i.shape is not None else None, i.type) for i in items]


def main() -> None:
    from multiprocessing.connection import Client

    address, family, runtime_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    conn = Client(address, family=family, authkey=bytes.fromhex(os.environ.pop("VISAGE_WORKER_KEY")))
    sys.path.insert(0, runtime_dir)
    _prepare_dlls(cuda_lib_dirs())
    try:
        import onnxruntime as ort
    except Exception as e:
        conn.send(("error", f"cannot import onnxruntime from {runtime_dir}: {type(e).__name__}: {e}"))
        return
    if not os.path.abspath(ort.__file__).startswith(os.path.abspath(runtime_dir)):
        conn.send(("error", f"loaded the wrong onnxruntime ({ort.__file__})"))
        return
    conn.send(("ready", {"version": ort.__version__, "providers": list(ort.get_available_providers())}))

    sessions: dict[str, object] = {}
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break  # Forge went away
        op = msg[0]
        try:
            if op == "load":
                _, path, providers = msg
                sess = sessions.get(path)
                if sess is None:
                    opts = ort.SessionOptions()
                    opts.log_severity_level = 3
                    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    sess = ort.InferenceSession(path, sess_options=opts, providers=providers)
                    sessions[path] = sess
                value = {"inputs": _io(sess.get_inputs()), "outputs": _io(sess.get_outputs()),
                         "providers": list(sess.get_providers())}
            elif op == "run":
                _, path, names, feed = msg
                value = sessions[path].run(names, feed)
            elif op == "unload":
                paths = msg[1]
                victims = list(sessions) if paths is None else [p for p in paths if p in sessions]
                for p in victims:
                    del sessions[p]
                value = len(victims)
            elif op == "exit":
                conn.send(("ok", None))
                break
            else:
                raise ValueError(f"unknown request {op!r}")
            conn.send(("ok", value))
        except Exception as e:
            conn.send(("error", _message(e)))
    conn.close()


def _message(e: Exception) -> str:
    # onnxruntime errors can carry Windows text in the system code page; pybind11 then fails to
    # decode them and raises UnicodeDecodeError with the original bytes
    if isinstance(e, UnicodeDecodeError) and isinstance(e.object, bytes):
        return e.object.decode("utf-8", "replace").strip()
    return f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    main()
