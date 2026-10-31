"""Visage's own onnxruntime builds, kept outside Forge's venv.

Forge decides what goes into its venv (the ControlNet preprocessors pull the CPU-only
`onnxruntime` in through insightface, for example). Visage never adds, removes or replaces
anything there. When it needs a build the venv does not have, it installs it with
`pip install --target` into `<extension>/runtime/<flavour>`:

  * `cpu`  only when the venv has no onnxruntime at all; loaded in Forge's process;
  * `gpu`  when the user picks the GPU in settings; loaded in a separate worker process,
           so it never meets the venv's onnxruntime in the same interpreter.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime")
MARKER = "visage-runtime.txt"

_lock = threading.Lock()
_state: dict[str, str] = {}  # flavour -> "installing" | "error: ..." while a background install runs


def path(flavour: str) -> str:
    return os.path.join(ROOT, flavour)


def gpu_package(nvidia: bool, cuda_major: int | None, rocm: bool = False) -> str:
    """The onnxruntime build for this machine's GPU (``VISAGE_ORT_PACKAGE`` overrides it)."""
    if os.environ.get("VISAGE_ORT_PACKAGE"):
        return os.environ["VISAGE_ORT_PACKAGE"]
    if sys.platform == "darwin":
        return "onnxruntime"  # the default macOS build has CoreML
    if nvidia and not rocm:
        # onnxruntime-gpu on PyPI is built for CUDA 13 from 1.27 on, CUDA 12 before that
        return "onnxruntime-gpu<1.27" if cuda_major is not None and cuda_major < 13 else "onnxruntime-gpu>=1.27"
    if os.name == "nt":
        return "onnxruntime-directml"  # AMD / Intel GPUs on Windows
    return "onnxruntime"


def installed(flavour: str, package: str | None = None) -> bool:
    """True when the flavour is fully installed (and, if given, with that package spec)."""
    marker = os.path.join(path(flavour), MARKER)
    if not os.path.isfile(os.path.join(path(flavour), "onnxruntime", "__init__.py")) or not os.path.isfile(marker):
        return False
    if package is None:
        return True
    with open(marker, encoding="utf-8") as f:
        return f.read().strip() == package


def status(flavour: str) -> str | None:
    return _state.get(flavour)


def install(flavour: str, package: str, log=print) -> None:
    """Install ``package`` into ``runtime/<flavour>``. Blocking; safe to call from several threads.

    Dependencies are not installed: onnxruntime only needs numpy at inference time, and
    the numpy that Forge ships is the one both sides must use anyway.
    """
    with _lock:
        if installed(flavour, package):
            return
        _state[flavour] = "installing"
        target = path(flavour)
        staging = target + ".partial"
        shutil.rmtree(staging, ignore_errors=True)
        log(f"[Visage] Installing {package} into {target} (Forge's venv is not touched)...")
        cmd = [sys.executable, "-m", "pip", "install", "--no-deps", "--no-warn-script-location",
               "--disable-pip-version-check", "--target", staging, package]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 creationflags=0x08000000 if os.name == "nt" else 0)
            if out.returncode != 0:
                raise RuntimeError((out.stderr or out.stdout).strip().splitlines()[-1] if (out.stderr or out.stdout) else "pip failed")
            with open(os.path.join(staging, MARKER), "w", encoding="utf-8") as f:
                f.write(package)
            shutil.rmtree(target, ignore_errors=True)
            os.replace(staging, target)
            _state.pop(flavour, None)
            log(f"[Visage] {package} is ready.")
        except Exception as e:
            shutil.rmtree(staging, ignore_errors=True)
            _state[flavour] = f"error: {e}"
            log(f"[Visage] Could not install {package}: {e}")
            raise


def install_in_background(flavour: str, package: str, log=print) -> None:
    if installed(flavour, package) or _state.get(flavour) == "installing":
        return
    _state[flavour] = "installing"

    def run():
        try:
            install(flavour, package, log)
        except Exception:
            pass

    threading.Thread(target=run, name=f"visage-ort-{flavour}", daemon=True).start()


def venv_has_onnxruntime() -> bool:
    """Whether some onnxruntime build is importable from Forge's own environment."""
    from importlib import util

    spec = util.find_spec("onnxruntime")
    return spec is not None and not (spec.origin or "").startswith(ROOT)
