"""Visage dependency check, run by Forge at startup.

Rules (learned from other face swap extensions breaking Forge installs):
  * never uninstall or replace an onnxruntime build that is already there;
  * install exactly one onnxruntime distribution, only when none is present;
  * no insightface, no compilers, no pkg_resources.
"""

import os
import shutil
import subprocess
import sys
from importlib import metadata

import launch

TAG = "[Visage]"

ORT_DISTS = (
    "onnxruntime-gpu",
    "onnxruntime-directml",
    "onnxruntime-rocm",
    "onnxruntime-migraphx",
    "onnxruntime-openvino",
    "onnxruntime-silicon",
    "onnxruntime-qnn",
    "onnxruntime",
)


def installed(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def has_nvidia() -> bool:
    exe = shutil.which("nvidia-smi")
    if exe is None and os.name == "nt":
        candidate = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")
        exe = candidate if os.path.isfile(candidate) else None
    if exe is None:
        return False
    try:
        out = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=10,
                             creationflags=0x08000000 if os.name == "nt" else 0)
        return "GPU" in (out.stdout or "")
    except Exception:
        return False


def torch_flavour() -> str:
    try:
        version = metadata.version("torch")
    except metadata.PackageNotFoundError:
        return "unknown"
    if "+rocm" in version:
        return "rocm"
    if "+cu" in version:
        return "cuda"
    return "other"


def ensure_onnxruntime():
    present = {d: v for d in ORT_DISTS if (v := installed(d))}
    if present:
        if len(present) > 1:
            listing = ", ".join(f"{d} {v}" for d, v in present.items())
            print(f"{TAG} Warning: several onnxruntime builds are installed ({listing}).")
            print(f"{TAG}   They share one module and can break each other. Visage will not touch them;")
            print(f"{TAG}   if face swap fails, uninstall all of them and install just one.")
        return

    if os.environ.get("VISAGE_ORT_PACKAGE"):
        package = os.environ["VISAGE_ORT_PACKAGE"]
    elif sys.platform == "darwin":
        package = "onnxruntime"
    elif has_nvidia() and torch_flavour() != "rocm":
        package = "onnxruntime-gpu"
    elif os.name == "nt":
        package = "onnxruntime-directml"  # AMD / Intel GPUs on Windows
    else:
        package = "onnxruntime"
    launch.run_pip(f"install {package}", f"{package} for Visage")


def ensure(dist: str, spec: str | None = None, desc: str | None = None):
    if installed(dist) is None:
        launch.run_pip(f"install {spec or dist}", desc or f"{dist} for Visage")


try:
    ensure_onnxruntime()
    ensure("onnx")  # reads the identity projection out of inswapper-style models
    ensure("safetensors")  # opencv, numpy and Pillow always ship with Forge
except Exception as e:  # never block Forge from starting
    print(f"{TAG} dependency check failed: {e}")
