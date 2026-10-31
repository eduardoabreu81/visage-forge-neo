"""Visage dependency check, run by Forge at startup.

Rules (learned from other face swap extensions breaking Forge installs):
  * never install, uninstall or replace anything in Forge's venv;
  * the onnxruntime builds Visage needs go into <extension>/runtime (lib_visage/ortenv.py);
  * no insightface, no compilers, no pkg_resources.
"""

import json
import os
import shutil
import subprocess
import sys
from importlib import metadata

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib_visage import ortenv  # noqa: E402

TAG = "[Visage]"


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


def torch_version() -> str:
    try:
        return metadata.version("torch")
    except metadata.PackageNotFoundError:
        return ""


def torch_cuda_major() -> int | None:
    """13 for torch 2.x+cu130, 12 for +cu128, None when torch is not a CUDA build."""
    version = torch_version()
    if "+cu" not in version:
        return None
    digits = version.split("+cu", 1)[1]
    return int(digits[:-1]) if len(digits) >= 3 and digits.isdigit() else None


def settings() -> dict:
    try:
        from modules import launch_utils

        if launch_utils.args.skip_install:
            return {"_skip": True}
        path = launch_utils.args.ui_settings_file
    except Exception:
        path = "config.json"
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    opts = settings()
    if opts.get("_skip"):
        return
    if not ortenv.venv_has_onnxruntime() and not ortenv.installed("cpu"):
        ortenv.install("cpu", "onnxruntime")
    if opts.get("visage_device") == "GPU":
        rocm = "+rocm" in torch_version()
        package = ortenv.gpu_package(nvidia=has_nvidia() and not rocm, cuda_major=torch_cuda_major(), rocm=rocm)
        if not ortenv.installed("gpu", package):
            ortenv.install("gpu", package)


try:
    main()
except Exception as e:  # never block Forge from starting
    print(f"{TAG} dependency check failed: {e}")
