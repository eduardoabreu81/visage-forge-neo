"""Model registry and on-demand downloads.

Visage never redistributes model weights. Every model is fetched from its
upstream host the first time it is needed and stored under the models
directory (``<forge>/models/visage`` by default). Several of these models are
licensed for non-commercial / research use only; the license is listed next
to each entry and surfaced in the UI.
"""

from __future__ import annotations

import os
import threading
import urllib.request
from dataclasses import dataclass, field

_FF_HF = "https://huggingface.co/facefusion/{release}/resolve/main/{file}"
_FF_GH = "https://github.com/facefusion/facefusion-assets/releases/download/{release}/{file}"
_REACTOR_HF = "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/{file}"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    kind: str  # detector | recognizer | landmarker | swapper | restorer | occluder | parser
    file: str
    urls: tuple[str, ...]
    license: str
    label: str = ""
    # model specific knobs
    template: str = ""
    size: int = 0
    mean: float = 0.0
    std: float = 1.0
    arch: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def title(self) -> str:
        return self.label or self.key


def _ff(release: str, file: str) -> tuple[str, ...]:
    return (_FF_HF.format(release=release, file=file), _FF_GH.format(release=release, file=file))


REGISTRY: dict[str, ModelSpec] = {}


def _add(spec: ModelSpec) -> None:
    REGISTRY[spec.key] = spec


# --- detection / analysis -------------------------------------------------
_add(ModelSpec("retinaface_10g", "detector", "detection/retinaface_10g.onnx", _ff("models-3.0.0", "retinaface_10g.onnx"),
               "MIT (InsightFace SCRFD-10G)", "RetinaFace / SCRFD 10G", arch="scrfd"))
_add(ModelSpec("scrfd_2.5g", "detector", "detection/scrfd_2.5g.onnx", _ff("models-3.0.0", "scrfd_2.5g.onnx"),
               "MIT (InsightFace)", "SCRFD 2.5G (fast)", arch="scrfd"))
_add(ModelSpec("yoloface_8n", "detector", "detection/yoloface_8n.onnx", _ff("models-3.0.0", "yoloface_8n.onnx"),
               "GPL-3.0 (YOLOv8-face)", "YOLOv8n face", arch="yoloface"))
_add(ModelSpec("arcface_w600k_r50", "recognizer", "recognition/arcface_w600k_r50.onnx", _ff("models-3.0.0", "arcface_w600k_r50.onnx"),
               "Non-commercial (InsightFace)", "ArcFace w600k r50", template="arcface_112", size=112))
_add(ModelSpec("fan_68_5", "landmarker", "detection/fan_68_5.onnx", _ff("models-3.0.0", "fan_68_5.onnx"),
               "Apache-2.0 (FaceFusion)", "5 to 68 landmark estimator"))
_add(ModelSpec("fairface", "classifier", "detection/fairface.onnx", _ff("models-3.0.0", "fairface.onnx"),
               "CC BY 4.0 (FairFace)", "FairFace gender / age", template="arcface_112", size=224))

# --- swappers ---------------------------------------------------------------
_add(ModelSpec("hyperswap_1a_256", "swapper", "swappers/hyperswap_1a_256.onnx", _ff("models-3.3.0", "hyperswap_1a_256.onnx"),
               "ResearchRAIL (FaceFusion)", "HyperSwap 1A 256", template="arcface_128", size=256, mean=0.5, std=0.5, arch="hyperswap"))
_add(ModelSpec("hyperswap_1b_256", "swapper", "swappers/hyperswap_1b_256.onnx", _ff("models-3.3.0", "hyperswap_1b_256.onnx"),
               "ResearchRAIL (FaceFusion)", "HyperSwap 1B 256", template="arcface_128", size=256, mean=0.5, std=0.5, arch="hyperswap"))
_add(ModelSpec("hyperswap_1c_256", "swapper", "swappers/hyperswap_1c_256.onnx", _ff("models-3.3.0", "hyperswap_1c_256.onnx"),
               "ResearchRAIL (FaceFusion)", "HyperSwap 1C 256", template="arcface_128", size=256, mean=0.5, std=0.5, arch="hyperswap"))
_add(ModelSpec("inswapper_128", "swapper", "swappers/inswapper_128.onnx", _ff("models-3.0.0", "inswapper_128.onnx"),
               "Non-commercial (InsightFace)", "inswapper 128", template="arcface_128", size=128, arch="inswapper"))
_add(ModelSpec("inswapper_128_fp16", "swapper", "swappers/inswapper_128_fp16.onnx", _ff("models-3.0.0", "inswapper_128_fp16.onnx"),
               "Non-commercial (InsightFace)", "inswapper 128 fp16", template="arcface_128", size=128, arch="inswapper"))
_add(ModelSpec("reswapper_128", "swapper", "swappers/reswapper_128.onnx", (_REACTOR_HF.format(file="reswapper_128.onnx"),),
               "Non-commercial (ReSwapper)", "ReSwapper 128", template="arcface_128", size=128, arch="inswapper"))
_add(ModelSpec("reswapper_256", "swapper", "swappers/reswapper_256.onnx", (_REACTOR_HF.format(file="reswapper_256.onnx"),),
               "Non-commercial (ReSwapper)", "ReSwapper 256", template="arcface_128", size=256, arch="inswapper"))

# --- restorers --------------------------------------------------------------
_add(ModelSpec("gfpgan_1.4", "restorer", "restorers/gfpgan_1.4.onnx", _ff("models-3.0.0", "gfpgan_1.4.onnx"),
               "Apache-2.0 (TencentARC)", "GFPGAN 1.4", template="ffhq_512", size=512))
_add(ModelSpec("codeformer", "restorer", "restorers/codeformer.onnx", _ff("models-3.0.0", "codeformer.onnx"),
               "S-Lab 1.0 (non-commercial)", "CodeFormer", template="ffhq_512", size=512))
_add(ModelSpec("gpen_bfr_256", "restorer", "restorers/gpen_bfr_256.onnx", _ff("models-3.0.0", "gpen_bfr_256.onnx"),
               "Non-commercial (GPEN)", "GPEN-BFR 256", template="arcface_128", size=256))
_add(ModelSpec("gpen_bfr_512", "restorer", "restorers/gpen_bfr_512.onnx", _ff("models-3.0.0", "gpen_bfr_512.onnx"),
               "Non-commercial (GPEN)", "GPEN-BFR 512", template="ffhq_512", size=512))
_add(ModelSpec("gpen_bfr_1024", "restorer", "restorers/gpen_bfr_1024.onnx", _ff("models-3.0.0", "gpen_bfr_1024.onnx"),
               "Non-commercial (GPEN)", "GPEN-BFR 1024", template="ffhq_512", size=1024))
_add(ModelSpec("gpen_bfr_2048", "restorer", "restorers/gpen_bfr_2048.onnx", _ff("models-3.0.0", "gpen_bfr_2048.onnx"),
               "Non-commercial (GPEN)", "GPEN-BFR 2048", template="ffhq_512", size=2048))
_add(ModelSpec("restoreformer_plus_plus", "restorer", "restorers/restoreformer_plus_plus.onnx", _ff("models-3.0.0", "restoreformer_plus_plus.onnx"),
               "Apache-2.0 (RestoreFormer++)", "RestoreFormer++", template="ffhq_512", size=512))

# --- masks ------------------------------------------------------------------
_add(ModelSpec("xseg_1", "occluder", "masks/xseg_1.onnx", _ff("models-3.1.0", "xseg_1.onnx"),
               "GPL-3.0 (DeepFaceLab XSeg)", "XSeg 1", size=256))
_add(ModelSpec("xseg_2", "occluder", "masks/xseg_2.onnx", _ff("models-3.1.0", "xseg_2.onnx"),
               "GPL-3.0 (DeepFaceLab XSeg)", "XSeg 2", size=256))
_add(ModelSpec("bisenet_resnet_34", "parser", "masks/bisenet_resnet_34.onnx", _ff("models-3.0.0", "bisenet_resnet_34.onnx"),
               "MIT (yakhyo face-parsing)", "BiSeNet ResNet-34", size=512))
_add(ModelSpec("bisenet_resnet_18", "parser", "masks/bisenet_resnet_18.onnx", _ff("models-3.1.0", "bisenet_resnet_18.onnx"),
               "MIT (yakhyo face-parsing)", "BiSeNet ResNet-18", size=512))


def keys_of(kind: str) -> list[str]:
    return [k for k, s in REGISTRY.items() if s.kind == kind]


# --- paths / downloads --------------------------------------------------------
_models_dir: str | None = None
_download_lock = threading.Lock()
allow_download = True


def set_models_dir(path: str) -> None:
    global _models_dir
    _models_dir = path


def models_dir() -> str:
    if _models_dir:
        return _models_dir
    return os.path.join(os.getcwd(), "models", "visage")


def local_path(key: str) -> str:
    return os.path.join(models_dir(), *REGISTRY[key].file.split("/"))


def is_downloaded(key: str) -> bool:
    return os.path.isfile(local_path(key))


class ModelUnavailable(RuntimeError):
    pass


def ensure(key: str, progress=None) -> str:
    """Return the local path of `key`, downloading it first when allowed."""
    if key not in REGISTRY:
        raise ModelUnavailable(f"Unknown model '{key}'")
    path = local_path(key)
    if os.path.isfile(path):
        return path
    if not allow_download:
        raise ModelUnavailable(f"Model '{key}' is missing at {path} and automatic downloads are disabled")

    with _download_lock:
        if os.path.isfile(path):
            return path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        errors = []
        for url in REGISTRY[key].urls:
            try:
                _download(url, path, progress)
                return path
            except Exception as e:  # try the next mirror
                errors.append(f"{url}: {e}")
        raise ModelUnavailable(f"Could not download '{key}':\n  " + "\n  ".join(errors))


def _download(url: str, path: str, progress=None) -> None:
    part = path + ".part"
    name = os.path.basename(path)
    req = urllib.request.Request(url, headers={"User-Agent": "visage-forge-neo"})
    print(f"[Visage] Downloading {name} ...")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            last = -1
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    if pct != last and pct % 10 == 0:
                        print(f"[Visage]   {name}: {pct}% of {total >> 20} MB")
                        last = pct
                    if progress is not None:
                        progress(done / total, desc=f"Downloading {name}")
        if total and done != total:
            raise IOError(f"incomplete download ({done} of {total} bytes)")
        os.replace(part, path)
    finally:
        if os.path.exists(part):
            try:
                os.remove(part)
            except OSError:
                pass


def delete(key: str) -> None:
    path = local_path(key)
    if os.path.isfile(path):
        os.remove(path)


def disk_usage() -> int:
    root = models_dir()
    total = 0
    for base, _, files in os.walk(root):
        for f in files:
            total += os.path.getsize(os.path.join(base, f))
    return total


__all__ = ["REGISTRY", "ModelSpec", "ModelUnavailable", "ensure", "keys_of", "local_path", "is_downloaded",
           "set_models_dir", "models_dir"]
