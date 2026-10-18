"""Face restoration (GFPGAN, CodeFormer, GPEN, RestoreFormer++) through ONNX."""

from __future__ import annotations

import cv2
import numpy as np

from . import models, runtime
from .geometry import warp_face


def restore_crop(key: str, crop: np.ndarray, codeformer_weight: float = 0.5, progress=None) -> np.ndarray:
    """Restore an already aligned crop (float/uint8 RGB, any size) and return it at the same size."""
    spec = models.REGISTRY[key]
    sess = runtime.session(key, progress)
    size = crop.shape[0]
    x = cv2.resize(crop.astype(np.float32), (spec.size, spec.size), interpolation=cv2.INTER_CUBIC if size < spec.size else cv2.INTER_AREA)
    x = (x / 255.0 - 0.5) / 0.5
    feed = {}
    for inp in sess.get_inputs():
        if inp.name == "input":
            feed[inp.name] = x.transpose(2, 0, 1)[None].astype(np.float32)
        elif inp.name == "weight":
            feed[inp.name] = np.array(codeformer_weight, dtype=np.float64)
    out = sess.run(None, feed)[0][0].clip(-1, 1)
    out = ((out + 1) / 2 * 255.0).transpose(1, 2, 0)
    if spec.size != size:
        out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA if spec.size > size else cv2.INTER_CUBIC)
    return out.astype(np.float32)


def restore_face(key: str, frame: np.ndarray, kps: np.ndarray, codeformer_weight: float = 0.5,
                 progress=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align with the restorer's own template, restore. Returns (crop, restored, matrix)."""
    spec = models.REGISTRY[key]
    crop, matrix = warp_face(frame, kps, spec.template, spec.size)
    restored = restore_crop(key, crop, codeformer_weight, progress)
    return crop.astype(np.float32), restored, matrix
