"""Optional SFW filter (off by default).

Three independent NSFW models vote. An image counts as NSFW only when at least
two agree, which keeps false positives low. The models are downloaded only when
the filter is switched on.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import models, runtime

MODES = ["off", "skip", "blur"]  # off: no check | skip: leave NSFW images untouched | blur: blur NSFW results


def _fit_contain(frame: np.ndarray, size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=resized.dtype)
    y, x = (size - nh) // 2, (size - nw) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def _run(key: str, frame: np.ndarray, progress=None) -> np.ndarray:
    spec = models.REGISTRY[key]
    sess = runtime.session(key, progress)
    x = _fit_contain(frame, spec.size).astype(np.float32) / 255.0
    x = (x - np.asarray(spec.extra["mean"], np.float32)) / np.asarray(spec.extra["std"], np.float32)
    return sess.run(None, {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0]


def votes(frame: np.ndarray, progress=None) -> dict[str, bool]:
    """Per-model verdicts for an RGB frame."""
    d1 = _run("nsfw_1", frame, progress)  # 1 x (4 + classes) x anchors, detector style
    s1 = float(np.max(np.amax(d1[:, 4:], axis=1)))
    d2 = _run("nsfw_2", frame, progress)[0]
    s2 = float(d2[0] - d2[1])
    d3 = _run("nsfw_3", frame, progress)[0]
    s3 = float((d3[2] + d3[3]) - (d3[0] + d3[1]))
    return {"nsfw_1": s1 > 0.2, "nsfw_2": s2 > 0.25, "nsfw_3": s3 > 10.5}


def is_nsfw(frame: np.ndarray, progress=None) -> bool:
    return sum(votes(frame, progress).values()) >= 2


def blur(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    k = max(h, w) // 12
    small = cv2.resize(frame, (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), (0, 0), k / 3)
