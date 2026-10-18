"""Identity swappers: inswapper / ReSwapper / HyperSwap families."""

from __future__ import annotations

import threading

import cv2
import numpy as np

from . import models, onnxread, runtime
from .geometry import explode_pixel_boost, implode_pixel_boost, warp_face

_emaps: dict[str, np.ndarray] = {}
_emap_lock = threading.Lock()


def _emap(key: str) -> np.ndarray:
    """The 512x512 identity projection stored as the last initializer of inswapper-style graphs."""
    with _emap_lock:
        if key not in _emaps:
            _emaps[key] = onnxread.last_initializer(models.local_path(key)).astype(np.float32)
        return _emaps[key]


def blend_identity(source: np.ndarray, target: np.ndarray | None, strength: float) -> np.ndarray:
    """Interpolate (or extrapolate when strength > 1) between target and source identity."""
    s = source / (np.linalg.norm(source) + 1e-12)
    if target is None or abs(strength - 1.0) < 1e-6:
        return s
    t = target / (np.linalg.norm(target) + 1e-12)
    mixed = t + strength * (s - t)
    return mixed / (np.linalg.norm(mixed) + 1e-12)


# Visage keeps identities as unit vectors (face models store them that way). The CrossFace
# converters and AlphaFace are scale-invariant (verified: identical output at norm 1 and ~23),
# so the unit vector is passed as is.


def _convert(converter: str, identity: np.ndarray, progress=None) -> np.ndarray:
    sess = runtime.session(converter, progress)
    x = identity.reshape(1, -1).astype(np.float32)
    return sess.run(None, {sess.get_inputs()[0].name: x})[0].reshape(1, -1).astype(np.float32)


def source_input(key: str, identity: np.ndarray, progress=None) -> np.ndarray:
    """What each swapper family expects as its `source` input."""
    spec = models.REGISTRY[key]
    if spec.arch == "inswapper":
        latent = identity.reshape(1, -1) @ _emap(key)
        return (latent / (np.linalg.norm(latent) + 1e-12)).astype(np.float32)
    if spec.arch == "ghost":
        return _convert(spec.extra["converter"], identity, progress)
    if spec.arch in ("simswap", "hififace"):
        conv = _convert(spec.extra["converter"], identity, progress)
        return conv / (np.linalg.norm(conv) + 1e-12)
    return identity.reshape(1, -1).astype(np.float32)  # hyperswap: the normed ArcFace vector


DENORMALISE_OUTPUT = {"hyperswap", "ghost", "hififace"}


def swap_crop(key: str, frame: np.ndarray, kps: np.ndarray, identity: np.ndarray, pixel_boost: int = 1,
              progress=None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Swap one face.

    Returns (original_crop, swapped_crop, model_mask_or_None, matrix); crops are float32 RGB 0..255
    at ``spec.size * pixel_boost``.
    """
    spec = models.REGISTRY[key]
    sess = runtime.session(key, progress)
    size = spec.size
    boost = max(1, int(pixel_boost))
    crop, matrix = warp_face(frame, kps, spec.template, size * boost)
    crop_f = crop.astype(np.float32)
    src = source_input(key, identity, progress)
    mean = np.asarray(spec.extra.get("mean", spec.mean), dtype=np.float32)
    std = np.asarray(spec.extra.get("std", spec.std), dtype=np.float32)

    names = runtime.input_names(sess)
    out_names = [o.name for o in sess.get_outputs()]
    mask_index = out_names.index("mask") if "mask" in out_names else None
    outs, masks = [], []
    for tile in implode_pixel_boost(crop_f, boost, size):
        x = (tile / 255.0 - mean) / std
        feed = {}
        for name in names:
            if name == "target":
                feed[name] = x.transpose(2, 0, 1)[None].astype(np.float32)
            elif name == "source":
                feed[name] = src
        result = sess.run(None, feed)
        img = result[0][0].transpose(1, 2, 0)
        if spec.arch in DENORMALISE_OUTPUT:
            img = img * std + mean
        outs.append(img.clip(0, 1) * 255.0)
        if mask_index is not None:
            m = result[mask_index][0, 0]
            masks.append(np.repeat(m[..., None], 3, axis=2))

    swapped = explode_pixel_boost(outs, boost, size)
    mask = None
    if masks:
        mask = explode_pixel_boost(masks, boost, size)[..., 0].clip(0, 1).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, size * boost / 128))
    return crop_f, swapped.astype(np.float32), mask, matrix
