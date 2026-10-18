"""Crop-space masks: feathered box, XSeg occlusion and BiSeNet face-parsing regions."""

from __future__ import annotations

import cv2
import numpy as np

from . import runtime

# CelebAMask-HQ label ids produced by the BiSeNet face parsers.
REGIONS = {
    "skin": 1,
    "left-eyebrow": 2,
    "right-eyebrow": 3,
    "left-eye": 4,
    "right-eye": 5,
    "glasses": 6,
    "nose": 10,
    "mouth": 11,
    "upper-lip": 12,
    "lower-lip": 13,
    "hair": 17,
}
DEFAULT_REGIONS = ["skin", "left-eyebrow", "right-eyebrow", "left-eye", "right-eye", "glasses", "nose", "mouth",
                   "upper-lip", "lower-lip"]


def _feather(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Soft edge that stays at 1 inside the shape (blur, then re-stretch the upper half)."""
    if sigma <= 0:
        return mask.clip(0, 1)
    return ((cv2.GaussianBlur(mask.clip(0, 1), (0, 0), sigma).clip(0.5, 1) - 0.5) * 2).astype(np.float32)


def box_mask(size: int, blur: float = 0.3, padding: tuple[int, int, int, int] = (0, 0, 0, 0)) -> np.ndarray:
    """Rectangle covering the crop with feathered edges.

    blur: 0..1 fraction of half the crop used as feather; padding: top, right, bottom, left in percent.
    """
    blur_px = int(size * 0.5 * blur)
    edge = max(blur_px // 2, 1)
    m = np.ones((size, size), dtype=np.float32)
    t, r, b, lft = padding
    m[: max(edge, int(size * t / 100)), :] = 0
    m[-max(edge, int(size * b / 100)):, :] = 0
    m[:, : max(edge, int(size * lft / 100))] = 0
    m[:, -max(edge, int(size * r / 100)):] = 0
    if blur_px > 0:
        m = cv2.GaussianBlur(m, (0, 0), blur_px * 0.25)
    return m


def occlusion_mask(crop: np.ndarray, model: str = "xseg_1", progress=None) -> np.ndarray:
    """1 where the face is visible, 0 where hands, hair, objects cover it."""
    sess = runtime.session(model, progress)
    size = crop.shape[0]
    x = cv2.resize(crop.astype(np.float32), (256, 256))[:, :, ::-1] / 255.0  # XSeg expects BGR
    out = sess.run(None, {sess.get_inputs()[0].name: x[None].astype(np.float32)})[0][0]
    m = cv2.resize(out.reshape(256, 256).clip(0, 1).astype(np.float32), (size, size))
    return _feather(m, 5 * size / 256)


def parse(crop: np.ndarray, model: str = "bisenet_resnet_34", progress=None) -> np.ndarray:
    """Per-pixel CelebAMask label map at crop size."""
    sess = runtime.session(model, progress)
    size = crop.shape[0]
    x = cv2.resize(crop.astype(np.float32), (512, 512)) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    logits = sess.run(None, {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
    labels = logits.argmax(0).astype(np.uint8)
    return cv2.resize(labels, (size, size), interpolation=cv2.INTER_NEAREST)


def region_mask(labels: np.ndarray, regions: list[str], dilate: int = 0) -> np.ndarray:
    ids = [REGIONS[r] for r in regions if r in REGIONS]
    m = np.isin(labels, ids).astype(np.float32)
    if dilate:
        k = abs(int(dilate))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        m = cv2.dilate(m, kernel) if dilate > 0 else cv2.erode(m, kernel)
    return _feather(m, 5 * labels.shape[0] / 256)


def combine(masks: list[np.ndarray]) -> np.ndarray:
    if not masks:
        raise ValueError("no masks")
    return np.minimum.reduce([m.astype(np.float32) for m in masks]).clip(0, 1)
