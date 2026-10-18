"""Face detection, recognition and classification (ONNX only, no insightface).

Frames are RGB uint8 HxWx3 arrays throughout Visage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

from . import models, runtime
from .geometry import warp_face


@dataclass
class Face:
    bbox: np.ndarray  # x1, y1, x2, y2
    kps: np.ndarray  # 5x2
    score: float
    embedding: np.ndarray | None = None  # raw ArcFace embedding (512,)
    gender: str | None = None  # "male" | "female"
    age: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def area(self) -> float:
        return float(max(0.0, self.bbox[2] - self.bbox[0]) * max(0.0, self.bbox[3] - self.bbox[1]))

    @property
    def normed_embedding(self) -> np.ndarray | None:
        if self.embedding is None:
            return None
        return self.embedding / (np.linalg.norm(self.embedding) + 1e-12)


# --- detection ----------------------------------------------------------------

@lru_cache(maxsize=32)
def _anchors(stride: int, num_anchors: int, h: int, w: int) -> np.ndarray:
    ys, xs = np.mgrid[:h, :w]
    centers = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32) * stride
    return np.repeat(centers, num_anchors, axis=0)


def _letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    canvas[:nh, :nw] = resized
    return canvas, scale


def _detect_scrfd(sess, frame: np.ndarray, size: int, threshold: float):
    canvas, scale = _letterbox(frame, size)
    blob = ((canvas - 127.5) / 128.0).transpose(2, 0, 1)[None].astype(np.float32)
    outputs = sess.run(None, {sess.get_inputs()[0].name: blob})
    boxes, scores, kpss = [], [], []
    fmc = 3
    for idx, stride in enumerate((8, 16, 32)):
        s = outputs[idx].reshape(-1)
        b = outputs[idx + fmc] * stride
        k = outputs[idx + fmc * 2] * stride
        keep = np.where(s >= threshold)[0]
        if keep.size == 0:
            continue
        a = _anchors(stride, 2, size // stride, size // stride)[keep]
        bb = b[keep]
        boxes.append(np.stack([a[:, 0] - bb[:, 0], a[:, 1] - bb[:, 1], a[:, 0] + bb[:, 2], a[:, 1] + bb[:, 3]], axis=-1))
        kpss.append(k[keep].reshape(-1, 5, 2) + a[:, None, :])
        scores.append(s[keep])
    if not boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0, 5, 2))
    return np.concatenate(boxes) / scale, np.concatenate(scores), np.concatenate(kpss) / scale


def _detect_yoloface(sess, frame: np.ndarray, size: int, threshold: float):
    size = 640  # exported with a fixed input
    canvas, scale = _letterbox(frame, size)
    blob = (canvas / 255.0).transpose(2, 0, 1)[None].astype(np.float32)
    pred = sess.run(None, {sess.get_inputs()[0].name: blob})[0][0].T  # 8400 x 20
    keep = np.where(pred[:, 4] > threshold)[0]
    if keep.size == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0, 5, 2))
    p = pred[keep]
    cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)
    kps = p[:, 5:].reshape(-1, 5, 3)[:, :, :2]
    return boxes / scale, p[:, 4], kps / scale


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    if len(boxes) == 0:
        return []
    xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
    idx = cv2.dnn.NMSBoxes(xywh, [float(s) for s in scores], 0.0, iou)
    return [int(i) for i in np.array(idx).reshape(-1)]


def _rotate(frame: np.ndarray, angle: int) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the frame and return the 2x3 matrix mapping rotated coords back."""
    h, w = frame.shape[:2]
    if angle == 0:
        return frame, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    if angle == 90:  # clockwise: (x, y) <- (y', h - 1 - x')
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE), np.array([[0, 1, 0], [-1, 0, h - 1]], dtype=np.float32)
    if angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180), np.array([[-1, 0, w - 1], [0, -1, h - 1]], dtype=np.float32)
    return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE), np.array([[0, -1, w - 1], [1, 0, 0]], dtype=np.float32)


def detect(frame: np.ndarray, detector: str = "retinaface_10g", size: int = 640, threshold: float = 0.5,
           try_rotations: bool = False, progress=None, pad_retry: bool = True) -> list[Face]:
    spec = models.REGISTRY[detector]
    sess = runtime.session(detector, progress)
    run = _detect_yoloface if spec.arch == "yoloface" else _detect_scrfd

    # With try_rotations every orientation is scored and the best one wins, so
    # sideways / upside-down faces get upright landmarks (better swaps).
    best: list[Face] = []
    best_score = 0.0
    for angle in ((0, 90, 270, 180) if try_rotations else (0,)):
        img, back = _rotate(frame, angle)
        boxes, scores, kpss = run(sess, img, size, threshold)
        faces = []
        for i in _nms(boxes, scores, 0.4):
            kps = cv2.transform(kpss[i].reshape(1, -1, 2).astype(np.float32), back).reshape(-1, 2)
            corners = np.array([[boxes[i][0], boxes[i][1]], [boxes[i][2], boxes[i][3]]], dtype=np.float32)
            corners = cv2.transform(corners.reshape(1, -1, 2), back).reshape(-1, 2)
            bbox = np.array([corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max()])
            faces.append(Face(bbox=bbox, kps=kps, score=float(scores[i])))
        total = sum(f.score for f in faces)
        if total > best_score * 1.05:  # prefer upright on ties
            best, best_score = faces, total
    if not best and pad_retry:
        # Tight face crops (typical for source photos) have no context for the
        # detector anchors; give them a border and retry once.
        h, w = frame.shape[:2]
        p = max(h, w) // 2
        padded = cv2.copyMakeBorder(frame, p, p, p, p, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        best = detect(padded, detector, size, threshold, try_rotations, progress, pad_retry=False)
        for f in best:
            f.bbox = f.bbox - p
            f.kps = f.kps - p
    return best


# --- recognition ------------------------------------------------------------------

def embed(frame: np.ndarray, face: Face, recognizer: str = "arcface_w600k_r50", progress=None) -> np.ndarray:
    if face.embedding is not None:
        return face.embedding
    spec = models.REGISTRY[recognizer]
    sess = runtime.session(recognizer, progress)
    crop, _ = warp_face(frame, face.kps, spec.template, spec.size)
    blob = (crop.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]
    face.embedding = sess.run(None, {sess.get_inputs()[0].name: blob})[0][0].astype(np.float32)
    return face.embedding


AGE_BUCKETS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]


def classify(frame: np.ndarray, face: Face, progress=None) -> Face:
    """Fill gender / age using FairFace (downloaded on first use)."""
    if face.gender is not None:
        return face
    spec = models.REGISTRY["fairface"]
    sess = runtime.session("fairface", progress)
    crop, _ = warp_face(frame, face.kps, spec.template, spec.size)
    x = crop.astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    _race, gender, age = sess.run(None, {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32)})
    face.gender = "female" if int(np.ravel(gender)[0]) == 1 else "male"
    face.age = AGE_BUCKETS[min(int(np.ravel(age)[0]), len(AGE_BUCKETS) - 1)]
    return face


# --- selection ----------------------------------------------------------------------

SORT_ORDERS = ["large-small", "small-large", "left-right", "right-left", "top-bottom", "bottom-top", "best-worst"]


def sort_faces(faces: list[Face], order: str) -> list[Face]:
    keys = {
        "large-small": lambda f: -f.area,
        "small-large": lambda f: f.area,
        "left-right": lambda f: f.bbox[0],
        "right-left": lambda f: -f.bbox[0],
        "top-bottom": lambda f: f.bbox[1],
        "bottom-top": lambda f: -f.bbox[1],
        "best-worst": lambda f: -f.score,
    }
    key = keys.get(order)
    return sorted(faces, key=key) if key else list(faces)


def parse_indexes(text) -> list[int] | None:
    """'0, 2' / '0-2' -> list; '' / 'all' / None -> None (meaning every face)."""
    if text is None:
        return None
    text = str(text).strip().lower()
    if text in ("", "all", "*"):
        return None
    out = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))
