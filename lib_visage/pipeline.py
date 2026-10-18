"""End-to-end face processing: select faces, swap, mask, colour-match, restore, paste."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, fields

import cv2
import numpy as np

from . import analysis, masks, models, restorer, swapper
from .facemodel import Identity, average
from .geometry import paste_back, warp_mask_to_frame


@dataclass
class Options:
    swapper: str = "hyperswap_1a_256"
    detector: str = "retinaface_10g"
    det_size: int = 640
    det_threshold: float = 0.5
    try_rotations: bool = False

    target_indexes: str = "0"  # "0", "0,2", "all"
    sort_order: str = "large-small"
    gender: str = "any"  # any | male | female
    min_face_size: int = 0  # px of the shorter bbox side

    strength: float = 1.0
    pixel_boost: int = 1  # 1, 2 or 4

    mask_box: bool = True
    mask_blur: float = 0.3
    mask_padding: tuple = (0, 0, 0, 0)
    mask_model: bool = True  # use the swapper's own mask when it has one (HyperSwap)
    mask_occlusion: bool = False
    occluder: str = "xseg_1"
    mask_region: bool = False
    parser: str = "bisenet_resnet_34"
    regions: list = field(default_factory=lambda: list(masks.DEFAULT_REGIONS))
    region_dilate: int = 0
    keep_mouth: bool = False  # keep the target's own mouth (open mouths, teeth, tongue)

    color_match: bool = False

    restorer: str = "none"
    restorer_visibility: float = 1.0
    codeformer_weight: float = 0.5
    face_boost: bool = False  # restore inside the swap crop before pasting back

    report_similarity: bool = False  # measure ArcFace similarity of each result face to its source

    @classmethod
    def from_dict(cls, data: dict | None) -> "Options":
        data = data or {}
        names = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if k in names}
        if "mask_padding" in clean:
            clean["mask_padding"] = tuple(int(x) for x in clean["mask_padding"])
        return cls(**clean)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Result:
    image: np.ndarray
    faces_found: int = 0
    swapped: list = field(default_factory=list)  # target indexes that were swapped
    face_mask: np.ndarray | None = None  # full frame union of what was replaced (0..1)
    seconds: float = 0.0
    message: str = ""
    similarity: list = field(default_factory=list)  # per swapped face, when Options.report_similarity


def identity_from_images(images: list[np.ndarray], opts: Options | None = None, face_index: int = 0,
                         name: str = "", progress=None) -> tuple[Identity | None, list[str]]:
    """Average the chosen face of every image into one identity. Returns (identity, warnings)."""
    opts = opts or Options()
    found, notes = [], []
    for n, img in enumerate(images):
        faces = analysis.sort_faces(analysis.detect(img, opts.detector, opts.det_size, opts.det_threshold,
                                                    opts.try_rotations, progress), opts.sort_order)
        if len(faces) <= face_index:
            notes.append(f"source image {n + 1}: no face #{face_index} (found {len(faces)})")
            continue
        emb = analysis.embed(img, faces[face_index], progress=progress)
        found.append(Identity(emb))
    if not found:
        return None, notes
    return average(found, name=name), notes


def identities_from_image(image: np.ndarray, indexes: list[int], opts: Options | None = None,
                          progress=None) -> tuple[list[Identity], list[str]]:
    """Several identities from one source image (multi-face mapping)."""
    opts = opts or Options()
    faces = analysis.sort_faces(analysis.detect(image, opts.detector, opts.det_size, opts.det_threshold,
                                                opts.try_rotations, progress), opts.sort_order)
    out, notes = [], []
    for i in indexes:
        if i < len(faces):
            out.append(Identity(analysis.embed(image, faces[i], progress=progress)))
        else:
            notes.append(f"source has no face #{i} (found {len(faces)})")
    return out, notes


def _color_match(swapped: np.ndarray, original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Match LAB mean/std of the swapped face to the original face inside the mask."""
    sel = mask > 0.5
    if sel.sum() < 64:
        return swapped
    s = cv2.cvtColor(swapped.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    o = cv2.cvtColor(original.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    for c in range(3):
        sm, ss = s[..., c][sel].mean(), s[..., c][sel].std() + 1e-6
        om, os_ = o[..., c][sel].mean(), o[..., c][sel].std() + 1e-6
        s[..., c] = (s[..., c] - sm) * (os_ / ss) + om
    return cv2.cvtColor(s.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)


def _build_mask(opts: Options, crop: np.ndarray, swapped: np.ndarray, model_mask: np.ndarray | None,
                progress=None) -> np.ndarray:
    size = crop.shape[0]
    parts = []
    if opts.mask_box or not (opts.mask_occlusion or opts.mask_region or (opts.mask_model and model_mask is not None)):
        parts.append(masks.box_mask(size, opts.mask_blur, tuple(opts.mask_padding)))
    if opts.mask_model and model_mask is not None:
        parts.append(model_mask)
    if opts.mask_occlusion:
        parts.append(masks.occlusion_mask(crop, opts.occluder, progress))
    if opts.mask_region or opts.keep_mouth:
        labels = masks.parse(crop, opts.parser, progress)
        if opts.mask_region:
            parts.append(masks.region_mask(labels, list(opts.regions), opts.region_dilate))
        if opts.keep_mouth:
            mouth = masks.region_mask(labels, ["mouth"], 2)
            parts.append(1.0 - mouth)
    return masks.combine(parts)


def select_targets(frame: np.ndarray, opts: Options, progress=None) -> tuple[list[analysis.Face], list[int]]:
    faces = analysis.detect(frame, opts.detector, opts.det_size, opts.det_threshold, opts.try_rotations, progress)
    if opts.min_face_size:
        faces = [f for f in faces if min(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]) >= opts.min_face_size]
    if opts.gender in ("male", "female"):
        faces = [f for f in faces if analysis.classify(frame, f, progress).gender == opts.gender]
    faces = analysis.sort_faces(faces, opts.sort_order)
    wanted = analysis.parse_indexes(opts.target_indexes)
    indexes = list(range(len(faces))) if wanted is None else [i for i in wanted if i < len(faces)]
    return faces, indexes


def process(frame: np.ndarray, identities: list[Identity], opts: Options, progress=None) -> Result:
    """Swap the selected target faces of `frame` (RGB uint8) with `identities` (cycled in order)."""
    t0 = time.time()
    if not identities:
        return Result(frame, message="no source identity")
    if opts.swapper not in models.REGISTRY:
        raise ValueError(f"unknown swapper {opts.swapper}")

    faces, indexes = select_targets(frame, opts, progress)
    if not indexes:
        return Result(frame, faces_found=len(faces), seconds=time.time() - t0,
                      message=f"no matching target face (found {len(faces)})")

    out = frame.copy()
    union = np.zeros(frame.shape[:2], dtype=np.float32)
    restore = opts.restorer if opts.restorer and opts.restorer != "none" else None

    for n, idx in enumerate(indexes):
        face = faces[idx]
        identity = identities[n % len(identities)]
        target_emb = analysis.embed(frame, face, progress=progress) if abs(opts.strength - 1.0) > 1e-6 else None
        ident = swapper.blend_identity(identity.normed, target_emb, opts.strength)

        crop, swapped, model_mask, matrix = swapper.swap_crop(opts.swapper, out, face.kps, ident, opts.pixel_boost, progress)
        mask = _build_mask(opts, crop, swapped, model_mask, progress)

        if opts.color_match:
            swapped = _color_match(swapped, crop, mask)

        if restore and opts.face_boost:
            fixed = restorer.restore_crop(restore, swapped, opts.codeformer_weight, progress)
            v = float(np.clip(opts.restorer_visibility, 0, 1))
            swapped = swapped * (1 - v) + fixed * v

        out = paste_back(out, swapped, mask, matrix)
        union = np.maximum(union, warp_mask_to_frame(mask, matrix, frame.shape))

        if restore and not opts.face_boost:
            rcrop, restored, rmatrix = restorer.restore_face(restore, out, face.kps, opts.codeformer_weight, progress)
            rmask = masks.box_mask(rcrop.shape[0], 0.3) * float(np.clip(opts.restorer_visibility, 0, 1))
            out = paste_back(out, restored, rmask, rmatrix)

    similarity = []
    if opts.report_similarity:
        for n, idx in enumerate(indexes):
            probe = analysis.Face(bbox=faces[idx].bbox, kps=faces[idx].kps, score=faces[idx].score)
            emb = analysis.embed(out, probe, progress=progress)
            similarity.append(analysis.cosine_similarity(emb, identities[n % len(identities)].normed))

    return Result(out, faces_found=len(faces), swapped=indexes, face_mask=union, seconds=time.time() - t0,
                  message=f"swapped {len(indexes)} of {len(faces)} face(s) in {time.time() - t0:.1f}s", similarity=similarity)


def restore_only(frame: np.ndarray, opts: Options, progress=None) -> Result:
    """Run just the restorer on the selected faces (no swap)."""
    t0 = time.time()
    faces, indexes = select_targets(frame, opts, progress)
    out = frame.copy()
    if opts.restorer in (None, "", "none"):
        return Result(out, faces_found=len(faces), message="no restorer selected")
    for idx in indexes:
        rcrop, restored, rmatrix = restorer.restore_face(opts.restorer, out, faces[idx].kps, opts.codeformer_weight, progress)
        rmask = masks.box_mask(rcrop.shape[0], 0.3) * float(np.clip(opts.restorer_visibility, 0, 1))
        out = paste_back(out, restored, rmask, rmatrix)
    return Result(out, faces_found=len(faces), swapped=indexes, seconds=time.time() - t0,
                  message=f"restored {len(indexes)} face(s)")
