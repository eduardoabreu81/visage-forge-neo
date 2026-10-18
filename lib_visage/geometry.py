"""Face alignment templates, affine warps and paste-back.

All images handled here are HxWx3 uint8 or float32 arrays. Landmarks are
5-point (left eye, right eye, nose, left mouth corner, right mouth corner)
in pixel coordinates of the frame they came from.
"""

from __future__ import annotations

import cv2
import numpy as np

# Normalised (0..1) 5-point templates.
#   arcface_112 - the ArcFace recognition crop (insightface arcface_dst / 112)
#   arcface_128 - the inswapper / reswapper / hyperswap crop (arcface_dst shifted by 8px, / 128)
#   ffhq_512    - the FFHQ crop used by GFPGAN, CodeFormer, GPEN, RestoreFormer
TEMPLATES: dict[str, np.ndarray] = {
    "arcface_112": np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    / 112.0,
    "arcface_128": (
        np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        + np.array([8.0, 0.0], dtype=np.float32)
    )
    / 128.0,
    # SimSwap / Ghost crop (FaceFusion's arcface_112_v1)
    "arcface_112_v1": np.array(
        [
            [0.35473214, 0.45658929],
            [0.64526786, 0.45658929],
            [0.50000000, 0.61154464],
            [0.37913393, 0.77687500],
            [0.62086607, 0.77687500],
        ],
        dtype=np.float32,
    ),
    # HiFiFace crop
    "mtcnn_512": np.array(
        [
            [0.36562865, 0.46733799],
            [0.63305391, 0.46585885],
            [0.50019127, 0.61942959],
            [0.39032951, 0.77598822],
            [0.61178945, 0.77476328],
        ],
        dtype=np.float32,
    ),
    "ffhq_512": np.array(
        [
            [192.98138, 239.94708],
            [318.90277, 240.19360],
            [256.63416, 314.01935],
            [201.26117, 371.41043],
            [313.08905, 371.15118],
        ],
        dtype=np.float32,
    )
    / 512.0,
}


def estimate_matrix(kps: np.ndarray, template: str, size: int) -> np.ndarray:
    """Similarity transform mapping frame landmarks onto the template at `size`."""
    dst = TEMPLATES[template] * size
    matrix, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), dst, method=cv2.LMEDS)
    if matrix is None:  # degenerate landmarks; fall back to least squares
        matrix = cv2.estimateAffinePartial2D(kps.astype(np.float32), dst, method=0)[0]
    return matrix


def warp_face(frame: np.ndarray, kps: np.ndarray, template: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = estimate_matrix(kps, template, size)
    crop = cv2.warpAffine(frame, matrix, (size, size), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
    return crop, matrix


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.transform(points.reshape(1, -1, 2).astype(np.float32), matrix).reshape(-1, 2)


def paste_back(frame: np.ndarray, crop: np.ndarray, mask: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Inverse-warp `crop` into `frame`, blending with `mask` (crop-space, 0..1).

    Only the bounding region of the crop is touched so large frames stay cheap.
    """
    h, w = frame.shape[:2]
    ch, cw = crop.shape[:2]
    inverse = cv2.invertAffineTransform(matrix)
    corners = transform_points(np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype=np.float32), inverse)
    x1, y1 = np.clip(np.floor(corners.min(axis=0)).astype(int), 0, [w, h])
    x2, y2 = np.clip(np.ceil(corners.max(axis=0)).astype(int), 0, [w, h])
    if x2 <= x1 or y2 <= y1:
        return frame

    local = inverse.copy()
    local[0, 2] -= x1
    local[1, 2] -= y1
    size = (int(x2 - x1), int(y2 - y1))

    inv_mask = cv2.warpAffine(mask.astype(np.float32), local, size).clip(0, 1)[..., None]
    inv_crop = cv2.warpAffine(crop.astype(np.float32), local, size, borderMode=cv2.BORDER_REPLICATE)

    out = frame.copy()
    region = out[y1:y2, x1:x2].astype(np.float32)
    out[y1:y2, x1:x2] = (region * (1.0 - inv_mask) + inv_crop * inv_mask).round().clip(0, 255).astype(frame.dtype)
    return out


def warp_mask_to_frame(mask: np.ndarray, matrix: np.ndarray, frame_shape: tuple[int, ...]) -> np.ndarray:
    """Inverse-warp a crop-space mask to a full-frame float mask."""
    h, w = frame_shape[:2]
    inverse = cv2.invertAffineTransform(matrix)
    return cv2.warpAffine(mask.astype(np.float32), inverse, (w, h)).clip(0, 1)


def implode_pixel_boost(crop: np.ndarray, factor: int, model_size: int) -> list[np.ndarray]:
    """Split a (model_size*factor)^2 crop into factor^2 sub-pixel phase crops of model_size.

    Each sub-crop is a full, slightly shifted view of the face, so a low resolution
    swapper can be run several times and the results interleaved back together.
    """
    if factor == 1:
        return [crop]
    tiles = crop.reshape(model_size, factor, model_size, factor, 3).transpose(1, 3, 0, 2, 4)
    return [tiles[i, j] for i in range(factor) for j in range(factor)]


def explode_pixel_boost(tiles: list[np.ndarray], factor: int, model_size: int) -> np.ndarray:
    if factor == 1:
        return tiles[0]
    stack = np.stack(tiles).reshape(factor, factor, model_size, model_size, 3)
    return stack.transpose(2, 0, 3, 1, 4).reshape(model_size * factor, model_size * factor, 3)
