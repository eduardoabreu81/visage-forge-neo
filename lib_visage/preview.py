"""Annotated previews: draw detected faces with the index Visage would use."""

from __future__ import annotations

import cv2
import numpy as np

from . import analysis
from .pipeline import Options


def annotate(frame: np.ndarray, opts: Options | None = None, classify: bool = False, highlight: list[int] | None = None,
             progress=None) -> tuple[np.ndarray, str]:
    """Return (image with numbered boxes, markdown summary). Indexes follow opts.sort_order."""
    opts = opts or Options()
    faces = analysis.sort_faces(analysis.detect(frame, opts.detector, opts.det_size, opts.det_threshold,
                                                opts.try_rotations, progress), opts.sort_order)
    out = frame.copy()
    h, w = out.shape[:2]
    thick = max(2, round(min(h, w) / 300))
    rows = []
    for i, f in enumerate(faces):
        if classify:
            analysis.classify(frame, f, progress)
        on = highlight is None or i in highlight
        color = (64, 220, 120) if on else (150, 150, 150)
        x1, y1, x2, y2 = [int(round(v)) for v in f.bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        for x, y in f.kps:
            cv2.circle(out, (int(x), int(y)), thick, (255, 200, 0), -1)
        label = str(i) + (f" {f.gender[0].upper()}" if f.gender else "")
        scale = max(0.6, (x2 - x1) / 140)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        ty = max(th + 4, y1 - 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), color, -1)
        cv2.putText(out, label, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
        extra = f" · {f.gender}, {f.age}" if f.gender else ""
        rows.append(f"- `{i}` {x2 - x1}×{y2 - y1}px · score {f.score:.2f}{extra}")
    summary = f"**{len(faces)} face(s)** in order `{opts.sort_order}`" + ("\n\n" + "  \n".join(rows) if rows else "")
    return out, summary
