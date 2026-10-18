"""Visage REST API (enabled when Forge runs with --api).

POST /visage/v1/swap        {source_image | face_model, target_image, options} -> {image, message, faces_found, swapped}
POST /visage/v1/restore     {image, options}                                   -> {image, message}
POST /visage/v1/detect      {image, detector?, threshold?, classify?}          -> {faces: [...]}
POST /visage/v1/facemodels  {name, images: [...], face_index?}                 -> {path}
GET  /visage/v1/facemodels                                                     -> {face_models: [...]}
GET  /visage/v1/models                                                         -> {models: [...], providers, active_provider}
POST /visage/v1/unload                                                         -> {unloaded}

Images are base64 (PNG/JPEG, optionally as data URLs). `options` takes the same
field names as lib_visage.pipeline.Options.
"""

import os
import sys
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from modules import script_callbacks

from lib_visage import __version__, analysis, facemodel, models, pipeline, runtime
from lib_visage import forge as vf


class SwapRequest(BaseModel):
    target_image: str
    source_image: str | None = None
    source_indexes: str = "0"
    face_model: str | None = None
    options: dict = {}


class RestoreRequest(BaseModel):
    image: str
    options: dict = {}


class DetectRequest(BaseModel):
    image: str
    detector: str = "retinaface_10g"
    threshold: float = 0.5
    classify: bool = False


class FaceModelRequest(BaseModel):
    name: str
    images: list[str]
    face_index: int = 0
    overwrite: bool = False


def _image(b64: str, what: str):
    arr = vf.to_rgb(b64)
    if arr is None:
        raise HTTPException(status_code=400, detail=f"could not decode {what}")
    return arr


def _options(data: dict) -> pipeline.Options:
    try:
        return pipeline.Options.from_dict(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"bad options: {e}")


def _guard(fn):
    try:
        return fn()
    except HTTPException:
        raise
    except models.ModelUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def mount(_demo, app: FastAPI):
    @app.get("/visage/v1/models")
    def list_models():
        vf.apply_settings()
        return {
            "version": __version__,
            "models_dir": models.models_dir(),
            "providers": runtime.available_providers(),
            "active_provider": runtime.active_provider() if runtime.available_providers() else None,
            "models": [
                {"key": k, "kind": s.kind, "label": s.title, "license": s.license, "downloaded": models.is_downloaded(k)}
                for k, s in models.REGISTRY.items()
            ],
        }

    @app.get("/visage/v1/facemodels")
    def list_face_models():
        vf.apply_settings()
        return {"face_models": vf.face_model_choices()}

    @app.post("/visage/v1/facemodels")
    def build_face_model(req: FaceModelRequest):
        def run():
            vf.apply_settings()
            imgs = [_image(i, f"image {n}") for n, i in enumerate(req.images)]
            ident, notes = pipeline.identity_from_images(imgs, pipeline.Options(), req.face_index, req.name)
            if ident is None:
                raise HTTPException(status_code=422, detail="; ".join(notes) or "no face found")
            path = facemodel.save(ident, vf.faces_dir(), req.name, overwrite=req.overwrite)
            return {"path": path, "notes": notes}

        return _guard(run)

    @app.post("/visage/v1/swap")
    def swap(req: SwapRequest):
        def run():
            vf.apply_settings()
            opts = _options(req.options)
            if "sfw" not in req.options:
                opts.sfw = vf.sfw_mode()
            target = _image(req.target_image, "target_image")
            if req.face_model:
                idents, notes = vf.resolve_identities("Face model", None, None, req.face_model, "0", opts)
            elif req.source_image:
                idents, notes = vf.resolve_identities("Image", _image(req.source_image, "source_image"), None, None, req.source_indexes, opts)
            else:
                raise HTTPException(status_code=400, detail="give source_image or face_model")
            if not idents:
                raise HTTPException(status_code=422, detail="; ".join(notes) or "no source face")
            r = pipeline.process(target, idents, opts)
            return {"image": vf.to_base64(r.image), "message": r.message, "faces_found": r.faces_found,
                    "swapped": r.swapped, "similarity": r.similarity, "flagged": r.flagged, "notes": notes}

        return _guard(run)

    @app.post("/visage/v1/restore")
    def restore(req: RestoreRequest):
        def run():
            vf.apply_settings()
            opts = _options({"target_indexes": "all", "restorer": "gfpgan_1.4", **req.options})
            r = pipeline.restore_only(_image(req.image, "image"), opts)
            return {"image": vf.to_base64(r.image), "message": r.message}

        return _guard(run)

    @app.post("/visage/v1/detect")
    def detect(req: DetectRequest):
        def run():
            vf.apply_settings()
            img = _image(req.image, "image")
            faces = analysis.sort_faces(analysis.detect(img, req.detector, 640, req.threshold), "large-small")
            out = []
            for i, f in enumerate(faces):
                if req.classify:
                    analysis.classify(img, f)
                out.append({"index": i, "bbox": [round(float(v), 1) for v in f.bbox], "score": round(f.score, 3),
                            "kps": [[round(float(x), 1), round(float(y), 1)] for x, y in f.kps],
                            "gender": f.gender, "age": f.age})
            return {"faces": out}

        return _guard(run)

    @app.post("/visage/v1/unload")
    def unload():
        return {"unloaded": runtime.unload()}


script_callbacks.on_app_started(mount)
