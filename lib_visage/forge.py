"""Glue between the Forge-independent core and Forge Neo (settings, paths, conversions, refine pass).

Only import this module from inside Forge.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os

import cv2
import numpy as np
from PIL import Image

from modules import paths, shared

from . import facemodel, models, pipeline, runtime

SECTION = ("visage", "Visage")

SWAPPERS = models.keys_of("swapper")
DETECTORS = models.keys_of("detector")
RESTORERS = ["none"] + models.keys_of("restorer")
OCCLUDERS = models.keys_of("occluder")
PARSERS = models.keys_of("parser")


# --- settings -------------------------------------------------------------------

def register_settings():
    import gradio as gr

    def add(key, default, label, component=None, component_args=None, info=None):
        opt = shared.OptionInfo(default, label, component, component_args, section=SECTION)
        if info:
            opt = opt.info(info)
        shared.opts.add_option(key, opt)

    add("visage_models_dir", "", "Models folder", info=f"empty = {os.path.join(paths.models_path, 'visage')}")
    add("visage_auto_download", True, "Download missing models automatically")
    add("visage_provider", "auto", "ONNX execution provider", gr.Dropdown,
        {"choices": list(runtime.PROVIDER_CHOICES)}, info="auto picks CUDA > ROCm > MIGraphX > DirectML > CoreML > CPU")
    add("visage_unload_after", False, "Unload Visage models after every generation", info="frees VRAM at the cost of reload time")
    add("visage_save_original", False, "Also save the image before Visage processing (suffix -before-visage)")
    add("visage_default_swapper", "hyperswap_1a_256", "Default swapper", gr.Dropdown, {"choices": SWAPPERS})


def apply_settings():
    models.set_models_dir(models_root())
    models.allow_download = bool(getattr(shared.opts, "visage_auto_download", True))
    runtime.provider_preference = getattr(shared.opts, "visage_provider", "auto") or "auto"


def models_root() -> str:
    cmd = getattr(shared.cmd_opts, "visage_models_dir", None)
    opt = getattr(shared.opts, "visage_models_dir", "") if hasattr(shared, "opts") else ""
    return cmd or opt or os.path.join(paths.models_path, "visage")


def faces_dir() -> str:
    return os.path.join(models_root(), "faces")


def face_model_folders() -> list[str]:
    return [faces_dir(), os.path.join(paths.models_path, "reactor", "faces")]


def face_model_choices() -> list[str]:
    return list(facemodel.list_models(face_model_folders()))


def face_model_path(label: str) -> str | None:
    return facemodel.list_models(face_model_folders()).get(label)


# --- conversions ------------------------------------------------------------------

def to_rgb(value) -> np.ndarray | None:
    """Accept numpy / PIL / base64 / data-url / file path / gradio dicts and return RGB uint8."""
    if value is None:
        return None
    if isinstance(value, dict):  # gradio ImageEditor / file dicts
        value = value.get("composite") or value.get("image") or value.get("background") or value.get("path") or value.get("name")
        if value is None:
            return None
    if isinstance(value, Image.Image):
        return np.array(value.convert("RGB"))
    if isinstance(value, np.ndarray):
        arr = value
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return np.ascontiguousarray(arr.astype(np.uint8))
    if hasattr(value, "name") and isinstance(getattr(value, "name"), str):  # tempfile wrappers
        value = value.name
    if isinstance(value, str):
        if os.path.isfile(value):
            return np.array(Image.open(value).convert("RGB"))
        data = value.split(",", 1)[1] if value.startswith("data:") else value
        try:
            return np.array(Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB"))
        except Exception:
            return None
    return None


def to_base64(arr: np.ndarray, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def fingerprint(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(str(p.shape).encode())
            h.update(p[::7, ::7].tobytes())
        else:
            h.update(repr(p).encode())
    return h.hexdigest()


# --- identities -------------------------------------------------------------------

_identity_cache: dict[str, list] = {}


def resolve_identities(source_type: str, source_image, source_files, face_model: str, source_indexes: str,
                       opts: pipeline.Options) -> tuple[list[facemodel.Identity], list[str]]:
    """Turn the UI source settings into identities (cached by content)."""
    notes: list[str] = []
    if source_type == "Face model":
        path = face_model_path(face_model) if face_model else None
        if not path:
            return [], [f"face model '{face_model}' not found"]
        return [facemodel.load(path)], notes

    from .analysis import parse_indexes

    indexes = parse_indexes(source_indexes) or [0]
    if source_type == "Multiple images":
        images = [to_rgb(f) for f in (source_files or [])]
        images = [i for i in images if i is not None]
        if not images:
            return [], ["no source images"]
        key = fingerprint("multi", indexes[0], opts.detector, *images)
        if key not in _identity_cache:
            ident, notes = pipeline.identity_from_images(images, opts, indexes[0])
            _identity_cache[key] = [ident] if ident else []
        return _identity_cache[key], notes

    image = to_rgb(source_image)
    if image is None:
        return [], ["no source image"]
    key = fingerprint("single", tuple(indexes), opts.detector, image)
    if key not in _identity_cache:
        idents, notes = pipeline.identities_from_image(image, indexes, opts)
        _identity_cache[key] = idents
    if len(_identity_cache) > 16:
        _identity_cache.pop(next(iter(_identity_cache)))
    return _identity_cache[key], notes


# --- refine pass (Forge only) ----------------------------------------------------------

def refine(p, image: Image.Image, face_mask: np.ndarray, denoise: float, steps: int, padding: int,
           prompt: str, negative: str, index: int) -> Image.Image:
    """Low-denoise inpaint of the swapped area with the loaded checkpoint, so the new face
    picks up the scene's lighting, grain and style. Runs a nested img2img with scripts disabled."""
    from modules.processing import StableDiffusionProcessingImg2Img, process_images

    mask = (cv2.dilate((face_mask > 0.15).astype(np.uint8) * 255, np.ones((9, 9), np.uint8)))
    if mask.max() == 0:
        return image
    mask_img = Image.fromarray(mask, mode="L")

    def pick(values, fallback):
        try:
            return values[index] if values else fallback
        except (IndexError, TypeError):
            return fallback

    seed = pick(getattr(p, "all_seeds", None), p.seed)
    i2i = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=p.outpath_samples,
        outpath_grids=p.outpath_grids,
        prompt=prompt or pick(getattr(p, "all_prompts", None), p.prompt),
        negative_prompt=negative or pick(getattr(p, "all_negative_prompts", None), p.negative_prompt),
        styles=[],
        seed=seed,
        subseed=pick(getattr(p, "all_subseeds", None), p.subseed),
        subseed_strength=0,
        sampler_name=p.sampler_name,
        scheduler=getattr(p, "scheduler", None),
        batch_size=1,
        n_iter=1,
        steps=int(steps) if steps else p.steps,
        cfg_scale=p.cfg_scale,
        distilled_cfg_scale=getattr(p, "distilled_cfg_scale", 3.5),
        width=image.width,
        height=image.height,
        restore_faces=False,
        tiling=False,
        do_not_save_samples=True,
        do_not_save_grid=True,
        init_images=[image],
        mask=mask_img,
        mask_blur=4,
        inpainting_fill=1,
        resize_mode=0,
        denoising_strength=float(denoise),
        inpaint_full_res=True,
        inpaint_full_res_padding=int(padding),
        inpainting_mask_invert=0,
        override_settings={},
    )
    i2i.scripts = None
    i2i.script_args = []
    i2i._visage_nested = True
    try:
        processed = process_images(i2i)
    finally:
        i2i.close()
    if processed is not None and processed.images:
        return processed.images[0]
    return image
