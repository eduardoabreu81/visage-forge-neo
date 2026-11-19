"""The Visage panel, shared by txt2img / img2img (scripts/visage.py) and the Extras tab (scripts/visage_extras.py).

Forge glue like forge.py: imports Gradio and `modules`, so only import it from inside Forge.
"""

from __future__ import annotations

import dataclasses
import os
import traceback

import gradio as gr

from modules import shared

from . import analysis, facemodel, masks, models, pipeline, preview, runtime
from . import __version__
from . import forge as vf
from .refine_args import FROM_PICTURE

# Order of the values returned by VisageScript.ui(); process hooks receive them positionally.
# Part of the API: append new keys at the end only.
ARG_KEYS = [
    "enabled", "source_type", "source_image", "source_files", "face_model", "source_indexes",
    "target_indexes", "sort_order", "gender", "swap_generated", "swap_init",
    "swapper", "strength", "pixel_boost", "color_match",
    "detector", "det_threshold", "det_size", "try_rotations", "min_face_size",
    "mask_box", "mask_blur", "pad_top", "pad_right", "pad_bottom", "pad_left", "mask_model",
    "mask_occlusion", "occluder", "mask_region", "regions", "region_dilate", "keep_mouth",
    "restorer", "restorer_visibility", "codeformer_weight", "face_boost",
    "refine", "refine_denoise", "refine_steps", "refine_padding", "refine_prompt", "refine_negative",
    "refine_reswap", "refine_extend_down", "restore_guard", "refine_texture",
]

# txt2img / img2img only: the Extras picture is already the input and the output.
GENERATION_ONLY_KEYS = ("swap_generated", "swap_init")

# Extras only: there is no generation to borrow these from (see refine_args.py).
EXTRAS_KEYS = ["mode", "refine_sampler", "refine_scheduler", "refine_cfg", "refine_seed"]

# Extras keeps the defaults it always had: a finished picture, so all faces, sharper and restored.
EXTRAS_DEFAULTS = {"target_indexes": "all", "pixel_boost": "2x", "restorer": "gfpgan_1.4", "restorer_visibility": 0.8}


def build_options(s: dict) -> pipeline.Options:
    return pipeline.Options(
        swapper=s.get("swapper") or "hyperswap_1a_256",
        detector=s.get("detector") or "retinaface_10g",
        det_size=int(s.get("det_size") or 640),
        det_threshold=float(s.get("det_threshold") or 0.5),
        try_rotations=bool(s.get("try_rotations")),
        target_indexes=str(s.get("target_indexes") if s.get("target_indexes") is not None else "0"),
        sort_order=s.get("sort_order") or "large-small",
        gender=(s.get("gender") or "any").lower(),
        min_face_size=int(s.get("min_face_size") or 0),
        strength=float(s.get("strength") if s.get("strength") is not None else 1.0),
        pixel_boost=int(str(s.get("pixel_boost") or "1").rstrip("x")),
        color_match=bool(s.get("color_match")),
        mask_box=bool(s.get("mask_box", True)),
        mask_blur=float(s.get("mask_blur") if s.get("mask_blur") is not None else 0.3),
        mask_padding=(int(s.get("pad_top") or 0), int(s.get("pad_right") or 0), int(s.get("pad_bottom") or 0), int(s.get("pad_left") or 0)),
        mask_model=bool(s.get("mask_model", True)),
        mask_occlusion=bool(s.get("mask_occlusion")),
        occluder=s.get("occluder") or "xseg_1",
        mask_region=bool(s.get("mask_region")),
        regions=list(s.get("regions") or masks.DEFAULT_REGIONS),
        region_dilate=int(s.get("region_dilate") or 0),
        keep_mouth=bool(s.get("keep_mouth")),
        restorer=s.get("restorer") or "none",
        restorer_visibility=float(s.get("restorer_visibility") if s.get("restorer_visibility") is not None else 1.0),
        codeformer_weight=float(s.get("codeformer_weight") if s.get("codeformer_weight") is not None else 0.5),
        face_boost=bool(s.get("face_boost")),
        restore_guard=float(s.get("restore_guard") or 0),
        sfw=vf.sfw_mode(),
    )


def first_pass_options(opts: pipeline.Options, s: dict) -> pipeline.Options:
    """With "Keep the refine's skin texture", the swap before the refine skips the restorer: a restored face is
    smooth, and the refine keeps that smoothness instead of drawing skin texture. The second swap restores as usual."""
    texture = s.get("refine") and s.get("refine_reswap", True) and float(s.get("refine_texture") or 0) > 0
    if texture and opts.restorer not in (None, "", "none"):
        return dataclasses.replace(opts, restorer="none")
    return opts


def similarity_note(result) -> str:
    if not result.similarity:
        return ""
    return " · identity match " + ", ".join(f"{v:.2f}" for v in result.similarity)


def status_models() -> str:
    rows = []
    for kind in ("swapper", "detector", "restorer", "occluder", "parser", "recognizer", "classifier", "converter", "safety"):
        for key in models.keys_of(kind):
            spec = models.REGISTRY[key]
            mark = "✅" if models.is_downloaded(key) else "⬇️"
            rows.append(f"| {mark} | {spec.title} | {kind} | {spec.license} |")
    providers = ", ".join(runtime.available_providers()) or "onnxruntime not available"
    note = vf.device_note()
    return (
        (f"ℹ️ {note}\n\n" if note else "")
        + f"**Visage {__version__}** · models: `{models.models_dir()}` · runs on: {runtime.backend()} · "
        f"active provider: `{_safe_provider()}`\n\n"
        f"Available providers: {providers}\n\n"
        "| | Model | Kind | License |\n|---|---|---|---|\n" + "\n".join(rows)
    )


def verify_models() -> str:
    ok, bad = [], []
    for key in models.REGISTRY:
        res = models.verify(key)
        if res is True:
            ok.append(key)
        elif res is False:
            bad.append(key)
    if bad:
        head = f"⚠️ {len(bad)} model(s) do not match their pinned SHA-256; delete and re-download: " + ", ".join(f"`{b}`" for b in bad)
    else:
        head = f"✅ Verified {len(ok)} downloaded model(s): all match their pinned SHA-256."
    return head + "\n\n" + status_models()


def _safe_provider() -> str:
    try:
        return runtime.active_provider()
    except Exception as e:
        return f"error: {e}"


def _sampler_choices() -> list[str]:
    try:
        from modules import sd_samplers

        return [FROM_PICTURE] + list(sd_samplers.visible_sampler_names())
    except Exception:
        return [FROM_PICTURE]


def _scheduler_choices() -> list[str]:
    try:
        from modules import sd_schedulers

        return [FROM_PICTURE] + [s.label for s in sd_schedulers.schedulers]
    except Exception:
        return [FROM_PICTURE]


def build(eid, *, is_img2img: bool = False, extras: bool = False) -> dict:
    """Create the panel's tabs inside the caller's InputAccordion and wire its buttons.

    Returns {key: component} for every key in ARG_KEYS except "enabled" (the caller's accordion),
    minus GENERATION_ONLY_KEYS in Extras, plus EXTRAS_KEYS in Extras.
    """
    defaults = EXTRAS_DEFAULTS if extras else {}
    default_swapper = getattr(shared.opts, "visage_default_swapper", "hyperswap_1a_256")
    if default_swapper not in vf.SWAPPERS:
        default_swapper = "hyperswap_1a_256"
    c = {}

    if extras:
        c["mode"] = gr.Radio(["Swap", "Restore only"], value="Swap", label="Mode", elem_id=eid("mode"),
                             info="Restore only runs the face restorer on the target faces, without a source")

    with gr.Tabs():
        with gr.Tab("Source"):
            c["source_type"] = gr.Radio(["Image", "Multiple images", "Face model"], value="Image", label="Source", elem_id=eid("source_type"))
            with gr.Row():
                c["source_image"] = gr.Image(label="Source face", type="numpy", image_mode="RGB", height=240, elem_id=eid("source_image"))
                c["source_files"] = gr.File(label="Source images (averaged into one identity)", file_count="multiple",
                                            file_types=["image"], type="filepath", visible=False, elem_id=eid("source_files"))
            with gr.Row():
                c["face_model"] = gr.Dropdown(vf.face_model_choices(), label="Face model", visible=False, elem_id=eid("face_model"))
                refresh = gr.Button("🔄", visible=False, scale=0, min_width=40)
            c["source_indexes"] = gr.Textbox("0", label="Source face index(es)", elem_id=eid("source_indexes"),
                                             info="which face of the source image; '0,1' maps source faces to target faces in order")
            with gr.Accordion("🔍 Show detected source faces", open=False):
                src_prev_btn = gr.Button("Detect faces in the source image")
                with gr.Row():
                    src_prev_img = gr.Image(label="Source faces", interactive=False, height=260)
                    src_prev_md = gr.Markdown("")

            def _toggle(kind):
                return (gr.update(visible=kind == "Image"), gr.update(visible=kind == "Multiple images"),
                        gr.update(visible=kind == "Face model"), gr.update(visible=kind == "Face model"))

            c["source_type"].change(_toggle, c["source_type"], [c["source_image"], c["source_files"], c["face_model"], refresh], show_progress=False)
            refresh.click(lambda: gr.update(choices=vf.face_model_choices()), None, c["face_model"], show_progress=False)

        with gr.Tab("Target"):
            with gr.Row():
                c["target_indexes"] = gr.Textbox(defaults.get("target_indexes", "0"), label="Target face index(es)",
                                                 info="'0', '0,2', '1-3' or 'all'", elem_id=eid("target_indexes"))
                c["sort_order"] = gr.Dropdown(analysis.SORT_ORDERS, value="large-small", label="Face order", elem_id=eid("sort_order"))
                c["gender"] = gr.Dropdown(["Any", "Female", "Male"], value="Any", label="Only gender", elem_id=eid("gender"))
            if not extras:
                with gr.Row():
                    c["swap_generated"] = gr.Checkbox(True, label="Swap in generated image", elem_id=eid("swap_generated"))
                    c["swap_init"] = gr.Checkbox(False, label="Swap in img2img input before generation", visible=is_img2img, elem_id=eid("swap_init"))
            with gr.Accordion("Detection", open=False):
                with gr.Row():
                    c["detector"] = gr.Dropdown(vf.DETECTORS, value="retinaface_10g", label="Detector", elem_id=eid("detector"))
                    c["det_threshold"] = gr.Slider(0.1, 0.95, 0.5, step=0.05, label="Confidence", elem_id=eid("det_threshold"))
                    c["det_size"] = gr.Dropdown([320, 480, 640, 800, 1024], value=640, label="Detection size", elem_id=eid("det_size"))
                with gr.Row():
                    c["try_rotations"] = gr.Checkbox(False, label="Detect rotated / lying faces", elem_id=eid("try_rotations"))
                    c["min_face_size"] = gr.Slider(0, 256, 0, step=8, label="Ignore faces smaller than (px)", elem_id=eid("min_face_size"))
            if not extras:
                with gr.Accordion("🧪 Test on an image", open=False):
                    gr.Markdown("Check which index each face gets, or try the current settings without generating.")
                    test_img = gr.Image(label="Test image", type="numpy", image_mode="RGB", height=260)
                    with gr.Row():
                        test_prev_btn = gr.Button("Show face indexes")
                        test_swap_btn = gr.Button("Try swap with current settings", variant="primary")
                    with gr.Row():
                        test_out = gr.Image(label="Result", interactive=False, height=320)
                        test_md = gr.Markdown("")

        with gr.Tab("Swap"):
            with gr.Row():
                c["swapper"] = gr.Dropdown(vf.SWAPPERS, value=default_swapper, label="Swap model", elem_id=eid("swapper"))
                c["pixel_boost"] = gr.Radio(["1x", "2x", "4x"], value=defaults.get("pixel_boost", "1x"), label="Pixel boost",
                                            elem_id=eid("pixel_boost"), info="runs the swapper on sub-pixel tiles for a sharper face")
            with gr.Row():
                c["strength"] = gr.Slider(0.0, 1.5, 1.0, step=0.05, label="Identity strength", elem_id=eid("strength"),
                                          info="0 keeps the target, 1 = source, >1 exaggerates the source")
                c["color_match"] = gr.Checkbox(False, label="Match skin tone to target", elem_id=eid("color_match"))

        with gr.Tab("Mask"):
            with gr.Row():
                c["mask_box"] = gr.Checkbox(True, label="Feathered box", elem_id=eid("mask_box"))
                c["mask_model"] = gr.Checkbox(True, label="Swapper mask (HyperSwap)", elem_id=eid("mask_model"))
                c["mask_occlusion"] = gr.Checkbox(False, label="Occlusion (hands, hair, objects)", elem_id=eid("mask_occlusion"))
                c["keep_mouth"] = gr.Checkbox(False, label="Keep target mouth", elem_id=eid("keep_mouth"))
            with gr.Row():
                c["mask_blur"] = gr.Slider(0.0, 1.0, 0.3, step=0.05, label="Box feather", elem_id=eid("mask_blur"))
                c["occluder"] = gr.Dropdown(vf.OCCLUDERS, value="xseg_1", label="Occlusion model", elem_id=eid("occluder"))
            with gr.Row():
                c["pad_top"] = gr.Slider(0, 50, 0, step=1, label="Pad top %", elem_id=eid("pad_top"))
                c["pad_right"] = gr.Slider(0, 50, 0, step=1, label="Pad right %", elem_id=eid("pad_right"))
                c["pad_bottom"] = gr.Slider(0, 50, 0, step=1, label="Pad bottom %", elem_id=eid("pad_bottom"))
                c["pad_left"] = gr.Slider(0, 50, 0, step=1, label="Pad left %", elem_id=eid("pad_left"))
            with gr.Row():
                c["mask_region"] = gr.Checkbox(False, label="Face parsing regions", elem_id=eid("mask_region"))
                c["region_dilate"] = gr.Slider(-20, 20, 0, step=1, label="Grow / shrink regions (px)", elem_id=eid("region_dilate"))
            c["regions"] = gr.CheckboxGroup(list(masks.REGIONS), value=list(masks.DEFAULT_REGIONS), label="Regions to replace", elem_id=eid("regions"))

        with gr.Tab("Restore"):
            with gr.Row():
                c["restorer"] = gr.Dropdown(vf.RESTORERS, value=defaults.get("restorer", "none"), label="Face restorer", elem_id=eid("restorer"))
                c["restorer_visibility"] = gr.Slider(0.0, 1.0, defaults.get("restorer_visibility", 1.0), step=0.05, label="Visibility",
                                                     elem_id=eid("restorer_visibility"))
                c["codeformer_weight"] = gr.Slider(0.0, 1.0, 0.5, step=0.05, label="CodeFormer fidelity", elem_id=eid("codeformer_weight"))
            c["face_boost"] = gr.Checkbox(False, label="Face boost (restore inside the swap crop before pasting)", elem_id=eid("face_boost"))
            c["restore_guard"] = gr.Slider(0.0, 0.1, 0.0, step=0.01, label="Likeness guard (0 = off)",
                                           elem_id=eid("restore_guard"),
                                           info="if the restorer makes the face less like the source by more than this, "
                                                "Visage lowers the restorer visibility for that face; 0.02-0.03 works well")
            gr.Markdown("In Swap mode, restoration only touches the faces Visage swapped." if extras
                        else "Restoration only touches the faces Visage swapped.")

        with gr.Tab("Refine"):
            if extras:
                gr.Markdown("A low-denoise inpaint of the swapped face with the loaded checkpoint, "
                            "so the new face takes on the picture's lighting and style. Swap mode only. "
                            "Empty fields come from the picture's generation info (PNG Info), "
                            "or from the UI preset's img2img settings when the picture has none.")
            else:
                gr.Markdown("A low-denoise inpaint of the swapped face with the loaded checkpoint, "
                            "so the new face takes on the image's lighting and style. Costs one extra small img2img per face set.")
            with gr.Row():
                c["refine"] = gr.Checkbox(False, label="Refine with checkpoint", elem_id=eid("refine"))
                c["refine_denoise"] = gr.Slider(0.02, 0.6, 0.15, step=0.01, label="Denoise", elem_id=eid("refine_denoise"),
                                                info="turbo / few-step models need less (0.05-0.15)")
                c["refine_steps"] = gr.Slider(0, 60, 0, step=1, elem_id=eid("refine_steps"),
                                              label="Steps (0 = from the picture)" if extras else "Steps (0 = same as generation)")
                c["refine_padding"] = gr.Slider(0, 256, 48, step=8, label="Context padding (px)", elem_id=eid("refine_padding"))
            if extras:
                with gr.Row():
                    c["refine_sampler"] = gr.Dropdown(_sampler_choices(), value=FROM_PICTURE, label="Sampler", elem_id=eid("refine_sampler"))
                    c["refine_scheduler"] = gr.Dropdown(_scheduler_choices(), value=FROM_PICTURE, label="Schedule type",
                                                        elem_id=eid("refine_scheduler"))
                    c["refine_cfg"] = gr.Slider(0.0, 30.0, 0.0, step=0.5, label="CFG scale (0 = from the picture)", elem_id=eid("refine_cfg"))
                    c["refine_seed"] = gr.Number(-1, precision=0, label="Seed (-1 = from the picture, or random)", elem_id=eid("refine_seed"))
            c["refine_prompt"] = gr.Textbox("", lines=2, elem_id=eid("refine_prompt"),
                                            label="Refine prompt (empty = the picture's prompt)" if extras
                                            else "Refine prompt (empty = generation prompt)")
            c["refine_negative"] = gr.Textbox("", lines=1, elem_id=eid("refine_negative"),
                                              label="Refine negative prompt (empty = the picture's negative)" if extras
                                              else "Refine negative prompt (empty = generation negative)")
            c["refine_reswap"] = gr.Checkbox(True, label="Swap again after refining (keeps the identity, keeps the new lighting)",
                                             elem_id=eid("refine_reswap"))
            c["refine_texture"] = gr.Slider(0.0, 1.0, 0.0, step=0.05, label="Keep the refine's skin texture",
                                            elem_id=eid("refine_texture"),
                                            info="for large faces that come out soft: keeps the identity from the swap and "
                                                 "the pores and lashes from the refine. Use 1 with Swap again on and a denoise of 0.4-0.5")
            c["refine_extend_down"] = gr.Slider(0, 60, 0, step=5, label="Extend the refine area below the jaw (% of face height)",
                                                elem_id=eid("refine_extend_down"),
                                                info="the swapper only redraws the face; use 30-50 with a refine prompt "
                                                     "that describes the beard to carry a full beard over")

        with gr.Tab("Face models"):
            gr.Markdown("Save the current source as a reusable face model, or blend saved models into a new one.")
            with gr.Row():
                fm_name = gr.Textbox(label="Name")
                fm_save = gr.Button("💾 Save source as face model")
            with gr.Row():
                fm_blend = gr.Dropdown(vf.face_model_choices(), multiselect=True, label="Blend these models")
                fm_blend_name = gr.Textbox(label="Blend name")
                fm_blend_btn = gr.Button("Blend")
            fm_status = gr.Markdown("")

            def _save(stype, simg, sfiles, fmodel, sidx, detector, name):
                try:
                    opts = pipeline.Options(detector=detector or "retinaface_10g")
                    idents, notes = vf.resolve_identities(stype, simg, sfiles, fmodel, sidx, opts)
                    if not idents:
                        return "⚠️ " + ("; ".join(notes) or "no face found in the source"), gr.update(), gr.update()
                    ident = facemodel.average(idents, name=name)
                    path = facemodel.save(ident, vf.faces_dir(), name)
                    choices = vf.face_model_choices()
                    return f"✅ Saved `{os.path.basename(path)}`", gr.update(choices=choices), gr.update(choices=choices)
                except Exception as e:
                    return f"❌ {e}", gr.update(), gr.update()

            fm_save.click(_save, [c["source_type"], c["source_image"], c["source_files"], c["face_model"], c["source_indexes"], c["detector"], fm_name],
                          [fm_status, c["face_model"], fm_blend])

            def _blend(labels, name):
                try:
                    if not labels or len(labels) < 2:
                        return "⚠️ pick at least two models", gr.update(), gr.update()
                    idents = [facemodel.load(vf.face_model_path(lbl)) for lbl in labels]
                    path = facemodel.save(facemodel.average(idents, name=name), vf.faces_dir(), name or "blend")
                    choices = vf.face_model_choices()
                    return f"✅ Saved `{os.path.basename(path)}`", gr.update(choices=choices), gr.update(choices=choices)
                except Exception as e:
                    return f"❌ {e}", gr.update(), gr.update()

            fm_blend_btn.click(_blend, [fm_blend, fm_blend_name], [fm_status, c["face_model"], fm_blend])

        with gr.Tab("Models"):
            status = gr.Markdown(status_models)
            with gr.Row():
                gr.Button("🔄 Refresh").click(lambda: status_models(), None, status, show_progress=False)
                gr.Button("🧹 Unload models from memory").click(lambda: f"Unloaded {runtime.unload()} session(s).\n\n" + status_models(), None, status)
                gr.Button("🔐 Verify downloaded models (SHA-256)").click(verify_models, None, status)

    keys = [k for k in ARG_KEYS if k in c]
    inputs = [c[k] for k in keys]

    def _src_preview(*vals):
        st = dict(zip(keys, vals))
        img = vf.to_rgb(st.get("source_image"))
        if img is None:
            return None, "⚠️ add a source image first"
        try:
            vf.apply_settings()
            wanted = analysis.parse_indexes(st.get("source_indexes"))
            return preview.annotate(img, build_options(st), highlight=wanted or [0])
        except Exception as e:
            return None, f"❌ {e}"

    src_prev_btn.click(_src_preview, inputs, [src_prev_img, src_prev_md])

    if not extras:
        def _test_preview(img, *vals):
            st = dict(zip(keys, vals))
            img = vf.to_rgb(img)
            if img is None:
                return None, "⚠️ add a test image first"
            try:
                vf.apply_settings()
                opts = build_options(st)
                faces, chosen = pipeline.select_targets(img, opts)
                annotated, summary = preview.annotate(img, opts, classify=opts.gender != "any", highlight=None)
                return annotated, summary + f"\n\nWith the current settings Visage would swap: `{chosen}`"
            except Exception as e:
                return None, f"❌ {e}"

        def _test_swap(img, *vals):
            st = dict(zip(keys, vals))
            img = vf.to_rgb(img)
            if img is None:
                return None, "⚠️ add a test image first"
            try:
                vf.apply_settings()
                opts = build_options(st)
                opts.report_similarity = True
                idents, notes = vf.resolve_identities(st.get("source_type") or "Image", st.get("source_image"), st.get("source_files"),
                                                      st.get("face_model"), st.get("source_indexes"), opts)
                if not idents:
                    return None, "⚠️ " + ("; ".join(notes) or "no source identity")
                r = pipeline.process(img, idents, opts)
                return r.image, f"{r.message}{similarity_note(r)}" + ("\n\n" + "; ".join(notes) if notes else "")
            except Exception as e:
                traceback.print_exc()
                return None, f"❌ {e}"

        test_prev_btn.click(_test_preview, [test_img] + inputs, [test_out, test_md])
        test_swap_btn.click(_test_swap, [test_img] + inputs, [test_out, test_md])

    return c
