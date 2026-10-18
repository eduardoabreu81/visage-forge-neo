"""Visage in the Extras tab: swap or restore faces on any image (single or batch)."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gradio as gr
import numpy as np
from PIL import Image

from modules import scripts_postprocessing
from modules.ui_components import InputAccordion

from lib_visage import analysis, pipeline
from lib_visage import forge as vf


class ScriptPostprocessingVisage(scripts_postprocessing.ScriptPostprocessing):
    name = "Visage"
    order = 1900

    def ui(self):
        with InputAccordion(False, label="Visage", elem_id="extras_visage") as enabled:
            with gr.Row():
                mode = gr.Radio(["Swap", "Restore only"], value="Swap", label="Mode", elem_id="extras_visage_mode")
                source_type = gr.Radio(["Image", "Face model"], value="Image", label="Source", elem_id="extras_visage_source_type")
            with gr.Row():
                source_image = gr.Image(label="Source face", type="numpy", image_mode="RGB", height=200, elem_id="extras_visage_source")
                with gr.Column():
                    face_model = gr.Dropdown(vf.face_model_choices(), label="Face model", elem_id="extras_visage_face_model")
                    gr.Button("🔄 Refresh face models").click(lambda: gr.update(choices=vf.face_model_choices()), None, face_model, show_progress=False)
                    source_indexes = gr.Textbox("0", label="Source face index(es)", elem_id="extras_visage_source_indexes")
                    target_indexes = gr.Textbox("all", label="Target face index(es)", elem_id="extras_visage_target_indexes")
            with gr.Row():
                swapper = gr.Dropdown(vf.SWAPPERS, value="hyperswap_1a_256", label="Swap model", elem_id="extras_visage_swapper")
                pixel_boost = gr.Radio(["1x", "2x", "4x"], value="2x", label="Pixel boost", elem_id="extras_visage_boost")
                strength = gr.Slider(0.0, 1.5, 1.0, step=0.05, label="Identity strength", elem_id="extras_visage_strength")
            with gr.Row():
                sort_order = gr.Dropdown(analysis.SORT_ORDERS, value="large-small", label="Face order", elem_id="extras_visage_order")
                occlusion = gr.Checkbox(False, label="Occlusion mask", elem_id="extras_visage_occlusion")
                color_match = gr.Checkbox(False, label="Match skin tone", elem_id="extras_visage_color")
            with gr.Row():
                restorer = gr.Dropdown(vf.RESTORERS, value="gfpgan_1.4", label="Face restorer", elem_id="extras_visage_restorer")
                visibility = gr.Slider(0.0, 1.0, 0.8, step=0.05, label="Restorer visibility", elem_id="extras_visage_visibility")
                codeformer_weight = gr.Slider(0.0, 1.0, 0.5, step=0.05, label="CodeFormer fidelity", elem_id="extras_visage_cf")

        return {
            "enabled": enabled, "mode": mode, "source_type": source_type, "source_image": source_image,
            "face_model": face_model, "source_indexes": source_indexes, "target_indexes": target_indexes,
            "swapper": swapper, "pixel_boost": pixel_boost, "strength": strength, "sort_order": sort_order,
            "occlusion": occlusion, "color_match": color_match, "restorer": restorer, "visibility": visibility,
            "codeformer_weight": codeformer_weight,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enabled=False, mode="Swap", source_type="Image",
                source_image=None, face_model=None, source_indexes="0", target_indexes="all", swapper="hyperswap_1a_256",
                pixel_boost="2x", strength=1.0, sort_order="large-small", occlusion=False, color_match=False,
                restorer="none", visibility=0.8, codeformer_weight=0.5):
        if not enabled:
            return
        vf.apply_settings()
        opts = pipeline.Options(
            swapper=swapper, target_indexes=target_indexes, sort_order=sort_order, strength=float(strength),
            pixel_boost=int(str(pixel_boost).rstrip("x")), mask_occlusion=bool(occlusion), color_match=bool(color_match),
            restorer=restorer or "none", restorer_visibility=float(visibility), codeformer_weight=float(codeformer_weight),
        )
        arr = np.array(pp.image.convert("RGB"))
        if mode == "Restore only":
            result = pipeline.restore_only(arr, opts)
        else:
            idents, notes = vf.resolve_identities(source_type, source_image, None, face_model, source_indexes, opts)
            for n in notes:
                print(f"[Visage] {n}")
            if not idents:
                print("[Visage] Extras: no source identity, skipped")
                return
            result = pipeline.process(arr, idents, opts)
        print(f"[Visage] Extras: {result.message}")
        if result.swapped:
            pp.image = Image.fromarray(result.image)
            pp.info["Visage"] = f"{mode.lower()}, {swapper if mode == 'Swap' else restorer}"
