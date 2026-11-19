"""Visage in the Extras tab: swap, restore and refine faces on any image (single or batch), with the full panel."""

import os
import sys
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from PIL import Image

from modules import scripts_postprocessing
from modules.ui_components import InputAccordion

from lib_visage import panel, pipeline
from lib_visage import forge as vf


class ScriptPostprocessingVisage(scripts_postprocessing.ScriptPostprocessing):
    name = "Visage"
    order = 1900

    def ui(self):
        eid = lambda name: f"extras_visage_{name}"  # noqa: E731
        with InputAccordion(False, label="Visage", elem_id="extras_visage") as enabled:
            c = panel.build(eid, extras=True)
        return {"enabled": enabled, **c}

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enabled=False, mode="Swap", **s):
        if not enabled:
            return
        for key, value in panel.EXTRAS_DEFAULTS.items():
            s.setdefault(key, value)
        try:
            vf.apply_settings()
            opts = panel.build_options(s)
            source = pp.image
            arr = np.array(source.convert("RGB"))
            if mode == "Restore only":
                result = pipeline.restore_only(arr, opts)
                print(f"[Visage] Extras: {result.message}")
                if result.swapped:
                    pp.image = Image.fromarray(result.image)
                    pp.info["Visage"] = f"restore only, {opts.restorer}"
                return

            idents, notes = vf.resolve_identities(s.get("source_type") or "Image", s.get("source_image"), s.get("source_files"),
                                                  s.get("face_model"), s.get("source_indexes"), opts)
            for n in notes:
                print(f"[Visage] {n}")
            if not idents:
                print("[Visage] Extras: no source identity, skipped (check the Source tab)")
                return
            result = pipeline.process(arr, idents, opts)
            print(f"[Visage] Extras: {result.message}{panel.similarity_note(result)}")
            if not result.swapped:
                return
            out = Image.fromarray(result.image)
            info = f"swap, {opts.swapper}"
            if s.get("refine") and result.face_mask is not None:
                out, refined = self._refine(source, out, result.face_mask, idents, opts, s)
                if refined:
                    info += f", refine {float(s.get('refine_denoise') or 0.15):g}"
            pp.image = out
            pp.info["Visage"] = info
        except Exception as e:
            print(f"[Visage] Extras failed: {e}")
            traceback.print_exc()

    @staticmethod
    def _refine(source, out, face_mask, idents, opts, s):
        """Refine with the loaded checkpoint; the settings come from the panel, the picture's PNG Info or the preset."""
        try:
            fields = {"prompt": s.get("refine_prompt"), "negative": s.get("refine_negative"), "steps": s.get("refine_steps"),
                      "sampler": s.get("refine_sampler"), "scheduler": s.get("refine_scheduler"), "cfg": s.get("refine_cfg"),
                      "seed": s.get("refine_seed")}
            settings, sources = vf.refine_settings_for_picture(source, fields)
            from_picture = sorted(k for k, v in sources.items() if v == "picture")
            print(f"[Visage] Extras refine: {settings.sampler_name} / {settings.scheduler}, {settings.steps} steps, "
                  f"CFG {settings.cfg_scale:g}" + (f" (from the picture: {', '.join(from_picture)})" if from_picture else ""))
            refined = vf.refine(settings, out, face_mask, s.get("refine_denoise") or 0.15, int(s.get("refine_padding") or 48),
                                float(s.get("refine_extend_down") or 0) / 100)
            if s.get("refine_reswap", True):
                again = pipeline.process(np.array(refined.convert("RGB")), idents, opts)
                print(f"[Visage] Extras after refine: {again.message}{panel.similarity_note(again)}")
                if again.swapped:
                    refined = Image.fromarray(again.image)
            return refined, True
        except Exception as e:
            print(f"[Visage] Extras refine failed, keeping the plain swap: {e}")
            traceback.print_exc()
            return out, False
