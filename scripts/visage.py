"""Visage: face swap, face restore and refine for Forge Neo (txt2img / img2img)."""

import os
import sys
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from PIL import Image

from modules import images, script_callbacks, scripts, shared
from modules.ui_components import InputAccordion

from lib_visage import panel, pipeline, runtime
from lib_visage import forge as vf

ARG_KEYS = panel.ARG_KEYS  # positional script args; see lib_visage/panel.py
build_options = panel.build_options


def _settings(args) -> dict:
    return dict(zip(ARG_KEYS, args))


def infotext_params(opts: pipeline.Options, s: dict) -> dict:
    """Generation-info entries, read back by PNG Info / Send to (see paste_fields)."""
    params = {
        "Visage": opts.swapper,
        "Visage strength": f"{opts.strength:g}",
        "Visage faces": opts.target_indexes,
    }
    if opts.pixel_boost > 1:
        params["Visage boost"] = f"{opts.pixel_boost}x"
    if opts.restorer and opts.restorer != "none":
        params["Visage restorer"] = opts.restorer
        params["Visage restorer visibility"] = f"{opts.restorer_visibility:g}"
        if opts.restore_guard > 0:
            params["Visage likeness guard"] = f"{opts.restore_guard:g}"
    if s.get("refine"):
        params["Visage refine"] = f"{float(s.get('refine_denoise') or 0.15):g}"
        if s.get("refine_extend_down"):
            params["Visage refine extend"] = f"{float(s.get('refine_extend_down')):g}"
        if s.get("refine_reswap", True) and s.get("refine_texture"):
            params["Visage refine texture"] = f"{float(s.get('refine_texture')):g}"
    return params


def paste_fields(c: dict) -> list:
    """(component, infotext key or callable) pairs: pasting an infotext restores the Visage panel."""
    return [
        (c["enabled"], lambda d: "Visage" in d),
        (c["swapper"], lambda d: d.get("Visage") if d.get("Visage") in vf.SWAPPERS else None),
        (c["strength"], "Visage strength"),
        (c["target_indexes"], "Visage faces"),
        (c["pixel_boost"], lambda d: d.get("Visage boost", "1x") if "Visage" in d else None),
        (c["restorer"], lambda d: d.get("Visage restorer", "none") if "Visage" in d else None),
        (c["restorer_visibility"], "Visage restorer visibility"),
        (c["restore_guard"], lambda d: d.get("Visage likeness guard", 0) if "Visage" in d else None),
        (c["refine"], lambda d: "Visage refine" in d if "Visage" in d else None),
        (c["refine_denoise"], "Visage refine"),
        (c["refine_extend_down"], lambda d: d.get("Visage refine extend", 0) if "Visage" in d else None),
        (c["refine_texture"], lambda d: d.get("Visage refine texture", 0) if "Visage" in d else None),
    ]


_similarity_note = panel.similarity_note


class VisageScript(scripts.Script):
    sorting_priority = 20

    def title(self):
        return "Visage"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # --- UI -------------------------------------------------------------------------
    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"
        eid = lambda name: f"visage_{tab}_{name}"  # noqa: E731
        with InputAccordion(False, label="Visage", elem_id=eid("enabled")) as enabled:
            c = {"enabled": enabled, **panel.build(eid, is_img2img=is_img2img)}

        self.infotext_fields = paste_fields(c)
        self.paste_field_names = [key for _, key in self.infotext_fields if isinstance(key, str)]

        missing = [k for k in ARG_KEYS if k not in c]
        assert not missing, f"Visage UI is missing components: {missing}"
        return [c[k] for k in ARG_KEYS]

    # --- processing -------------------------------------------------------------
    def _prepare(self, p, s):
        vf.apply_settings()
        opts = build_options(s)
        idents, notes = vf.resolve_identities(s.get("source_type") or "Image", s.get("source_image"), s.get("source_files"),
                                              s.get("face_model"), s.get("source_indexes"), opts)
        for n in notes:
            print(f"[Visage] {n}")
        return opts, idents

    def process(self, p, *args):
        if getattr(p, "_visage_nested", False):
            return
        s = _settings(args)
        s.update(getattr(p, "_visage_xyz", None) or {})  # X/Y/Z plot axes (scripts/visage_xyz.py)
        self._job = None
        self._early = False
        if not s.get("enabled"):
            return
        try:
            opts, idents = self._prepare(p, s)
            opts.report_similarity = bool(getattr(shared.opts, "visage_report_similarity", True))
        except Exception as e:
            print(f"[Visage] disabled for this job: {e}")
            traceback.print_exc()
            return
        if not idents:
            print("[Visage] no source identity - nothing to do (check the Source tab)")
            return
        self._job = (opts, idents, s)
        p.extra_generation_params.update(infotext_params(opts, s))
        self._early = False
        try:
            self._early = self._swap_early(p)
            if self._early:
                print(f"[Visage] swapping before {', '.join(self._enhancers_before(p))}, so it also improves the new face")
        except Exception as e:  # never let the ordering check break a generation
            print(f"[Visage] could not check other extensions: {e}")

        if s.get("swap_init") and getattr(p, "init_images", None):
            swapped = []
            for img in p.init_images:
                try:
                    arr = np.array(img.convert("RGB"))
                    r = pipeline.process(arr, idents, opts)
                    print(f"[Visage] input: {r.message}")
                    swapped.append(Image.fromarray(r.image))
                except Exception as e:
                    print(f"[Visage] input swap failed: {e}")
                    swapped.append(img)
            p.init_images = swapped

    def _enhancers_before(self, p) -> list[str]:
        """Enabled extensions that change the finished picture in postprocess_image_after_composite and run before
        Visage (Forge runs them in folder-name order). DLSS 5 is one: without care it enhances the image first and
        the swapped face is pasted onto it afterwards, untouched."""
        runner = getattr(p, "scripts", None)
        found = []
        for script in getattr(runner, "alwayson_scripts", None) or []:
            if script is self:
                break
            if type(script).postprocess_image_after_composite is scripts.Script.postprocess_image_after_composite:
                continue
            args = (getattr(p, "script_args", None) or [])[script.args_from:script.args_to]
            if args and args[0] is False:  # the usual enable checkbox / InputAccordion is off
                continue
            found.append(script.title())
        return found

    def _swap_early(self, p) -> bool:
        """Swap in postprocess_image, before those extensions, when the picture is already final there
        (no inpaint mask to composite). Otherwise keep the usual postprocess_image_after_composite."""
        if getattr(p, "image_mask", None) is not None or getattr(p, "paste_to", None) is not None:
            return False
        return bool(self._enhancers_before(p))

    def postprocess_image(self, p, pp, *args):
        if getattr(self, "_early", False):
            self._swap(p, pp)

    def postprocess_image_after_composite(self, p, pp, *args):
        if not getattr(self, "_early", False):
            self._swap(p, pp)

    def _swap(self, p, pp):
        job = getattr(self, "_job", None)
        if job is None or getattr(p, "_visage_nested", False):
            return
        opts, idents, s = job
        if not s.get("swap_generated", True):
            return
        if shared.state.interrupted or shared.state.skipped:
            return
        try:
            original = pp.image
            arr = np.array(original.convert("RGB"))
            first = panel.first_pass_options(opts, s)
            result = pipeline.process(arr, idents, first)
            print(f"[Visage] {result.message}{_similarity_note(result)}")
            if not result.swapped:
                return
            refined = False
            if getattr(shared.opts, "visage_save_original", False) and p.save_samples():
                try:
                    images.save_image(original, p.outpath_samples, "", p.all_seeds[pp.index] if p.all_seeds else p.seed,
                                      p.prompt, shared.opts.samples_format, p=p, suffix="-before-visage")
                except Exception as e:
                    print(f"[Visage] could not save the original: {e}")
            out = Image.fromarray(result.image)
            if s.get("refine") and result.face_mask is not None:
                try:
                    settings = vf.refine_settings_from_p(p, pp.index, int(s.get("refine_steps") or 0),
                                                         s.get("refine_prompt") or "", s.get("refine_negative") or "")
                    out = vf.refine(settings, out, result.face_mask, s.get("refine_denoise") or 0.15,
                                    int(s.get("refine_padding") or 48), float(s.get("refine_extend_down") or 0) / 100)
                    if s.get("refine_reswap", True):
                        again = pipeline.process(np.array(out.convert("RGB")), idents, opts)
                        print(f"[Visage] after refine: {again.message}{_similarity_note(again)}")
                        if again.swapped:
                            out = Image.fromarray(pipeline.keep_texture(again.image, np.array(out.convert("RGB")),
                                                                        again.face_mask, float(s.get("refine_texture") or 0)))
                    refined = True
                except Exception as e:
                    print(f"[Visage] refine failed, keeping the plain swap: {e}")
                    traceback.print_exc()
            if not refined and first is not opts:  # the first pass skipped the restorer for the refine
                out = Image.fromarray(pipeline.process(arr, idents, opts).image)
            out.info = dict(original.info)
            pp.image = out
        except Exception as e:
            print(f"[Visage] swap failed: {e}")
            traceback.print_exc()

    def postprocess(self, p, processed, *args):
        if getattr(p, "_visage_nested", False):
            return
        self._job = None
        self._early = False
        if getattr(shared.opts, "visage_unload_after", False):
            runtime.unload()


def _startup_notice(*_):
    try:
        vf.apply_settings()
        note = runtime.note() if runtime.device == "GPU" else None  # the CPU default needs no warning
    except Exception:
        note = None
    if note:
        print(f"[Visage] {note.removeprefix('Visage: ')}")


script_callbacks.on_ui_settings(vf.register_settings)
script_callbacks.on_app_started(_startup_notice)
