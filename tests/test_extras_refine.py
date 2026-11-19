"""Extras refine settings and the shared panel's argument order, without importing Forge."""

import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib_visage import refine_args  # noqa: E402
from lib_visage.refine_args import FROM_PICTURE  # noqa: E402

PANEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib_visage", "panel.py")

PRESET = {"sampler": "Euler", "scheduler": "Simple", "steps": 8, "cfg": 1.0, "dcfg": 3.0}
INFO = {"Prompt": "a portrait, soft light", "Negative prompt": "blurry", "Steps": "30", "Sampler": "DPM++ 2M",
        "Schedule type": "Karras", "CFG scale": "5.5", "Seed": "1234", "Distilled CFG Scale": "3.5"}
EMPTY = {"prompt": "", "negative": "", "steps": 0, "sampler": FROM_PICTURE, "scheduler": FROM_PICTURE, "cfg": 0, "seed": -1}


def test_empty_fields_take_the_picture_info():
    r = refine_args.resolve(EMPTY, INFO, PRESET)
    assert r["prompt"] == "a portrait, soft light"
    assert r["negative_prompt"] == "blurry"
    assert (r["steps"], r["sampler_name"], r["scheduler"]) == (30, "DPM++ 2M", "Karras")
    assert (r["cfg_scale"], r["distilled_cfg_scale"], r["seed"]) == (5.5, 3.5, 1234)
    assert set(r["sources"].values()) == {"picture"}


def test_fields_win_over_the_picture():
    fields = {"prompt": "a man with a beard", "negative": "", "steps": 12, "sampler": "Euler a", "scheduler": FROM_PICTURE,
              "cfg": 2.0, "seed": 7}
    r = refine_args.resolve(fields, INFO, PRESET)
    assert (r["prompt"], r["steps"], r["sampler_name"], r["cfg_scale"], r["seed"]) == ("a man with a beard", 12, "Euler a", 2.0, 7)
    assert r["scheduler"] == "Karras" and r["sources"]["scheduler"] == "picture"
    assert r["negative_prompt"] == "blurry"


def test_picture_without_info_uses_the_preset():
    r = refine_args.resolve(EMPTY, {}, PRESET)
    assert (r["prompt"], r["negative_prompt"]) == ("", "")
    assert (r["steps"], r["sampler_name"], r["scheduler"], r["cfg_scale"], r["distilled_cfg_scale"]) == (8, "Euler", "Simple", 1.0, 3.0)
    assert r["seed"] == -1
    assert r["sources"]["steps"] == "preset"


def test_shift_counts_as_the_distilled_value():
    r = refine_args.resolve(EMPTY, {"Shift": "9"}, PRESET)
    assert r["distilled_cfg_scale"] == 9.0


def test_bad_numbers_in_the_info_are_ignored():
    r = refine_args.resolve(EMPTY, {"Steps": "lots", "CFG scale": "", "Seed": "x"}, PRESET)
    assert (r["steps"], r["cfg_scale"], r["seed"]) == (8, 1.0, -1)


def test_preset_steps_zero_falls_back_to_20():
    r = refine_args.resolve(EMPTY, {}, {**PRESET, "steps": 0})
    assert r["steps"] == 20


def _panel_list(name):
    tree = ast.parse(open(PANEL, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in panel.py")


def test_arg_keys_order_is_unchanged():
    # Script args are positional (AGENTS.md invariant 7): these 46 keys keep their order forever.
    assert _panel_list("ARG_KEYS")[:46] == [
        "enabled", "source_type", "source_image", "source_files", "face_model", "source_indexes",
        "target_indexes", "sort_order", "gender", "swap_generated", "swap_init",
        "swapper", "strength", "pixel_boost", "color_match",
        "detector", "det_threshold", "det_size", "try_rotations", "min_face_size",
        "mask_box", "mask_blur", "pad_top", "pad_right", "pad_bottom", "pad_left", "mask_model",
        "mask_occlusion", "occluder", "mask_region", "regions", "region_dilate", "keep_mouth",
        "restorer", "restorer_visibility", "codeformer_weight", "face_boost",
        "refine", "refine_denoise", "refine_steps", "refine_padding", "refine_prompt", "refine_negative",
        "refine_reswap", "refine_extend_down", "restore_guard",
    ]


def test_extras_keys_do_not_clash_with_arg_keys():
    assert not set(_panel_list("EXTRAS_KEYS")) & set(_panel_list("ARG_KEYS"))
