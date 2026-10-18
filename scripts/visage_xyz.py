"""Visage axes for the built-in X/Y/Z plot script.

Each axis stores its value on `p._visage_xyz`; scripts/visage.py merges those over the
panel settings, so Visage still has to be enabled (with a source) for the axes to apply.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modules import script_callbacks, scripts

from lib_visage import forge as vf


def _set(key, convert=None):
    def apply(p, x, xs):
        overrides = dict(getattr(p, "_visage_xyz", None) or {})
        overrides[key] = convert(x) if convert else x
        if key == "refine_denoise":
            overrides["refine"] = True
        p._visage_xyz = overrides

    return apply


def _find_xyz_module():
    for data in scripts.scripts_data:
        if os.path.basename(data.path) == "xyz_grid.py":
            return data.module
    return None


def add_axes():
    xyz = _find_xyz_module()
    if xyz is None or any(getattr(o, "label", "").startswith("[Visage]") for o in xyz.axis_options):
        return
    fmt = xyz.format_value_add_label
    xyz.axis_options.extend([
        xyz.AxisOption("[Visage] Swap model", str, _set("swapper"), format_value=xyz.format_value, choices=lambda: list(vf.SWAPPERS)),
        xyz.AxisOption("[Visage] Identity strength", float, _set("strength"), format_value=fmt),
        xyz.AxisOption("[Visage] Pixel boost", int, _set("pixel_boost"), format_value=fmt, choices=lambda: ["1", "2", "4"]),
        xyz.AxisOption("[Visage] Face restorer", str, _set("restorer"), format_value=xyz.format_value, choices=lambda: list(vf.RESTORERS)),
        xyz.AxisOption("[Visage] Restorer visibility", float, _set("restorer_visibility"), format_value=fmt),
        xyz.AxisOption("[Visage] Refine denoise", float, _set("refine_denoise"), format_value=fmt),
        xyz.AxisOption("[Visage] Target faces", str, _set("target_indexes"), format_value=fmt),
    ])


script_callbacks.on_before_ui(add_axes)
