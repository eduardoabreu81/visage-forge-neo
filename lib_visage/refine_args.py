"""Settings for a refine pass that has no generation behind it (the Extras tab).

In txt2img / img2img the refine img2img borrows the prompt, sampler, steps, CFG and seed of the
generation. In Extras there is none, so each value comes from, in order: the panel field, the
picture's own generation info (PNG Info), then the UI preset's img2img defaults.

Forge-independent: the caller passes the parsed infotext and the preset defaults in.
"""

from __future__ import annotations

FROM_PICTURE = "From the picture"


def _number(value, kind=float):
    try:
        return kind(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def resolve(fields: dict, info: dict | None, preset: dict | None) -> dict:
    """Return prompt, negative_prompt, steps, sampler_name, scheduler, cfg_scale, distilled_cfg_scale, seed
    and `sources`, which says where each value came from ("field", "picture" or "preset")."""
    info = info or {}
    preset = preset or {}
    out, sources = {}, {}

    def pick(key, field_value, use_field, info_value, preset_value):
        if use_field:
            out[key], sources[key] = field_value, "field"
        elif info_value is not None:
            out[key], sources[key] = info_value, "picture"
        else:
            out[key], sources[key] = preset_value, "preset"

    prompt = (fields.get("prompt") or "").strip()
    pick("prompt", prompt, bool(prompt), info.get("Prompt"), "")
    negative = (fields.get("negative") or "").strip()
    pick("negative_prompt", negative, bool(negative), info.get("Negative prompt"), "")

    steps = _number(fields.get("steps"), int) or 0
    info_steps = _number(info.get("Steps"), int)
    pick("steps", steps, steps > 0, info_steps if info_steps and info_steps > 0 else None, int(preset.get("steps") or 20))

    sampler = fields.get("sampler") or FROM_PICTURE
    pick("sampler_name", sampler, sampler != FROM_PICTURE, info.get("Sampler") or None, preset.get("sampler") or "Euler")
    scheduler = fields.get("scheduler") or FROM_PICTURE
    pick("scheduler", scheduler, scheduler != FROM_PICTURE, info.get("Schedule type") or None, preset.get("scheduler") or "Simple")

    cfg = _number(fields.get("cfg")) or 0.0
    pick("cfg_scale", cfg, cfg > 0, _number(info.get("CFG scale")), float(preset.get("cfg") or 7.0))

    # Flow models store the same slider as "Shift"; Flux-style models as "Distilled CFG Scale".
    info_dcfg = _number(info.get("Distilled CFG Scale"))
    if info_dcfg is None:
        info_dcfg = _number(info.get("Shift"))
    pick("distilled_cfg_scale", None, False, info_dcfg, float(preset.get("dcfg") or 3.0))

    seed = _number(fields.get("seed"), int)
    seed = -1 if seed is None else seed
    info_seed = _number(info.get("Seed"), int)
    pick("seed", seed, seed >= 0, info_seed, -1)

    out["sources"] = sources
    return out
