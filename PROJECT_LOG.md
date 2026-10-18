# PROJECT_LOG

## 2026-10-18 — 0.1.0, first build

**Context.** A community request (tusharbhutt, on GitHub) asked for a ReActor that works with Forge Neo.
Research showed:
- `sd-webui-reactor` has not changed since Jan 2025, pins `insightface==0.7.3` (cp310 wheel only) and breaks on Neo (Py 3.13, numpy 2);
- Forge issue #856: ReActor swaps onnxruntime-gpu for CPU onnxruntime on every boot;
- issue #1461: ReActor's install.py uses `pkg_resources`;
- the ComfyUI ReActor 0.7 added a core without insightface, HyperSwap, ReSwapper, face boost, masking and swap weight;
- VisoMaster-Fusion (GPL-3) has the richest masking and restorer set.

Decision: a new extension with its own name (Visage), its own code (ONNX only), and features chosen
from both projects. Forge PR #1561 (open) removes `--onnxruntime-gpu`, because stable `onnxruntime-gpu`
now supports cu130. So Visage installs onnxruntime itself, but only when none is present.

**Done**
- Core `lib_visage`: detection (SCRFD/RetinaFace 10G, SCRFD 2.5G, YOLOv8-face, rotated faces, padding retry for tight crops),
  ArcFace, FairFace gender/age, swappers (HyperSwap 1a/1b/1c, inswapper, ReSwapper), pixel boost 1/2/4x,
  identity strength, masks (box, HyperSwap mask, XSeg, BiSeNet regions, keep mouth), LAB colour match, restorers
  (GFPGAN, CodeFormer, GPEN 256–2048, RestoreFormer++), face boost, face models (safetensors, blend, ReActor import).
- Forge: txt2img/img2img accordion (Source, Target, Swap, Mask, Restore, Refine, Face models, Models tabs),
  swap of the img2img input, refine pass (nested inpaint), Extras tab (swap / restore only), REST API, settings.
- install.py with the onnxruntime policy (see AGENTS.md invariant 1).
- 12 tests passing on Forge Neo's Python 3.13.12 / numpy 2.3.5 / onnxruntime 1.26 (CPU),
  including real swaps with an identity-similarity check.

- Loaded in a real Forge Neo 2.27 instance (local, port 7861, `--api`) with no errors. The txt2img and
  img2img accordions, the Extras accordion and Settings → Visage all render.
- API checked end to end: detect + FairFace gender/age, face model build, swap (face model and image, gender
  filter "female" → 3 of 3), restore (CodeFormer), list face models.
- `/sdapi/v1/txt2img` with `alwayson_scripts.Visage` (Anima realistic turbo, 768x1024): swap only, then swap + refine.
  Both complete and write `Visage` / `Visage refine` to the infotext. The refine pass changes the face area.

**Not verified yet**
- GPU execution providers. Tests ran on CPU: the dev machine is AMD and has plain `onnxruntime` installed.
  `onnxruntime-directml` should accelerate it on Windows.
- Swap of the img2img input, batch size > 1, and the Extras-tab path through the UI. The code paths exist but no click-test was done.

**Backlog**
- Optional SFW gate (FaceFusion NSFW models) if hosting on GitHub requires it.
- More swappers: SimSwap 512 / Ghost (need crossface embedding converters), AlphaFace.
- Face similarity readout in the UI; per-face preview of detected indexes.
- Video (Forge Neo video models / frame-by-frame).
- `localizations/pt_BR.json`.
