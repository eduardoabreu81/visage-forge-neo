# PROJECT_LOG

## 2026-10-18 — 0.2.0, more swappers and an optional SFW filter

- New swappers: AlphaFace 256 (0.85), GHOST 1/2/3 256 (GHOST 1: 0.82), SimSwap 256 (0.70) and unofficial 512 (0.62),
  HiFiFace 256 unofficial (0.61, has a mask output). Numbers are identity similarity on the test set at 1x on CPU.
- CrossFace converters (ArcFace to GHOST / SimSwap / HiFiFace identity space) are a new model kind, `converter`.
  Verified: the converters and AlphaFace give the same result for a unit-norm input and for a raw (~23 norm) one,
  so identities stay unit vectors everywhere.
- Fix: the model-mask output is now picked by name (`mask`). GHOST returns feature maps as extra outputs.
- New templates: `arcface_112_v1` (SimSwap/GHOST) and `mtcnn_512` (HiFiFace).
- Optional SFW filter `lib_visage/safety.py`: 3 NSFW models vote and 2 of 3 are needed. Modes: off (default) / skip / blur.
  Settings → Visage → "Optional SFW filter". API: `options.sfw` overrides the setting per request. With the filter off,
  no safety model is loaded (there is a test for this). There were no false positives on the 10 ordinary test images.
  True positives were not tested on purpose (no NSFW test data).
- Version bumped to 0.2.0.

## 2026-10-18 — 0.1.1, validation round

- Tested in Forge through the API: batch size 2 (2 images × 2 faces), img2img (swap the generated image), and img2img with swap only on the input.
  The last one works, but at 0.3 denoise the turbo model redraws the faces, so the input swap mostly disappears. That is expected.
- **Bug found:** with turbo models, a refine at 0.25 denoise wipes out the identity (similarity 0.75 → 0.15; at 0.10 it stays 0.74).
  **Fix:** "Swap again after refining" (new arg `refine_reswap`, on by default) brings it back to 0.74 / 0.76 while keeping
  the refined lighting. The default refine denoise is now 0.15.
- New: per-face ArcFace similarity in the console (`visage_report_similarity`) and in `Result.similarity`.
- New: Source → "Show detected source faces" and Target → "Test on an image" (face indexes, "try swap"), using `lib_visage/preview.py`.
- Added `localizations/pt_BR.json`, then removed it: the UI stays English-only, like the other extensions.
- Note: Forge replaces `print` with a tqdm logging handler, so lines printed right after a nested process_images can
  be missing from a redirected console log. The code still runs; this was verified with a file probe.

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
