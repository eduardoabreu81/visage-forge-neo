# AGENTS.md — Visage for Forge Neo

Notes for agents and contributors working on this repository.

## What this is

A Forge Neo extension for face swap, face restoration and checkpoint refine. It uses ONNX only
(no insightface). Target stack: Python 3.13+, numpy 2.x, Pillow 12/13, Gradio 4.40, onnxruntime ≥ 1.20.

## Layout

```
install.py            dependency check (onnxruntime policy, see invariants)
preload.py            --visage-models-dir
lib_visage/           core package, importable without Forge
  geometry.py         5-point templates, warp, paste_back, pixel-boost tiling
  models.py           model registry, URLs, licenses, downloads to <models>/visage
  runtime.py          onnxruntime sessions and provider selection
  analysis.py         detection (SCRFD/RetinaFace, YOLOv8-face), ArcFace embedding, FairFace, sorting
  swapper.py          inswapper / ReSwapper (emap projection) and HyperSwap
  onnxread.py         reads the emap initializer from the .onnx bytes (no `onnx` / `protobuf` dependency)
  masks.py            box, XSeg occlusion, BiSeNet regions
  restorer.py         GFPGAN / CodeFormer / GPEN / RestoreFormer++
  facemodel.py        Identity, average, .safetensors face models (and ReActor face models)
  pipeline.py         Options dataclass, process(), restore_only()
  forge.py            Forge glue: settings, paths, image conversion, identity cache, refine pass
scripts/
  visage.py           AlwaysVisible script (txt2img / img2img UI and hooks)
  visage_extras.py    Extras tab postprocessing
  visage_api.py       /visage/v1/* FastAPI routes
  visage_xyz.py       X/Y/Z plot axes (override panel values via p._visage_xyz)
lib_visage/hashes.py  pinned SHA-256 + size per model (regenerate from HF LFS metadata when adding models)
lib_visage/safety.py  optional SFW filter
lib_visage/preview.py annotated face previews
tools/sfw_check.py    measure the SFW filter on a local folder
tests/test_core.py    unit tests, plus model tests when VISAGE_MODELS_DIR and VISAGE_TEST_IMAGES are set
```

## Invariants (do not break)

1. **Never uninstall, downgrade or replace an installed onnxruntime build.** Install one only when none
   is present. (Other face swap extensions broke Forge by swapping `onnxruntime-gpu` for CPU onnxruntime on every boot.)
2. **No insightface, no `pkg_resources`, no compilers.** Everything must install from wheels on Python 3.13 and 3.14.
3. **No model weights in the repo** (`*.onnx` and `*.safetensors` are gitignored). Models download from upstream on first use.
4. `lib_visage/*` (except `forge.py`) must not import `modules.*`, so the core stays testable outside Forge.
5. Every frame in the pipeline is **RGB uint8**. XSeg is the only model fed BGR, and `masks.occlusion_mask` handles that conversion.
6. Use the import path `lib_visage`, not a bare `scripts.*`: other extensions have broken the shared `scripts`
   namespace in Forge before.
7. Script args are positional. `ARG_KEYS` in `scripts/visage.py` defines their order and it is part of the API.
   Append new keys at the end only.
8. A nested refine img2img sets `_visage_nested`, and the hooks must skip it.

## Model I/O facts (verified 2026-10-18)

- inswapper / ReSwapper: `target` 1x3xSxS RGB in [0,1]; `source` = normed ArcFace @ emap (the graph's last
  initializer), then renormalised. Output in [0,1]. Template arcface_128.
- HyperSwap: `source` = normed ArcFace (1x512); `target` normalised to [-1,1]. Outputs `output` in [-1,1]
  and `mask` (1x1x256x256).
- SCRFD / RetinaFace-10G: input (x-127.5)/128 RGB, letterboxed top-left; 9 outputs (scores, boxes, kps per stride 8/16/32, 2 anchors).
- ArcFace w600k_r50: 112 crop (arcface_112), x/127.5-1, RGB.
- Restorers: RGB in [-1,1]; CodeFormer has an extra `weight` (double) input.
- AlphaFace: target RGB in [0,1] at arcface_128/256; source = ArcFace vector (scale-invariant); output in [0,1].
- GHOST: arcface_112_v1, target in [-1,1], source = crossface_ghost(ArcFace); output in [-1,1]; extra feature-map outputs.
- SimSwap 256: arcface_112_v1, ImageNet mean/std; 512: [0,1]; source = normalised crossface_simswap(ArcFace); output in [0,1].
- HiFiFace: mtcnn_512, [-1,1], normalised crossface_hififace(ArcFace); outputs `output` and `mask`.
- NSFW (optional filter): nsfw_1 640 detector (score = max class > 0.2), nsfw_2 384 (p0-p1 > 0.25),
  nsfw_3 448 CLIP-norm ((l2+l3)-(l0+l1) > 10.5); NSFW when 2 of 3 agree.

## Tests

```
python -m pytest tests
VISAGE_MODELS_DIR=<forge>/models/visage VISAGE_TEST_IMAGES=<dir with group.jpg, source.png> python -m pytest tests
```

The identity check measures cosine similarity between the swapped face and the source. Baseline on
CPU at 1x: inswapper ≈ 0.87, HyperSwap 1A ≈ 0.75, ReSwapper 256 ≈ 0.63.
