# Visage for Forge Neo

Face swap, face restoration and checkpoint refine for
[Stable Diffusion WebUI Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo).
It only uses ONNX: **no insightface, no C++ build tools, no second Python install.** It runs on
Python 3.13+, numpy 2 and Pillow 12/13, which is the stack Neo ships with.

> **Status:** early preview (0.1.1). The core pipeline is tested; the Forge UI is new. Please report issues.

## Why another face swap extension?

The face swap extensions people use with A1111 pin `insightface==0.7.3`, which only has a
prebuilt wheel for Python 3.10. Some also replace `onnxruntime-gpu` with the CPU build on every
start, which breaks other extensions. Visage avoids both problems:

- the whole pipeline is plain `onnxruntime` + `numpy` + `opencv`;
- the installer adds an onnxruntime build **only when none is installed**, and never uninstalls or swaps one;
- models are downloaded on first use, into `models/visage`.

## Features

| | |
|---|---|
| **Swap models** | HyperSwap 1A / 1B / 1C (256), inswapper 128 (+fp16), ReSwapper 128 / 256 |
| **Pixel boost** | 1x / 2x / 4x sub-pixel tiling, so 128/256 px swappers give sharper faces |
| **Identity strength** | 0 → keep the target, 1 → source, up to 1.5 → exaggerate the source |
| **Face selection** | indexes (`0`, `0,2`, `1-3`, `all`), 7 sort orders, gender filter (FairFace), minimum size, rotated faces |
| **Multi-face mapping** | source faces `0,1` → target faces `0,1` in order |
| **Sources** | one image, several images averaged into one identity, or a saved face model |
| **Face models** | save / blend `.safetensors` identities; ReActor face models in `models/reactor/faces` are listed too |
| **Masks** | feathered box with padding, HyperSwap's own mask, XSeg occlusion (hands, hair, objects), BiSeNet face-parsing regions, "keep target mouth" |
| **Colour** | optional LAB skin-tone match to the target |
| **Restore** | GFPGAN 1.4, CodeFormer (fidelity), GPEN-BFR 256/512/1024/2048, RestoreFormer++. Only swapped faces are touched. Face boost restores inside the swap crop |
| **Refine (Forge only)** | low-denoise inpaint of the swapped face with the loaded checkpoint, so the face picks up the image's lighting, grain and style; then an optional second swap restores the identity the inpaint may have drifted from |
| **Preview & test** | numbered face boxes (with gender/age) for the source and any test image, and "try swap" without generating |
| **Identity check** | prints the ArcFace similarity of each swapped face to its source (≈0.75 is a solid HyperSwap result) |
| **Languages** | English, Português (Brasil) |
| **Where** | txt2img, img2img (generated image and/or the img2img input), Extras tab (swap or restore only, batch too), REST API |
| **Runtime** | CUDA, ROCm, MIGraphX, DirectML, CoreML or CPU, picked automatically or pinned in Settings |

## Install

1. In Forge Neo: **Extensions → Install from URL**, paste this repository's URL, then click **Install**.
2. Restart Forge. On first start Visage checks for onnxruntime:
   - if any onnxruntime build is installed, Visage leaves it as it is;
   - otherwise it installs `onnxruntime-gpu` (NVIDIA), `onnxruntime-directml` (AMD/Intel on Windows) or `onnxruntime` (others).
     Set `VISAGE_ORT_PACKAGE` to pick another package.
3. Open the **Visage** accordion in txt2img or img2img. Models download the first time they are used
   (HyperSwap is about 384 MB, the detector and recognizer about 180 MB).

To use a custom folder, pass `--visage-models-dir D:\models\visage` or set it in **Settings → Visage**.

## Quick start

1. Enable **Visage**, then drop a face photo into **Source**.
2. Generate. The largest face (index `0`) is swapped with HyperSwap 1A.
3. For a sharper result, set **Swap → Pixel boost** to `2x` and **Restore** to GFPGAN at about 0.7 visibility.
4. If the face looks pasted on, enable **Refine**. Use a denoise of 0.1–0.15 on turbo / few-step models
   and 0.2–0.3 on regular ones. Keep "Swap again after refining" on.
5. Not sure which face is index `1`? Open **Target → Test on an image** and click **Show face indexes**.

To keep an identity: **Face models → Save source as face model**, then choose **Source → Face model**.

## API

The routes are mounted on Forge's FastAPI app:

```
GET  /visage/v1/models
GET  /visage/v1/facemodels
POST /visage/v1/facemodels   {"name": "...", "images": ["<b64>", ...]}
POST /visage/v1/swap         {"target_image": "<b64>", "source_image": "<b64>" | "face_model": "name", "options": {...}}
POST /visage/v1/restore      {"image": "<b64>", "options": {"restorer": "codeformer"}}
POST /visage/v1/detect       {"image": "<b64>", "classify": true}
POST /visage/v1/unload
```

`options` takes the field names of [`lib_visage.pipeline.Options`](lib_visage/pipeline.py), for example
`{"swapper": "hyperswap_1b_256", "pixel_boost": 2, "target_indexes": "all", "restorer": "gfpgan_1.4"}`.

Visage also works in `/sdapi/v1/txt2img` through `alwayson_scripts.Visage.args`, in the order of
`ARG_KEYS` in [`scripts/visage.py`](scripts/visage.py). Pass the source image as base64.

## Model licenses

Visage does not include any weights. Each model comes from its upstream host and keeps its own license.
**Several are non-commercial or research-only:** inswapper, ReSwapper, the ArcFace recognizer,
CodeFormer, GPEN and HyperSwap (ResearchRAIL). **Settings → Visage → Models** lists every model and its license.

## Responsible use

Face swapping can hurt real people. Only use faces you have the right to use. Do not create
non-consensual intimate imagery, impersonate people, or make content meant to deceive. In many
places this is illegal. Mark synthetic media as such.

## Credits

Visage is an independent implementation. It draws on the published work of the
[InsightFace](https://github.com/deepinsight/insightface) project (SCRFD, ArcFace, inswapper),
[FaceFusion](https://github.com/facefusion/facefusion) (HyperSwap and the ONNX model ports it hosts),
[ReSwapper](https://github.com/somanchiu/ReSwapper), DeepFaceLab (XSeg), and the GFPGAN, CodeFormer, GPEN
and RestoreFormer authors. The feature set was shaped by what the community uses in
[ReActor](https://codeberg.org/Gourieff/comfyui-reactor-node) and
[VisoMaster-Fusion](https://github.com/VisoMasterFusion/VisoMaster-Fusion). Visage contains no code from those projects.

## License

AGPL-3.0, the same as Forge.
