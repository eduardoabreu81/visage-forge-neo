# 🎭 Visage for Forge Neo

<div align="center">

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
[![Gradio](https://img.shields.io/badge/Gradio-4.40.0-orange)](https://gradio.app/)
[![Version](https://img.shields.io/badge/Version-0.4.0-brightgreen)](https://github.com/eduardoabreu81/visage-forge-neo)
![Status](https://img.shields.io/badge/Status-Preview-orange)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](LICENSE)

> **Extension for [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)**

</div>

Swap, restore and refine faces inside Forge Neo. Visage works in txt2img, img2img and the Extras tab, using the Generate button you already know.
It is **ONNX only**: no insightface, no extra packages that fight over onnxruntime, and no second Python to install. It runs on the Python 3.13 that Forge Neo ships with.

<p align="center"><img src=".github/images/swap-fisherman.jpg" alt="Source: an older man with a white beard. Generated: a younger fisherman in a yellow raincoat. With Visage: the fisherman with the older man's face, plus a close-up of the face before and after" width="760"></p>

> [!Important]
> **Preview.** Tested end to end inside Forge Neo 2.30 (txt2img, img2img, batch, Extras, X/Y/Z, API), on CPU and on an NVIDIA RTX 2070. Please [report what you find](https://github.com/eduardoabreu81/visage-forge-neo/issues).

> [!Note]
> Every face in this README was generated. None of them shows a real person.

---

## 📋 Table of Contents

- [What's New](#-whats-new)
- [Why Visage](#-why-visage)
- [Features](#-features)
- [Natural or Likeness?](#%EF%B8%8F-natural-or-likeness)
- [Installation](#-installation)
- [Recommended Settings](#%EF%B8%8F-recommended-settings)
- [Swap Models](#-swap-models)
- [API](#-api)
- [Tips](#-tips)
- [Responsible Use](#%EF%B8%8F-responsible-use)
- [Credits](#-credits)

---

## 🆕 What's New

### v0.4.0 - NVIDIA GPU and Full Beards

- **Runs on NVIDIA GPUs**: tested on an RTX 2070, about 5 times faster than on the CPU
- **Tells you when the GPU is not used**: if your Forge has the CPU-only onnxruntime and an NVIDIA card, Visage says so at startup and shows the two commands that fix it
- **Installs the right GPU build**: on a fresh install it picks the onnxruntime-gpu that matches your Forge's CUDA version (12 or 13)
- **inswapper and ReSwapper no longer fail** with a `protobuf` / `runtime_version` error on some Forge installs
- **Refine can carry a full beard over**: extend the refine area below the jaw and describe the beard in the refine prompt

### v0.3.0 - Integrity, PNG Info and X/Y/Z

- **Pinned SHA-256 for every model**: a download that does not match is discarded, and **Models → Verify** re-checks the files on disk
- **PNG Info / Send to** restores the Visage panel: swap model, strength, faces, boost, restorer and refine
- **X/Y/Z plot axes**: compare swap models, strength, pixel boost, restorers or refine denoise in one grid

### v0.2.0 - More Swap Models and an Optional SFW Filter

- **AlphaFace, GHOST 1/2/3, SimSwap 256/512 and HiFiFace** join HyperSwap, inswapper and ReSwapper
- **Optional SFW filter**, off by default: when you turn it on, it skips or blurs images that at least two of three NSFW models flag

### v0.1.1 - Previews and Refine Fix

- **Face indexes preview** and **Try swap** on a test image, without generating
- **Swap again after refining**: keeps the identity when the refine pass runs on turbo models
- **Identity match** in the console: how close each swapped face is to its source

### v0.1.0 - First Release

- Face swap, face restore and checkpoint refine for Forge Neo, ONNX only

---

## 💡 Why Visage

The face swap extensions people used with A1111 depend on `insightface`. Older versions needed C++ build tools. The current one installs easily, but it requires the CPU-only `onnxruntime` and the `onnx` package. On an NVIDIA machine that puts a second onnxruntime next to `onnxruntime-gpu`, and `onnx` fails on some Forge installs because of its `protobuf` version. Some extensions also replace `onnxruntime-gpu` with the CPU build on every start. Visage was written from scratch to avoid all of this:

- the whole pipeline is plain `onnxruntime` + `numpy` + `opencv`, with no insightface and no `onnx` package
- the installer adds an onnxruntime build **only when none is installed**, and never uninstalls or replaces one
- models download the first time you use them, from their upstream hosts, into `models/visage`

---

## 🎯 Features

### 🔄 Face Swap

<p align="center"><img src=".github/images/swap-astronaut-v2.jpg" alt="Source: a woman with a black bob and bangs. Generated: a red-haired, freckled astronaut. With Visage: the astronaut with the source face, plus a close-up before and after" width="760"></p>

- **14 swap models**: HyperSwap 1A/1B/1C, AlphaFace, inswapper (+fp16), ReSwapper 128/256, GHOST 1/2/3, SimSwap 256/512 and HiFiFace
- **Pixel boost** 1x / 2x / 4x: the swapper runs on sub-pixel tiles, so 128 and 256 px models give a sharper face
- **Identity strength**: 0 keeps the generated face, 1 is the source face, and values in between mix the two
- **Sources**: one picture, several pictures averaged into one identity, or a saved face model
- Swap the generated image, the img2img input before generation, or both

### 🎚️ Masks and Colour

<p align="center"><img src=".github/images/swap-jazz-v2.jpg" alt="Source: a colour portrait of a woman. Generated: a black and white 1950s jazz singer. With Visage: the singer with the source face, still in black and white" width="760"></p>

- **Feathered box** with padding, plus **HyperSwap's own mask**
- **Occlusion** with XSeg, so hands, hair, microphones and glasses stay in front of the face
- **Face parsing regions** (BiSeNet): pick which parts get replaced, such as skin, eyes, nose or lips
- **Keep target mouth** for open mouths, teeth and tongues
- **Match skin tone** to the target, for black-and-white pictures, colour grading or different lighting

### ✨ Restore and Refine

<p align="center"><img src=".github/images/refine-beard.jpg" alt="Close-ups: the source with a full white beard, the plain swap with a short beard, the swap plus refine, and the refine with its area extended below the jaw, where the full beard appears" width="900"></p>

- **Face restorers**: GFPGAN 1.4, CodeFormer (with fidelity), GPEN-BFR 256/512/1024/2048 and RestoreFormer++. They touch only the faces Visage swapped
- **Face boost**: restore inside the swap crop before pasting it back
- **Refine with the checkpoint**: a low-denoise inpaint of the new face with the model you are using, so the face takes on the picture's lighting, grain and style. It then swaps again to keep the identity
- **Beards**: a swap model only redraws the face itself, so a full beard stops at the jaw. In **Refine**, set **Extend the refine area below the jaw** to 30–50, describe the beard in the refine prompt and use a denoise around 0.5

### 👥 Choose Your Faces

<p align="center"><img src=".github/images/face-indexes-v2.jpg" alt="Four friends on a sofa with numbered boxes on their faces, the same picture with only face number 2 swapped, and a close-up of face 2 before and after" width="900"></p>

- **Target faces** by index (`0`, `0,2`, `1-3` or `all`), in 7 sort orders
- **Show face indexes** on any test picture, or on the source, before you generate
- **Only gender** (FairFace), minimum face size, and rotated / lying faces
- **Several people at once**: source faces `0,1` go to target faces `0,1` in order

### 🧑 Face Models

- Save the current source as a small `.safetensors` face model and pick it later from a list
- Blend several face models into one
- ReActor face models in `models/reactor/faces` show up in the list too

### 🧪 Try It Before You Generate

- **Target → Test on an image**: drop any picture, see the face indexes, and **Try swap** with the current settings
- The console shows how close each swapped face is to its source (`identity match 0.78`)

### 🧰 Forge Integration

- **Extras tab**: swap or only restore faces on any picture, including batches
- **PNG Info / Send to** restores the Visage settings (the source face is not stored in the picture)
- **X/Y/Z plot** axes: swap model, identity strength, pixel boost, face restorer, restorer visibility, refine denoise and target faces
- **REST API** under `/visage/v1/`

### 🛡️ Safe by Design

- Changes no Forge Neo file; remove the extension and everything is as it was
- Never replaces an onnxruntime build that is already installed
- Every model has a pinned SHA-256; a download that does not match is discarded
- Works with CUDA, ROCm, MIGraphX, DirectML, CoreML or CPU, picked automatically or set in **Settings → Visage**
- **Optional SFW filter**, off by default. With it off, no safety model is downloaded or loaded

---

## ⚖️ Natural or Likeness?

<p align="center"><img src=".github/images/natural-likeness.jpg" alt="Close-ups of two swaps with the same source: the natural setting keeps a softer, cleaner look, and the likeness setting looks more like the source" width="900"></p>

No swap model is best at everything, so Visage lets you choose:

- **Natural** (the default): **HyperSwap 1A**, pixel boost 2x, GFPGAN at 0.6. The cleanest skin and eyes, keeps the expression and lighting, and is the hardest to spot as a swap. It keeps a little more of the generated face.
- **Likeness**: **inswapper 128**, pixel boost 2x, GFPGAN at 0.6. The face is recognisable at once. It can show small artefacts, such as reddish eyelids, that the natural setting does not.

The number under each picture is how close the face is to the source (1.0 would be identical).

---

## 📦 Installation

1. Open Forge Neo WebUI
2. Go to **Extensions** → **Install from URL**
3. Paste: `https://github.com/eduardoabreu81/visage-forge-neo`
4. Click **Install** and restart the WebUI
5. On the first start, Visage checks for onnxruntime:
   - if any onnxruntime build is installed, it is left as it is
   - if none is, Visage installs the GPU build for NVIDIA (matching your Forge's CUDA version), `onnxruntime-directml` on AMD / Intel under Windows, or `onnxruntime` otherwise. Set `VISAGE_ORT_PACKAGE` to choose another package
6. Open the **Visage** accordion in txt2img or img2img. Models download the first time they are needed: about 180 MB for detection and recognition, then the swap model you pick (HyperSwap is 384 MB)

To keep the models somewhere else, start Forge with `--visage-models-dir D:\models\visage` or set the folder in **Settings → Visage**.

### NVIDIA card, but Visage says it runs on the CPU?

Another extension may have installed the CPU-only onnxruntime before Visage. Visage never replaces it on its own, because swapping onnxruntime builds behind your back is what broke other face swap extensions. To use the GPU, close Forge, open a terminal in the Forge folder and run:

```
venv\Scripts\python -m pip uninstall -y onnxruntime
venv\Scripts\python -m pip install "onnxruntime-gpu>=1.27"
```

Use `"onnxruntime-gpu<1.27"` instead if your Forge runs on CUDA 12 (the startup message tells you which one). Start Forge again: the message is gone and **Settings → Visage → Models** shows `CUDAExecutionProvider`.

> [!Note]
> Tested on Forge Neo 2.30 with Python 3.13, PyTorch 2.13 (CUDA 13), on the CPU and on an RTX 2070 with onnxruntime-gpu 1.31.

---

## ⚙️ Recommended Settings

| Goal | Swap model | Pixel boost | Restorer | Other |
| :--- | :--- | :--- | :--- | :--- |
| Natural look (default) | HyperSwap 1A | 2x | GFPGAN 1.4 at 0.6 | |
| Strongest likeness | inswapper 128 | 2x | GFPGAN 1.4 at 0.6 | |
| Older faces, wrinkles, stubble | HyperSwap 1C | 2x | GFPGAN 1.4 at 0.6 | |
| Full beard on the source | HyperSwap 1C | 2x | GFPGAN 1.4 at 0.6 | Refine, extend below the jaw 30–50, denoise about 0.5, beard in the refine prompt |
| Black and white, strong grading | HyperSwap 1A | 2x | GFPGAN 1.4 at 0.6 | Match skin tone |
| Hands or objects over the face | any | 2x | any | Occlusion mask |
| Face looks pasted on | any | 2x | any | Refine, denoise 0.10–0.15 on turbo models, 0.2–0.3 on regular ones |

---

## 🧩 Swap Models

<p align="center"><img src=".github/images/swap-models-v2.jpg" alt="Close-ups of the source, the generated fisherman, and the fisherman swapped with six models, each labelled with its identity score" width="900"></p>

*Same source and picture, six models, no restorer. Identity is how close the face is to the source (higher is closer). A higher number is not always the more believable picture: look at the skin and the edges too.*

| Model | Size | License | Notes |
| :--- | :--- | :--- | :--- |
| HyperSwap 1A / 1B / 1C | 384 MB | ResearchRAIL | natural, keeps expression and lighting, has its own mask |
| AlphaFace 256 | 529 MB | Non-commercial | newest, strong likeness |
| inswapper 128 (+fp16) | 529 / 264 MB | Non-commercial | strongest likeness, softer: use pixel boost 2x |
| ReSwapper 128 / 256 | 528 MB | Non-commercial | keeps more of the target |
| GHOST 1 / 2 / 3 | 491–815 MB | Apache-2.0 | the only swap model with a permissive license |
| SimSwap 256 / 512 | 210 / 228 MB | Non-commercial | |
| HiFiFace 256 | 194 MB | Unknown | keeps face shape |

**Settings → Visage → Models** lists every model with its license and whether it is downloaded.

---

## 🔌 API

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

`options` takes the field names of [`lib_visage/pipeline.py`](lib_visage/pipeline.py) `Options`, for example
`{"swapper": "hyperswap_1a_256", "pixel_boost": 2, "target_indexes": "all", "restorer": "gfpgan_1.4", "report_similarity": true}`.

Visage also works in `/sdapi/v1/txt2img` and `/sdapi/v1/img2img` through `alwayson_scripts.Visage.args`, in the order of `ARG_KEYS` in [`scripts/visage.py`](scripts/visage.py).

---

## 💡 Tips

- A clear, front-facing source photo works best. For a stronger identity, use **Multiple images** with 3 to 5 photos of the same face
- Not sure which face is index `1`? Use **Target → Test on an image → Show face indexes**
- With turbo / few-step models, keep the refine denoise low (0.10–0.15) and leave **Swap again after refining** on
- Hair, beards and ears outside the face come from the generated picture, not the source. Describe them in your prompt, or use the refine with an extended area for beards
- Small faces in wide shots gain the most from pixel boost 2x and a restorer
- **Unload models from memory** in the Models tab frees the memory Visage uses, without restarting Forge

---

## ⚖️ Responsible Use

Face swapping can hurt real people. Only use faces you have the right to use. Do not create non-consensual intimate imagery, impersonate people or make content meant to deceive. In many places this is illegal. Label synthetic media as synthetic.

The **optional SFW filter** (**Settings → Visage**) can skip or blur NSFW results, for shared machines and public demos. It is off by default and does not limit anything while it is off.

---

## 📄 Credits

- **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** by Haoming02
- **[InsightFace](https://github.com/deepinsight/insightface)**: SCRFD, ArcFace and inswapper
- **[FaceFusion](https://github.com/facefusion/facefusion)** by Henry Ruhs: HyperSwap and the ONNX ports of most models used here
- **[ReSwapper](https://github.com/somanchiu/ReSwapper)**, **[GHOST](https://github.com/ai-forever/ghost)**, **[SimSwap](https://github.com/neuralchen/SimSwap)**, HiFiFace and AlphaFace: the swap models
- **[GFPGAN](https://github.com/TencentARC/GFPGAN)**, **[CodeFormer](https://github.com/sczhou/CodeFormer)**, **[GPEN](https://github.com/yangxy/GPEN)** and **[RestoreFormer++](https://github.com/wzhouxiff/RestoreFormerPlusPlus)**: face restoration
- **DeepFaceLab** (XSeg), **[face-parsing](https://github.com/yakhyo/face-parsing)** by yakhyo (BiSeNet) and **[FairFace](https://github.com/joojs/fairface)**
- **[ReActor](https://codeberg.org/Gourieff/comfyui-reactor-node)** by Gourieff and **[VisoMaster-Fusion](https://github.com/VisoMasterFusion/VisoMaster-Fusion)**: the features the community asked for

Visage is an independent implementation and contains no code from these projects. It does not include any model weights. Each model downloads from its upstream host and keeps its own license, and several are for non-commercial or research use only.

---

## 📜 License

AGPL-3.0 - see [LICENSE](LICENSE)

---

<div align="center">

Made with ❤️ for the Stable Diffusion community

**[Report Bug](https://github.com/eduardoabreu81/visage-forge-neo/issues)** • **[Request Feature](https://github.com/eduardoabreu81/visage-forge-neo/issues)** • **[☕ Ko-fi](https://ko-fi.com/eduardoabreu81)**

</div>
