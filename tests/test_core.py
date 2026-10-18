"""Core tests. Model-backed tests run only when models and sample images are available:

    VISAGE_MODELS_DIR=<forge>/models/visage
    VISAGE_TEST_IMAGES=<folder with group.jpg (several faces) and source.png (one face)>
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib_visage import analysis, facemodel, geometry, masks, models, onnxread, pipeline, swapper  # noqa: E402

MODELS = os.environ.get("VISAGE_MODELS_DIR")
IMAGES = os.environ.get("VISAGE_TEST_IMAGES")


def _pb_varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _pb_len(num, payload):
    return _pb_varint(num << 3 | 2) + _pb_varint(len(payload)) + payload


def _pb_int(num, value):
    return _pb_varint(num << 3) + _pb_varint(value)


def test_gpu_hint(monkeypatch):
    from lib_visage import runtime

    assert runtime.ort_gpu_spec(13) == "onnxruntime-gpu>=1.27"
    assert runtime.ort_gpu_spec(12) == "onnxruntime-gpu<1.27"
    monkeypatch.setattr(runtime, "available_providers", lambda: ["CPUExecutionProvider"])
    assert runtime.gpu_hint(False) is None
    assert "onnxruntime-gpu<1.27" in runtime.gpu_hint(True, 12)
    monkeypatch.setattr(runtime, "available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert runtime.gpu_hint(True, 13) is None


def test_onnxread_last_initializer(tmp_path):
    first = np.arange(6, dtype=np.float32).reshape(2, 3)
    emap = np.linspace(-1, 1, 12, dtype=np.float32).reshape(3, 4)
    half = np.array([0.5, -2.0], dtype=np.float16)

    def raw(a):  # packed dims, FLOAT, name, raw_data
        return (_pb_len(1, b"".join(_pb_varint(d) for d in a.shape)) + _pb_int(2, 1)
                + _pb_len(8, b"w") + _pb_len(9, a.astype("<f4").tobytes()))

    fp16 = _pb_int(1, 2) + _pb_int(2, 10) + _pb_len(5, b"".join(_pb_varint(int(v)) for v in half.view(np.uint16)))
    for tensors, expected in (([raw(first), raw(emap)], emap), ([raw(first), fp16], half)):
        graph = _pb_len(1, _pb_len(1, b"node")) + b"".join(_pb_len(5, t) for t in tensors)
        path = tmp_path / "m.onnx"
        path.write_bytes(_pb_int(1, 8) + _pb_len(7, graph) + _pb_len(8, _pb_int(2, 13)))
        got = onnxread.last_initializer(str(path))
        assert got.dtype == expected.dtype and np.array_equal(got, expected)


def test_pixel_boost_roundtrip():
    rng = np.random.default_rng(0)
    crop = rng.random((256, 256, 3)).astype(np.float32)
    tiles = geometry.implode_pixel_boost(crop, 2, 128)
    assert len(tiles) == 4 and tiles[0].shape == (128, 128, 3)
    assert np.array_equal(geometry.explode_pixel_boost(tiles, 2, 128), crop)


def test_warp_and_paste_identity():
    frame = np.full((300, 300, 3), 100, np.uint8)
    kps = geometry.TEMPLATES["arcface_128"] * 128 + 80
    crop, matrix = geometry.warp_face(frame, kps, "arcface_128", 128)
    assert crop.shape == (128, 128, 3)
    white = np.full_like(crop, 255, dtype=np.float32)
    out = geometry.paste_back(frame, white, np.ones((128, 128), np.float32), matrix)
    assert out[80 + 64, 80 + 64].tolist() == [255, 255, 255]
    assert out[5, 5].tolist() == [100, 100, 100]


def test_parse_indexes():
    assert analysis.parse_indexes("all") is None
    assert analysis.parse_indexes("") is None
    assert analysis.parse_indexes("0, 2") == [0, 2]
    assert analysis.parse_indexes("1-3") == [1, 2, 3]


def test_box_mask_shape_and_range():
    m = masks.box_mask(256, 0.3, (10, 0, 0, 0))
    assert m.shape == (256, 256)
    assert 0.0 <= m.min() and m.max() <= 1.0
    assert m[128, 128] > 0.99 and m[0, 128] < 0.01


def test_blend_identity_endpoints():
    rng = np.random.default_rng(1)
    s, t = rng.normal(size=512), rng.normal(size=512)
    np.testing.assert_allclose(swapper.blend_identity(s, t, 1.0), s / np.linalg.norm(s), rtol=1e-6)
    np.testing.assert_allclose(swapper.blend_identity(s, t, 0.0), t / np.linalg.norm(t), rtol=1e-6)


def test_facemodel_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    a = facemodel.Identity(rng.normal(size=512).astype(np.float32), "a")
    b = facemodel.Identity(rng.normal(size=512).astype(np.float32), "b")
    path = facemodel.save(facemodel.average([a, b], name="ab"), str(tmp_path), "ab")
    loaded = facemodel.load(path)
    assert loaded.name == "ab" and loaded.sources == 2
    assert abs(np.linalg.norm(loaded.embedding) - 1.0) < 1e-5
    with pytest.raises(FileExistsError):
        facemodel.save(loaded, str(tmp_path), "ab")
    assert list(facemodel.list_models([str(tmp_path)])) == ["ab"]


def test_options_from_dict_ignores_unknown():
    o = pipeline.Options.from_dict({"swapper": "inswapper_128", "nope": 1, "mask_padding": [1, 2, 3, 4]})
    assert o.swapper == "inswapper_128" and o.mask_padding == (1, 2, 3, 4)


def test_registry_is_consistent():
    for key, spec in models.REGISTRY.items():
        assert spec.urls and spec.file.endswith(".onnx"), key
        if spec.kind == "swapper":
            assert spec.template in geometry.TEMPLATES and spec.size in (128, 256, 512), key


needs_models = pytest.mark.skipif(not (MODELS and IMAGES), reason="set VISAGE_MODELS_DIR and VISAGE_TEST_IMAGES")


def _load(name):
    import cv2

    return cv2.cvtColor(cv2.imread(os.path.join(IMAGES, name)), cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="module")
def setup_models():
    models.set_models_dir(MODELS)
    models.allow_download = False


@needs_models
def test_preview_annotates(setup_models):
    from lib_visage import preview

    img = _load("group.jpg")
    out, md = preview.annotate(img, highlight=[0])
    assert out.shape == img.shape and not (out == img).all()
    assert "face(s)" in md and "`0`" in md


@needs_models
def test_similarity_report(setup_models):
    ident, _ = pipeline.identity_from_images([_load("source.png")])
    r = pipeline.process(_load("group.jpg"), [ident], pipeline.Options(report_similarity=True))
    assert len(r.similarity) == 1 and r.similarity[0] > 0.6


@needs_models
def test_detect_group(setup_models):
    faces = analysis.detect(_load("group.jpg"))
    assert len(faces) >= 2


@needs_models
@pytest.mark.parametrize("key,min_sim", [
    ("hyperswap_1a_256", 0.6), ("inswapper_128", 0.7), ("reswapper_256", 0.5), ("alphaface_256", 0.7),
    ("ghost_1_256", 0.7), ("simswap_256", 0.55), ("simswap_unofficial_512", 0.5), ("hififace_unofficial_256", 0.5),
])
def test_swap_moves_identity(setup_models, key, min_sim):
    if not models.is_downloaded(key):
        pytest.skip(f"{key} not downloaded")
    group, src = _load("group.jpg"), _load("source.png")
    ident, notes = pipeline.identity_from_images([src])
    assert ident is not None, notes
    result = pipeline.process(group, [ident], pipeline.Options(swapper=key))
    assert result.swapped == [0]
    face = analysis.sort_faces(analysis.detect(result.image), "large-small")[0]
    sim = analysis.cosine_similarity(analysis.embed(result.image, face), ident.normed)
    assert sim >= min_sim, f"{key}: identity similarity {sim:.3f}"


def test_sfw_skip_and_blur(monkeypatch):
    from lib_visage import safety

    frame = (np.random.default_rng(3).random((64, 64, 3)) * 255).astype(np.uint8)
    ident = facemodel.Identity(np.ones(512, np.float32))
    monkeypatch.setattr(safety, "is_nsfw", lambda f, progress=None: True)
    r = pipeline.process(frame, [ident], pipeline.Options(sfw="skip"))
    assert r.flagged and r.swapped == [] and (r.image == frame).all()

    monkeypatch.setattr(pipeline, "select_targets", lambda f, o, progress=None: ([], []))
    r = pipeline.process(frame, [ident], pipeline.Options(sfw="blur"))
    assert r.flagged


def test_sfw_off_never_loads_models(monkeypatch):
    from lib_visage import safety

    def boom(*a, **k):
        raise AssertionError("SFW models must not run when the filter is off")

    monkeypatch.setattr(safety, "is_nsfw", boom)
    monkeypatch.setattr(pipeline, "select_targets", lambda f, o, progress=None: ([], []))
    frame = np.zeros((32, 32, 3), np.uint8)
    r = pipeline.process(frame, [facemodel.Identity(np.ones(512, np.float32))], pipeline.Options())
    assert not r.flagged


def test_every_model_has_a_pinned_hash():
    from lib_visage.hashes import MODEL_HASHES

    missing = [k for k in models.REGISTRY if k not in MODEL_HASHES]
    assert not missing, missing
    assert all(len(sha) == 64 and size > 0 for sha, size in MODEL_HASHES.values())


def test_download_rejects_bad_hash(tmp_path):
    import hashlib

    src = tmp_path / "src.bin"
    src.write_bytes(b"visage" * 1000)
    url = src.resolve().as_uri()
    good = (hashlib.sha256(src.read_bytes()).hexdigest(), src.stat().st_size)

    dst = tmp_path / "ok.onnx"
    models._download(url, str(dst), expected=good)
    assert dst.read_bytes() == src.read_bytes()

    bad = tmp_path / "bad.onnx"
    with pytest.raises(IOError):
        models._download(url, str(bad), expected=("0" * 64, good[1]))
    assert not bad.exists() and not (tmp_path / "bad.onnx.part").exists()
