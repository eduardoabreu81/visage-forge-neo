"""Core tests. Model-backed tests run only when models and sample images are available:

    VISAGE_MODELS_DIR=<forge>/models/visage
    VISAGE_TEST_IMAGES=<folder with group.jpg (several faces) and source.png (one face)>
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib_visage import analysis, facemodel, geometry, masks, models, pipeline, swapper  # noqa: E402

MODELS = os.environ.get("VISAGE_MODELS_DIR")
IMAGES = os.environ.get("VISAGE_TEST_IMAGES")


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
def test_detect_group(setup_models):
    faces = analysis.detect(_load("group.jpg"))
    assert len(faces) >= 2


@needs_models
@pytest.mark.parametrize("key,min_sim", [("hyperswap_1a_256", 0.6), ("inswapper_128", 0.7), ("reswapper_256", 0.5)])
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
