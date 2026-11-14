"""WP2：中心裁剪坐标一致性与可见区域测试。"""

import math

import numpy as np
import pytest
from PIL import Image

from alacv.backtrack import (
    crop_region,
    reconstruct_from_visible,
    reconstruct_full_resolution,
    visible_region,
)
from alacv.imaging import fit_geometry, fit_to_input


def _img(w, h, value=128):
    arr = np.full((h, w, 3), value, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.mark.parametrize("size", [(1024, 768), (768, 1024), (518, 518), (1000, 400), (400, 1000)])
def test_fit_to_input_matches_geometry(size):
    img = _img(*size)
    g = fit_geometry(size[0], size[1], 518)
    expected = img.resize(g.resized, Image.Resampling.BICUBIC).crop(
        (g.left, g.top, g.left + 518, g.top + 518)
    )
    assert np.array_equal(np.asarray(expected), np.asarray(fit_to_input(img, 518)))


def test_geometry_values_landscape():
    g = fit_geometry(1024, 768, 518)
    assert g.scale == pytest.approx(518 / 768)
    assert g.resized == (691, 518)
    assert (g.left, g.top) == (86, 0)


def test_geometry_values_portrait():
    g = fit_geometry(768, 1024, 518)
    assert g.resized == (518, 691)
    assert (g.left, g.top) == (0, 86)


def test_visible_region_center_is_full():
    img = _img(1024, 768)
    r = visible_region((400, 300, 200, 200), img.size, 518)
    assert r.visible
    assert r.box == (400, 300, 200, 200)
    assert r.coverage == pytest.approx(1.0)


def test_visible_region_partial_at_crop_edge():
    img = _img(1024, 768)
    # 可见 x 约 [127.5, 895.5]；ROI 从 850 开始 → 只剩 45 px 宽
    r = visible_region((850, 100, 120, 120), img.size, 518)
    assert r.visible
    assert r.box[0] == 850
    assert r.box[2] == 45
    assert r.coverage == pytest.approx(45 / 120)


def test_visible_region_fully_outside():
    img = _img(1024, 768)
    r = visible_region((900, 400, 100, 100), img.size, 518)
    assert not r.visible
    assert r.coverage == 0.0
    assert r.box[2] == 0


def test_visible_region_portrait_outside():
    img = _img(768, 1024)
    r = visible_region((100, 950, 100, 60), img.size, 518)
    assert not r.visible


def test_reconstruct_raises_on_invisible():
    img = _img(1024, 768)
    r = visible_region((950, 400, 50, 50), img.size, 518)
    with pytest.raises(ValueError):
        reconstruct_from_visible(img, r)


def test_constant_image_reconstruction_is_lossless():
    img = _img(1024, 768, value=77)
    box = (300, 200, 200, 200)
    region = visible_region(box, img.size, 518)
    rec = reconstruct_from_visible(img, region)
    gt = crop_region(img, region.box)
    assert rec.shape == gt.shape
    assert np.array_equal(rec, gt)


def test_full_resolution_reconstruction_is_identity():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (768, 1024, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    box = (400, 300, 200, 200)
    region = visible_region(box, img.size, 518)
    rec = reconstruct_full_resolution(img, region)
    assert np.array_equal(rec, crop_region(img, region.box))


def test_reconstruction_uses_518_scale_not_original():
    """A 的重建必须来自 518 归一化图，而不是原图直接裁剪。"""
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 255, (768, 1024, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    box = (400, 300, 200, 200)
    region = visible_region(box, img.size, 518)
    rec = reconstruct_from_visible(img, region)
    direct = crop_region(img, region.box)
    assert rec.shape == direct.shape
    # 经过 518 采样再放大，必然与原图像素不同
    assert not np.array_equal(rec, direct)


def test_coverage_bounds():
    img = _img(640, 480)
    for box in [(0, 0, 640, 480), (600, 440, 40, 40), (320, 240, 20, 20)]:
        r = visible_region(box, img.size, 518)
        assert 0.0 <= r.coverage <= 1.0 + 1e-9
        if r.visible:
            assert r.box[2] > 0 and r.box[3] > 0
