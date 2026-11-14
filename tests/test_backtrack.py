"""WP2：中心裁剪坐标一致性与可见区域测试。"""

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from alacv.backtrack import (
    crop_region,
    reconstruct_from_visible,
    reconstruct_full_resolution,
    recover_gt_region,
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


def test_per_axis_scales_differ_and_cause_1px_shift():
    """P1：实际缩放是逐轴的，名义 scale 在长边方向会差 1 个输入像素。"""
    g = fit_geometry(1024, 768, 518)
    assert g.x_scale == pytest.approx(691 / 1024)
    assert g.y_scale == pytest.approx(518 / 768)
    assert g.x_scale != pytest.approx(g.y_scale)
    # x=816 时：逐轴映射 551，名义 scale 映射 550
    assert round(816 * g.x_scale) == 551
    assert round(816 * g.scale) == 550


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


def test_reconstruction_matches_per_axis_mapping():
    """重建必须使用逐轴比例，而不是名义 scale。"""
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 255, (768, 1024, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    box = (816, 200, 60, 200)  # 靠近右边缘，落在可见窗口内
    region = visible_region(box, img.size, 518)
    assert region.visible
    g = region.geometry
    window = img.resize(g.resized, Image.Resampling.BICUBIC).crop(
        (g.left, g.top, g.left + 518, g.top + 518)
    )
    x, y, w, h = region.box
    cx0 = int(round(x * g.x_scale)) - g.left
    cy0 = int(round(y * g.y_scale)) - g.top
    cx1 = int(round((x + w) * g.x_scale)) - g.left
    cy1 = int(round((y + h) * g.y_scale)) - g.top
    expected = np.asarray(window.crop((cx0, cy0, cx1, cy1)).resize((w, h), Image.Resampling.BICUBIC))
    assert np.array_equal(reconstruct_from_visible(img, region), expected)


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


class _StubCfg:
    def __init__(self, root):
        self.root = Path(root)

    def path(self, dotted):
        mapping = {
            "paths.processed_dir": self.root / "processed",
            "paths.raw_dir": self.root / "raw",
        }
        return mapping[dotted]

    def get(self, dotted, default=None):
        return {"model.input_size": 518}.get(dotted, default)


class _StubStore:
    def __init__(self, assets_by_kind, rois):
        self._assets = assets_by_kind
        self._rois = rois

    def get_image(self, image_id):
        return {"rel_path": "img.png"}

    def assets(self, image_id, kind=None):
        rows = self._assets.get(kind, [])
        return rows

    def rois_of(self, image_id):
        return self._rois


def test_c_gt_region_composes_roi_crop_over_mosaic(tmp_path):
    """C 的 GT 区域回溯：ROI 覆盖处用全分辨率裁片，其余用马赛克。"""
    import json

    root = tmp_path
    (root / "raw").mkdir()
    (root / "processed" / "img" / "foveated").mkdir(parents=True)
    (root / "processed" / "img" / "roi").mkdir(parents=True)

    base = np.full((300, 400, 3), 200, dtype=np.uint8)
    Image.fromarray(base).save(root / "raw" / "img.png")

    # 马赛克全黑；ROI 裁片全白
    Image.fromarray(np.zeros((150, 200, 3), dtype=np.uint8)).save(
        root / "processed" / "img" / "foveated" / "foveated.webp", format="WEBP", lossless=True
    )
    Image.fromarray(np.full((100, 100, 3), 255, dtype=np.uint8)).save(
        root / "processed" / "img" / "roi" / "roi_00.webp", format="WEBP", lossless=True
    )
    assets = {
        "foveated": [{"label": "foveated", "rel_path": "processed/img/foveated/foveated.webp"}],
        "roi_crop": [{"label": "roi_00", "rel_path": "processed/img/roi/roi_00.webp"}],
    }
    rois = [{"roi_index": 0, "x": 100, "y": 80, "w": 100, "h": 100}]
    cfg = _StubCfg(root)
    store = _StubStore(assets, rois)

    # GT 框跨越 ROI 边界：左半在 ROI 内（白），右半在 ROI 外（黑）
    gt = (150, 100, 100, 60)  # y 完全落在 ROI 的 80..180 内，仅 x 方向跨边界
    rec, region = recover_gt_region(cfg, store, "img", "C", gt)
    assert region.visible
    x, y, w, h = region.box
    split = (100 + 100) - x  # ROI 右边界 200 减去 GT 左边界
    assert 0 < split < rec.shape[1]
    inside = rec[:, :split]
    outside = rec[:, split:]
    assert inside.mean() > 240, inside.mean()
    assert outside.mean() < 15, outside.mean()
