import numpy as np
from PIL import Image

from alacv.foveation import build_foveated, gradient_energy_map, manual_roi, select_rois


def _checker_scene(w=256, h=256, px=80, py=80, ps=96):
    arr = np.full((h, w, 3), 120, dtype=np.uint8)
    yy, xx = np.mgrid[0:ps, 0:ps]
    checker = (((xx // 8) + (yy // 8)) % 2 * 255).astype(np.uint8)
    arr[py : py + ps, px : px + ps, 0] = checker
    arr[py : py + ps, px : px + ps, 1] = 255 - checker
    arr[py : py + ps, px : px + ps, 2] = 10
    return Image.fromarray(arr, mode="RGB")


def test_gradient_energy_peaks_on_texture():
    img = _checker_scene().convert("L")
    e = gradient_energy_map(np.asarray(img))
    assert e.shape == (256, 256)
    assert 0.0 <= e.min() and e.max() <= 1.0 + 1e-6
    # 纹理区能量显著高于平坦区
    assert e[80:176, 80:176].mean() > e[0:40, 0:40].mean() * 5


def test_select_roi_hits_texture():
    img = _checker_scene()
    roi = select_rois(img, roi_frac=96 / 256, max_rois=1)[0]
    # 与真值框有较大重叠
    gt = (80, 80, 96, 96)
    ix = max(0, min(roi.x + roi.w, gt[0] + gt[2]) - max(roi.x, gt[0]))
    iy = max(0, min(roi.y + roi.h, gt[1] + gt[3]) - max(roi.y, gt[1]))
    inter = ix * iy
    union = roi.w * roi.h + gt[2] * gt[3] - inter
    assert inter / union > 0.3


def test_select_roi_fallback_on_flat_image():
    img = Image.new("RGB", (200, 100), (128, 128, 128))
    roi = select_rois(img, roi_frac=0.25, max_rois=1)[0]
    assert "center_fallback" in roi.method
    assert roi.w > 0 and roi.h > 0


def test_manual_roi_clamped():
    img = Image.new("RGB", (100, 80), (0, 0, 0))
    roi = manual_roi(img, (90, 70, 100, 100))
    assert roi.x + roi.w <= 100
    assert roi.y + roi.h <= 80


def test_foveated_artifacts(cfg):
    img = _checker_scene(512, 512, 160, 160, 192)
    roi = select_rois(img, roi_frac=0.375, max_rois=1)[0]
    fov = build_foveated(img, roi, cfg)
    assert fov.mosaic.size == (256, 256), fov.mosaic.size
    assert fov.roi_crop.size == (roi.w, roi.h)
    assert fov.meta["canvas_scale"] == 0.5
    assert 0 < fov.meta["coverage_ratio"] <= 1
    # 马赛克在 ROI 内应保留比周边更多的高频能量
    arr = np.asarray(fov.mosaic.convert("L"), dtype=np.float32)
    roi_region = arr[roi.y // 2 : (roi.y + roi.h) // 2, roi.x // 2 : (roi.x + roi.w) // 2]
    corner = arr[0:40, 0:40]
    assert roi_region.std() > corner.std()


def test_roi_crop_is_full_resolution(cfg):
    img = _checker_scene(400, 300, 100, 80, 100)
    roi = select_rois(img, roi_frac=0.3, max_rois=1)[0]
    fov = build_foveated(img, roi, cfg)
    assert fov.roi_crop.size == (roi.w, roi.h)
    assert np.asarray(fov.roi_crop).shape[1] == roi.w
