"""Foveated ROI：重要性选择 + 分级合成 + 全分辨率 ROI 裁片。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image, ImageFilter

from .imaging import resolve_interp


@dataclass
class ROI:
    x: int
    y: int
    w: int
    h: int
    score: float
    method: str = "gradient_energy_v1"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class FoveatedResult:
    mosaic: Image.Image
    roi_crop: Image.Image
    rois: list[ROI]
    meta: dict = field(default_factory=dict)


def _integral(a: np.ndarray) -> np.ndarray:
    return np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def gradient_energy_map(gray: np.ndarray, blur_radius: float = 1.0) -> np.ndarray:
    """灰度 → 高斯模糊 → **中心差分**梯度幅值 → 归一化到 [0,1]。

    算子为 gx = I[:,2:] - I[:,:-2], gy = I[2:,:] - I[:-2,:]（见 ADR-0008），
    不是标准 3×3 Sobel；不要在此处改用 Sobel（会改变已冻结的实验输入）。
    """
    img = Image.fromarray(gray.astype(np.uint8), mode="L")
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)
    gx[:, 1:-1] = arr[:, 2:] - arr[:, :-2]
    gy[1:-1, :] = arr[2:, :] - arr[:-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    mx = float(mag.max())
    if mx <= 1e-8:
        return np.zeros_like(mag)
    return mag / mx


def select_rois(
    img: Image.Image,
    roi_frac: float = 0.25,
    max_rois: int = 2,
    iou_threshold: float = 0.3,
    method: str = "gradient_energy_v1",
) -> list[ROI]:
    """无监督重要性：窗口梯度能量最大处。"""
    W, H = img.size
    side = max(32, int(round(roi_frac * min(W, H))))
    side = min(side, W, H)
    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    energy = gradient_energy_map(gray)
    if float(energy.sum()) <= 1e-8:
        # 兜底：纯色/低对比 → 图像中心
        x, y = max(0, (W - side) // 2), max(0, (H - side) // 2)
        return [ROI(x, y, side, side, 0.0, method + "_center_fallback")]

    ii = _integral(energy)
    step = max(4, side // 8)
    ys = np.arange(0, max(1, H - side + 1), step)
    xs = np.arange(0, max(1, W - side + 1), step)
    if ys[-1] + side < H:
        ys = np.append(ys, H - side)
    if xs[-1] + side < W:
        xs = np.append(xs, W - side)

    yg, xg = np.meshgrid(ys, xs, indexing="ij")
    sums = (
        ii[yg + side, xg + side]
        - ii[yg, xg + side]
        - ii[yg + side, xg]
        + ii[yg, xg]
    )
    flat_idx = np.dstack(np.unravel_index(np.argsort(sums.ravel())[::-1], sums.shape))[0]

    chosen: list[ROI] = []
    for yi, xi in flat_idx:
        if len(chosen) >= max_rois:
            break
        y0, x0 = int(ys[yi]), int(xs[xi])
        cand = ROI(x0, y0, side, side, float(sums[yi, xi]), method)
        if all(_iou(cand, c) < iou_threshold for c in chosen):
            chosen.append(cand)
    if not chosen:
        chosen = [ROI(0, 0, side, side, 0.0, method)]
    return chosen


def _iou(a: ROI, b: ROI) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    iw = max(0, min(ax2, bx2) - max(a.x, b.x))
    ih = max(0, min(ay2, by2) - max(a.y, b.y))
    inter = iw * ih
    if inter == 0:
        return 0.0
    return inter / float(a.w * a.h + b.w * b.h - inter)


def _rect_mask(h: int, w: int, box: tuple[int, int, int, int]) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.float32)
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = 1.0
    return m


def _ring_mask(h: int, w: int, cx: float, cy: float, r_in: float, r_out: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return ((d >= r_in) & (d <= r_out)).astype(np.float32)


def build_foveated(img: Image.Image, roi: ROI, cfg) -> FoveatedResult:
    """生成 foveated 马赛克（画布 = 原图 * canvas_scale）与全分辨率 ROI 裁片。"""
    W, H = img.size
    cs = float(cfg.get("foveation.canvas_scale", 0.5))
    cw, ch = max(8, int(round(W * cs))), max(8, int(round(H * cs)))
    periph_s = float(cfg.get("foveation.periphery_scale", 0.25))
    feather = float(cfg.get("foveation.feather_px", 8))
    ring0, ring1 = cfg.get("foveation.ring_ratio", [1.0, 1.75])
    interp = resolve_interp(cfg.get("pyramid.method", "bicubic"))

    # 1) 周边：先降到 periphery_scale，再放大到画布
    periph = img.resize(
        (max(1, int(round(W * periph_s))), max(1, int(round(H * periph_s)))), interp
    ).resize((cw, ch), interp)
    # 2) 细节层：画布原生尺度（= 原图 * canvas_scale）
    detail = img.resize((cw, ch), interp)

    # 3) 掩膜：中央凹矩形 + 旁中央环
    fx0, fy0 = int(round(roi.x * cs)), int(round(roi.y * cs))
    fx1, fy1 = int(round((roi.x + roi.w) * cs)), int(round((roi.y + roi.h) * cs))
    fovea = _rect_mask(ch, cw, (fx0, fy0, fx1, fy1))
    cx, cy = (fx0 + fx1) / 2.0, (fy0 + fy1) / 2.0
    r_eq = 0.5 * math.sqrt(max(1.0, roi.w * cs) * max(1.0, roi.h * cs))
    ring = _ring_mask(ch, cw, cx, cy, r_eq * float(ring0), r_eq * float(ring1))
    mask = np.clip(np.maximum(fovea, ring * 0.85), 0.0, 1.0)

    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    if feather > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=feather))
    m = np.asarray(mask_img, dtype=np.float32)[:, :, None] / 255.0

    blend = np.asarray(periph, dtype=np.float32) * (1.0 - m) + np.asarray(detail, dtype=np.float32) * m
    mosaic = Image.fromarray(np.clip(blend, 0, 255).astype(np.uint8), mode="RGB")

    roi_crop = img.crop((roi.x, roi.y, roi.x + roi.w, roi.y + roi.h))
    meta = {
        "canvas": [cw, ch],
        "canvas_scale": cs,
        "periphery_scale": periph_s,
        "feather_px": feather,
        "ring_ratio": [float(ring0), float(ring1)],
        "roi": roi.as_dict(),
        "original": [W, H],
        "coverage_ratio": (roi.w * roi.h) / float(W * H),
    }
    return FoveatedResult(mosaic=mosaic, roi_crop=roi_crop, rois=[roi], meta=meta)


def manual_roi(img: Image.Image, box: tuple[int, int, int, int]) -> ROI:
    W, H = img.size
    x, y, w, h = box
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return ROI(x, y, w, h, 1.0, "manual")
