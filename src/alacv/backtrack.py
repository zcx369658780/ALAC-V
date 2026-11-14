"""派生表示可见像素保真度（WP2）。

语义边界：本模块衡量的是"某一变体保存的派生表示，在原图 ROI 区域内还能还原多少
可见像素细节"，**不是**从 DINOv2 embedding 解码重建图像，也不代表模型具备像素生成能力。

坐标契约：与 `alacv.imaging.fit_to_input()` 完全一致 —— 短边缩放到 518 后，
按 `(nw-518)//2, (nh-518)//2` 做中心裁剪。任何 ROI 只有落在该裁剪窗口内才可见；
A/B/C 三种变体的保真度比较必须使用同一"可见交集"，否则不是同一评价区域。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .imaging import FitGeometry, fit_geometry


@dataclass(frozen=True)
class VisibleRegion:
    """原图像素坐标下的可见交集，以及它相对原始 ROI 的覆盖率。"""

    box: tuple[int, int, int, int]  # (x, y, w, h)，原图坐标
    coverage: float                 # 可见面积 / ROI 面积，0..1
    geometry: FitGeometry

    @property
    def visible(self) -> bool:
        return self.box[2] > 0 and self.box[3] > 0 and self.coverage > 0


def visible_region(
    box: tuple[int, int, int, int],
    size: tuple[int, int],
    input_size: int,
) -> VisibleRegion:
    """计算 ROI 与编码器可见区域（518 中心裁剪窗口）的交集。

    `box` 为原图像素坐标 (x, y, w, h)；`size` 为原图 (W, H)。
    完全不可见时返回面积为 0 的 box 与 coverage=0，调用方必须排除，不得伪造像素。
    """
    W, H = size
    g = fit_geometry(W, H, input_size)
    x, y, w, h = box
    x0, y0, x1, y1 = x, y, x + w, y + h

    # 可见窗口在原图坐标下的范围（浮点，再向内取整，避免越界采样）
    vx0 = g.left / g.scale
    vy0 = g.top / g.scale
    vx1 = (g.left + g.input_size) / g.scale
    vy1 = (g.top + g.input_size) / g.scale

    ix0 = max(x0, int(math.ceil(vx0)))
    iy0 = max(y0, int(math.ceil(vy0)))
    ix1 = min(x1, int(math.floor(vx1)))
    iy1 = min(y1, int(math.floor(vy1)))
    ix0, iy0 = max(0, min(ix0, W)), max(0, min(iy0, H))
    ix1, iy1 = max(0, min(ix1, W)), max(0, min(iy1, H))

    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    area = max(1, w * h)
    return VisibleRegion(
        box=(ix0, iy0, iw, ih),
        coverage=(iw * ih) / area,
        geometry=g,
    )


def crop_region(img: Image.Image, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    return np.asarray(img.crop((x, y, x + w, y + h)))


def reconstruct_from_visible(
    img: Image.Image,
    region: VisibleRegion,
    method: str = "bicubic",
) -> np.ndarray:
    """模拟 A 变体：只保留 518 中心裁剪内可见的像素，再放大回该区域的原始像素尺寸。

    这是"派生表示可见像素保真度"的上界：A 不保存任何像素资产，能提供的最好细节
    就是 518 归一化输入里的那一份。
    """
    from .imaging import resolve_interp

    if not region.visible:
        raise ValueError("不可见区域不能重建")
    g = region.geometry
    interp = resolve_interp(method)
    resized = img.resize(g.resized, interp)
    window = resized.crop((g.left, g.top, g.left + g.input_size, g.top + g.input_size))
    x, y, w, h = region.box
    cx0 = int(round(x * g.scale)) - g.left
    cy0 = int(round(y * g.scale)) - g.top
    cx1 = int(round((x + w) * g.scale)) - g.left
    cy1 = int(round((y + h) * g.scale)) - g.top
    cx0, cy0 = max(0, cx0), max(0, cy0)
    cx1, cy1 = min(g.input_size, cx1), min(g.input_size, cy1)
    if cx1 <= cx0 or cy1 <= cy0:
        raise ValueError("可见交集在 518 窗口内为空")
    patch = window.crop((cx0, cy0, cx1, cy1))
    return np.asarray(patch.resize((w, h), interp))


def reconstruct_full_resolution(
    img: Image.Image,
    region: VisibleRegion,
    method: str = "bicubic",
) -> np.ndarray:
    """B/C 的派生像素资产（L0 或全分辨率 ROI 裁片）在可见交集上的还原。"""
    from .imaging import resolve_interp

    if not region.visible:
        raise ValueError("不可见区域不能重建")
    interp = resolve_interp(method)
    x, y, w, h = region.box
    return np.asarray(img.crop((x, y, x + w, y + h)).resize((w, h), interp))


def recover_variant(cfg, store, image_id: str, variant: str, box: tuple[int, int, int, int]):
    """按变体实际保存的表示还原 ROI，返回 (重建像素 | None, VisibleRegion)。

    这是实验脚本与 CLI 共用的唯一实现，避免坐标逻辑分叉。
    """
    import json

    from .imaging import open_image

    row = store.get_image(image_id)
    if row is None:
        raise KeyError(image_id)
    img = open_image(cfg.path("paths.raw_dir") / row["rel_path"]).image
    size = int(cfg.get("model.input_size", 518))
    region = visible_region(box, img.size, size)
    if not region.visible:
        return None, region

    if variant == "A":
        return reconstruct_from_visible(img, region), region

    if variant == "B":
        assets = {a["level"]: a for a in store.assets(image_id, "pyramid")}
        if 0 in assets:
            im = open_image(cfg.path("paths.processed_dir").parent / assets[0]["rel_path"]).image
            return reconstruct_full_resolution(im, region), region

    if variant == "C":
        for a in store.assets(image_id, "roi_crop"):
            if a["label"] == "roi_00":
                im = open_image(cfg.path("paths.processed_dir").parent / a["rel_path"]).image
                if im.size == (box[2], box[3]):
                    x, y, w, h = region.box
                    return np.asarray(im.crop((x - box[0], y - box[1], x - box[0] + w, y - box[1] + h))), region
        fov = [a for a in store.assets(image_id, "foveated")]
        if fov:
            meta_p = (cfg.path("paths.processed_dir").parent / fov[0]["rel_path"]).with_suffix(".json")
            cs = 0.5
            if meta_p.exists():
                cs = float(json.loads(meta_p.read_text(encoding="utf-8"))["canvas_scale"])
            im = open_image(cfg.path("paths.processed_dir").parent / fov[0]["rel_path"]).image
            x, y, w, h = region.box
            cb = (int(x * cs), int(y * cs), int((x + w) * cs), int((y + h) * cs))
            return np.asarray(im.crop(cb).resize((w, h), Image.Resampling.BICUBIC)), region

    return reconstruct_full_resolution(img, region), region
