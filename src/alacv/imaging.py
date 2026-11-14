"""解码、色彩归一、多尺度金字塔。"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

INTERPOLATIONS = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}


def resolve_interp(name: str) -> Image.Resampling:
    if name not in INTERPOLATIONS:
        raise ValueError(f"不支持的插值方法: {name}；可选 {sorted(INTERPOLATIONS)}")
    return INTERPOLATIONS[name]


@dataclass
class DecodedImage:
    image: Image.Image          # 已摆正、已转 RGB
    format: str                 # PIL format，如 JPEG
    original_mode: str          # 原始 PIL mode
    width: int
    height: int
    icc_profile: bytes | None
    exif_present: bool


def to_rgb(img: Image.Image) -> Image.Image:
    """统一到 RGB；含 alpha 的图像合成到白底（避免黑边）。"""
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    return img.convert("RGB")


def open_image(path: str | Path, *, exif_transpose: bool = True) -> DecodedImage:
    path = Path(path)
    with Image.open(path) as im:
        fmt = im.format or path.suffix.lstrip(".").upper()
        original_mode = im.mode
        icc = im.info.get("icc_profile")
        exif_present = bool(im.getexif())
        im.load()
        work = im
        if exif_transpose:
            work = ImageOps.exif_transpose(work)
        work = work.copy()
    rgb = to_rgb(work)
    return DecodedImage(
        image=rgb,
        format=fmt,
        original_mode=original_mode,
        width=rgb.width,
        height=rgb.height,
        icc_profile=icc,
        exif_present=exif_present,
    )


def fit_to_input(img: Image.Image, size: int, method: str = "bicubic") -> Image.Image:
    """DINOv2 标准评估式预处理：短边缩放到 size，再中心裁剪 size×size。"""
    interp = resolve_interp(method)
    w, h = img.size
    scale = size / min(w, h)
    nw, nh = max(size, int(round(w * scale))), max(size, int(round(h * scale)))
    resized = img.resize((nw, nh), interp)
    left = (nw - size) // 2
    top = (nh - size) // 2
    return resized.crop((left, top, left + size, top + size))


def build_pyramid(img: Image.Image, scales: list[int], method: str = "bicubic") -> dict[int, Image.Image]:
    """返回 {分母: 图像}，分母 1 表示原始分辨率副本。"""
    interp = resolve_interp(method)
    out: dict[int, Image.Image] = {}
    w, h = img.size
    for s in scales:
        tw, th = max(1, int(round(w / s))), max(1, int(round(h / s)))
        out[s] = img.copy() if s == 1 else img.resize((tw, th), interp)
    return out


def save_image(img: Image.Image, path: str | Path, fmt: str = "webp", quality: int = 95) -> int:
    """原子写入；返回字节数。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    params: dict = {}
    fmt_l = fmt.lower()
    if fmt_l == "webp":
        params = {"quality": quality, "method": 4}
    elif fmt_l in ("jpg", "jpeg"):
        fmt_l = "JPEG"
        params = {"quality": quality, "subsampling": 0}
    elif fmt_l == "png":
        fmt_l = "PNG"
        params = {"compress_level": 6}
    img.save(tmp, format=fmt_l.upper(), **params)
    os.replace(tmp, path)
    return path.stat().st_size


def pixel_sha256(img: Image.Image) -> str:
    return hashlib.sha256(np.asarray(img.convert("RGB")).tobytes()).hexdigest()


def pyramid_npz_digest(pyr: dict[int, Image.Image]) -> str:
    h = hashlib.sha256()
    for s in sorted(pyr):
        h.update(str(s).encode())
        h.update(np.asarray(pyr[s].convert("RGB")).tobytes())
    return h.hexdigest()
