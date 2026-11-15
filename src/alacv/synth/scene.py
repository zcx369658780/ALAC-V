"""DS-0002 v2 确定性程序化场景生成器（Gate 1 整改）。

v2 修复（对应 PR #8 审查 P1-1）：
- 纹理类型 / 调色板族 / 形状：记录**真正参与绘图**的参数，不再二次随机抽样。
- 遮挡：先画目标、再画干扰块；`occlusion_fraction` 由目标与干扰块的真实重叠面积计算。
- 形状：rect / ellipse / triangle 三种，宽高比随机，标签记录真实形状。
- 生成器版本升级为 `ds0002-v2`；旧 v1 数据不进入 EXP-0002。

确定性：`base_seed` 按 split 分段互斥；`view_seed = base_seed*1000 + view`；PCG64。
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

GENERATOR_VERSION = "ds0002-v3"
PRNG = "numpy.random.Generator(PCG64)"
WIDTH, HEIGHT = 1024, 768
SPLIT_SEED_BASE = {"train": 0, "validation": 100_000, "test": 200_000}
SPLIT_COUNTS = {"train": 2000, "validation": 300, "test": 900}
VIEW_COUNTS = {"train": 8, "validation": 5, "test": 5}
SHAPES = ("rect", "ellipse", "triangle")

HUES = [
    (220, 40, 40), (40, 200, 70), (240, 150, 30),
    (150, 60, 200), (230, 210, 40), (240, 90, 170),
]

_LOW_W, _LOW_H = 256, 192


def base_seed(split: str, index: int) -> int:
    return SPLIT_SEED_BASE[split] + index


def view_rng(split: str, index: int, view: int) -> np.random.Generator:
    return np.random.default_rng(base_seed(split, index) * 1000 + view)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_image(img: Image.Image, quality: int = 90) -> tuple[str, int]:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=2)
    return sha256_bytes(buf.getvalue()), len(buf.getvalue())


def _background(rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:_LOW_H, 0:_LOW_W].astype(np.float32)
    c1 = rng.uniform(60, 180, 3).astype(np.float32)
    c2 = rng.uniform(70, 200, 3).astype(np.float32)
    a = rng.uniform(0.2, 0.8)
    t = (xx / _LOW_W) * a + (yy / _LOW_H) * (1 - a)
    img = c1[None, None, :] * (1 - t[..., None]) + c2[None, None, :] * t[..., None]
    for _ in range(3):
        fx, fy = rng.uniform(0.3, 1.8), rng.uniform(0.3, 1.8)
        ph = rng.uniform(0, 2 * np.pi)
        wave = 0.5 + 0.5 * np.sin(2 * np.pi * (fx * xx / _LOW_W + fy * yy / _LOW_H) + ph)
        img += (wave[..., None] - 0.5) * rng.uniform(6, 20)
    img += rng.normal(0, 2.0, img.shape)
    low = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="RGB")
    return np.asarray(low.resize((WIDTH, HEIGHT), Image.Resampling.BICUBIC), dtype=np.float32)


def _palette_colors(rng: np.random.Generator, family: int) -> tuple[np.ndarray, np.ndarray]:
    c1 = np.array(HUES[family % 6], dtype=np.float32)
    c2 = np.array(HUES[(family + 3) % 6], dtype=np.float32)
    j = rng.uniform(-25, 25, 3).astype(np.float32)
    return np.clip(c1 + j, 0, 255), np.clip(c2 + j, 0, 255)


def _texture(rng: np.random.Generator, kind: int, w: int, h: int,
             c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if kind == 0:
        cell = int(rng.integers(8, 26))
        pat = ((xx // cell + yy // cell) % 2).astype(np.float32)
    elif kind == 1:
        freq = rng.uniform(2.0, 6.0)
        ang = rng.uniform(0, np.pi)
        proj = xx * np.cos(ang) + yy * np.sin(ang)
        pat = (0.5 + 0.5 * np.sin(2 * np.pi * freq * proj / max(w, h))).astype(np.float32)
    elif kind == 2:
        d = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
        pat = (0.5 + 0.5 * np.sin(d / rng.uniform(4.0, 9.0))).astype(np.float32)
    else:
        bw, bh = max(1, w // 8), max(1, h // 8)
        blocks = rng.integers(0, 2, (8, 8)).astype(np.float32)
        pat = np.kron(blocks, np.ones((bh, bw), dtype=np.float32))[:h, :w]
        if pat.shape != (h, w):
            pat = np.asarray(Image.fromarray((pat * 255).astype(np.uint8)).resize((w, h),
                              Image.Resampling.NEAREST), dtype=np.float32) / 255.0
    tex = c1[None, None, :] * pat[..., None] + c2[None, None, :] * (1 - pat[..., None])
    tex += rng.normal(0, 4.0, tex.shape)
    return np.clip(tex, 0, 255)


def _shape_mask(shape: str, w: int, h: int) -> np.ndarray:
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    if shape == "rect":
        d.rectangle([0, 0, w - 1, h - 1], fill=255)
    elif shape == "ellipse":
        d.ellipse([0, 0, w - 1, h - 1], fill=255)
    else:
        d.polygon([(w // 2, 0), (w - 1, h - 1), (0, h - 1)], fill=255)
    return np.asarray(m, dtype=np.float32)[..., None] / 255.0


@dataclass
class Scene:
    image: Image.Image
    scene_id: str
    base_seed: int
    factors: dict = field(default_factory=dict)
    targets: list = field(default_factory=list)
    distractors: list = field(default_factory=list)
    masks: list = field(default_factory=list)


def _overlap_fraction(box, other) -> float:
    x0 = max(box["x"], other["x"])
    y0 = max(box["y"], other["y"])
    x1 = min(box["x"] + box["w"], other["x"] + other["w"])
    y1 = min(box["y"] + box["h"], other["y"] + other["h"])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return ((x1 - x0) * (y1 - y0)) / float(box["w"] * box["h"])


def render_scene(split: str, index: int) -> Scene:
    """确定性渲染一个 base scene（1024×768, v3）。

    遮挡定义（P1-1）：
    - 目标按顺序绘制，后画目标会遮住先画目标；
    - 干扰块在目标之后绘制且**不透明**（alpha=1.0），是物理遮挡；
    - `occlusion_fraction = 1 - 可见目标 mask 像素 / 目标原始 mask 像素`，逐像素计算。
    """
    seed = base_seed(split, index)
    rng = np.random.default_rng(seed)
    canvas = _background(rng)

    n_targets = int(rng.integers(1, 4))
    targets, masks = [], []
    for _ in range(n_targets):
        shape = SHAPES[int(rng.integers(0, len(SHAPES)))]
        base = int(rng.integers(int(0.12 * min(WIDTH, HEIGHT)), int(0.30 * min(WIDTH, HEIGHT))))
        aspect = float(rng.uniform(0.65, 1.55))
        w, h = base, max(24, int(round(base * aspect)))
        kind = int(rng.integers(0, 4))
        family = int(rng.integers(0, 6))
        c1, c2 = _palette_colors(rng, family)
        tex = _texture(rng, kind, w, h, c1, c2)
        mask_local = _shape_mask(shape, w, h)[..., 0] > 0.5
        x = int(rng.integers(10, WIDTH - w - 10))
        y = int(rng.integers(10, HEIGHT - h - 10))
        m3 = mask_local[..., None].astype(np.float32)
        region = canvas[y:y + h, x:x + w]
        canvas[y:y + h, x:x + w] = region * (1 - m3) + tex * m3
        full = np.zeros((HEIGHT, WIDTH), dtype=bool)
        full[y:y + h, x:x + w] = mask_local
        targets.append({"x": x, "y": y, "w": w, "h": h, "shape": shape,
                        "texture_kind": kind, "palette_family": family})
        masks.append(full)

    n_distract = int(rng.integers(2, 5))
    distractors, rects = [], []
    for _ in range(n_distract):
        bw, bh = int(rng.integers(90, 280)), int(rng.integers(90, 280))
        bx, by = int(rng.integers(0, WIDTH - bw)), int(rng.integers(0, HEIGHT - bh))
        c = np.clip(np.array([rng.uniform(60, 200) for _ in range(3)]), 0, 255)
        canvas[by:by + bh, bx:bx + bw] = c[None, None, :]
        r = np.zeros((HEIGHT, WIDTH), dtype=bool)
        r[by:by + bh, bx:bx + bw] = True
        rects.append(r)
        distractors.append({"x": bx, "y": by, "w": bw, "h": bh, "alpha": 1.0})

    occluded = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for m in masks[1:]:
        occluded |= m
    for r in rects:
        occluded |= r
    total = int(masks[0].sum())
    occ_frac = 0.0 if total == 0 else 1.0 - float((masks[0] & ~occluded).sum()) / total

    lighting = float(rng.uniform(0.85, 1.15))
    canvas = np.clip(canvas * lighting, 0, 255)
    img = Image.fromarray(canvas.astype(np.uint8), mode="RGB")

    prim = targets[0]
    cx, cy = prim["x"] + prim["w"] / 2, prim["y"] + prim["h"] / 2
    octant = int((cy > HEIGHT / 2) * 4 + (cx > WIDTH / 2) * 2 + ((cx % 512) > 256))
    scale_bucket = "small" if prim["w"] < 130 else ("medium" if prim["w"] < 190 else "large")
    factors = {
        "palette_family": prim["palette_family"],
        "texture_kind": prim["texture_kind"],
        "shape": prim["shape"],
        "position_octant": octant,
        "scale_bucket": scale_bucket,
        "target_count": n_targets,
        "occlusion": bool(occ_frac >= 0.05),
        "occlusion_fraction": round(float(occ_frac), 4),
        "occlusion_definition": "1 - visible_target_mask_px / target_mask_px (opaque occluders)",
        "lighting": round(lighting, 4),
    }
    return Scene(image=img, scene_id=f"{split}_{index:05d}", base_seed=seed,
                 factors=factors, targets=targets, distractors=distractors, masks=masks)


def augment(img: Image.Image, rng: np.random.Generator, strong: bool,
            q_gallery: int = 92, q_strong: tuple[int, int] = (60, 84)) -> tuple[Image.Image, dict]:
    """确定性的视图增强；JPEG 质量由配置传入并记录。"""
    params: dict = {}
    if strong:
        scale = float(rng.uniform(0.80, 1.0))
        angle = float(rng.uniform(-10, 10))
        brightness = float(rng.uniform(0.80, 1.20))
        contrast = float(rng.uniform(0.80, 1.20))
        saturation = float(rng.uniform(0.80, 1.20))
        quality = int(rng.integers(int(q_strong[0]), int(q_strong[1]) + 1))
    else:
        scale = float(rng.uniform(0.95, 1.0))
        angle = float(rng.uniform(-3, 3))
        brightness = float(rng.uniform(0.95, 1.05))
        contrast = float(rng.uniform(0.95, 1.05))
        saturation = float(rng.uniform(0.95, 1.05))
        quality = int(q_gallery)

    if scale < 1.0:
        nw, nh = int(img.width * scale), int(img.height * scale)
        small = img.resize((nw, nh), Image.Resampling.BICUBIC)
        ox = int(rng.integers(0, img.width - nw + 1))
        oy = int(rng.integers(0, img.height - nh + 1))
        canvas = Image.new("RGB", img.size, (128, 128, 128))
        canvas.paste(small, (ox, oy))
        img = canvas
    if abs(angle) > 0.01:
        img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0))
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(saturation)
    params.update(
        scale=round(scale, 4), angle=round(angle, 4), brightness=round(brightness, 4),
        contrast=round(contrast, 4), saturation=round(saturation, 4), jpeg_quality=int(quality),
    )
    return img, params
