"""DS-0002 确定性程序化场景生成器（Gate 1）。

设计要点
--------
- 纯 numpy/Pillow，无外部素材；同一 (split, index, view) 在任意机器上产生相同像素。
- 场景种子按 split 分段，**分组隔离**：train ∈ [0, 2000)，validation ∈ [100000, 100300)，
  test ∈ [200000, 200900)，不存在跨 split 的 base scene。
- 生成器版本、PRNG 算法、配置与逐文件 SHA-256 均写入 manifest，可审计、可重放。
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageEnhance

GENERATOR_VERSION = "ds0002-v1"
PRNG = "numpy.random.Generator(PCG64)"
WIDTH, HEIGHT = 1024, 768
SPLIT_SEED_BASE = {"train": 0, "validation": 100_000, "test": 200_000}
SPLIT_COUNTS = {"train": 2000, "validation": 300, "test": 900}
VIEW_COUNTS = {"train": 8, "validation": 5, "test": 5}

HUES = [
    (220, 40, 40), (40, 200, 70), (240, 150, 30),
    (150, 60, 200), (230, 210, 40), (240, 90, 170),
]


def base_seed(split: str, index: int) -> int:
    return SPLIT_SEED_BASE[split] + index


def view_rng(split: str, index: int, view: int) -> np.random.Generator:
    seed = base_seed(split, index) * 1000 + view
    return np.random.default_rng(seed)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_image(img: Image.Image, quality: int = 90) -> tuple[str, int]:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=2)
    data = buf.getvalue()
    return sha256_bytes(data), len(data)


_LOW_W, _LOW_H = 256, 192


def _background(rng: np.random.Generator) -> np.ndarray:
    # 低频背景在 256×192 上生成再双三次放大：视觉等价，速度快一个数量级
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


def _palette(rng: np.random.Generator, family: int) -> tuple[np.ndarray, np.ndarray]:
    c1 = np.array(HUES[family % 6], dtype=np.float32)
    c2 = np.array(HUES[(family + 3) % 6], dtype=np.float32)
    j = rng.uniform(-25, 25, 3).astype(np.float32)
    return np.clip(c1 + j, 0, 255), np.clip(c2 + j, 0, 255)


def _texture(rng: np.random.Generator, kind: int, size: int, family: int) -> np.ndarray:
    c1, c2 = _palette(rng, family)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    if kind == 0:
        cell = int(rng.integers(8, 26))
        pat = ((xx // cell + yy // cell) % 2).astype(np.float32)
    elif kind == 1:
        freq = rng.uniform(2.0, 6.0)
        ang = rng.uniform(0, np.pi)
        proj = xx * np.cos(ang) + yy * np.sin(ang)
        pat = (0.5 + 0.5 * np.sin(2 * np.pi * freq * proj / size)).astype(np.float32)
    elif kind == 2:
        d = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2)
        pat = (0.5 + 0.5 * np.sin(d / rng.uniform(4.0, 9.0))).astype(np.float32)
    else:
        blocks = rng.integers(0, 2, (8, 8)).astype(np.float32)
        pat = np.kron(blocks, np.ones((size // 8, size // 8), dtype=np.float32))[:size, :size]
    if pat.shape[0] != size or pat.shape[1] != size:
        pat_img = Image.fromarray((np.clip(pat, 0, 1) * 255).astype(np.uint8), mode="L")
        pat = np.asarray(pat_img.resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    tex = c1[None, None, :] * pat[..., None] + c2[None, None, :] * (1 - pat[..., None])
    tex += rng.normal(0, 4.0, tex.shape)
    return np.clip(tex, 0, 255)


@dataclass
class Scene:
    image: Image.Image
    scene_id: str
    base_seed: int
    factors: dict = field(default_factory=dict)
    targets: list = field(default_factory=list)


def render_scene(split: str, index: int) -> Scene:
    """确定性渲染一个 base scene（1024×768）。"""
    seed = base_seed(split, index)
    rng = np.random.default_rng(seed)
    canvas = _background(rng)

    # 2–4 个干扰块（部分可能遮挡目标）
    n_distract = int(rng.integers(2, 5))
    for _ in range(n_distract):
        bw, bh = int(rng.integers(90, 280)), int(rng.integers(90, 280))
        bx, by = int(rng.integers(0, WIDTH - bw)), int(rng.integers(0, HEIGHT - bh))
        c = np.clip(np.array([rng.uniform(60, 200) for _ in range(3)]), 0, 255)
        patch = np.broadcast_to(c, (bh, bw, 3)).astype(np.float32)
        region = canvas[by:by + bh, bx:bx + bw]
        canvas[by:by + bh, bx:bx + bw] = region * 0.65 + patch * 0.35

    n_targets = int(rng.integers(1, 4))
    targets = []
    for k in range(n_targets):
        size = int(rng.integers(int(0.12 * min(WIDTH, HEIGHT)), int(0.30 * min(WIDTH, HEIGHT))))
        tex = _texture(rng, int(rng.integers(0, 4)), size, int(rng.integers(0, 6)))
        x, y = int(rng.integers(10, WIDTH - size - 10)), int(rng.integers(10, HEIGHT - size - 10))
        alpha = 1.0 if k == 0 else float(rng.uniform(0.85, 1.0))
        region = canvas[y:y + size, x:x + size]
        canvas[y:y + size, x:x + size] = region * (1 - alpha) + tex * alpha
        targets.append({"x": x, "y": y, "w": size, "h": size, "texture_kind": int(rng.integers(0, 4))})

    lighting = float(rng.uniform(0.85, 1.15))
    canvas = np.clip(canvas * lighting, 0, 255)
    img = Image.fromarray(canvas.astype(np.uint8), mode="RGB")

    primary = targets[0]
    cx, cy = primary["x"] + primary["w"] / 2, primary["y"] + primary["h"] / 2
    octant = int((cy > HEIGHT / 2) * 4 + (cx > WIDTH / 2) * 2 + ((cx % 512) > 256))
    scale_bucket = "small" if primary["w"] < 130 else ("medium" if primary["w"] < 190 else "large")
    factors = {
        "palette_family": int(seed % 6),
        "texture_kind": primary["texture_kind"],
        "position_octant": octant,
        "scale_bucket": scale_bucket,
        "target_count": n_targets,
        "occlusion": bool(rng.random() < 0.35),
        "lighting": round(lighting, 4),
    }
    return Scene(image=img, scene_id=f"{split}_{index:05d}", base_seed=seed, factors=factors, targets=targets)


def augment(img: Image.Image, rng: np.random.Generator, strong: bool) -> tuple[Image.Image, dict]:
    """确定性的视图增强；返回图像与增强参数（训练可见）。"""
    params: dict = {}
    if strong:
        scale = float(rng.uniform(0.80, 1.0))
        angle = float(rng.uniform(-10, 10))
        brightness = float(rng.uniform(0.80, 1.20))
        contrast = float(rng.uniform(0.80, 1.20))
        saturation = float(rng.uniform(0.80, 1.20))
        quality = int(rng.integers(60, 85))
    else:
        scale = float(rng.uniform(0.95, 1.0))
        angle = float(rng.uniform(-3, 3))
        brightness = float(rng.uniform(0.95, 1.05))
        contrast = float(rng.uniform(0.95, 1.05))
        saturation = float(rng.uniform(0.95, 1.05))
        quality = 92

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
