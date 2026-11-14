"""生成 DS-0001 合成数据集（程序化、确定性、无外部下载）+ 可选真实图补充集。"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from alacv.config import load_config  # noqa: E402


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def smooth_background(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    c1 = rng.uniform(70, 180, 3).astype(np.float32)
    c2 = rng.uniform(90, 210, 3).astype(np.float32)
    a = rng.uniform(0.25, 0.75)
    t = (xx / w) * a + (yy / h) * (1 - a)
    img = c1[None, None, :] * (1 - t[..., None]) + c2[None, None, :] * t[..., None]
    for _ in range(3):
        fx, fy = rng.uniform(0.4, 1.6), rng.uniform(0.4, 1.6)
        ph = rng.uniform(0, 2 * np.pi)
        wave = 0.5 + 0.5 * np.sin(2 * np.pi * (fx * xx / w + fy * yy / h) + ph)
        img += (wave[..., None] - 0.5) * rng.uniform(8, 22)
    img += rng.normal(0, 2.0, img.shape)
    return np.clip(img, 0, 255)


def vivid_palette(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    base_hue = rng.integers(0, 6)
    hues = [
        (220, 40, 40), (40, 200, 70), (240, 150, 30),
        (150, 60, 200), (230, 210, 40), (240, 90, 170),
    ]
    c1 = np.array(hues[base_hue], dtype=np.float32)
    c2 = np.array(hues[(base_hue + 3) % 6], dtype=np.float32)
    jitter = rng.uniform(-25, 25, 3).astype(np.float32)
    return np.clip(c1 + jitter, 0, 255), np.clip(c2 + jitter, 0, 255)


def target_texture(rng: np.random.Generator, size: int) -> np.ndarray:
    kind = int(rng.integers(0, 4))
    c1, c2 = vivid_palette(rng)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    if kind == 0:  # 棋盘
        cell = int(rng.integers(8, 24))
        pat = ((xx // cell + yy // cell) % 2).astype(np.float32)
    elif kind == 1:  # 条纹（可旋转）
        freq = rng.uniform(2.0, 6.0)
        ang = rng.uniform(0, np.pi)
        proj = xx * np.cos(ang) + yy * np.sin(ang)
        pat = (0.5 + 0.5 * np.sin(2 * np.pi * freq * proj / size)).astype(np.float32)
    elif kind == 2:  # 同心环
        d = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2)
        pat = (0.5 + 0.5 * np.sin(d / rng.uniform(4.0, 9.0))).astype(np.float32)
    else:  # 随机块
        blocks = rng.integers(0, 2, (8, 8)).astype(np.float32)
        pat = np.kron(blocks, np.ones((size // 8, size // 8), dtype=np.float32))[:size, :size]
    tex = c1[None, None, :] * pat[..., None] + c2[None, None, :] * (1 - pat[..., None])
    tex += rng.normal(0, 4.0, tex.shape)
    return np.clip(tex, 0, 255)


def paste(canvas: np.ndarray, patch: np.ndarray, x: int, y: int, alpha: float = 1.0) -> None:
    h, w = patch.shape[:2]
    region = canvas[y : y + h, x : x + w]
    canvas[y : y + h, x : x + w] = region * (1 - alpha) + patch * alpha


def build_scene(rng: np.random.Generator, w: int, h: int, target_size: int, n_targets: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    canvas = smooth_background(rng, w, h)
    # 低对比干扰色块（防止"只看背景颜色"就够用）
    for _ in range(int(rng.integers(2, 5))):
        bw, bh = int(rng.integers(90, 260)), int(rng.integers(90, 260))
        bx, by = int(rng.integers(0, w - bw)), int(rng.integers(0, h - bh))
        c = np.clip(np.array([rng.uniform(60, 200) for _ in range(3)]), 0, 255)
        patch = np.broadcast_to(c, (bh, bw, 3)).astype(np.float32)
        paste(canvas, patch, bx, by, alpha=0.35)

    margin = 24
    xs = rng.integers(margin, w - target_size - margin, size=n_targets)
    ys = rng.integers(margin, h - target_size - margin, size=n_targets)
    main_bbox = None
    for i in range(n_targets):
        tex = target_texture(rng, target_size)
        x, y = int(xs[i]), int(ys[i])
        paste(canvas, tex, x, y, alpha=1.0 if i == 0 else 0.9)
        if i == 0:
            main_bbox = (x, y, target_size, target_size)
    assert main_bbox is not None
    return np.clip(canvas, 0, 255), main_bbox


def make_query(rng: np.random.Generator, base: Image.Image) -> Image.Image:
    img = base.copy()
    op = int(rng.integers(0, 3))
    if op in (0, 2):
        img = img.rotate(float(rng.uniform(-8, 8)), resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0))
    if op in (1, 2):
        img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.75, 1.25)))
        img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.8, 1.2)))
        img = ImageEnhance.Color(img).enhance(float(rng.uniform(0.8, 1.25)))
    if op == 0:
        s = 0.85
        nw, nh = int(img.width * s), int(img.height * s)
        small = img.resize((nw, nh), Image.Resampling.BICUBIC)
        ox = int(rng.integers(0, max(1, img.width - nw)))
        oy = int(rng.integers(0, max(1, img.height - nh)))
        canvas = Image.new("RGB", img.size, (128, 128, 128))
        canvas.paste(small, (ox, oy))
        img = canvas
    return img


def build_synthetic(cfg, out_root: Path) -> dict:
    dcfg = cfg.get("dataset", {})
    seed = int(dcfg.get("seed", 20261114))
    n_base = int(dcfg.get("n_base", 24))
    W, H = int(dcfg.get("width", 1024)), int(dcfg.get("height", 768))
    tsize = int(dcfg.get("target_size", 256))
    n_q = int(dcfg.get("queries_per_base", 3))

    rng = np.random.default_rng(seed)
    (out_root / "base").mkdir(parents=True, exist_ok=True)
    (out_root / "query").mkdir(parents=True, exist_ok=True)

    files = []
    for i in range(n_base):
        scene, bbox = build_scene(rng, W, H, tsize, int(rng.integers(1, 4)))
        name = f"scene_{i:03d}_base.jpg"
        p = out_root / "base" / name
        Image.fromarray(scene.astype(np.uint8)).save(p, quality=92, subsampling=0)
        base_img = Image.open(p).convert("RGB")
        files.append(
            {
                "path": str(p.relative_to(ROOT)),
                "sha256": sha256_file(p),
                "role": "base",
                "base_id": f"scene_{i:03d}",
                "gt_bbox": list(bbox),
            }
        )
        for q in range(n_q):
            qi = make_query(rng, base_img)
            qname = f"scene_{i:03d}_q{q}.jpg"
            qp = out_root / "query" / qname
            qi.save(qp, quality=int(rng.integers(55, 75)), subsampling=2)
            files.append(
                {
                    "path": str(qp.relative_to(ROOT)),
                    "sha256": sha256_file(qp),
                    "role": "query",
                    "base_id": f"scene_{i:03d}",
                    "gt_bbox": None,
                }
            )
    return {
        "dataset_id": dcfg.get("id", "ds-0001-alacv-synthetic-v1"),
        "seed": seed,
        "width": W,
        "height": H,
        "target_size": tsize,
        "n_base": n_base,
        "files": files,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator": "scripts/make_dataset.py",
        "license": "CC0-1.0（程序化生成，无第三方素材）",
    }


def add_real(cfg, out_root: Path) -> dict:
    rcfg = cfg.get("real", {})
    if not rcfg.get("enabled", False):
        return {"files": []}
    exts = []
    for g in rcfg.get("globs", []):
        exts.extend([Path(g).parent / p for p in Path(g).parent.glob(Path(g).name)])
    paths = sorted({p for p in exts if p.exists()})[: int(rcfg.get("max_images", 8))]
    dest_dir = out_root / "real"
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for p in paths:
        try:
            im = Image.open(p)
            im.load()
        except Exception:
            continue
        dst = dest_dir / f"real_{len(files):02d}{p.suffix.lower()}"
        shutil.copyfile(p, dst)
        files.append(
            {
                "path": str(dst.relative_to(ROOT)),
                "sha256": sha256_file(dst),
                "role": "real",
                "base_id": None,
                "gt_bbox": None,
                "origin": str(p),
            }
        )
    return {"files": files, "note": "本机 Windows 壁纸，本地只读复制，未上传/未下载"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset_ds0001.yaml")
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    out_root = cfg.path("dataset.out_dir")

    ds = build_synthetic(cfg, out_root)
    real = add_real(cfg, out_root)
    ds["files"].extend(real.get("files", []))
    ds["real_note"] = real.get("note")

    manifest_dir = ROOT / "data" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out = manifest_dir / f"{ds['dataset_id']}.json"
    out.write_text(json.dumps(ds, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(out), "n_files": len(ds["files"]), "n_base": ds["n_base"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
