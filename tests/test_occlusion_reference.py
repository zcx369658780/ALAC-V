"""独立的遮挡参考实现（WP0-4）。

不导入生产 renderer 的 mask/occlusion 代码；直接从 evaluation sidecar 的几何
（targets: shape/x/y/w/h，distractors: 矩形）重建 mask 并计算：
    O = 1 - |mask_primary & ~(later targets | distractor rects)| / |mask_primary|
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

W, H = 1024, 768
LABELS = Path("data/ds0002/eval")


def shape_mask(shape: str, w: int, h: int) -> np.ndarray:
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    if shape == "rect":
        d.rectangle([0, 0, w - 1, h - 1], fill=255)
    elif shape == "ellipse":
        d.ellipse([0, 0, w - 1, h - 1], fill=255)
    elif shape == "triangle":
        d.polygon([(w // 2, 0), (w - 1, h - 1), (0, h - 1)], fill=255)
    else:
        raise ValueError(shape)
    return np.asarray(img) > 127


def reference_occlusion(scene: dict) -> tuple[float, bool]:
    masks = []
    for t in scene["targets"]:
        m = np.zeros((H, W), dtype=bool)
        m[t["y"]:t["y"] + t["h"], t["x"]:t["x"] + t["w"]] = shape_mask(t["shape"], t["w"], t["h"])
        masks.append(m)
    occluded = np.zeros((H, W), dtype=bool)
    for m in masks[1:]:
        occluded |= m
    for d in scene["distractors"]:
        occluded[d["y"]:d["y"] + d["h"], d["x"]:d["x"] + d["w"]] = True
    total = int(masks[0].sum())
    frac = 0.0 if total == 0 else 1.0 - float((masks[0] & ~occluded).sum()) / total
    return frac, bool(frac >= 0.05)


def _pick_scenes(scenes: list[dict], max_n: int = 240) -> list[dict]:
    near = [s for s in scenes if abs(s["factors"]["occlusion_fraction"] - 0.05) <= 0.01]
    others = [s for s in scenes if s not in near]
    chosen = near + others[: max(0, max_n - len(near))]
    return chosen


@pytest.mark.skipif(not (LABELS / "labels_validation.json").exists(), reason="DS-0002 labels not present")
@pytest.mark.parametrize("split", ["validation", "test"])
def test_reference_matches_sidecar(split: str):
    labels = json.loads((LABELS / f"labels_{split}.json").read_text(encoding="utf-8"))
    scenes = _pick_scenes(labels["scenes"], 120)
    shapes = {s["factors"]["shape"] for s in labels["scenes"]}
    assert shapes == {"rect", "ellipse", "triangle"}
    for sc in scenes:
        frac, flag = reference_occlusion(sc)
        assert sc["factors"]["occlusion_fraction"] == pytest.approx(frac, abs=2e-3), sc["scene_id"]
        assert sc["factors"]["occlusion"] == flag, sc["scene_id"]


@pytest.mark.skipif(not (LABELS / "labels_train.json").exists(), reason="DS-0002 labels not present")
def test_threshold_boundary_scenes():
    labels = json.loads((LABELS / "labels_train.json").read_text(encoding="utf-8"))
    near = [s for s in labels["scenes"] if abs(s["factors"]["occlusion_fraction"] - 0.05) <= 0.01]
    assert near, "需要阈值附近的场景来验证 0.05 判定"
    below = [s for s in near if s["factors"]["occlusion_fraction"] < 0.05]
    above = [s for s in near if s["factors"]["occlusion_fraction"] >= 0.05]
    assert below and above, "阈值两侧都应存在样本"
    for sc in near:
        frac, flag = reference_occlusion(sc)
        assert sc["factors"]["occlusion"] == (frac >= 0.05)
