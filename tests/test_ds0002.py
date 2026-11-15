"""DS-0002 v2 生成器与读取层测试（含标签隔离负向测试）。"""

import builtins
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from alacv.ds0002 import ALLOWED_ITEM_KEYS, Ds0002Dataset
from alacv.synth import SPLIT_SEED_BASE, augment, base_seed, render_scene, view_rng


def test_render_deterministic_and_labels_match_pixels():
    a = render_scene("train", 3)
    b = render_scene("train", 3)
    assert np.array_equal(np.asarray(a.image), np.asarray(b.image))
    t = a.targets[0]
    # 标签必须来自真实绘制参数
    assert t["shape"] in ("rect", "ellipse", "triangle")
    assert 0 <= t["texture_kind"] < 4
    assert 0 <= t["palette_family"] < 6
    assert a.factors["texture_kind"] == t["texture_kind"]
    assert a.factors["palette_family"] == t["palette_family"]
    assert a.factors["shape"] == t["shape"]


def test_shapes_vary_across_scenes():
    shapes = {render_scene("train", i).targets[0]["shape"] for i in range(12)}
    assert len(shapes) >= 2


def test_occlusion_reflects_geometry():
    for i in range(6):
        sc = render_scene("train", i)
        t = sc.targets[0]
        expected = 0.0
        for d in sc.distractors:
            x0, y0 = max(t["x"], d["x"]), max(t["y"], d["y"])
            x1 = min(t["x"] + t["w"], d["x"] + d["w"])
            y1 = min(t["y"] + t["h"], d["y"] + d["h"])
            if x1 > x0 and y1 > y0:
                expected = max(expected, (x1 - x0) * (y1 - y0) / (t["w"] * t["h"]))
        assert sc.factors["occlusion_fraction"] == pytest.approx(expected, abs=1e-3)


def test_split_seed_ranges_disjoint():
    r = {"train": (SPLIT_SEED_BASE["train"], SPLIT_SEED_BASE["train"] + 2000),
         "validation": (SPLIT_SEED_BASE["validation"], SPLIT_SEED_BASE["validation"] + 300),
         "test": (SPLIT_SEED_BASE["test"], SPLIT_SEED_BASE["test"] + 900)}
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    for a, b in pairs:
        assert r[a][1] <= r[b][0], (a, b)


def test_augment_keeps_quality_label():
    sc = render_scene("train", 5)
    img, p = augment(sc.image, view_rng("train", 5, 1), strong=True)
    assert 60 <= p["jpeg_quality"] <= 84
    img0, p0 = augment(sc.image, view_rng("train", 5, 0), strong=False)
    assert p0["jpeg_quality"] == 92


def _mini(tmp_path: Path, bad_field: bool = False) -> Path:
    root = tmp_path / "ds"
    (root / "images" / "train" / "00000").mkdir(parents=True)
    Image.fromarray(np.full((32, 40, 3), 100, dtype=np.uint8)).save(root / "images/train/00000/v0.jpg")
    view = {
        "view_id": "train_00000_v0",
        "path": "images/train/00000/v0.jpg",
        "augment": {"scale": 1.0, "jpeg_quality": 92},
        "pair_key": "gabc123",
    }
    if bad_field:
        view["scene_id"] = "train_00000"
    payload = {"split": "train", "n_views": 1, "views": [view]}
    p = root / "index_train.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_loader_only_exposes_allowlisted_item_keys(tmp_path):
    ds = Ds0002Dataset(_mini(tmp_path), tmp_path / "ds")
    item = ds[0]
    assert set(item) == ALLOWED_ITEM_KEYS
    for forbidden in ("scene_id", "base_seed", "targets", "factors", "pair_key", "role"):
        assert forbidden not in item


def test_loader_holds_no_label_manifest(tmp_path):
    ds = Ds0002Dataset(_mini(tmp_path), tmp_path / "ds")
    attrs = {k for k in vars(ds)}
    assert "manifest" not in attrs
    assert not hasattr(ds, "manifest")
    # 索引中不允许出现标签字段
    for v in ds._views:
        assert set(v) <= {"view_id", "path", "augment", "pair_key"}


def test_loader_rejects_index_with_label_fields(tmp_path):
    with pytest.raises(ValueError):
        Ds0002Dataset(_mini(tmp_path, bad_field=True), tmp_path / "ds")


def test_loader_never_opens_eval_labels(tmp_path, monkeypatch):
    ds = Ds0002Dataset(_mini(tmp_path), tmp_path / "ds")
    real_open = builtins.open

    def guarded(path, *a, **k):
        if "labels_" in str(path) or "labels" in str(path).lower():
            raise AssertionError(f"训练路径尝试读取评估标签: {path}")
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", guarded)
    _ = ds[0]
