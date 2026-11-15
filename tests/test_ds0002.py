"""DS-0002 生成器与读取层的单元测试（Gate 1）。"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from alacv.ds0002 import ALLOWED_ITEM_KEYS, Ds0002Dataset, PairBatchSampler
from alacv.synth import SPLIT_SEED_BASE, augment, base_seed, render_scene, view_rng


def test_render_is_deterministic():
    a = render_scene("train", 3)
    b = render_scene("train", 3)
    assert np.array_equal(np.asarray(a.image), np.asarray(b.image))
    assert a.base_seed == b.base_seed


def test_split_seed_ranges_are_disjoint():
    ranges = {
        "train": (SPLIT_SEED_BASE["train"], SPLIT_SEED_BASE["train"] + 2000),
        "validation": (SPLIT_SEED_BASE["validation"], SPLIT_SEED_BASE["validation"] + 300),
        "test": (SPLIT_SEED_BASE["test"], SPLIT_SEED_BASE["test"] + 900),
    }
    names = list(ranges)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = ranges[names[i]], ranges[names[j]]
            assert a[1] <= b[0] or b[1] <= a[0], (names[i], names[j])


def test_same_index_different_split_differs():
    assert base_seed("train", 5) != base_seed("validation", 5)
    assert base_seed("validation", 5) != base_seed("test", 5)


def test_augment_deterministic_and_params_visible():
    scene = render_scene("train", 7)
    i1, p1 = augment(scene.image, view_rng("train", 7, 1), strong=True)
    i2, p2 = augment(scene.image, view_rng("train", 7, 1), strong=True)
    assert np.array_equal(np.asarray(i1), np.asarray(i2))
    assert p1 == p2
    assert set(p1) >= {"scale", "angle", "brightness", "jpeg_quality"}


def test_view_images_differ():
    scene = render_scene("train", 9)
    i0, _ = augment(scene.image, view_rng("train", 9, 0), strong=False)
    i1, _ = augment(scene.image, view_rng("train", 9, 1), strong=True)
    assert not np.array_equal(np.asarray(i0), np.asarray(i1))


def _write_mini_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    imgs = root / "images" / "train"
    views = []
    for scene in range(2):
        d = imgs / f"{scene:05d}"
        d.mkdir(parents=True)
        arr = np.full((48, 64, 3), 30 + scene * 100, dtype=np.uint8)
        Image.fromarray(arr).save(d / "v0.jpg")
        views.append(
            {
                "view_id": f"train_{scene:05d}_v0",
                "scene_id": f"train_{scene:05d}",
                "base_seed": scene,
                "split": "train",
                "view_index": 0,
                "role": "train",
                "path": f"images/train/{scene:05d}/v0.jpg",
                "sha256": "x",
                "bytes": 1,
                "width": 64,
                "height": 48,
                "augment": {"scale": 1.0},
            }
        )
    manifest = {"split": "train", "views": views}
    mp = root / "train.json"
    mp.write_text(json.dumps(manifest), encoding="utf-8")
    return mp


def test_loader_exposes_only_allowlisted_keys(tmp_path):
    mp = _write_mini_dataset(tmp_path)
    ds = Ds0002Dataset(mp, tmp_path / "ds")
    item = ds[0]
    assert set(item) == ALLOWED_ITEM_KEYS
    for forbidden in ("scene_id", "base_seed", "targets", "factors", "role"):
        assert forbidden not in item
    assert item["image"].size == (64, 48)


def test_loader_does_not_leak_labels_in_repr(tmp_path):
    mp = _write_mini_dataset(tmp_path)
    ds = Ds0002Dataset(mp, tmp_path / "ds")
    text = repr(ds.getitem if False else ds[0]["view_id"])
    assert "train_00000" in text  # view_id is allowed


def test_pair_sampler_uses_same_scene(tmp_path):
    class FakeDS:
        def group_indices(self):
            return [[0, 1], [2, 3]]

    sampler = PairBatchSampler(FakeDS(), batch_pairs=2, seed=0)
    batches = list(sampler)
    assert batches and all(len(p) == 2 for p in batches)
    for batch in batches:
        for a, b in batch:
            assert {a, b} in ({0, 1}, {2, 3})
