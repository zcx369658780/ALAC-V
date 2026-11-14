import json

import numpy as np
import pytest

from alacv.store import VisualStore, safe_source

MODEL = {
    "model_id": "m1",
    "display_name": "test",
    "arch": "test",
    "params": 1,
    "source": "test",
    "revision": "r",
    "weights_sha256": "w",
    "license": "Apache-2.0",
    "license_file": None,
    "input_size": 518,
    "patch_size": 14,
    "cls_dim": 4,
    "norm_mean": "[]",
    "norm_std": "[]",
    "created_at": "2026-11-14T00:00:00Z",
}


def _img_record(image_id="a" * 16, sha="a" * 64):
    return {
        "image_id": image_id,
        "sha256": sha,
        "rel_path": f"{sha[:2]}/{sha}.jpg",
        "source_name": "x.jpg",
        "ext": "jpg",
        "mime": "image/jpeg",
        "width": 100,
        "height": 80,
        "channels": 3,
        "pil_mode": "RGB",
        "color_space": "sRGB",
        "icc_sha256": None,
        "file_size": 1234,
        "exif_json": json.dumps({"present": False}),
        "dataset_id": "ds-test",
        "status": "ok",
        "reject_reason": None,
        "ingested_at": "2026-11-14T00:00:00Z",
    }


@pytest.fixture()
def model(store):
    store.upsert_model(dict(MODEL))
    return "m1"


def _add(store, image_id, source, cls, variant="A"):
    patch = np.zeros((1, len(cls)), dtype=np.float32)
    w = store.write_features(image_id, source, np.asarray(cls, np.float32), patch, {"grid": [1, 1]})
    store.add_feature(
        {
            "image_id": image_id,
            "variant": variant,
            "source": source,
            "level": None,
            "model_id": "m1",
            "precision": "fp16",
            "n_patches": 1,
            "patch_grid": 1,
            "cls_dim": len(cls),
            "rel_path": w["rel_path"],
            "bytes": w["bytes"],
            "sha256": w["sha256"],
            "config_sha256": "cfg",
            "created_at": "2026-11-14T00:00:00Z",
        }
    )
    return w


def test_schema_version_and_tables(store):
    ver = store.conn.execute("PRAGMA user_version").fetchone()[0]
    assert ver >= 1
    names = {r[0] for r in store.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"images", "assets", "features", "rois", "models", "runs", "run_items"} <= names


def test_feature_roundtrip(store, model):
    store.add_image(_img_record())
    cls = np.arange(8, dtype=np.float32)
    patch = np.arange(4 * 8, dtype=np.float32).reshape(4, 8)
    written = store.write_features("a" * 16, "original", cls, patch, {"grid": [2, 2]})
    store.add_feature(
        {
            "image_id": "a" * 16,
            "variant": "A",
            "source": "original",
            "level": None,
            "model_id": "m1",
            "precision": "fp16",
            "n_patches": written["n_patches"],
            "patch_grid": written["patch_grid"],
            "cls_dim": written["cls_dim"],
            "rel_path": written["rel_path"],
            "bytes": written["bytes"],
            "sha256": written["sha256"],
            "config_sha256": "cfg",
            "created_at": "2026-11-14T00:00:00Z",
        }
    )
    loaded = store.load_feature("a" * 16, "original")
    assert loaded["cls"].shape == (8,)
    assert loaded["patch"].shape == (4, 8)
    assert loaded["grid"] == (2, 2)
    assert np.allclose(loaded["cls"], cls, atol=1e-2)
    assert loaded["meta"]["grid"] == [2, 2]


def test_feature_dedup(store, model):
    store.add_image(_img_record())
    _add(store, "a" * 16, "original", [0, 0, 0, 0])
    assert store.stats()["features"] == 1
    assert store.has_feature("a" * 16, "original", "cfg") is True


def test_descriptor_is_unit_norm(store, model):
    store.add_image(_img_record())
    _add(store, "a" * 16, "a", [1.0, 2.0, 3.0])
    _add(store, "a" * 16, "b", [2.0, 2.0, 3.0])
    d = store.descriptor("a" * 16, ["a", "b"])
    assert d.shape == (6,)
    assert pytest.approx(1.0, abs=1e-5) == float(np.linalg.norm(d))


def test_assets_unique(store):
    store.add_image(_img_record())
    rec = {
        "image_id": "a" * 16, "kind": "pyramid", "level": 0, "label": "L0",
        "rel_path": "processed/x/L0.webp", "width": 100, "height": 80, "format": "webp",
        "bytes": 10, "sha256": "h", "config_sha256": "cfg", "created_at": "z",
    }
    store.add_asset(rec)
    rec["bytes"] = 20
    store.add_asset(rec)
    rows = store.assets("a" * 16, "pyramid")
    assert len(rows) == 1 and rows[0]["bytes"] == 20


def test_safe_source():
    assert safe_source("pyramid:L0") == "pyramid-L0"
    assert safe_source("roi:roi_00") == "roi-roi_00"


def test_missing_feature_raises(store):
    store.add_image(_img_record())
    with pytest.raises(FileNotFoundError):
        store.load_feature("a" * 16, "nope")
