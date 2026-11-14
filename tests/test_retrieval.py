import json

import numpy as np
import pytest

from alacv.retrieval import (
    RetrievalIndex,
    average_precision,
    evaluate_queries,
    recall_at_k,
    similarity_separation,
)
from alacv.store import VisualStore


def _add(store, image_id, vec, source="original", variant="A"):
    v = np.asarray(vec, dtype=np.float32)
    cls_dim = v.shape[0]
    store.upsert_model(
        {
            "model_id": "m1", "display_name": "t", "arch": "t", "params": 1, "source": "t",
            "revision": "r", "weights_sha256": "w", "license": "Apache-2.0", "license_file": None,
            "input_size": 518, "patch_size": 14, "cls_dim": cls_dim, "norm_mean": "[]", "norm_std": "[]",
            "created_at": "z",
        }
    )
    w = store.write_features(image_id, source, v, np.zeros((1, cls_dim), dtype=np.float32), {"grid": [1, 1]})
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
            "cls_dim": cls_dim,
            "rel_path": w["rel_path"],
            "bytes": w["bytes"],
            "sha256": w["sha256"],
            "config_sha256": "cfg",
            "created_at": "2026-11-14T00:00:00Z",
        }
    )


def _mk(store, iid, vec):
    store.add_image(
        {
            "image_id": iid,
            "sha256": (iid * 64)[:64],
            "rel_path": f"{iid[:2]}/{iid}.jpg",
            "source_name": iid,
            "ext": "jpg",
            "mime": "image/jpeg",
            "width": 10,
            "height": 10,
            "channels": 3,
            "pil_mode": "RGB",
            "color_space": "sRGB",
            "icc_sha256": None,
            "file_size": 1,
            "exif_json": "{}",
            "dataset_id": "d",
            "status": "ok",
            "reject_reason": None,
            "ingested_at": "z",
        }
    )
    _add(store, iid, vec)


def test_index_returns_nearest(store):
    _mk(store, "a" * 16, [1, 0, 0, 0])
    _mk(store, "b" * 16, [0, 1, 0, 0])
    _mk(store, "c" * 16, [0, 0, 1, 0])
    idx = RetrievalIndex(store, ["a" * 16, "b" * 16, "c" * 16], ["original"])
    res = idx.search(np.array([0.9, 0.1, 0, 0], dtype=np.float32), k=2)
    assert res[0][0] == "a" * 16
    assert res[0][1] > res[1][1]
    assert idx.dim == 4


def test_recall_and_ap_helpers():
    assert recall_at_k(["x", "y"], {"y"}, 1) == 0.0
    assert recall_at_k(["x", "y"], {"y"}, 2) == 1.0
    assert pytest.approx(1.0) == average_precision(["y", "z"], {"y"}, 10)


def test_evaluate_queries(store):
    _mk(store, "a" * 16, [1, 0, 0, 0])
    _mk(store, "b" * 16, [0, 1, 0, 0])
    idx = RetrievalIndex(store, ["a" * 16, "b" * 16], ["original"])
    out = evaluate_queries(idx, [{"image_id": "a" * 16, "gt": {"a" * 16}}])
    assert out["recall@1"] == 1.0
    assert out["n_queries"] == 1


def test_similarity_separation(store):
    _mk(store, "a" * 16, [1, 0, 0, 0])
    _mk(store, "b" * 16, [0.9, 0.1, 0, 0])
    _mk(store, "c" * 16, [0, 0, 1, 0])
    idx = RetrievalIndex(store, ["a" * 16, "b" * 16, "c" * 16], ["original"])
    sep = similarity_separation(idx, [("a" * 16, "b" * 16)], [("a" * 16, "c" * 16)])
    assert sep["auc"] == 1.0
    assert sep["gap"] > 0.5
