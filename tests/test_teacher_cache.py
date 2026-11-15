"""教师特征缓存 v2 测试：完整校验、fail-closed、有界缓存。"""

import json

import numpy as np
import pytest

from alacv.teacher_cache import CLS_DIM, PATCH_TOKENS, TeacherCacheReader, TeacherCacheWriter


def _meta():
    return {"model_id": "facebook/dinov2-small", "revision": "r", "weights_sha256": "w"}


def _write(tmp_path, n=3, shard_size=2):
    w = TeacherCacheWriter(tmp_path, "train", _meta(), "m", shard_size=shard_size)
    for i in range(n):
        w.add(f"v{i}", np.full(CLS_DIM, i, np.float16), np.zeros((PATCH_TOKENS, CLS_DIM), np.float16))
    return w.close()


def test_roundtrip_and_index_metadata(tmp_path):
    idx = _write(tmp_path)
    assert idx["n_views"] == 3 and idx["n_shards"] == 2
    assert idx["shards"][0]["tensor_spec"]["patch"] == [2, PATCH_TOKENS, CLS_DIM]
    r = TeacherCacheReader(tmp_path, "train")
    c, p = r.get("v1")
    assert c.shape == (CLS_DIM,) and p.shape == (PATCH_TOKENS, CLS_DIM)
    assert float(c[0]) == 1.0


def test_reader_fails_on_missing_index(tmp_path):
    with pytest.raises(FileNotFoundError):
        TeacherCacheReader(tmp_path, "test")


def test_reader_detects_tampered_shard(tmp_path):
    idx = _write(tmp_path)
    shard = tmp_path / idx["shards"][0]["path"]
    shard.write_bytes(shard.read_bytes() + b"x")
    with pytest.raises(RuntimeError, match="shard"):
        TeacherCacheReader(tmp_path, "train")


def test_reader_rejects_duplicate_view(tmp_path):
    idx = _write(tmp_path)
    p = tmp_path / "index_train.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    data["shards"][1]["views"][0] = data["shards"][0]["views"][0]
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RuntimeError, match="重复"):
        TeacherCacheReader(tmp_path, "train")


def test_bounded_cache_keeps_at_most_two_shards(tmp_path):
    idx = _write(tmp_path, n=6, shard_size=1)
    r = TeacherCacheReader(tmp_path, "train", verify_sha=False, max_cached_shards=2)
    for i in range(6):
        r.get(f"v{i}")
    assert len(r._cache) <= 2
