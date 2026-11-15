"""教师特征缓存读写与 fail-closed 测试。"""

import numpy as np
import pytest

from alacv.teacher_cache import CLS_DIM, PATCH_TOKENS, TeacherCacheReader, TeacherCacheWriter


def _meta():
    return {"model_id": "facebook/dinov2-small", "revision": "r", "weights_sha256": "w"}


def test_writer_reader_roundtrip_and_alignment(tmp_path):
    w = TeacherCacheWriter(tmp_path, "train", _meta(), "manifest-sha", shard_size=2)
    cls = [np.arange(CLS_DIM, dtype=np.float32) + i for i in range(3)]
    patch = [np.zeros((PATCH_TOKENS, CLS_DIM), dtype=np.float16) for _ in range(3)]
    for i in range(3):
        w.add(f"v{i}", cls[i], patch[i])
    idx = w.close()
    assert idx["n_views"] == 3 and idx["n_shards"] == 2

    r = TeacherCacheReader(tmp_path, "train")
    c, p = r.get("v1")
    assert c.shape == (CLS_DIM,) and p.shape == (PATCH_TOKENS, CLS_DIM)
    assert np.allclose(c.astype(np.float32), cls[1], atol=1)
    with pytest.raises(KeyError):
        r.get("missing")


def test_reader_detects_missing_index(tmp_path):
    with pytest.raises(FileNotFoundError):
        TeacherCacheReader(tmp_path, "test")


def test_bad_shape_rejected(tmp_path):
    w = TeacherCacheWriter(tmp_path, "train", _meta(), "m", shard_size=8)
    with pytest.raises(AssertionError):
        w.add("v0", np.zeros(CLS_DIM - 1, dtype=np.float16), np.zeros((PATCH_TOKENS, CLS_DIM), np.float16))
