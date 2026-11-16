"""A1/F0 不变量：在真实缓存上逐位一致（Gate 3 证据）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from alacv.ssl.arms import A1, F0
from alacv.ssl.checkpoint import descriptor_sha256, kendall_tau, recall_at_k

ROOT = Path("data/ds0002")
CACHE = ROOT / "features" / "teacher"


@pytest.mark.skipif(not (CACHE / "index_validation.json").exists(), reason="teacher cache not present")
def test_a1_equals_f0_bitwise_on_real_cache():
    from alacv.teacher_cache import TeacherCacheReader

    reader = TeacherCacheReader(CACHE, "validation", verify_sha=False)
    ids = list(reader.view_to_pos)[:8]
    cls = torch.stack([torch.from_numpy(reader.get(v)[0].astype("float32")) for v in ids])
    patch = torch.stack([torch.from_numpy(reader.get(v)[1].astype("float32")) for v in ids])

    a1, f0 = A1().eval(), F0().eval()
    with torch.no_grad():
        d1 = a1.eval_cls(cls, patch)
        d0 = f0.eval_cls(cls, patch)
    assert torch.equal(d1, d0)
    assert descriptor_sha256(d1) == descriptor_sha256(d0)

    target = torch.arange(len(ids))
    assert recall_at_k(d1, d0, target, 1) == recall_at_k(d0, d0, target, 1) == 1.0
    assert kendall_tau(d1[:, 0], d0[:, 0]) == 1.0
