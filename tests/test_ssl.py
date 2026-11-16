"""Gate 3 六臂实现测试（合成张量；不加载真实图像）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from alacv.ssl import arms
from alacv.ssl.checkpoint import (
    CheckpointRecord,
    EarlyStopping,
    VicRegState,
    descriptor_sha256,
    kendall_tau,
    recall_at_k,
    select_checkpoint,
)
from alacv.ssl.masking import (
    MASK_PATCHES,
    N_PATCHES,
    contiguous_block_mask,
    mask_seed_for,
    mask_sha256,
    patch_pixels,
    random_mask,
)
from alacv.ssl.modules import Adapter, B2Predictor, Predictor

D = 384
B = 2


def _batch(arm: str, mask: np.ndarray | None = None) -> dict:
    torch.manual_seed(0)
    teacher_patch = torch.randn(B, N_PATCHES, D)
    student_patch = torch.randn(B, N_PATCHES, D)
    m = torch.zeros(B, N_PATCHES, dtype=torch.bool) if mask is None else torch.from_numpy(
        np.stack([mask, mask])).bool()
    return {
        "teacher_cls": torch.randn(B, D),
        "teacher_patch": teacher_patch,
        "student_cls": torch.randn(B, D),
        "student_patch": student_patch,
        "teacher_cls2": torch.randn(B, D),
        "teacher_patch2": torch.randn(B, N_PATCHES, D),
        "mask": m,
    }


def test_mask_counts_and_determinism():
    m1 = contiguous_block_mask(2026 * 1000 + 1)
    m2 = contiguous_block_mask(2026 * 1000 + 1)
    assert m1.sum() == MASK_PATCHES == 821
    assert np.array_equal(m1, m2)
    assert mask_sha256(m1) == mask_sha256(m2)
    r = random_mask(2026 * 1000 + 1)
    assert r.sum() == 821 and not np.array_equal(r, m1)


def test_block_mask_has_four_components():
    m = contiguous_block_mask(7).reshape(37, 37)
    seen = np.zeros_like(m, dtype=bool)
    comps = []
    for i in range(37):
        for j in range(37):
            if m[i, j] and not seen[i, j]:
                stack = [(i, j)]
                seen[i, j] = True
                size = 0
                while stack:
                    r, c = stack.pop()
                    size += 1
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < 37 and 0 <= cc < 37 and m[rr, cc] and not seen[rr, cc]:
                            seen[rr, cc] = True
                            stack.append((rr, cc))
                comps.append(size)
    assert len(comps) == 4, comps
    assert all(s >= 196 for s in comps), comps


def test_patch_pixels_mask_before_student():
    m = random_mask(1)
    px = patch_pixels(m, image_size=518, patch=14)
    assert px.shape == (518, 518)
    assert px.sum() == 821 * 14 * 14
    x = torch.ones(3, 518, 518)
    x[:, torch.from_numpy(px)] = 0.0
    assert float(x[:, torch.from_numpy(px)].abs().sum()) == 0.0
    assert float(x[:, torch.from_numpy(~px)].sum()) == float((~torch.from_numpy(px)).sum()) * 3


def test_parameter_budgets():
    a2 = arms.A2()
    n_adapter = sum(p.numel() for p in a2.adapter.parameters())
    n_pred = sum(p.numel() for p in a2.predictor.parameters())
    assert 3.2e6 < n_adapter < 3.8e6, n_adapter
    assert 3.2e6 < n_pred < 3.8e6, n_pred
    assert a2.trainable_parameters() <= 10_000_000
    b2 = arms.B2()
    assert b2.trainable_parameters() <= 10_000_000
    assert isinstance(b2.predictor, B2Predictor)


def test_arm_forward_shapes():
    masks = {"A2": contiguous_block_mask(1), "C2": random_mask(1), "A1": contiguous_block_mask(1)}
    for name in ("A1", "A2", "C2", "B2"):
        arm = arms.build_arm(name)
        out = arm(_batch(name, masks.get(name)))
        assert torch.isfinite(out["loss"]), name
        if name == "B2":
            assert out["pred"].shape == (B, 768)
            assert out["target"].shape == (B, 768)
        else:
            assert out["pred"].shape[-1] == D


def test_gradient_flow_and_teacher_isolation():
    arm = arms.A2()
    batch = _batch("A2", contiguous_block_mask(3))
    loss = arm(batch)["loss"]
    loss.backward()
    assert arm.adapter.mask_token.grad is not None
    assert arm.predictor.blocks[0].attn.in_proj_weight.grad is not None
    for k in ("teacher_patch", "student_patch"):
        assert not batch[k].requires_grad
    assert arm.adapter.mask_token.grad.abs().sum() > 0


def test_a1_eval_equals_f0():
    teacher_cls = torch.randn(6, D)
    teacher_patch = torch.randn(6, N_PATCHES, D)
    a1, f0 = arms.A1(), arms.F0()
    d1 = a1.eval_cls(teacher_cls, teacher_patch)
    d0 = f0.eval_cls(teacher_cls, teacher_patch)
    assert torch.equal(d1, d0)
    assert descriptor_sha256(d1) == descriptor_sha256(d0)
    gal = d1.clone()
    q = d1.clone()
    t = torch.arange(6)
    assert recall_at_k(d1, gal, t, 1) == recall_at_k(d0, gal, t, 1) == 1.0
    assert kendall_tau(d1[:, 0], d0[:, 0]) == 1.0


def test_r0_is_frozen_random_adapter():
    r0 = arms.R0()
    assert r0.trainable_parameters() == 0


def test_checkpoint_selection_rules():
    recs = [
        CheckpointRecord(1, 0.50, 1.0),
        CheckpointRecord(2, 0.55, 0.9),
        CheckpointRecord(3, 0.55, 0.8),
        CheckpointRecord(4, 0.55, 0.8),
    ]
    best = select_checkpoint(recs)
    assert best.epoch == 3  # 同 recall、同 loss → 更早 epoch
    with pytest.raises(ValueError):
        select_checkpoint([])


def test_early_stopping_patience_three():
    es = EarlyStopping(patience=3)
    assert not es.update(1.0)
    assert not es.update(1.1)
    assert not es.update(1.2)
    assert es.update(1.3)
    assert es.stopped


def test_vicreg_rule_two_consecutive_epochs():
    st = VicRegState()
    assert not st.update(1, 0.59)
    assert st.update(2, 0.58)
    assert st.active and st.activation_epoch == 3


def test_mask_seed_rule():
    assert mask_seed_for(2026, 5) == 2026 * 1000 + 5
