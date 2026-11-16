"""六个实验臂：F0 / A1 / A2 / B2 / C2 / R0。

所有臂的检索 readout 均为 384 维 CLS（L2 归一化见 retrieval.py）。
Teacher 特征一律视为常量（stop-gradient），不参与梯度。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Adapter, B2Predictor, Predictor

D = 384


def latent_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target.detach(), reduction="mean")


def cosine_loss(q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(q, t.detach(), dim=-1)).mean()


class Arm(nn.Module):
    name = "base"

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def eval_cls(self, teacher_cls: torch.Tensor, teacher_patch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class F0(Arm):
    name = "F0"

    def eval_cls(self, teacher_cls, teacher_patch):
        return teacher_cls


class R0(Arm):
    """随机初始化 adapter，冻结，不训练。"""

    name = "R0"

    def __init__(self):
        super().__init__()
        self.adapter = Adapter()
        for p in self.parameters():
            p.requires_grad_(False)

    def eval_cls(self, teacher_cls, teacher_patch):
        b, n, _ = teacher_patch.shape
        idx = torch.arange(n, device=teacher_patch.device)[None].expand(b, -1)
        empty = torch.zeros(b, 0, dtype=torch.long, device=teacher_patch.device)
        seq, _ = self.adapter(teacher_cls, teacher_patch, idx, empty)
        return seq[:, 0]


class MaskedLatentArm(Arm):
    """A2 / C2 共用结构；差异仅在 mask 几何。"""

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.adapter = Adapter()
        self.predictor = Predictor()

    def forward(self, batch: dict) -> dict:
        mask = batch["mask"]  # (B,1369) bool
        if int(mask.sum()) == 0:
            return {"loss": batch["teacher_cls"].sum() * 0.0, "pred": None, "target": None}
        student_patch = batch["student_patch"]
        vis_idx = (~mask).nonzero(as_tuple=False)
        rows = vis_idx[:, 0]
        vis_pos = vis_idx[:, 1].view(mask.shape[0], -1)
        mask_pos = mask.nonzero(as_tuple=False)[:, 1].view(mask.shape[0], -1)
        vis_feats = student_patch[rows, vis_pos.reshape(-1)].view(mask.shape[0], vis_pos.shape[1], -1)

        seq, mslice = self.adapter(batch["student_cls"], vis_feats, vis_pos, mask_pos)
        pos_idx = torch.cat([torch.zeros(mask.shape[0], 1, dtype=torch.long, device=mask.device),
                             vis_pos, mask_pos], dim=1)
        pred = self.predictor(seq, pos_idx, mslice)
        target = batch["teacher_patch"][mask.nonzero(as_tuple=False)[:, 0], mask_pos.reshape(-1)]
        target = target.view(mask.shape[0], mask_pos.shape[1], -1)
        loss = latent_l2(pred, target)
        return {"loss": loss, "pred": pred, "target": target}

    def eval_cls(self, teacher_cls, teacher_patch):
        b, n, _ = teacher_patch.shape
        idx = torch.arange(n, device=teacher_patch.device)[None].expand(b, -1)
        empty = torch.zeros(b, 0, dtype=torch.long, device=teacher_patch.device)
        seq, _ = self.adapter(teacher_cls, teacher_patch, idx, empty)
        return seq[:, 0]


class A2(MaskedLatentArm):
    def __init__(self):
        super().__init__("A2")


class C2(MaskedLatentArm):
    def __init__(self):
        super().__init__("C2")


class A1(Arm):
    """adapter = Identity；predictor-only；检索 CLS 与 F0 逐位相同。"""

    name = "A1"

    def __init__(self):
        super().__init__()
        self.predictor = Predictor()

    def forward(self, batch: dict) -> dict:
        mask = batch["mask"]
        if int(mask.sum()) == 0:
            return {"loss": batch["teacher_cls"].sum() * 0.0, "pred": None, "target": None}
        teacher_patch = batch["teacher_patch"]
        b, n, _ = teacher_patch.shape
        idx = torch.arange(n, device=teacher_patch.device)[None].expand(b, -1)
        seq = torch.cat([batch["teacher_cls"][:, None, :], teacher_patch], dim=1)
        pos_idx = torch.cat([torch.zeros(b, 1, dtype=torch.long, device=teacher_patch.device), idx], dim=1)
        mask_pos = mask.nonzero(as_tuple=False)[:, 1].view(b, -1)
        rows = mask.nonzero(as_tuple=False)[:, 0]
        pred_full = self.predictor(seq, pos_idx, slice(0, 1 + n))
        pred = pred_full[rows, mask_pos.reshape(-1)].view(b, mask_pos.shape[1], -1)
        target = teacher_patch[rows, mask_pos.reshape(-1)].view(b, mask_pos.shape[1], -1)
        return {"loss": latent_l2(pred, target), "pred": pred, "target": target}

    def eval_cls(self, teacher_cls, teacher_patch):
        return teacher_cls


class B2(Arm):
    """跨视图自蒸馏：q 768 维，target = [CLS2; mean(Patch2)] 768 维。"""

    name = "B2"

    def __init__(self):
        super().__init__()
        self.adapter = Adapter()
        self.predictor = B2Predictor()

    def forward(self, batch: dict) -> dict:
        teacher_patch = batch["teacher_patch"]
        b, n, _ = teacher_patch.shape
        idx = torch.arange(n, device=teacher_patch.device)[None].expand(b, -1)
        empty = torch.zeros(b, 0, dtype=torch.long, device=teacher_patch.device)
        seq, _ = self.adapter(batch["teacher_cls"], teacher_patch, idx, empty)
        q = self.predictor(seq[:, 0])
        t = torch.cat([batch["teacher_cls2"], batch["teacher_patch2"].mean(dim=1)], dim=-1)
        return {"loss": cosine_loss(q, t), "pred": q, "target": t}

    def eval_cls(self, teacher_cls, teacher_patch):
        b, n, _ = teacher_patch.shape
        idx = torch.arange(n, device=teacher_patch.device)[None].expand(b, -1)
        empty = torch.zeros(b, 0, dtype=torch.long, device=teacher_patch.device)
        seq, _ = self.adapter(teacher_cls, teacher_patch, idx, empty)
        return seq[:, 0]


def build_arm(name: str) -> Arm:
    name = name.upper()
    return {"F0": F0, "A1": A1, "A2": A2, "B2": B2, "C2": C2, "R0": R0}[name]()
